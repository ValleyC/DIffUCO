"""
Evaluate trained chip placement model and visualize results

Usage:
    python eval_and_visualize.py --checkpoint Checkpoints/90r9zq60/best_90r9zq60.pickle --dataset Chip_small --n_samples 5
"""

import sys
import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

sys.path.append(".")

from train import TrainMeanField


def load_checkpoint(checkpoint_path):
    """Load trained model checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def load_test_data(dataset_name="Chip_small", mode="test"):
    """Load test dataset"""
    data_path = f"DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_ChipPlacement_seed_123_solutions.pickle"

    print(f"Loading test data from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {len(data['H_graphs'])} instances")
    return data


def compute_hpwl(positions, graph):
    """Compute Half-Perimeter Wirelength"""
    senders = graph.senders
    receivers = graph.receivers

    total_hpwl = 0.0

    for sender, receiver in zip(senders, receivers):
        pos1 = positions[sender]
        pos2 = positions[receiver]

        x_coords = [pos1[0], pos2[0]]
        y_coords = [pos1[1], pos2[1]]

        bbox_width = max(x_coords) - min(x_coords)
        bbox_height = max(y_coords) - min(y_coords)

        total_hpwl += bbox_width + bbox_height

    return total_hpwl


def compute_overlap_penalty(positions, component_sizes):
    """
    Compute overlap penalty between components.

    Args:
        positions: component positions [n_components, 2]
        component_sizes: component sizes [n_components, 2] (width, height)

    Returns:
        total_overlap_area: sum of overlap areas
    """
    n_components = len(positions)
    half_sizes = component_sizes / 2.0

    # Component bounding boxes
    x_min = positions[:, 0] - half_sizes[:, 0]
    y_min = positions[:, 1] - half_sizes[:, 1]
    x_max = positions[:, 0] + half_sizes[:, 0]
    y_max = positions[:, 1] + half_sizes[:, 1]

    total_overlap = 0.0

    # Check all pairs
    for i in range(n_components):
        for j in range(i + 1, n_components):
            # Compute overlap
            overlap_width = max(0.0, min(x_max[i], x_max[j]) - max(x_min[i], x_min[j]))
            overlap_height = max(0.0, min(y_max[i], y_max[j]) - max(y_min[i], y_min[j]))
            overlap_area = overlap_width * overlap_height
            total_overlap += overlap_area

    return total_overlap


def compute_boundary_penalty(positions, component_sizes, canvas_x_min=-1.0, canvas_y_min=-1.0,
                             canvas_width=2.0, canvas_height=2.0):
    """
    Compute boundary violation penalty.

    Args:
        positions: component positions [n_components, 2]
        component_sizes: component sizes [n_components, 2]
        canvas_x_min, canvas_y_min: canvas minimum coordinates
        canvas_width, canvas_height: canvas dimensions

    Returns:
        total_boundary_violation: sum of out-of-bounds areas
    """
    half_sizes = component_sizes / 2.0

    # Component bounding boxes
    x_min = positions[:, 0] - half_sizes[:, 0]
    y_min = positions[:, 1] - half_sizes[:, 1]
    x_max = positions[:, 0] + half_sizes[:, 0]
    y_max = positions[:, 1] + half_sizes[:, 1]

    # Canvas boundaries
    canvas_x_max = canvas_x_min + canvas_width
    canvas_y_max = canvas_y_min + canvas_height

    # Out-of-bounds violations
    left_violation = np.maximum(0.0, canvas_x_min - x_min)
    right_violation = np.maximum(0.0, x_max - canvas_x_max)
    bottom_violation = np.maximum(0.0, canvas_y_min - y_min)
    top_violation = np.maximum(0.0, y_max - canvas_y_max)

    # Approximate out-of-bounds "area"
    x_violation = (left_violation + right_violation) * component_sizes[:, 1]
    y_violation = (bottom_violation + top_violation) * component_sizes[:, 0]

    total_boundary_violation = np.sum(x_violation + y_violation)

    return total_boundary_violation


def visualize_comparison(initial_pos, generated_pos, graph, component_sizes,
                         initial_metrics, generated_metrics, instance_id, save_path=None):
    """
    Visualize initial vs generated placement side-by-side

    Args:
        initial_metrics: dict with 'hpwl', 'overlap', 'boundary'
        generated_metrics: dict with 'hpwl', 'overlap', 'boundary'
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    placements = [initial_pos, generated_pos]
    titles = ["Initial Placement (Random)", "Generated Placement (Trained Model)"]
    metrics = [initial_metrics, generated_metrics]

    canvas_min, canvas_max = -1.0, 1.0

    for ax_idx, (ax, positions, title, metric) in enumerate(zip(axes, placements, titles, metrics)):
        ax.set_xlim(canvas_min - 0.1, canvas_max + 0.1)
        ax.set_ylim(canvas_min - 0.1, canvas_max + 0.1)
        ax.set_aspect('equal')
        title_text = f"{title}\n"
        title_text += f"HPWL = {metric['hpwl']:.2f} | "
        title_text += f"Overlap = {metric['overlap']:.2f} | "
        title_text += f"Out-of-Bound = {metric['boundary']:.2f}"
        ax.set_title(title_text, fontsize=12, fontweight='bold')
        ax.set_xlabel('X position', fontsize=12)
        ax.set_ylabel('Y position', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Canvas boundary
        canvas_rect = Rectangle(
            (canvas_min, canvas_min),
            canvas_max - canvas_min,
            canvas_max - canvas_min,
            fill=False,
            edgecolor='black',
            linewidth=2,
            linestyle='--'
        )
        ax.add_patch(canvas_rect)

        # Draw nets first
        senders = graph.senders
        receivers = graph.receivers

        for sender, receiver in zip(senders, receivers):
            x1, y1 = positions[sender]
            x2, y2 = positions[receiver]
            ax.plot([x1, x2], [y1, y2],
                   color='gray', alpha=0.2, linewidth=0.5, zorder=0)

        # Draw components
        n_components = positions.shape[0]
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_components, 20)))
        if n_components > 20:
            colors = np.tile(colors, (n_components // 20 + 1, 1))[:n_components]

        for i in range(n_components):
            x, y = positions[i]
            width, height = component_sizes[i]

            rect = Rectangle(
                (x - width/2, y - height/2),
                width, height,
                facecolor=colors[i],
                edgecolor='black',
                linewidth=1,
                alpha=0.7,
                zorder=1
            )
            ax.add_patch(rect)

            if n_components <= 50:
                ax.text(x, y, str(i), ha='center', va='center',
                       fontsize=max(6, min(10, 200 // n_components)),
                       fontweight='bold', color='white', zorder=2)

    # Compute improvements
    hpwl_improvement = (initial_metrics['hpwl'] - generated_metrics['hpwl']) / max(initial_metrics['hpwl'], 1e-6) * 100
    overlap_improvement = (initial_metrics['overlap'] - generated_metrics['overlap']) / max(initial_metrics['overlap'], 1e-6) * 100 if initial_metrics['overlap'] > 0 else 0
    boundary_improvement = (initial_metrics['boundary'] - generated_metrics['boundary']) / max(initial_metrics['boundary'], 1e-6) * 100 if initial_metrics['boundary'] > 0 else 0

    fig.suptitle(
        f"Instance {instance_id} - {n_components} Components\n"
        f"HPWL: {initial_metrics['hpwl']:.1f} → {generated_metrics['hpwl']:.1f} ({hpwl_improvement:+.1f}%) | "
        f"Overlap: {initial_metrics['overlap']:.2f} → {generated_metrics['overlap']:.2f} ({overlap_improvement:+.1f}%) | "
        f"Out-of-Bound: {initial_metrics['boundary']:.2f} → {generated_metrics['boundary']:.2f} ({boundary_improvement:+.1f}%)",
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    return fig


def evaluate_instance(trainer, instance_data, instance_id):
    """
    Evaluate a single instance using the trained model

    Args:
        trainer: TrainMeanField object with loaded parameters
        instance_data: dict with 'H_graphs', 'positions', etc.
        instance_id: index of instance to evaluate

    Returns:
        result dict with initial and generated placements
    """
    # Get instance data
    graph = instance_data['H_graphs'][instance_id]
    initial_positions = instance_data['positions'][instance_id]
    component_sizes = graph.nodes
    n_components = component_sizes.shape[0]  # Store original number of components

    print(f"\nEvaluating instance {instance_id}...")
    print(f"  Components: {n_components}")

    # Compute initial metrics
    initial_hpwl = compute_hpwl(initial_positions, graph)
    initial_overlap = compute_overlap_penalty(initial_positions, component_sizes)
    initial_boundary = compute_boundary_penalty(initial_positions, component_sizes)

    print(f"  Initial HPWL: {initial_hpwl:.2f}")
    print(f"  Initial Overlap: {initial_overlap:.2f}")
    print(f"  Initial Out-of-Bound: {initial_boundary:.2f}")

    # Prepare batch for model inference
    # Create batch dict in the format expected by trainer._prepare_graphs
    # Note: _prepare_graphs expects lists of graphs (as from the dataloader's collate function)
    batch_dict = {
        "input_graph": [graph],  # Wrap in list
        "energy_graph": [graph]  # Use same graph for both input and energy
    }

    # Use trainer's method to properly prepare and batch graphs for pmap
    graph_dict_batch, energy_graph_batch = trainer._prepare_graphs(batch_dict, mode="val")

    # Generate key
    key = jax.random.PRNGKey(instance_id)
    batched_key = jax.random.split(key, num=len(jax.devices()))

    # Run inference
    try:
        print(f"  Running model inference...")
        loss, (log_dict, _) = trainer.TrainerClass.evaluation_step(
            trainer.params,
            graph_dict_batch,
            energy_graph_batch,
            trainer.T,
            batched_key,
            mode="eval",
            epoch=0,
            epochs=1
        )

        # Extract generated positions
        # log_dict["X_0"] has shape [n_devices, n_nodes_padded, n_basis_states, continuous_dim]
        X_0 = log_dict["X_0"]

        # Take first device, only original components (exclude padding), first basis state
        generated_positions = np.array(X_0[0, :n_components, 0, :])  # [n_components, 2]

        # Compute generated metrics
        generated_hpwl = compute_hpwl(generated_positions, graph)
        generated_overlap = compute_overlap_penalty(generated_positions, component_sizes)
        generated_boundary = compute_boundary_penalty(generated_positions, component_sizes)

        hpwl_improvement = (initial_hpwl - generated_hpwl) / initial_hpwl * 100

        print(f"  Generated HPWL: {generated_hpwl:.2f}")
        print(f"  Generated Overlap: {generated_overlap:.2f}")
        print(f"  Generated Out-of-Bound: {generated_boundary:.2f}")
        print(f"  HPWL Improvement: {hpwl_improvement:.1f}%")

    except Exception as e:
        print(f"  Error during inference: {e}")
        import traceback
        traceback.print_exc()
        generated_positions = initial_positions
        generated_hpwl = initial_hpwl
        generated_overlap = initial_overlap
        generated_boundary = initial_boundary
        hpwl_improvement = 0.0

    # Create metric dictionaries
    initial_metrics = {
        'hpwl': initial_hpwl,
        'overlap': initial_overlap,
        'boundary': initial_boundary
    }

    generated_metrics = {
        'hpwl': generated_hpwl,
        'overlap': generated_overlap,
        'boundary': generated_boundary
    }

    result = {
        'instance_id': instance_id,
        'initial_positions': initial_positions,
        'initial_metrics': initial_metrics,
        'generated_positions': generated_positions,
        'generated_metrics': generated_metrics,
        'graph': graph,
        'component_sizes': component_sizes,
        'hpwl_improvement': hpwl_improvement,
    }

    return result


def print_summary(results):
    """Print summary statistics"""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    initial_hpwls = [r['initial_metrics']['hpwl'] for r in results]
    generated_hpwls = [r['generated_metrics']['hpwl'] for r in results]
    initial_overlaps = [r['initial_metrics']['overlap'] for r in results]
    generated_overlaps = [r['generated_metrics']['overlap'] for r in results]
    initial_boundaries = [r['initial_metrics']['boundary'] for r in results]
    generated_boundaries = [r['generated_metrics']['boundary'] for r in results]
    hpwl_improvements = [r['hpwl_improvement'] for r in results]

    print(f"\nNumber of instances: {len(results)}")

    print(f"\nInitial HPWL (random placement):")
    print(f"  Mean: {np.mean(initial_hpwls):.2f}")
    print(f"  Std:  {np.std(initial_hpwls):.2f}")
    print(f"  Min:  {np.min(initial_hpwls):.2f}")
    print(f"  Max:  {np.max(initial_hpwls):.2f}")

    print(f"\nGenerated HPWL (trained model):")
    print(f"  Mean: {np.mean(generated_hpwls):.2f}")
    print(f"  Std:  {np.std(generated_hpwls):.2f}")
    print(f"  Min:  {np.min(generated_hpwls):.2f}")
    print(f"  Max:  {np.max(generated_hpwls):.2f}")

    print(f"\nHPWL Improvement:")
    print(f"  Mean: {np.mean(hpwl_improvements):.1f}%")
    print(f"  Std:  {np.std(hpwl_improvements):.1f}%")
    print(f"  Min:  {np.min(hpwl_improvements):.1f}%")
    print(f"  Max:  {np.max(hpwl_improvements):.1f}%")

    print(f"\nInitial Overlap (random placement):")
    print(f"  Mean: {np.mean(initial_overlaps):.2f}")
    print(f"  Std:  {np.std(initial_overlaps):.2f}")
    print(f"  Min:  {np.min(initial_overlaps):.2f}")
    print(f"  Max:  {np.max(initial_overlaps):.2f}")

    print(f"\nGenerated Overlap (trained model):")
    print(f"  Mean: {np.mean(generated_overlaps):.2f}")
    print(f"  Std:  {np.std(generated_overlaps):.2f}")
    print(f"  Min:  {np.min(generated_overlaps):.2f}")
    print(f"  Max:  {np.max(generated_overlaps):.2f}")

    print(f"\nInitial Out-of-Bound (random placement):")
    print(f"  Mean: {np.mean(initial_boundaries):.2f}")
    print(f"  Std:  {np.std(initial_boundaries):.2f}")
    print(f"  Min:  {np.min(initial_boundaries):.2f}")
    print(f"  Max:  {np.max(initial_boundaries):.2f}")

    print(f"\nGenerated Out-of-Bound (trained model):")
    print(f"  Mean: {np.mean(generated_boundaries):.2f}")
    print(f"  Std:  {np.std(generated_boundaries):.2f}")
    print(f"  Min:  {np.min(generated_boundaries):.2f}")
    print(f"  Max:  {np.max(generated_boundaries):.2f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate and visualize chip placement')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--dataset', type=str, default='Chip_small',
                       help='Dataset name')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'val'],
                       help='Dataset split')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of instances to evaluate')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save visualizations')
    parser.add_argument('--show', action='store_true',
                       help='Display plots interactively')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint)

    # Create config from checkpoint
    config = checkpoint.get('config', {})
    config['dataset_name'] = args.dataset
    config['wandb'] = False  # Disable wandb for evaluation

    print("\nInitializing trainer...")
    trainer = TrainMeanField(config)

    # Load parameters
    trainer.params = checkpoint['params']
    print("Loaded model parameters")

    # Load test data
    test_data = load_test_data(args.dataset, args.mode)

    # Evaluate instances
    results = []
    n_instances = min(args.n_samples, len(test_data['H_graphs']))

    for i in range(n_instances):
        result = evaluate_instance(trainer, test_data, i)
        results.append(result)

        # Visualize
        save_path = output_dir / f"instance_{i}_comparison.png"
        visualize_comparison(
            result['initial_positions'],
            result['generated_positions'],
            result['graph'],
            result['component_sizes'],
            result['initial_metrics'],
            result['generated_metrics'],
            i,
            save_path=save_path
        )

    # Print summary
    print_summary(results)

    if args.show:
        plt.show()
    else:
        plt.close('all')

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == '__main__':
    main()

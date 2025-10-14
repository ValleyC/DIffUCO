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


def visualize_comparison(initial_pos, generated_pos, graph, component_sizes,
                         initial_hpwl, generated_hpwl,
                         initial_energy, generated_energy,
                         initial_overlap, generated_overlap,
                         initial_boundary, generated_boundary,
                         instance_id, save_path=None):
    """
    Visualize initial vs generated placement side-by-side
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    placements = [initial_pos, generated_pos]
    titles = ["Initial Placement (Random)", "Generated Placement (Trained Model)"]
    hpwls = [initial_hpwl, generated_hpwl]
    energies = [initial_energy, generated_energy]
    overlaps = [initial_overlap, generated_overlap]
    boundaries = [initial_boundary, generated_boundary]

    canvas_min, canvas_max = -1.0, 1.0

    for ax_idx, (ax, positions, title, hpwl, energy, overlap, boundary) in enumerate(
        zip(axes, placements, titles, hpwls, energies, overlaps, boundaries)):
        ax.set_xlim(canvas_min - 0.1, canvas_max + 0.1)
        ax.set_ylim(canvas_min - 0.1, canvas_max + 0.1)
        ax.set_aspect('equal')

        # Enhanced title with all energy components
        title_text = f"{title}\n"
        title_text += f"HPWL = {hpwl:.2f} | "
        title_text += f"Overlap = {overlap:.2f} | "
        title_text += f"Boundary = {boundary:.2f}\n"
        title_text += f"Total Energy = {energy:.2f}"

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

    improvement = (initial_hpwl - generated_hpwl) / initial_hpwl * 100

    fig.suptitle(
        f"Instance {instance_id} - {n_components} Components - "
        f"Improvement: {improvement:.1f}% (HPWL: {initial_hpwl:.1f} → {generated_hpwl:.1f})",
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    return fig


def compute_full_energy(positions, graph, component_sizes, energy_fn):
    """Compute HPWL, overlap, boundary penalties and total energy"""
    node_gr_idx = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    n_graph = 1

    # Compute total energy and components
    energy, _, violations = energy_fn.calculate_Energy(graph, positions, node_gr_idx, component_sizes)

    # Compute individual components
    hpwl = energy_fn._compute_hpwl(graph, positions, node_gr_idx, n_graph)
    overlap = energy_fn._compute_overlap_penalty(positions, component_sizes, node_gr_idx, n_graph)
    boundary = energy_fn._compute_boundary_penalty(positions, component_sizes, node_gr_idx, n_graph)

    return {
        'hpwl': float(hpwl[0]),
        'overlap_penalty': float(overlap[0]),
        'boundary_penalty': float(boundary[0]),
        'total_energy': float(energy[0, 0]),
        'overlap_weighted': float(energy_fn.overlap_weight * overlap[0]),
        'boundary_weighted': float(energy_fn.boundary_weight * boundary[0])
    }


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

    print(f"\nEvaluating instance {instance_id}...")
    print(f"  Components: {component_sizes.shape[0]}")

    # Compute initial metrics
    initial_energy_dict = compute_full_energy(initial_positions, graph, component_sizes, trainer.EnergyClass)
    initial_hpwl = initial_energy_dict['hpwl']

    print(f"  Initial HPWL: {initial_hpwl:.2f}")
    print(f"  Initial Overlap (raw): {initial_energy_dict['overlap_penalty']:.4f}")
    print(f"  Initial Boundary (raw): {initial_energy_dict['boundary_penalty']:.4f}")
    print(f"  Initial Total Energy: {initial_energy_dict['total_energy']:.2f}")

    # Prepare batch for model inference
    # Unwrap params from pmap (take first device replica)
    params_single = jax.tree_util.tree_map(lambda x: x[0] if (isinstance(x, jnp.ndarray) and x.shape[0] == 1) else x, trainer.params)

    # Prepare graph in expected format
    energy_graph = graph
    graph_dict = {"graphs": [graph]}

    # Generate key
    key = jax.random.PRNGKey(instance_id)

    # Run inference
    try:
        print(f"  Running model inference...")

        # Use the non-pmapped sample function directly
        loss, (log_dict, _) = trainer.TrainerClass.sample(
            params_single,
            graph_dict,
            energy_graph,
            trainer.T,
            key
        )

        # Extract generated positions
        # Check different possible locations for X_0
        if "metrics" in log_dict and "X_0" in log_dict["metrics"]:
            X_0 = log_dict["metrics"]["X_0"]
        elif "X_0" in log_dict:
            X_0 = log_dict["X_0"]
        else:
            raise ValueError(f"Cannot find X_0 in log_dict. Keys: {log_dict.keys()}")

        # Debug: print shape
        print(f"  X_0 shape: {X_0.shape}")

        # Handle different possible shapes
        if len(X_0.shape) == 3:
            # [n_nodes, n_basis_states, continuous_dim]
            generated_positions = np.array(X_0[:, 0, :])  # Take first basis state
        elif len(X_0.shape) == 2:
            # [n_nodes, continuous_dim] - already in correct shape
            generated_positions = np.array(X_0)
        else:
            raise ValueError(f"Unexpected X_0 shape: {X_0.shape}")

        print(f"  Generated positions shape: {generated_positions.shape}")

        # Compute generated metrics
        generated_energy_dict = compute_full_energy(generated_positions, graph, component_sizes, trainer.EnergyClass)
        generated_hpwl = generated_energy_dict['hpwl']

        hpwl_improvement = (initial_hpwl - generated_hpwl) / initial_hpwl * 100
        energy_improvement = (initial_energy_dict['total_energy'] - generated_energy_dict['total_energy']) / initial_energy_dict['total_energy'] * 100

        print(f"  Generated HPWL: {generated_hpwl:.2f}")
        print(f"  Generated Overlap (raw): {generated_energy_dict['overlap_penalty']:.4f}")
        print(f"  Generated Boundary (raw): {generated_energy_dict['boundary_penalty']:.4f}")
        print(f"  Generated Total Energy: {generated_energy_dict['total_energy']:.2f}")
        print(f"  HPWL Improvement: {hpwl_improvement:.1f}%")
        print(f"  Energy Improvement: {energy_improvement:.1f}%")

    except Exception as e:
        print(f"  Error during inference: {e}")
        import traceback
        traceback.print_exc()
        generated_positions = initial_positions
        generated_energy_dict = initial_energy_dict
        generated_hpwl = initial_hpwl
        hpwl_improvement = 0.0
        energy_improvement = 0.0

    result = {
        'instance_id': instance_id,
        'initial_positions': initial_positions,
        'initial_hpwl': initial_hpwl,
        'initial_energy': initial_energy_dict['total_energy'],
        'initial_overlap': initial_energy_dict['overlap_penalty'],
        'initial_boundary': initial_energy_dict['boundary_penalty'],
        'generated_positions': generated_positions,
        'generated_hpwl': generated_hpwl,
        'generated_energy': generated_energy_dict['total_energy'],
        'generated_overlap': generated_energy_dict['overlap_penalty'],
        'generated_boundary': generated_energy_dict['boundary_penalty'],
        'graph': graph,
        'component_sizes': component_sizes,
        'hpwl_improvement': hpwl_improvement,
        'energy_improvement': energy_improvement,
    }

    return result


def print_summary(results):
    """Print summary statistics"""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    initial_hpwls = [r['initial_hpwl'] for r in results]
    generated_hpwls = [r['generated_hpwl'] for r in results]
    improvements = [r['hpwl_improvement'] for r in results]
    energy_improvements = [r['energy_improvement'] for r in results]

    initial_overlaps = [r['initial_overlap'] for r in results]
    generated_overlaps = [r['generated_overlap'] for r in results]
    initial_boundaries = [r['initial_boundary'] for r in results]
    generated_boundaries = [r['generated_boundary'] for r in results]

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
    print(f"  Mean: {np.mean(improvements):.1f}%")
    print(f"  Std:  {np.std(improvements):.1f}%")
    print(f"  Min:  {np.min(improvements):.1f}%")
    print(f"  Max:  {np.max(improvements):.1f}%")

    print(f"\nTotal Energy Improvement:")
    print(f"  Mean: {np.mean(energy_improvements):.1f}%")
    print(f"  Std:  {np.std(energy_improvements):.1f}%")

    print(f"\nOverlap Penalties:")
    print(f"  Initial (mean): {np.mean(initial_overlaps):.4f}")
    print(f"  Generated (mean): {np.mean(generated_overlaps):.4f}")
    print(f"  ⚠ Generated overlaps should be NEAR ZERO for valid placements!")

    print(f"\nBoundary Penalties:")
    print(f"  Initial (mean): {np.mean(initial_boundaries):.4f}")
    print(f"  Generated (mean): {np.mean(generated_boundaries):.4f}")
    print(f"  ⚠ Generated boundaries should be NEAR ZERO for valid placements!")

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
            result['initial_hpwl'],
            result['generated_hpwl'],
            result['initial_energy'],
            result['generated_energy'],
            result['initial_overlap'],
            result['generated_overlap'],
            result['initial_boundary'],
            result['generated_boundary'],
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

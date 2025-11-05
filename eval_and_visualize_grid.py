"""
Evaluate trained GRID chip placement model and visualize results

This is the grid-based version of eval_and_visualize.py
Handles discrete grid cell assignments and converts them to continuous positions for visualization.

Usage:
    python eval_and_visualize_grid.py --checkpoint Checkpoints/xxxxx/best_xxxxx.pickle --dataset Chip_small --n_samples 5
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
from chipdiffusion_utils import denormalize_positions_and_sizes


def load_checkpoint(checkpoint_path):
    """Load trained model checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def load_test_data(dataset_name="Chip_small", mode="test"):
    """Load test dataset"""
    # GridChipPlacement uses ChipPlacement datasets (same graph structure)
    data_path = f"DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_ChipPlacement_seed_123_solutions.pickle"

    # Fallback to train mode if test mode doesn't exist
    if not Path(data_path).exists():
        fallback_mode = "train"
        fallback_path = f"DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{fallback_mode}_ChipPlacement_seed_123_solutions.pickle"
        if Path(fallback_path).exists():
            print(f"Warning: {mode} mode not found, falling back to {fallback_mode} mode")
            data_path = fallback_path
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path} or {fallback_path}")

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


def compute_total_energy(positions, graph, component_sizes,
                         overlap_weight=1.0, boundary_weight=1.0):
    """
    Compute total energy = HPWL + overlap_penalty + boundary_penalty
    """
    hpwl = compute_hpwl(positions, graph)
    overlap = compute_overlap_penalty(positions, component_sizes)
    boundary = compute_boundary_penalty(positions, component_sizes)

    total_energy = hpwl + overlap_weight * overlap + boundary_weight * boundary
    return total_energy


def compute_collision_count(grid_indices):
    """
    Count number of grid cell collisions (multiple components in same cell).

    Args:
        grid_indices: [n_components, 1] or [n_components,] grid cell assignments

    Returns:
        n_collisions: number of components involved in collisions
    """
    if len(grid_indices.shape) > 1:
        grid_indices = grid_indices.squeeze()

    unique_cells = np.unique(grid_indices)
    n_collisions = 0

    for cell in unique_cells:
        count = np.sum(grid_indices == cell)
        if count > 1:
            n_collisions += count  # All components in this cell are in collision

    return n_collisions


def select_best_sample(X_0_samples, graph, component_sizes, energy_func):
    """
    Evaluate all K samples and pick the one with lowest energy.

    For grid-based placement:
    - X_0_samples contains discrete grid indices [n_components, n_basis_states, 1]
    - Convert each sample to continuous positions
    - Evaluate energy (HPWL only, since grid guarantees no overlap/boundary violations)

    Args:
        X_0_samples: [n_components, n_basis_states, 1] discrete grid indices
        graph: jraph.GraphsTuple
        component_sizes: [n_components, 2]
        energy_func: GridChipPlacementEnergyClass instance

    Returns:
        best_sample_continuous: [n_components, 2] - continuous positions of best sample
        best_sample_discrete: [n_components, 1] - grid indices of best sample
        best_energy: float
        best_idx: int - Index of best sample
    """
    n_basis_states = X_0_samples.shape[1]

    best_sample_continuous = None
    best_sample_discrete = None
    best_energy = float('inf')
    best_idx = 0

    for k in range(n_basis_states):
        # Get discrete grid indices for this sample
        sample_grid_indices = X_0_samples[:, k, :]  # [n_components, 1]

        # Convert to continuous positions
        sample_positions = energy_func.get_continuous_positions(sample_grid_indices)  # [n_components, 2]

        # Compute energy (HPWL only for grid)
        energy = compute_hpwl(sample_positions, graph)

        if energy < best_energy:
            best_energy = energy
            best_sample_continuous = sample_positions
            best_sample_discrete = sample_grid_indices
            best_idx = k

    return best_sample_continuous, best_sample_discrete, best_energy, best_idx


def visualize_comparison(legal_pos, raw_pos, generated_pos, graph, component_sizes,
                         legal_metrics, raw_metrics, generated_metrics, instance_id, save_path=None):
    """
    Visualize ground truth, raw model output, and final output side-by-side

    For GridChipPlacement:
    - legal_pos: ground truth (continuous)
    - raw_pos: raw model output (continuous, converted from grid indices)
    - generated_pos: same as raw_pos (no decoder needed for grid)

    Args:
        legal_pos: (V, 2) array of legal positions (ground truth)
        raw_pos: (V, 2) array of raw model output (converted from grid indices)
        generated_pos: (V, 2) array of final output (same as raw for grid)
        legal_metrics: dict with 'hpwl', 'overlap', 'boundary' for ground truth
        raw_metrics: dict with 'hpwl', 'overlap', 'boundary', 'collisions' for raw
        generated_metrics: dict with 'hpwl', 'overlap', 'boundary', 'collisions' for final
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    placements = [legal_pos, raw_pos, generated_pos]
    titles = ["Ground Truth (Legal Placement)", "Raw Model Output (Grid Assignment)", "Final Output (Grid Assignment)"]
    metrics = [legal_metrics, raw_metrics, generated_metrics]

    canvas_min, canvas_max = -1.0, 1.0

    for ax_idx, (ax, positions, title, metric) in enumerate(zip(axes, placements, titles, metrics)):
        ax.set_xlim(canvas_min - 0.1, canvas_max + 0.1)
        ax.set_ylim(canvas_min - 0.1, canvas_max + 0.1)
        ax.set_aspect('equal')
        title_text = f"{title}\n"
        title_text += f"HPWL = {metric['hpwl']:.2f} | "
        title_text += f"Overlap = {metric['overlap']:.2f} | "
        title_text += f"Out-of-Bound = {metric['boundary']:.2f}"
        if 'collisions' in metric:
            title_text += f" | Collisions = {metric['collisions']}"
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

    # Compute improvements from raw to decoded (should be same for grid)
    hpwl_improvement = (raw_metrics['hpwl'] - generated_metrics['hpwl']) / max(raw_metrics['hpwl'], 1e-6) * 100
    overlap_improvement = (raw_metrics['overlap'] - generated_metrics['overlap']) / max(raw_metrics['overlap'], 1e-6) * 100 if raw_metrics['overlap'] > 0 else 0
    boundary_improvement = (raw_metrics['boundary'] - generated_metrics['boundary']) / max(raw_metrics['boundary'], 1e-6) * 100 if raw_metrics['boundary'] > 0 else 0

    # How close to ground truth
    hpwl_gap_to_gt = abs(generated_metrics['hpwl'] - legal_metrics['hpwl']) / max(legal_metrics['hpwl'], 1e-6) * 100

    n_components = len(component_sizes)
    collision_info = f"Collisions: {raw_metrics['collisions']}/{n_components} components" if 'collisions' in raw_metrics else ""

    fig.suptitle(
        f"Instance {instance_id} - {n_components} Components (Grid-Based Placement)\n"
        f"HPWL: GT={legal_metrics['hpwl']:.1f} | Grid={generated_metrics['hpwl']:.1f} "
        f"(Gap to GT: {hpwl_gap_to_gt:.1f}%) | {collision_info}\n"
        f"Overlap: GT={legal_metrics['overlap']:.2f} | Grid={generated_metrics['overlap']:.2f}\n"
        f"Out-of-Bound: GT={legal_metrics['boundary']:.2f} | Grid={generated_metrics['boundary']:.2f}",
        fontsize=12,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    return fig


def evaluate_instance(trainer, instance_data, instance_id):
    """
    Evaluate a single instance using the trained GRID model

    Args:
        trainer: TrainMeanField object with loaded parameters
        instance_data: dict with 'H_graphs', 'positions', etc.
        instance_id: index of instance to evaluate

    Returns:
        result dict with initial and generated placements
    """
    # Get instance data
    graph = instance_data['H_graphs'][instance_id]
    initial_positions = instance_data['positions'][instance_id]  # Randomized positions
    legal_positions = instance_data['legal_positions'][instance_id]  # Ground truth
    component_sizes = graph.nodes
    n_components = component_sizes.shape[0]

    # Get chip_size for denormalization (if available)
    chip_size = instance_data.get('chip_sizes', [None])[instance_id] if 'chip_sizes' in instance_data else None
    has_chip_size = chip_size is not None

    # Get energy function instance (for grid-to-continuous conversion)
    energy_func = trainer.EnergyClass

    print(f"\nEvaluating instance {instance_id}...")
    print(f"  Components: {n_components}")
    print(f"  Grid size: {energy_func.grid_width}×{energy_func.grid_height} = {energy_func.n_grid_cells} cells")
    if has_chip_size:
        print(f"  Canvas: {chip_size[2]-chip_size[0]:.1f} x {chip_size[3]-chip_size[1]:.1f} um")

    # Compute legal (ground truth) metrics in NORMALIZED space
    legal_hpwl_norm = compute_hpwl(legal_positions, graph)
    legal_overlap = compute_overlap_penalty(legal_positions, component_sizes)
    legal_boundary = compute_boundary_penalty(legal_positions, component_sizes)

    # Compute REAL-SCALE HPWL for comparison
    if has_chip_size:
        legal_positions_real, legal_sizes_real = denormalize_positions_and_sizes(
            legal_positions, component_sizes, chip_size
        )
        legal_hpwl_real = compute_hpwl(legal_positions_real, graph)
        print(f"  Ground Truth HPWL (normalized): {legal_hpwl_norm:.2f}")
        print(f"  Ground Truth HPWL (real-scale): {legal_hpwl_real:.2f} um")
    else:
        legal_hpwl_real = legal_hpwl_norm
        print(f"  Ground Truth HPWL: {legal_hpwl_norm:.2f}")

    print(f"  Ground Truth Overlap: {legal_overlap:.2f}")
    print(f"  Ground Truth Out-of-Bound: {legal_boundary:.2f}")

    # Compute initial (randomized) metrics
    initial_hpwl = compute_hpwl(initial_positions, graph)
    initial_overlap = compute_overlap_penalty(initial_positions, component_sizes)
    initial_boundary = compute_boundary_penalty(initial_positions, component_sizes)

    print(f"  Initial (Random) HPWL: {initial_hpwl:.2f}")
    print(f"  Initial Overlap: {initial_overlap:.2f}")
    print(f"  Initial Out-of-Bound: {initial_boundary:.2f}")

    # Prepare batch for model inference
    batch_dict = {
        "input_graph": [graph],
        "energy_graph": [graph]
    }

    # Use trainer's method to properly prepare and batch graphs
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

        # Extract generated DISCRETE grid indices
        # For discrete SDDS (grid), X_0 has shape [n_devices, n_nodes_padded, n_basis_states, 1]
        # where the last dimension is the discrete grid cell index
        X_0 = log_dict["X_0"]

        # Extract ALL K samples from first device, only original components
        X_0_samples = np.array(X_0[0, :n_components, :, :])  # [n_components, n_basis_states, 1]

        print(f"  Model generated {X_0_samples.shape[1]} samples per component")
        print(f"  Sample shape: {X_0_samples.shape} (discrete grid indices)")

        # Step 1: Select best sample by energy
        best_sample_continuous, best_sample_discrete, best_energy, best_idx = select_best_sample(
            X_0_samples, graph, component_sizes, energy_func
        )
        print(f"  Best sample: #{best_idx} with energy {best_energy:.2f}")

        # Compute collision statistics for best sample
        n_collisions = compute_collision_count(best_sample_discrete)
        print(f"  Grid collisions: {n_collisions}/{n_components} components in same cells")

        # Compute raw model output metrics
        raw_hpwl_norm = compute_hpwl(best_sample_continuous, graph)
        raw_overlap = compute_overlap_penalty(best_sample_continuous, component_sizes)
        raw_boundary = compute_boundary_penalty(best_sample_continuous, component_sizes)

        # Compute real-scale HPWL
        if has_chip_size:
            raw_positions_real, _ = denormalize_positions_and_sizes(best_sample_continuous, component_sizes, chip_size)
            raw_hpwl_real = compute_hpwl(raw_positions_real, graph)
            print(f"  Grid Model HPWL (normalized): {raw_hpwl_norm:.2f}")
            print(f"  Grid Model HPWL (real-scale): {raw_hpwl_real:.2f} um")
        else:
            raw_hpwl_real = raw_hpwl_norm
            print(f"  Grid Model HPWL: {raw_hpwl_norm:.2f}")

        print(f"  Grid Model Overlap: {raw_overlap:.2f}")
        print(f"  Grid Model Out-of-Bound: {raw_boundary:.2f}")

        # Store raw model output
        raw_positions = best_sample_continuous.copy()

        # For grid placement: NO DECODER needed!
        # Grid guarantees feasibility (if no collisions)
        # Raw output = Final output
        print(f"  No decoder needed for grid placement (grid guarantees feasibility)")
        generated_positions = best_sample_continuous.copy()
        generated_hpwl_norm = raw_hpwl_norm
        generated_overlap = raw_overlap
        generated_boundary = raw_boundary
        generated_hpwl_real = raw_hpwl_real

        print(f"  Final HPWL (normalized): {generated_hpwl_norm:.2f}")
        if has_chip_size:
            print(f"  Final HPWL (real-scale): {generated_hpwl_real:.2f} um")
        print(f"  Final Overlap: {generated_overlap:.2f}")
        print(f"  Final Out-of-Bound: {generated_boundary:.2f}")

        hpwl_improvement = (initial_hpwl - generated_hpwl_norm) / initial_hpwl * 100
        print(f"  HPWL Improvement (random→grid): {hpwl_improvement:.1f}%")

        # Report real-scale comparison with ground truth
        if has_chip_size:
            hpwl_vs_gt = ((generated_hpwl_real - legal_hpwl_real) / legal_hpwl_real) * 100
            print(f"  vs Ground Truth (real-scale): {hpwl_vs_gt:+.1f}%")

    except Exception as e:
        print(f"  Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raw_positions = initial_positions
        raw_hpwl_norm = initial_hpwl
        raw_hpwl_real = legal_hpwl_real if has_chip_size else initial_hpwl
        raw_overlap = initial_overlap
        raw_boundary = initial_boundary
        n_collisions = 0
        generated_positions = initial_positions
        generated_hpwl_norm = initial_hpwl
        generated_hpwl_real = legal_hpwl_real if has_chip_size else initial_hpwl
        generated_overlap = initial_overlap
        generated_boundary = initial_boundary
        hpwl_improvement = 0.0

    # Create metric dictionaries
    legal_metrics = {
        'hpwl': legal_hpwl_norm,
        'hpwl_real': legal_hpwl_real if has_chip_size else legal_hpwl_norm,
        'overlap': legal_overlap,
        'boundary': legal_boundary
    }

    initial_metrics = {
        'hpwl': initial_hpwl,
        'overlap': initial_overlap,
        'boundary': initial_boundary
    }

    raw_metrics = {
        'hpwl': raw_hpwl_norm if 'raw_hpwl_norm' in locals() else 0.0,
        'hpwl_real': raw_hpwl_real if has_chip_size and 'raw_hpwl_real' in locals() else 0.0,
        'overlap': raw_overlap if 'raw_overlap' in locals() else 0.0,
        'boundary': raw_boundary if 'raw_boundary' in locals() else 0.0,
        'collisions': n_collisions if 'n_collisions' in locals() else 0,
    }

    generated_metrics = {
        'hpwl': generated_hpwl_norm if 'generated_hpwl_norm' in locals() else 0.0,
        'hpwl_real': generated_hpwl_real if has_chip_size and 'generated_hpwl_real' in locals() else 0.0,
        'overlap': generated_overlap if 'generated_overlap' in locals() else 0.0,
        'boundary': generated_boundary if 'generated_boundary' in locals() else 0.0,
        'collisions': n_collisions if 'n_collisions' in locals() else 0,
    }

    result = {
        'instance_id': instance_id,
        'legal_positions': legal_positions,
        'legal_metrics': legal_metrics,
        'initial_positions': initial_positions,
        'initial_metrics': initial_metrics,
        'raw_positions': raw_positions,
        'raw_metrics': raw_metrics,
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
    print("EVALUATION SUMMARY (Grid-Based Placement)")
    print("=" * 80)

    legal_hpwls = [r['legal_metrics']['hpwl'] for r in results]
    initial_hpwls = [r['initial_metrics']['hpwl'] for r in results]
    generated_hpwls = [r['generated_metrics']['hpwl'] for r in results]
    legal_overlaps = [r['legal_metrics']['overlap'] for r in results]
    initial_overlaps = [r['initial_metrics']['overlap'] for r in results]
    generated_overlaps = [r['generated_metrics']['overlap'] for r in results]
    legal_boundaries = [r['legal_metrics']['boundary'] for r in results]
    initial_boundaries = [r['initial_metrics']['boundary'] for r in results]
    generated_boundaries = [r['generated_metrics']['boundary'] for r in results]
    hpwl_improvements = [r['hpwl_improvement'] for r in results]
    collisions = [r['generated_metrics']['collisions'] for r in results]

    print(f"\nNumber of instances: {len(results)}")

    print(f"\nGround Truth (Legal) HPWL:")
    print(f"  Mean: {np.mean(legal_hpwls):.2f}")
    print(f"  Std:  {np.std(legal_hpwls):.2f}")
    print(f"  Min:  {np.min(legal_hpwls):.2f}")
    print(f"  Max:  {np.max(legal_hpwls):.2f}")

    print(f"\nInitial HPWL (random placement):")
    print(f"  Mean: {np.mean(initial_hpwls):.2f}")
    print(f"  Std:  {np.std(initial_hpwls):.2f}")
    print(f"  Min:  {np.min(initial_hpwls):.2f}")
    print(f"  Max:  {np.max(initial_hpwls):.2f}")

    print(f"\nGenerated HPWL (grid model):")
    print(f"  Mean: {np.mean(generated_hpwls):.2f}")
    print(f"  Std:  {np.std(generated_hpwls):.2f}")
    print(f"  Min:  {np.min(generated_hpwls):.2f}")
    print(f"  Max:  {np.max(generated_hpwls):.2f}")

    print(f"\nHPWL Improvement:")
    print(f"  Mean: {np.mean(hpwl_improvements):.1f}%")
    print(f"  Std:  {np.std(hpwl_improvements):.1f}%")
    print(f"  Min:  {np.min(hpwl_improvements):.1f}%")
    print(f"  Max:  {np.max(hpwl_improvements):.1f}%")

    print(f"\nGrid Collisions:")
    print(f"  Mean: {np.mean(collisions):.1f} components")
    print(f"  Std:  {np.std(collisions):.1f}")
    print(f"  Min:  {np.min(collisions):.0f}")
    print(f"  Max:  {np.max(collisions):.0f}")

    print(f"\nGround Truth (Legal) Overlap:")
    print(f"  Mean: {np.mean(legal_overlaps):.2f}")
    print(f"  Std:  {np.std(legal_overlaps):.2f}")
    print(f"  Min:  {np.min(legal_overlaps):.2f}")
    print(f"  Max:  {np.max(legal_overlaps):.2f}")

    print(f"\nInitial Overlap (random placement):")
    print(f"  Mean: {np.mean(initial_overlaps):.2f}")
    print(f"  Std:  {np.std(initial_overlaps):.2f}")
    print(f"  Min:  {np.min(initial_overlaps):.2f}")
    print(f"  Max:  {np.max(initial_overlaps):.2f}")

    print(f"\nGenerated Overlap (grid model):")
    print(f"  Mean: {np.mean(generated_overlaps):.2f}")
    print(f"  Std:  {np.std(generated_overlaps):.2f}")
    print(f"  Min:  {np.min(generated_overlaps):.2f}")
    print(f"  Max:  {np.max(generated_overlaps):.2f}")

    print(f"\nGround Truth (Legal) Out-of-Bound:")
    print(f"  Mean: {np.mean(legal_boundaries):.2f}")
    print(f"  Std:  {np.std(legal_boundaries):.2f}")
    print(f"  Min:  {np.min(legal_boundaries):.2f}")
    print(f"  Max:  {np.max(legal_boundaries):.2f}")

    print(f"\nInitial Out-of-Bound (random placement):")
    print(f"  Mean: {np.mean(initial_boundaries):.2f}")
    print(f"  Std:  {np.std(initial_boundaries):.2f}")
    print(f"  Min:  {np.min(initial_boundaries):.2f}")
    print(f"  Max:  {np.max(initial_boundaries):.2f}")

    print(f"\nGenerated Out-of-Bound (grid model):")
    print(f"  Mean: {np.mean(generated_boundaries):.2f}")
    print(f"  Std:  {np.std(generated_boundaries):.2f}")
    print(f"  Min:  {np.min(generated_boundaries):.2f}")
    print(f"  Max:  {np.max(generated_boundaries):.2f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate and visualize GRID chip placement')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to GridChipPlacement checkpoint file')
    parser.add_argument('--dataset', type=str, default='Chip_small',
                       help='Dataset name')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'val'],
                       help='Dataset split')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of instances to evaluate')
    parser.add_argument('--output_dir', type=str, default='evaluation_results_grid',
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

    # Verify this is a GridChipPlacement checkpoint
    if config.get('problem_name') != 'GridChipPlacement':
        print(f"WARNING: Checkpoint problem_name is '{config.get('problem_name')}', expected 'GridChipPlacement'")
        print("This script is designed for GridChipPlacement checkpoints only.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # CRITICAL: Preserve exact training config for model architecture
    print(f"\n=== Checkpoint Config ===")
    print(f"problem_name: {config.get('problem_name')}")
    print(f"n_diffusion_steps: {config.get('n_diffusion_steps')}")
    print(f"n_random_node_features: {config.get('n_random_node_features')}")
    print(f"n_bernoulli_features: {config.get('n_bernoulli_features')}")
    print(f"grid_width: {config.get('grid_width')}")
    print(f"grid_height: {config.get('grid_height')}")
    print(f"time_encoding: {config.get('time_encoding')}")

    # Only override dataset and wandb
    config['dataset_name'] = args.dataset
    config['wandb'] = False  # Disable wandb for evaluation

    print("\nInitializing trainer...")
    trainer = TrainMeanField(config)

    # Verify config wasn't overridden
    print(f"\n=== After Trainer Init ===")
    print(f"n_diffusion_steps: {trainer.config.get('n_diffusion_steps')}")
    print(f"n_bernoulli_features: {trainer.config.get('n_bernoulli_features')}")

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
        save_path = output_dir / f"instance_{i}_grid_comparison.png"
        visualize_comparison(
            result['legal_positions'],
            result['raw_positions'],
            result['generated_positions'],
            result['graph'],
            result['component_sizes'],
            result['legal_metrics'],
            result['raw_metrics'],
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

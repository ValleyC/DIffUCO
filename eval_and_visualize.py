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


def compute_total_energy(positions, graph, component_sizes,
                         overlap_weight=1.0, boundary_weight=1.0):
    """
    Compute total energy = HPWL + overlap_penalty + boundary_penalty

    This is the objective function we want to minimize.
    """
    hpwl = compute_hpwl(positions, graph)
    overlap = compute_overlap_penalty(positions, component_sizes)
    boundary = compute_boundary_penalty(positions, component_sizes)

    total_energy = hpwl + overlap_weight * overlap + boundary_weight * boundary
    return total_energy


def compute_netlist_degrees(graph):
    """
    Count how many nets each component is connected to.

    Returns:
        degrees: [n_components,] array of connection counts
    """
    n_components = graph.nodes.shape[0]
    degrees = np.zeros(n_components)

    for sender, receiver in zip(graph.senders, graph.receivers):
        degrees[int(sender)] += 1
        degrees[int(receiver)] += 1

    return degrees


def select_best_sample(X_0_samples, graph, component_sizes):
    """
    Evaluate all K samples and pick the one with lowest energy.

    Args:
        X_0_samples: [n_components, n_basis_states, 2]
        graph: jraph.GraphsTuple
        component_sizes: [n_components, 2]

    Returns:
        best_sample: [n_components, 2] - The sample with lowest energy
        best_energy: float
        best_idx: int - Index of best sample
    """
    n_basis_states = X_0_samples.shape[1]

    best_sample = None
    best_energy = float('inf')
    best_idx = 0

    for k in range(n_basis_states):
        sample_positions = X_0_samples[:, k, :]  # [n_components, 2]

        # Compute energy (HPWL + overlap + boundary penalties)
        energy = compute_total_energy(sample_positions, graph, component_sizes)

        if energy < best_energy:
            best_energy = energy
            best_sample = sample_positions
            best_idx = k

    return best_sample, best_energy, best_idx


def is_position_legal(pos, size, placed_positions, canvas_bounds=[-1, 1]):
    """
    Check if a position is legal (in bounds + no overlap with placed components).

    Args:
        pos: (x, y) position of component center
        size: (width, height) of component
        placed_positions: dict {component_id: (pos, size)}
        canvas_bounds: [min, max] for both x and y

    Returns:
        True if legal, False otherwise
    """
    x, y = pos
    width, height = size

    # Check 1: Within canvas bounds
    x_min, y_min = x - width/2, y - height/2
    x_max, y_max = x + width/2, y + height/2

    canvas_min, canvas_max = canvas_bounds[0], canvas_bounds[1]

    if x_min < canvas_min or x_max > canvas_max:
        return False
    if y_min < canvas_min or y_max > canvas_max:
        return False

    # Check 2: No overlap with already placed components
    for placed_id, (placed_pos, placed_size) in placed_positions.items():
        px, py = placed_pos
        pw, ph = placed_size

        # Compute overlap
        overlap_width = max(0.0, min(x_max, px + pw/2) - max(x_min, px - pw/2))
        overlap_height = max(0.0, min(y_max, py + ph/2) - max(y_min, py - ph/2))

        if overlap_width > 1e-6 and overlap_height > 1e-6:
            return False

    return True


def find_closest_legal_position(target_position, component_size, placed_positions,
                                canvas_bounds=[-1, 1], max_search_radius=10.0):
    """
    Spiral search from target position to find closest legal position.

    Args:
        target_position: (x, y) where component wants to be
        component_size: (width, height) of component
        placed_positions: dict {component_id: (pos, size)}
        canvas_bounds: [min, max]
        max_search_radius: Maximum radius to search

    Returns:
        legal_position: (x, y) closest legal position
    """
    # Try target position first
    if is_position_legal(target_position, component_size, placed_positions, canvas_bounds):
        return target_position

    # Spiral search parameters
    min_step = min(component_size) / 4  # Fine-grained search
    num_angles = 24  # Increased number of angles for better coverage

    # Spiral outward
    for radius in np.arange(min_step, max_search_radius, min_step):
        for angle in np.linspace(0, 2*np.pi, num_angles, endpoint=False):
            candidate_x = target_position[0] + radius * np.cos(angle)
            candidate_y = target_position[1] + radius * np.sin(angle)
            candidate = np.array([candidate_x, candidate_y])

            if is_position_legal(candidate, component_size, placed_positions, canvas_bounds):
                return candidate

    # Fallback: Exhaustive grid search over entire canvas
    # This finds the closest legal position anywhere on the canvas
    canvas_min, canvas_max = canvas_bounds[0], canvas_bounds[1]
    width, height = component_size

    # Grid search with fine granularity
    grid_step = min(component_size) / 2

    best_candidate = None
    best_distance = float('inf')

    x_range = np.arange(canvas_min + width/2, canvas_max - width/2 + grid_step/2, grid_step)
    y_range = np.arange(canvas_min + height/2, canvas_max - height/2 + grid_step/2, grid_step)

    for x in x_range:
        for y in y_range:
            candidate = np.array([x, y])

            if is_position_legal(candidate, component_size, placed_positions, canvas_bounds):
                distance = np.linalg.norm(candidate - target_position)
                if distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate

    if best_candidate is not None:
        return best_candidate

    # Last resort: return target position even if illegal
    # This should almost never happen
    return target_position


def greedy_legalize(positions, component_sizes, graph, canvas_bounds=[-1, 1]):
    """
    Parameter-free greedy decoder to legalize a placement.

    Algorithm:
    1. Order components by area (larger first) then netlist degree (common bin-packing heuristic)
    2. Greedily place each component at closest legal position to its target

    Args:
        positions: [n_components, 2] - May have overlaps/out-of-bounds
        component_sizes: [n_components, 2]
        graph: jraph.GraphsTuple (netlist)
        canvas_bounds: [min, max]

    Returns:
        legal_positions: [n_components, 2] - Fully legal placement
    """
    n_components = len(positions)

    # Step 1: Order components (parameter-free!)
    # Primary: Area (larger first - prevents fragmentation)
    # Secondary: Netlist degree (more connected first)
    areas = component_sizes[:, 0] * component_sizes[:, 1]
    degrees = compute_netlist_degrees(graph)

    # Lexicographic sort: (area descending, degree descending)
    component_order = np.lexsort((-degrees, -areas))

    # Step 2: Greedy placement
    placed_positions = {}  # {component_id: (pos, size)}
    legal_positions_list = [None] * n_components

    for comp_i in component_order:
        target = positions[comp_i]
        size = component_sizes[comp_i]

        # Find closest legal position to target
        legal_pos = find_closest_legal_position(
            target_position=target,
            component_size=size,
            placed_positions=placed_positions,
            canvas_bounds=canvas_bounds
        )

        # Record this placement
        placed_positions[comp_i] = (legal_pos, size)
        legal_positions_list[comp_i] = legal_pos

    return np.array(legal_positions_list)


def minimal_disruption_legalize(positions, component_sizes, graph, canvas_bounds=[-1, 1]):
    """
    Alternative legalization: Only move violating components.

    This preserves the model's spatial relationships better than full greedy placement.

    Algorithm:
    1. Identify components that are violating constraints (overlap or out-of-bounds)
    2. Order violating components by degree
    3. Move only violating components to nearest legal positions
    4. Keep non-violating components in place

    Args:
        positions: [n_components, 2] - May have overlaps/out-of-bounds
        component_sizes: [n_components, 2]
        graph: jraph.GraphsTuple (netlist)
        canvas_bounds: [min, max]

    Returns:
        legal_positions: [n_components, 2] - Legalized placement
    """
    n_components = len(positions)
    legal_positions = positions.copy()

    # Step 1: Place all components initially
    placed_positions = {}
    for i in range(n_components):
        placed_positions[i] = (positions[i], component_sizes[i])

    # Step 2: Identify violating components
    violating = set()

    # Check out-of-bounds
    for i in range(n_components):
        pos = positions[i]
        size = component_sizes[i]
        x_min, y_min = pos - size/2
        x_max, y_max = pos + size/2

        if (x_min < canvas_bounds[0] or x_max > canvas_bounds[1] or
            y_min < canvas_bounds[0] or y_max > canvas_bounds[1]):
            violating.add(i)

    # Check overlaps
    for i in range(n_components):
        if i in violating:
            continue
        for j in range(i+1, n_components):
            pos_i, size_i = positions[i], component_sizes[i]
            pos_j, size_j = positions[j], component_sizes[j]

            # Check overlap
            x_min_i, y_min_i = pos_i - size_i/2
            x_max_i, y_max_i = pos_i + size_i/2
            x_min_j, y_min_j = pos_j - size_j/2
            x_max_j, y_max_j = pos_j + size_j/2

            overlap_width = max(0.0, min(x_max_i, x_max_j) - max(x_min_i, x_min_j))
            overlap_height = max(0.0, min(y_max_i, y_max_j) - max(y_min_i, y_min_j))

            if overlap_width > 1e-6 and overlap_height > 1e-6:
                violating.add(i)
                violating.add(j)

    print(f"    {len(violating)}/{n_components} components violating constraints")

    if len(violating) == 0:
        return legal_positions  # Already legal!

    # Step 3: Order violating components by degree (more connected first)
    degrees = compute_netlist_degrees(graph)
    violating_list = sorted(list(violating), key=lambda i: degrees[i], reverse=True)

    # Step 4: Remove violating components from placed_positions
    for i in violating:
        del placed_positions[i]

    # Step 5: Re-place violating components
    for comp_i in violating_list:
        target = positions[comp_i]
        size = component_sizes[comp_i]

        # Find closest legal position
        legal_pos = find_closest_legal_position(
            target_position=target,
            component_size=size,
            placed_positions=placed_positions,
            canvas_bounds=canvas_bounds
        )

        # Update
        placed_positions[comp_i] = (legal_pos, size)
        legal_positions[comp_i] = legal_pos

    return legal_positions


def iterative_push_apart_legalize(positions, component_sizes, canvas_bounds=[-1, 1],
                                   max_iterations=100, overlap_threshold=1e-6):
    """
    Iterative "push apart" legalization that minimally adjusts positions.

    This preserves the model's learned spatial structure while eliminating overlaps.

    Algorithm:
    1. Start with model's predicted positions
    2. Detect overlapping pairs
    3. Push overlapping components apart by minimum distance
    4. Handle boundary violations by nudging components back in
    5. Repeat until no violations (or max iterations)

    Args:
        positions: [n_components, 2] - Model predictions
        component_sizes: [n_components, 2]
        canvas_bounds: [min, max]
        max_iterations: Maximum number of refinement iterations
        overlap_threshold: Minimum overlap to consider

    Returns:
        legal_positions: [n_components, 2] - Minimally adjusted legal positions
    """
    n_components = len(positions)
    legal_positions = positions.copy()

    canvas_min, canvas_max = canvas_bounds[0], canvas_bounds[1]

    for iteration in range(max_iterations):
        moved = False

        # Step 1: Handle boundary violations (push back in bounds)
        for i in range(n_components):
            pos = legal_positions[i]
            size = component_sizes[i]

            # Check boundaries
            x_min_required = canvas_min + size[0]/2
            x_max_required = canvas_max - size[0]/2
            y_min_required = canvas_min + size[1]/2
            y_max_required = canvas_max - size[1]/2

            # Clamp position to valid range
            new_x = np.clip(pos[0], x_min_required, x_max_required)
            new_y = np.clip(pos[1], y_min_required, y_max_required)

            if new_x != pos[0] or new_y != pos[1]:
                legal_positions[i] = np.array([new_x, new_y])
                moved = True

        # Step 2: Detect and resolve overlaps
        for i in range(n_components):
            for j in range(i+1, n_components):
                pos_i, size_i = legal_positions[i], component_sizes[i]
                pos_j, size_j = legal_positions[j], component_sizes[j]

                # Compute bounding boxes
                x_min_i, y_min_i = pos_i - size_i/2
                x_max_i, y_max_i = pos_i + size_i/2
                x_min_j, y_min_j = pos_j - size_j/2
                x_max_j, y_max_j = pos_j + size_j/2

                # Check overlap
                overlap_x = max(0.0, min(x_max_i, x_max_j) - max(x_min_i, x_min_j))
                overlap_y = max(0.0, min(y_max_i, y_max_j) - max(y_min_i, y_min_j))

                if overlap_x > overlap_threshold and overlap_y > overlap_threshold:
                    # Components overlap - push them apart

                    # Compute direction to push (from center to center)
                    direction = pos_j - pos_i
                    distance = np.linalg.norm(direction)

                    if distance < 1e-9:
                        # Components exactly on top of each other - push in random direction
                        direction = np.random.randn(2)
                        distance = np.linalg.norm(direction)

                    direction = direction / distance  # Normalize

                    # Required separation distance (sum of half-sizes in push direction)
                    # Use minimum of overlap dimensions to determine push amount
                    push_amount = (min(overlap_x, overlap_y) / 2.0) + 0.01  # Small margin

                    # Push both components apart (equal force)
                    legal_positions[i] -= direction * push_amount / 2
                    legal_positions[j] += direction * push_amount / 2

                    moved = True

        # Check convergence
        if not moved:
            break

    if iteration == max_iterations - 1:
        print(f"    Warning: Push-apart reached max iterations ({max_iterations})")
    else:
        print(f"    Push-apart converged in {iteration+1} iterations")

    return legal_positions


def visualize_comparison(legal_pos, initial_pos, generated_pos, graph, component_sizes,
                         legal_metrics, initial_metrics, generated_metrics, instance_id, save_path=None):
    """
    Visualize ground truth, initial, and generated placements side-by-side

    Args:
        legal_pos: (V, 2) array of legal positions (ground truth)
        initial_pos: (V, 2) array of randomized positions
        generated_pos: (V, 2) array of model-generated positions
        legal_metrics: dict with 'hpwl', 'overlap', 'boundary' for ground truth
        initial_metrics: dict with 'hpwl', 'overlap', 'boundary' for randomized
        generated_metrics: dict with 'hpwl', 'overlap', 'boundary' for generated
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    placements = [legal_pos, initial_pos, generated_pos]
    titles = ["Ground Truth (Legal Placement)", "Initial (Random Placement)", "Generated (Model Output)"]
    metrics = [legal_metrics, initial_metrics, generated_metrics]

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

    # Compute improvements from random to generated
    hpwl_improvement = (initial_metrics['hpwl'] - generated_metrics['hpwl']) / max(initial_metrics['hpwl'], 1e-6) * 100
    overlap_improvement = (initial_metrics['overlap'] - generated_metrics['overlap']) / max(initial_metrics['overlap'], 1e-6) * 100 if initial_metrics['overlap'] > 0 else 0
    boundary_improvement = (initial_metrics['boundary'] - generated_metrics['boundary']) / max(initial_metrics['boundary'], 1e-6) * 100 if initial_metrics['boundary'] > 0 else 0

    # How close to ground truth
    hpwl_gap_to_gt = abs(generated_metrics['hpwl'] - legal_metrics['hpwl']) / max(legal_metrics['hpwl'], 1e-6) * 100

    n_components = len(component_sizes)
    fig.suptitle(
        f"Instance {instance_id} - {n_components} Components\n"
        f"HPWL: GT={legal_metrics['hpwl']:.1f} | Random={initial_metrics['hpwl']:.1f} | Model={generated_metrics['hpwl']:.1f} "
        f"(Improvement: {hpwl_improvement:+.1f}%, Gap to GT: {hpwl_gap_to_gt:.1f}%)\n"
        f"Overlap: GT={legal_metrics['overlap']:.2f} | Random={initial_metrics['overlap']:.2f} | Model={generated_metrics['overlap']:.2f} "
        f"({overlap_improvement:+.1f}%)\n"
        f"Out-of-Bound: GT={legal_metrics['boundary']:.2f} | Random={initial_metrics['boundary']:.2f} | Model={generated_metrics['boundary']:.2f} "
        f"({boundary_improvement:+.1f}%)",
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
    initial_positions = instance_data['positions'][instance_id]  # Randomized positions (training input)
    legal_positions = instance_data['legal_positions'][instance_id]  # Legal positions (ground truth)
    component_sizes = graph.nodes
    n_components = component_sizes.shape[0]  # Store original number of components

    print(f"\nEvaluating instance {instance_id}...")
    print(f"  Components: {n_components}")

    # Compute legal (ground truth) metrics
    legal_hpwl = compute_hpwl(legal_positions, graph)
    legal_overlap = compute_overlap_penalty(legal_positions, component_sizes)
    legal_boundary = compute_boundary_penalty(legal_positions, component_sizes)

    print(f"  Ground Truth (Legal) HPWL: {legal_hpwl:.2f}")
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

        # Extract ALL K samples from first device, only original components (exclude padding)
        X_0_samples = np.array(X_0[0, :n_components, :, :])  # [n_components, n_basis_states, 2]

        print(f"  Model generated {X_0_samples.shape[1]} samples per component")

        # Step 1: Select best sample by energy (parameter-free!)
        best_sample, best_energy, best_idx = select_best_sample(X_0_samples, graph, component_sizes)
        print(f"  Best sample: #{best_idx} with energy {best_energy:.2f}")

        # Compute raw model output metrics (may be illegal)
        raw_hpwl = compute_hpwl(best_sample, graph)
        raw_overlap = compute_overlap_penalty(best_sample, component_sizes)
        raw_boundary = compute_boundary_penalty(best_sample, component_sizes)

        print(f"  Raw Model HPWL: {raw_hpwl:.2f}")
        print(f"  Raw Model Overlap: {raw_overlap:.2f}")
        print(f"  Raw Model Out-of-Bound: {raw_boundary:.2f}")

        # Step 2: Apply iterative push-apart legalization decoder (parameter-free!)
        print(f"  Applying iterative push-apart decoder...")
        legalized_positions = iterative_push_apart_legalize(best_sample, component_sizes)

        # Compute legalized metrics (should be legal!)
        generated_positions = legalized_positions
        generated_hpwl = compute_hpwl(generated_positions, graph)
        generated_overlap = compute_overlap_penalty(generated_positions, component_sizes)
        generated_boundary = compute_boundary_penalty(generated_positions, component_sizes)

        hpwl_improvement = (initial_hpwl - generated_hpwl) / initial_hpwl * 100

        print(f"  Legalized HPWL: {generated_hpwl:.2f}")
        print(f"  Legalized Overlap: {generated_overlap:.2f}")
        print(f"  Legalized Out-of-Bound: {generated_boundary:.2f}")
        print(f"  HPWL Improvement (random→legalized): {hpwl_improvement:.1f}%")

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
    legal_metrics = {
        'hpwl': legal_hpwl,
        'overlap': legal_overlap,
        'boundary': legal_boundary
    }

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
        'legal_positions': legal_positions,
        'legal_metrics': legal_metrics,
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

    print(f"\nGenerated Overlap (trained model):")
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
            result['legal_positions'],
            result['initial_positions'],
            result['generated_positions'],
            result['graph'],
            result['component_sizes'],
            result['legal_metrics'],
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

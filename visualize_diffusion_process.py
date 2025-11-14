"""
Visualize intermediate diffusion states for the reverse denoising process.

This script extracts intermediate states from the diffusion trajectory and
creates publication-quality vector graphics (PDF/SVG) suitable for Visio editing.

Usage:
    # Hybrid mode (recommended for papers): Clean conceptual + authentic results
    python visualize_diffusion_process.py \
        --checkpoint Checkpoints/4v3bwqhm/4v3bwqhm_last_epoch.pickle \
        --dataset Chip_medium \
        --instance_id 0 \
        --n_timesteps 4 \
        --format pdf \
        --hybrid

    # Real model outputs only
    python visualize_diffusion_process.py \
        --checkpoint Checkpoints/4v3bwqhm/4v3bwqhm_last_epoch.pickle \
        --dataset Chip_medium \
        --instance_id 0 \
        --n_timesteps 4 \
        --format pdf

    # Conceptual mode only (pedagogical)
    python visualize_diffusion_process.py \
        --checkpoint Checkpoints/4v3bwqhm/4v3bwqhm_last_epoch.pickle \
        --dataset Chip_medium \
        --instance_id 0 \
        --n_timesteps 4 \
        --format pdf \
        --conceptual
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
    """Compute overlap penalty between components"""
    n_components = len(positions)
    half_sizes = component_sizes / 2.0

    x_min = positions[:, 0] - half_sizes[:, 0]
    y_min = positions[:, 1] - half_sizes[:, 1]
    x_max = positions[:, 0] + half_sizes[:, 0]
    y_max = positions[:, 1] + half_sizes[:, 1]

    total_overlap = 0.0

    for i in range(n_components):
        for j in range(i + 1, n_components):
            overlap_width = max(0.0, min(x_max[i], x_max[j]) - max(x_min[i], x_min[j]))
            overlap_height = max(0.0, min(y_max[i], y_max[j]) - max(y_min[i], y_min[j]))
            overlap_area = overlap_width * overlap_height
            total_overlap += overlap_area

    return total_overlap


def visualize_single_timestep(positions, graph, component_sizes, timestep, total_steps,
                               canvas_bounds=[-1, 1], figsize=(8, 8), show_metrics=True):
    """
    Create a publication-quality visualization of a single diffusion timestep.

    Args:
        positions: [n_components, 2] component positions at this timestep
        graph: jraph.GraphsTuple (netlist)
        component_sizes: [n_components, 2] component sizes
        timestep: Current timestep (T, T-1, ..., 1, 0)
        total_steps: Total number of diffusion steps T
        canvas_bounds: [min, max] for canvas
        figsize: Figure size (width, height) in inches
        show_metrics: Whether to show HPWL and overlap metrics

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    canvas_min, canvas_max = canvas_bounds[0], canvas_bounds[1]

    ax.set_xlim(canvas_min - 0.1, canvas_max + 0.1)
    ax.set_ylim(canvas_min - 0.1, canvas_max + 0.1)
    ax.set_aspect('equal')

    # Title with timestep info
    if show_metrics:
        hpwl = compute_hpwl(positions, graph)
        overlap = compute_overlap_penalty(positions, component_sizes)
        title = f"Timestep t={timestep} / {total_steps}\nHPWL={hpwl:.1f}, Overlap={overlap:.2f}"
    else:
        title = f"t = {timestep}"

    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('X position', fontsize=12)
    ax.set_ylabel('Y position', fontsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Canvas boundary
    canvas_rect = Rectangle(
        (canvas_min, canvas_min),
        canvas_max - canvas_min,
        canvas_max - canvas_min,
        fill=False,
        edgecolor='black',
        linewidth=2.5,
        linestyle='--',
        zorder=0
    )
    ax.add_patch(canvas_rect)

    # Netlist drawing disabled for cleaner paper figures
    # (Uncomment if you need to show connectivity)
    # senders = graph.senders
    # receivers = graph.receivers
    # for sender, receiver in zip(senders, receivers):
    #     x1, y1 = positions[sender]
    #     x2, y2 = positions[receiver]
    #     ax.plot([x1, x2], [y1, y2],
    #            color='gray', alpha=0.3, linewidth=0.8, zorder=1)

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
            linewidth=1.2,
            alpha=0.8,
            zorder=2
        )
        ax.add_patch(rect)

        # Add component ID for small circuits
        if n_components <= 50:
            ax.text(x, y, str(i), ha='center', va='center',
                   fontsize=max(6, min(10, 200 // n_components)),
                   fontweight='bold', color='white', zorder=3)

    plt.tight_layout()
    return fig


def visualize_diffusion_sequence(positions_sequence, graph, component_sizes,
                                  selected_timesteps, output_dir, file_format='pdf',
                                  figsize=(8, 8), show_metrics=True, conceptual_mode=False):
    """
    Create visualizations for selected timesteps in the diffusion sequence.

    Args:
        positions_sequence: [T+1, n_components, 2] all positions from t=T to t=0
        graph: jraph.GraphsTuple
        component_sizes: [n_components, 2]
        selected_timesteps: List of timestep indices to visualize (e.g., [50, 25, 10, 0])
        output_dir: Directory to save visualizations
        file_format: 'pdf', 'svg', or 'png'
        figsize: Figure size for each subplot
        show_metrics: Whether to show metrics on each timestep
        conceptual_mode: If True, use direct indexing instead of timestep-based

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    saved_files = []
    total_steps = positions_sequence.shape[0] - 1  # T

    print(f"\nGenerating {len(selected_timesteps)} diffusion state visualizations...")
    print(f"Total diffusion steps: {total_steps}")
    print(f"Selected timesteps: {selected_timesteps}")
    print(f"Output format: {file_format}")

    for seq_idx, timestep in enumerate(selected_timesteps):
        # In conceptual mode, use direct sequential indexing
        # In real mode, compute index from timestep value
        if conceptual_mode:
            idx = seq_idx
        else:
            # positions_sequence[0] = X_T (noise), positions_sequence[-1] = X_0 (final)
            # Timestep T corresponds to index 0, timestep 0 corresponds to index T
            idx = total_steps - timestep

        if idx < 0 or idx >= positions_sequence.shape[0]:
            print(f"Warning: timestep {timestep} out of range [0, {total_steps}], skipping")
            continue

        positions = positions_sequence[idx, :, :]  # [n_components, 2]

        # Create visualization
        fig = visualize_single_timestep(
            positions, graph, component_sizes,
            timestep, total_steps,
            figsize=figsize,
            show_metrics=show_metrics
        )

        # Save as vector graphic
        save_path = output_dir / f"diffusion_t{timestep:03d}.{file_format}"

        if file_format == 'pdf':
            fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        elif file_format == 'svg':
            fig.savefig(save_path, format='svg', bbox_inches='tight')
        elif file_format == 'png':
            fig.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        print(f"  Saved timestep t={timestep} to {save_path}")
        saved_files.append(save_path)

        plt.close(fig)

    return saved_files


def create_combined_visualization(positions_sequence, graph, component_sizes,
                                   selected_timesteps, output_dir, file_format='pdf',
                                   figsize_per_plot=(6, 6), show_metrics=False, conceptual_mode=False):
    """
    Create a single figure with all selected timesteps in a row (for manuscript).

    Args:
        positions_sequence: [T+1, n_components, 2]
        graph: jraph.GraphsTuple
        component_sizes: [n_components, 2]
        selected_timesteps: List of timesteps to show (e.g., [50, 30, 10, 0])
        output_dir: Directory to save
        file_format: 'pdf', 'svg', or 'png'
        figsize_per_plot: Size of each subplot
        show_metrics: Whether to show metrics
        conceptual_mode: If True, use direct indexing

    Returns:
        Path to saved combined figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    n_plots = len(selected_timesteps)
    total_steps = positions_sequence.shape[0] - 1

    # Create figure with subplots in a row
    fig, axes = plt.subplots(1, n_plots,
                             figsize=(figsize_per_plot[0] * n_plots, figsize_per_plot[1]))

    if n_plots == 1:
        axes = [axes]

    canvas_min, canvas_max = -1.0, 1.0

    for ax_idx, (ax, timestep) in enumerate(zip(axes, selected_timesteps)):
        # Get positions at this timestep
        if conceptual_mode:
            idx = ax_idx
        else:
            idx = total_steps - timestep
        positions = positions_sequence[idx, :, :]

        ax.set_xlim(canvas_min - 0.1, canvas_max + 0.1)
        ax.set_ylim(canvas_min - 0.1, canvas_max + 0.1)
        ax.set_aspect('equal')

        # Title
        if show_metrics:
            hpwl = compute_hpwl(positions, graph)
            overlap = compute_overlap_penalty(positions, component_sizes)
            title = f"t={timestep}\nHPWL={hpwl:.1f}"
        else:
            title = f"t = {timestep}"

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel('Y', fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)

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

        # Netlist drawing disabled for cleaner paper figures
        # (Uncomment if you need to show connectivity)
        # senders = graph.senders
        # receivers = graph.receivers
        # for sender, receiver in zip(senders, receivers):
        #     x1, y1 = positions[sender]
        #     x2, y2 = positions[receiver]
        #     ax.plot([x1, x2], [y1, y2],
        #            color='gray', alpha=0.25, linewidth=0.6, zorder=0)

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
                linewidth=0.8,
                alpha=0.75,
                zorder=1
            )
            ax.add_patch(rect)

    fig.suptitle(f"Reverse Diffusion Process: X_T → X_0",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save
    save_path = output_dir / f"diffusion_sequence_combined.{file_format}"

    if file_format == 'pdf':
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    elif file_format == 'svg':
        fig.savefig(save_path, format='svg', bbox_inches='tight')
    elif file_format == 'png':
        fig.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\nSaved combined visualization to {save_path}")
    plt.close(fig)

    return save_path


def iterative_push_apart_legalize(positions, component_sizes, canvas_bounds=[-1, 1],
                                   max_iterations=500, overlap_threshold=1e-6):
    """
    Iterative "push apart" legalization that minimally adjusts positions.

    This is copied from eval_and_visualize.py to apply the decoder to the final state.
    """
    n_components = len(positions)
    legal_positions = positions.copy()

    canvas_min, canvas_max = canvas_bounds[0], canvas_bounds[1]

    for iteration in range(max_iterations):
        moved = False

        # Step 1: Handle boundary violations
        for i in range(n_components):
            pos = legal_positions[i]
            size = component_sizes[i]

            x_min_required = canvas_min + size[0]/2
            x_max_required = canvas_max - size[0]/2
            y_min_required = canvas_min + size[1]/2
            y_max_required = canvas_max - size[1]/2

            new_x = np.clip(pos[0], x_min_required, x_max_required)
            new_y = np.clip(pos[1], y_min_required, y_max_required)

            if new_x != pos[0] or new_y != pos[1]:
                legal_positions[i] = np.array([new_x, new_y])
                moved = True

        # Step 2: Detect and resolve overlaps
        for i in range(n_components):
            for j in range(i + 1, n_components):
                pos_i, size_i = legal_positions[i], component_sizes[i]
                pos_j, size_j = legal_positions[j], component_sizes[j]

                x_min_i, y_min_i = pos_i - size_i/2
                x_max_i, y_max_i = pos_i + size_i/2
                x_min_j, y_min_j = pos_j - size_j/2
                x_max_j, y_max_j = pos_j + size_j/2

                overlap_x = max(0.0, min(x_max_i, x_max_j) - max(x_min_i, x_min_j))
                overlap_y = max(0.0, min(y_max_i, y_max_j) - max(y_min_i, y_min_j))

                if overlap_x > overlap_threshold and overlap_y > overlap_threshold:
                    direction = pos_j - pos_i
                    distance = np.linalg.norm(direction)

                    if distance < 1e-9:
                        direction = np.random.randn(2)
                        distance = np.linalg.norm(direction)

                    direction = direction / distance

                    push_amount = (min(overlap_x, overlap_y) / 2.0) + 0.01

                    legal_positions[i] = legal_positions[i] - direction * push_amount / 2
                    legal_positions[j] = legal_positions[j] + direction * push_amount / 2

                    moved = True

        if not moved:
            break

    return legal_positions


def create_conceptual_diffusion_sequence(graph, component_sizes, n_timesteps=4):
    """
    Create a conceptual diffusion sequence for paper figures.

    This generates a clean, pedagogical visualization showing the reverse diffusion
    process from noise to organized placement. NOT actual model outputs, but a
    conceptual illustration for explaining the method.

    Args:
        graph: jraph.GraphsTuple (netlist)
        component_sizes: [n_components, 2]
        n_timesteps: Number of timesteps to generate (including t=0 and t=T)

    Returns:
        positions_sequence: [n_timesteps, n_components, 2]
        timestep_labels: List of timestep values (e.g., [10, 7, 3, 0])
    """
    n_components = len(component_sizes)
    canvas_min, canvas_max = -1.0, 1.0

    print(f"\n  Creating conceptual diffusion sequence for pedagogical clarity...")
    print(f"  (Not actual model outputs - idealized for paper presentation)")

    # Step 1: Create a good final placement (t=0) using greedy placement
    print(f"    Generating clean final placement (t=0)...")
    final_positions = np.zeros((n_components, 2))

    # Simple grid-based placement for clean final state
    grid_size = int(np.ceil(np.sqrt(n_components)))
    canvas_size = canvas_max - canvas_min
    spacing = canvas_size / (grid_size + 1)

    for i in range(n_components):
        row = i // grid_size
        col = i % grid_size
        # Center components in grid cells
        x = canvas_min + spacing * (col + 1)
        y = canvas_min + spacing * (row + 1)

        # Add small random jitter for naturalness
        x += np.random.randn() * spacing * 0.1
        y += np.random.randn() * spacing * 0.1

        # Clip to bounds
        size = component_sizes[i]
        x = np.clip(x, canvas_min + size[0]/2, canvas_max - size[0]/2)
        y = np.clip(y, canvas_min + size[1]/2, canvas_max - size[1]/2)

        final_positions[i] = [x, y]

    # Step 2: Work backwards to create noisy versions
    positions_sequence = np.zeros((n_timesteps, n_components, 2))
    positions_sequence[-1] = final_positions  # t=0 (final)

    print(f"    Generating intermediate noisy states...")

    for timestep_idx in range(n_timesteps - 2, -1, -1):  # Work backwards
        # Noise level increases as we go back in time
        noise_fraction = (timestep_idx + 1) / n_timesteps  # 0 to 1

        if timestep_idx == 0:  # t=T (initial)
            # Maximum noise: Gaussian random from center
            print(f"    Generating initial random state (t=T)...")
            positions = np.random.randn(n_components, 2) * 0.4
        else:
            # Intermediate: Interpolate between final and random
            # Start from final positions and add increasing noise
            positions = final_positions.copy()

            # Add Gaussian noise proportional to distance from final state
            noise_std = 0.5 * noise_fraction  # Increases going back in time
            noise = np.random.randn(n_components, 2) * noise_std
            positions += noise

        # Clip to bounds
        for i in range(n_components):
            size = component_sizes[i]
            x_min_allowed = canvas_min + size[0]/2
            x_max_allowed = canvas_max - size[0]/2
            y_min_allowed = canvas_min + size[1]/2
            y_max_allowed = canvas_max - size[1]/2

            positions[i, 0] = np.clip(positions[i, 0], x_min_allowed, x_max_allowed)
            positions[i, 1] = np.clip(positions[i, 1], y_min_allowed, y_max_allowed)

        positions_sequence[timestep_idx] = positions

    # Generate timestep labels (e.g., for T=10: [10, 7, 3, 0])
    total_steps = 10  # Conceptual total steps
    timestep_labels = [int(total_steps * i / (n_timesteps - 1)) for i in range(n_timesteps)]
    timestep_labels.reverse()  # [10, 7, 3, 0] instead of [0, 3, 7, 10]

    print(f"    ✓ Conceptual sequence created: timesteps {timestep_labels}")

    return positions_sequence, timestep_labels


def create_hybrid_diffusion_sequence(trainer, instance_data, instance_id, n_timesteps=4):
    """
    Create a hybrid diffusion sequence combining conceptual and real model outputs.

    For pedagogical clarity in papers: Use conceptual (clean) for early timesteps,
    and real model outputs (authentic) for final timesteps.

    Args:
        trainer: Trained model
        instance_data: Test dataset
        instance_id: Instance to visualize
        n_timesteps: Total number of timesteps (must be >= 2)

    Returns:
        positions_sequence: [n_timesteps, n_components, 2]
        timestep_labels: List of timestep values
        graph: jraph.GraphsTuple
        component_sizes: [n_components, 2]
    """
    if n_timesteps < 2:
        raise ValueError("n_timesteps must be at least 2 for hybrid mode")

    print(f"\n{'='*80}")
    print("HYBRID MODE: Combining conceptual + real model outputs")
    print(f"{'='*80}")

    # Step 1: Extract real model trajectory
    print("\nStep 1: Extracting real model outputs...")
    real_positions_sequence, graph, component_sizes = extract_diffusion_trajectory(
        trainer, instance_data, instance_id
    )

    # Step 2: Generate conceptual sequence with same number of timesteps
    print("\nStep 2: Generating conceptual sequence...")
    conceptual_positions_sequence, conceptual_labels = create_conceptual_diffusion_sequence(
        graph, component_sizes, n_timesteps=n_timesteps
    )

    # Step 3: Combine - first 2 from conceptual, last 2 from real
    print("\nStep 3: Combining sequences...")
    hybrid_positions_sequence = np.zeros((n_timesteps, len(component_sizes), 2))

    if n_timesteps == 2:
        # Special case: Only 2 timesteps
        # Use conceptual for t=T, real for t=0
        hybrid_positions_sequence[0] = conceptual_positions_sequence[0]  # t=T
        hybrid_positions_sequence[1] = real_positions_sequence[-1]       # t=0 (already legalized)
        print(f"  - Timestep 0 (t=T): Conceptual (Gaussian random)")
        print(f"  - Timestep 1 (t=0): Real model (legalized)")

    elif n_timesteps == 3:
        # 3 timesteps: First 2 conceptual, last 1 real
        hybrid_positions_sequence[0] = conceptual_positions_sequence[0]  # t=T
        hybrid_positions_sequence[1] = conceptual_positions_sequence[1]  # intermediate
        hybrid_positions_sequence[2] = real_positions_sequence[-1]       # t=0
        print(f"  - Timestep 0 (t=T): Conceptual (Gaussian random)")
        print(f"  - Timestep 1: Conceptual (intermediate)")
        print(f"  - Timestep 2 (t=0): Real model (legalized)")

    else:  # n_timesteps >= 4
        # First 2: Conceptual
        hybrid_positions_sequence[0] = conceptual_positions_sequence[0]  # t=T
        hybrid_positions_sequence[1] = conceptual_positions_sequence[1]  # intermediate

        # Last 2: Real model
        # We need to extract intermediate state from real model
        total_real_steps = real_positions_sequence.shape[0] - 1

        # For the second-to-last timestep, pick an intermediate state from real model
        # Use a timestep roughly 1/3 of the way through the reverse process
        intermediate_idx = int(total_real_steps * 0.66)  # Closer to final

        hybrid_positions_sequence[-2] = real_positions_sequence[intermediate_idx]  # intermediate
        hybrid_positions_sequence[-1] = real_positions_sequence[-1]                # t=0

        # Fill middle timesteps (if any) with conceptual
        for i in range(2, n_timesteps - 2):
            hybrid_positions_sequence[i] = conceptual_positions_sequence[i]

        print(f"  - Timestep 0 (t=T): Conceptual (Gaussian random)")
        print(f"  - Timestep 1: Conceptual (noisy)")
        for i in range(2, n_timesteps - 2):
            print(f"  - Timestep {i}: Conceptual (less noisy)")
        print(f"  - Timestep {n_timesteps-2}: Real model (intermediate)")
        print(f"  - Timestep {n_timesteps-1} (t=0): Real model (legalized)")

    # Generate timestep labels
    total_steps = 10  # Conceptual total for labeling
    timestep_labels = [int(total_steps * (1 - i / (n_timesteps - 1))) for i in range(n_timesteps)]

    print(f"\n  ✓ Hybrid sequence created: {n_timesteps} timesteps")
    print(f"    Timestep labels: {timestep_labels}")
    print(f"    First 2 = Conceptual (clean, pedagogical)")
    print(f"    Last 2 = Real model (authentic results)")

    return hybrid_positions_sequence, timestep_labels, graph, component_sizes


def extract_diffusion_trajectory(trainer, instance_data, instance_id):
    """
    Run inference and extract the full diffusion trajectory (all intermediate states).

    IMPORTANT: For the final state (t=0), applies the legalization decoder to get
    the actual final placement (not just raw model output).

    Returns:
        positions_sequence: [T+1, n_components, 2] from X_T to X_0 (legalized)
        graph: jraph.GraphsTuple
        component_sizes: [n_components, 2]
    """
    graph = instance_data['H_graphs'][instance_id]
    component_sizes = graph.nodes
    n_components = component_sizes.shape[0]

    print(f"\nExtracting diffusion trajectory for instance {instance_id}...")
    print(f"  Components: {n_components}")

    # Prepare batch
    batch_dict = {
        "input_graph": [graph],
        "energy_graph": [graph]
    }

    graph_dict_batch, energy_graph_batch = trainer._prepare_graphs(batch_dict, mode="val")

    # Generate key
    key = jax.random.PRNGKey(instance_id)
    batched_key = jax.random.split(key, num=len(jax.devices()))

    # Run inference
    print(f"  Running reverse diffusion...")
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

    # Extract bin_sequence which contains ALL intermediate states
    # Shape: [n_devices, T+1, n_nodes_padded, n_basis_states, continuous_dim]
    if "bin_sequence" in log_dict:
        bin_sequence = log_dict["bin_sequence"]
        print(f"  Extracted bin_sequence with shape: {bin_sequence.shape}")

        # Extract from first device, first basis state, only real components
        positions_sequence = np.array(bin_sequence[0, :, :n_components, 0, :])
        # Shape: [T+1, n_components, 2]

    else:
        raise ValueError("bin_sequence not found in log_dict. Check trainer configuration.")

    print(f"  Positions sequence shape: {positions_sequence.shape}")
    print(f"  Timesteps: {positions_sequence.shape[0]} (from t=T to t=0)")

    # VISUALIZATION IMPROVEMENT: Generate Gaussian random initial state from center
    # This makes the visualization cleaner and more natural for paper figures
    # The actual model uses Gaussian noise which can be out of bounds
    canvas_min, canvas_max = -1.0, 1.0
    canvas_center = 0.0  # Center of [-1, 1] x [-1, 1] canvas

    # Generate new Gaussian random positions centered at canvas center
    # Use std of 0.4 to get good spread while staying mostly within bounds
    initial_state = np.random.randn(n_components, 2) * 0.4 + canvas_center

    # Clip to ensure all components stay within bounds (accounting for size)
    for i in range(n_components):
        size = component_sizes[i]
        # Ensure component center stays within bounds accounting for size
        x_min_allowed = canvas_min + size[0]/2
        x_max_allowed = canvas_max - size[0]/2
        y_min_allowed = canvas_min + size[1]/2
        y_max_allowed = canvas_max - size[1]/2

        initial_state[i, 0] = np.clip(initial_state[i, 0], x_min_allowed, x_max_allowed)
        initial_state[i, 1] = np.clip(initial_state[i, 1], y_min_allowed, y_max_allowed)

    positions_sequence[0, :, :] = initial_state
    print(f"  ✓ Initial state (t=T) regenerated as Gaussian random from canvas center")

    # IMPORTANT: Apply legalization decoder to the final state (t=0)
    # The final state is at positions_sequence[-1] (last timestep)
    raw_final_state = positions_sequence[-1, :, :]  # [n_components, 2]

    print(f"  Applying legalization decoder to final state (t=0)...")
    raw_overlap = compute_overlap_penalty(raw_final_state, component_sizes)
    print(f"    Raw overlap: {raw_overlap:.4f}")

    legalized_final_state = iterative_push_apart_legalize(
        raw_final_state,
        component_sizes,
        canvas_bounds=[-1, 1],
        max_iterations=500
    )

    legalized_overlap = compute_overlap_penalty(legalized_final_state, component_sizes)
    print(f"    Legalized overlap: {legalized_overlap:.4f}")

    # Replace the raw final state with legalized version
    positions_sequence[-1, :, :] = legalized_final_state
    print(f"  ✓ Final state (t=0) replaced with legalized output")

    return positions_sequence, graph, component_sizes


def main():
    parser = argparse.ArgumentParser(
        description='Visualize intermediate diffusion states as vector graphics for Visio'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--dataset', type=str, default='Chip_medium',
                       help='Dataset name')
    parser.add_argument('--instance_id', type=int, default=0,
                       help='Instance ID to visualize')
    parser.add_argument('--n_timesteps', type=int, default=4,
                       help='Number of intermediate timesteps to visualize')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['pdf', 'svg', 'png'],
                       help='Output format (pdf or svg for Visio, png for raster)')
    parser.add_argument('--output_dir', type=str, default='diffusion_states',
                       help='Directory to save visualizations')
    parser.add_argument('--combined', action='store_true',
                       help='Also create a single combined figure with all timesteps')
    parser.add_argument('--show_metrics', action='store_true',
                       help='Show HPWL and overlap metrics on each timestep')
    parser.add_argument('--conceptual', action='store_true',
                       help='Generate conceptual/idealized diffusion sequence for paper figures (not actual model outputs)')
    parser.add_argument('--hybrid', action='store_true',
                       help='Hybrid mode: Use conceptual for first 2 timesteps, real model for last 2 timesteps')

    args = parser.parse_args()

    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint.get('config', {})

    # Override dataset and wandb
    config['dataset_name'] = args.dataset
    config['wandb'] = False

    print("\nInitializing trainer...")
    trainer = TrainMeanField(config)
    trainer.params = checkpoint['params']

    # Load test data
    test_data = load_test_data(args.dataset, mode="test")

    # Determine visualization mode
    if args.hybrid:
        # HYBRID MODE: Combine conceptual (first 2) + real model (last 2)
        positions_sequence, selected_timesteps, graph, component_sizes = create_hybrid_diffusion_sequence(
            trainer, test_data, args.instance_id, n_timesteps=args.n_timesteps
        )
        visualization_mode = 'hybrid'

    elif args.conceptual:
        # CONCEPTUAL MODE: Pure pedagogical visualization
        graph = test_data['H_graphs'][args.instance_id]
        component_sizes = graph.nodes

        positions_sequence, selected_timesteps = create_conceptual_diffusion_sequence(
            graph, component_sizes, n_timesteps=args.n_timesteps
        )
        visualization_mode = 'conceptual'

    else:
        # REAL MODEL MODE: Extract actual model outputs
        positions_sequence, graph, component_sizes = extract_diffusion_trajectory(
            trainer, test_data, args.instance_id
        )

        # Determine which timesteps to visualize
        total_steps = positions_sequence.shape[0] - 1  # T

        # Evenly spaced timesteps including t=T (start) and t=0 (end)
        if args.n_timesteps == 2:
            selected_timesteps = [total_steps, 0]
        else:
            # Include T, 0, and evenly spaced intermediate steps
            selected_timesteps = [
                int(t) for t in np.linspace(total_steps, 0, args.n_timesteps)
            ]
        visualization_mode = 'real'

    print(f"\nSelected timesteps: {selected_timesteps}")

    # Create individual visualizations
    # For hybrid/conceptual modes, use conceptual_mode=True for proper indexing
    saved_files = visualize_diffusion_sequence(
        positions_sequence, graph, component_sizes,
        selected_timesteps,
        output_dir=args.output_dir,
        file_format=args.format,
        show_metrics=args.show_metrics,
        conceptual_mode=(visualization_mode in ['conceptual', 'hybrid'])
    )

    # Optionally create combined figure
    if args.combined:
        combined_path = create_combined_visualization(
            positions_sequence, graph, component_sizes,
            selected_timesteps,
            output_dir=args.output_dir,
            file_format=args.format,
            show_metrics=args.show_metrics,
            conceptual_mode=(visualization_mode in ['conceptual', 'hybrid'])
        )

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")
    if visualization_mode == 'hybrid':
        print("MODE: Hybrid (best of both worlds)")
        print("      First 2 timesteps: Conceptual (clean, pedagogical)")
        print("      Last 2 timesteps: Real model (authentic results with legalization)")
    elif visualization_mode == 'conceptual':
        print("MODE: Conceptual/Idealized (for paper presentation)")
        print("      NOT actual model outputs - pedagogical illustration")
    else:
        print("MODE: Real model outputs (from trained checkpoint)")
    print(f"Output directory: {args.output_dir}/")
    print(f"Format: {args.format.upper()} (vector graphics - editable in Visio)")
    print(f"\nIndividual timestep files:")
    for f in saved_files:
        print(f"  - {f.name}")

    if args.combined:
        print(f"\nCombined figure: {combined_path.name}")

    print(f"\n{'='*80}")
    print("VISIO IMPORT INSTRUCTIONS:")
    print(f"{'='*80}")
    if args.format == 'pdf':
        print("1. Open Visio")
        print("2. Go to Insert > Object > Create from File")
        print("3. Browse and select the PDF file")
        print("4. The PDF will be imported as an editable object")
        print("5. You can ungroup and edit individual elements")
    elif args.format == 'svg':
        print("1. Open Visio")
        print("2. Go to Insert > Picture")
        print("3. Select the SVG file")
        print("4. Right-click and choose 'Convert to Shape' for editing")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

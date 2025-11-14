"""
Visualize intermediate diffusion states for the reverse denoising process.

This script extracts intermediate states from the diffusion trajectory and
creates publication-quality vector graphics (PDF/SVG) suitable for Visio editing.

Usage:
    python visualize_diffusion_process.py \
        --checkpoint Checkpoints/4v3bwqhm/4v3bwqhm_last_epoch.pickle \
        --dataset Chip_medium \
        --instance_id 0 \
        --n_timesteps 4 \
        --format pdf
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

    # Draw nets first (in background)
    senders = graph.senders
    receivers = graph.receivers

    for sender, receiver in zip(senders, receivers):
        x1, y1 = positions[sender]
        x2, y2 = positions[receiver]
        ax.plot([x1, x2], [y1, y2],
               color='gray', alpha=0.3, linewidth=0.8, zorder=1)

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
                                  figsize=(8, 8), show_metrics=True):
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

    for timestep in selected_timesteps:
        # positions_sequence[0] = X_T (noise), positions_sequence[-1] = X_0 (final)
        # Timestep T corresponds to index 0, timestep 0 corresponds to index T
        idx = total_steps - timestep

        if idx < 0 or idx >= positions_sequence.shape[0]:
            print(f"Warning: timestep {timestep} out of range [0, {total_steps}], skipping")
            continue

        positions = positions_sequence[idx, :, :, 0]  # [n_components, 2]

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
                                   figsize_per_plot=(6, 6), show_metrics=False):
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
        idx = total_steps - timestep
        positions = positions_sequence[idx, :, :, 0]

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

        # Draw nets
        senders = graph.senders
        receivers = graph.receivers

        for sender, receiver in zip(senders, receivers):
            x1, y1 = positions[sender]
            x2, y2 = positions[receiver]
            ax.plot([x1, x2], [y1, y2],
                   color='gray', alpha=0.25, linewidth=0.6, zorder=0)

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

    # Extract diffusion trajectory
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

    print(f"\nSelected timesteps: {selected_timesteps}")

    # Create individual visualizations
    saved_files = visualize_diffusion_sequence(
        positions_sequence, graph, component_sizes,
        selected_timesteps,
        output_dir=args.output_dir,
        file_format=args.format,
        show_metrics=args.show_metrics
    )

    # Optionally create combined figure
    if args.combined:
        combined_path = create_combined_visualization(
            positions_sequence, graph, component_sizes,
            selected_timesteps,
            output_dir=args.output_dir,
            file_format=args.format,
            show_metrics=args.show_metrics
        )

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")
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

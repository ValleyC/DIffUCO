"""
Visualize training data instances from the chip placement dataset.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse


def load_dataset(dataset_path):
    """Load dataset from pickle file."""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return data


def visualize_instance(instance, save_path=None, title="Chip Placement Instance"):
    """
    Visualize a single chip placement instance.

    Args:
        instance: Dictionary containing:
            - 'positions': [N, 2] component center positions
            - 'component_sizes': [N, 2] component (width, height)
            - 'adjacency': [N, N] or edge list for connections
            - 'energy' (optional): energy metrics
        save_path: Path to save the figure
        title: Figure title
    """
    positions = instance.get('positions', instance.get('H_pos'))
    component_sizes = instance.get('component_sizes', instance.get('nodes'))

    # Handle different data formats
    if component_sizes.shape[1] > 2:
        # Extract just width and height (first 2 columns)
        component_sizes = component_sizes[:, :2]

    num_components = len(positions)

    # Calculate actual density
    canvas_width = 2.0
    canvas_height = 2.0
    canvas_area = canvas_width * canvas_height
    component_areas = component_sizes[:, 0] * component_sizes[:, 1]
    total_component_area = np.sum(component_areas)
    density = total_component_area / canvas_area

    # Calculate HPWL if edges available
    hpwl = None
    edges = None
    if 'adjacency' in instance:
        adjacency = instance['adjacency']
        if len(adjacency.shape) == 2:
            # Adjacency matrix format
            edges = np.argwhere(adjacency > 0)
        else:
            edges = adjacency
    elif 'edges' in instance:
        edges = instance['edges']

    if edges is not None and len(edges) > 0:
        hpwl = 0.0
        for edge in edges:
            if len(edge) >= 2:
                src, dst = edge[0], edge[1]
                if src < len(positions) and dst < len(positions):
                    pos1 = positions[src]
                    pos2 = positions[dst]
                    # Half-perimeter wirelength
                    hpwl += abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # Calculate overlaps
    overlap_area = 0.0
    for i in range(num_components):
        for j in range(i + 1, num_components):
            x1_min = positions[i, 0] - component_sizes[i, 0] / 2
            x1_max = positions[i, 0] + component_sizes[i, 0] / 2
            y1_min = positions[i, 1] - component_sizes[i, 1] / 2
            y1_max = positions[i, 1] + component_sizes[i, 1] / 2

            x2_min = positions[j, 0] - component_sizes[j, 0] / 2
            x2_max = positions[j, 0] + component_sizes[j, 0] / 2
            y2_min = positions[j, 1] - component_sizes[j, 1] / 2
            y2_max = positions[j, 1] + component_sizes[j, 1] / 2

            # Calculate overlap
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            overlap_area += x_overlap * y_overlap

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw canvas boundary
    canvas_rect = patches.Rectangle(
        (-1, -1), canvas_width, canvas_height,
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(canvas_rect)

    # Draw connections (nets) first (so they appear behind components)
    if edges is not None and len(edges) > 0:
        for edge in edges:
            if len(edge) >= 2:
                src, dst = edge[0], edge[1]
                if src < len(positions) and dst < len(positions):
                    pos1 = positions[src]
                    pos2 = positions[dst]
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                           'gray', alpha=0.3, linewidth=0.5, zorder=1)

    # Draw components
    colors = plt.cm.tab20(np.linspace(0, 1, num_components))
    for i in range(num_components):
        center = positions[i]
        size = component_sizes[i]

        # Component rectangle (centered at position)
        rect = patches.Rectangle(
            (center[0] - size[0]/2, center[1] - size[1]/2),
            size[0], size[1],
            linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.7,
            zorder=2
        )
        ax.add_patch(rect)

        # Component ID at center
        ax.text(center[0], center[1], str(i),
               ha='center', va='center', fontsize=6, zorder=3)

    # Set axis properties
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Title with metrics
    metrics_str = f"Components: {num_components} | Density: {density:.1%}"
    if hpwl is not None:
        metrics_str += f" | HPWL: {hpwl:.2f}"
    if overlap_area > 0:
        metrics_str += f" | Overlap: {overlap_area:.3f}"

    ax.set_title(f"{title}\n{metrics_str}", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()

    return {
        'num_components': num_components,
        'density': density,
        'hpwl': hpwl,
        'overlap_area': overlap_area
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize chip placement training data')
    parser.add_argument('--dataset_dir', type=str,
                       default='DIffUCO/DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm',
                       help='Directory containing dataset pickle files')
    parser.add_argument('--dataset_prefix', type=str, default='Chip',
                       help='Dataset file prefix (e.g., Chip, Chip_high_density)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split to visualize')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='dataset_visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index for samples')

    args = parser.parse_args()

    # Find dataset file
    dataset_pattern = f"{args.dataset_prefix}_*_{args.mode}.pickle"
    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return

    # Find matching dataset files
    dataset_files = list(dataset_dir.glob(dataset_pattern))
    if not dataset_files:
        print(f"Error: No dataset files matching pattern '{dataset_pattern}' in {dataset_dir}")
        print(f"Available files:")
        for f in sorted(dataset_dir.glob("*.pickle")):
            print(f"  - {f.name}")
        return

    dataset_file = dataset_files[0]
    print(f"Loading dataset: {dataset_file}")

    # Load dataset
    data = load_dataset(dataset_file)
    print(f"Dataset contains {len(data)} instances")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Visualize samples
    num_to_visualize = min(args.num_samples, len(data) - args.start_idx)

    print(f"\nVisualizing {num_to_visualize} instances starting from index {args.start_idx}...")

    all_metrics = []
    for i in range(args.start_idx, args.start_idx + num_to_visualize):
        instance = data[i]
        save_path = output_dir / f"{args.dataset_prefix}_{args.mode}_instance_{i:04d}.png"

        metrics = visualize_instance(
            instance,
            save_path=save_path,
            title=f"{args.dataset_prefix} {args.mode.capitalize()} Instance {i}"
        )
        all_metrics.append(metrics)

        print(f"Instance {i}: " +
              f"Components={metrics['num_components']}, " +
              f"Density={metrics['density']:.1%}, " +
              (f"HPWL={metrics['hpwl']:.2f}, " if metrics['hpwl'] is not None else "") +
              f"Overlap={metrics['overlap_area']:.3f}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    densities = [m['density'] for m in all_metrics]
    print(f"Density: min={min(densities):.1%}, max={max(densities):.1%}, mean={np.mean(densities):.1%}")

    if all_metrics[0]['hpwl'] is not None:
        hpwls = [m['hpwl'] for m in all_metrics if m['hpwl'] is not None]
        if hpwls:
            print(f"HPWL: min={min(hpwls):.2f}, max={max(hpwls):.2f}, mean={np.mean(hpwls):.2f}")

    overlaps = [m['overlap_area'] for m in all_metrics]
    print(f"Overlap: min={min(overlaps):.3f}, max={max(overlaps):.3f}, mean={np.mean(overlaps):.3f}")

    print(f"\nVisualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()

"""
Visualize newly generated unsupervised chip placement data instances
Show netlist, placement, and HPWL for comparison
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import sys
sys.path.append('..')

from loadGraphDatasets.ChipDatasetGenerator import ChipDatasetGenerator

def visualize_instance(positions, sizes, edge_index, edge_attr, hpwl, density, ax, title):
    """Visualize one chip placement instance"""

    # Draw chip boundary
    boundary = Rectangle((-1, -1), 2, 2, fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(boundary)

    # Draw components as rectangles
    colors = plt.cm.Set3(np.linspace(0, 1, len(positions)))

    for i in range(len(positions)):
        x_center, y_center = positions[i]
        width, height = sizes[i]

        # Rectangle (bottom-left corner)
        x_left = x_center - width/2
        y_bottom = y_center - height/2

        rect = Rectangle((x_left, y_bottom), width, height,
                        facecolor=colors[i], edgecolor='blue', linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)

        # Component label
        ax.text(x_center, y_center, str(i), ha='center', va='center',
               fontsize=8, fontweight='bold')

    # Draw netlist connections
    # Only draw a subset to avoid clutter
    num_edges_to_draw = min(50, edge_index.shape[1] // 2)

    for i in range(num_edges_to_draw):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()

        # Get terminal positions (center + offset)
        src_pos = positions[src_idx] + edge_attr[i, :2].numpy()
        dst_pos = positions[dst_idx] + edge_attr[i, 2:4].numpy()

        # Draw connection
        arrow = FancyArrowPatch(src_pos, dst_pos,
                               arrowstyle='-', linewidth=0.5,
                               color='red', alpha=0.3, linestyle='--')
        ax.add_patch(arrow)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\nComponents: {len(positions)}, Density: {density:.3f}, HPWL: {hpwl:.1f}',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)

def main():
    print("="*80)
    print("VISUALIZING NEW UNSUPERVISED CHIP PLACEMENT DATA")
    print("="*80)

    config = {
        "dataset_name": "Chip_dummy",
        "max_instance": 50,
        "max_attempts_per_instance": 1000,
        "seed": 456,
        "parent": False,
        "save": False,
        "mode": "train",
        "problem": "ChipPlacement",
        "diff_ps": False,
        "gurobi_solve": False,
        "licence_base_path": ".",
        "time_limit": 1000,
        "thread_fraction": 1.0
    }

    generator = ChipDatasetGenerator(config)
    generator.mode = "train"

    print(f"\nGenerating 6 instances for visualization...")

    # Generate instances
    instances = []
    for i in range(6):
        print(f"Generating instance {i+1}/6...")
        positions, data, density, hpwl = generator.sample_chip_instance_unsupervised()

        instances.append({
            'positions': positions.numpy(),
            'sizes': data.x.numpy(),
            'edge_index': data.edge_index,
            'edge_attr': data.edge_attr,
            'hpwl': hpwl,
            'density': density
        })

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Unsupervised Chip Placement Training Data\n' +
                 'Netlist: Random graph (Barabasi-Albert) | Placement: Random legal (independent)\n' +
                 'Red dashed lines = netlist connections (high HPWL = poor baseline)',
                 fontsize=14, fontweight='bold')

    axes = axes.flatten()

    for idx, inst in enumerate(instances):
        visualize_instance(
            inst['positions'],
            inst['sizes'],
            inst['edge_index'],
            inst['edge_attr'],
            inst['hpwl'],
            inst['density'],
            axes[idx],
            f'Instance {idx+1}'
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('d:/Codes/Unsupervised/new_data_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n" + "="*80)
    print("Visualization saved to: d:/Codes/Unsupervised/new_data_visualization.png")
    print("="*80)

    # Print statistics
    hpwl_values = [inst['hpwl'] for inst in instances]
    densities = [inst['density'] for inst in instances]
    num_components = [len(inst['positions']) for inst in instances]

    print(f"\nStatistics (6 instances):")
    print(f"  HPWL:       mean={np.mean(hpwl_values):.1f}, std={np.std(hpwl_values):.1f}, range=[{np.min(hpwl_values):.1f}, {np.max(hpwl_values):.1f}]")
    print(f"  Density:    mean={np.mean(densities):.3f}, std={np.std(densities):.3f}")
    print(f"  Components: mean={np.mean(num_components):.1f}, range=[{np.min(num_components)}, {np.max(num_components)}]")

    print(f"\nKey observations:")
    print(f"  - Components are legally placed (no overlaps, within bounds)")
    print(f"  - Red dashed lines show netlist connections")
    print(f"  - Many connections span long distances (high HPWL)")
    print(f"  - This is CORRECT for unsupervised learning (poor baseline)")
    print(f"  - Model will learn to reduce HPWL by placing connected components closer")
    print("="*80)

if __name__ == "__main__":
    main()

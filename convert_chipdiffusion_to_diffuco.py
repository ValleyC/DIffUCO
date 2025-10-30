"""
Convert ChipDiffusion clustered benchmarks to DIffUCO format

ChipDiffusion format (PyTorch Geometric):
- graph*.pickle: PyG Data with x, edge_index, is_macros
- output*.pickle: Numpy array of positions

DIffUCO format (jraph):
- Expects jraph.GraphsTuple with nodes, edges, senders, receivers, n_node, n_edge
- Positions stored separately
"""

import pickle
import numpy as np
import jraph
import torch
from pathlib import Path
import argparse


def convert_pyg_to_jraph(pyg_data, positions):
    """
    Convert PyTorch Geometric Data to jraph GraphsTuple

    Args:
        pyg_data: PyG Data object with x, edge_index, is_macros
        positions: numpy array of positions (N, 2)

    Returns:
        jraph_data: jraph.GraphsTuple
        positions: torch tensor of positions
    """
    # Extract data
    node_features = pyg_data.x.cpu().numpy()  # (N, 2) - component sizes
    edge_index = pyg_data.edge_index.cpu().numpy()  # (2, E)
    is_macros = pyg_data.is_macros.cpu().numpy()  # (N,)

    n_nodes = node_features.shape[0]
    n_edges = edge_index.shape[1]

    # Create jraph GraphsTuple
    # In jraph, edges are represented by senders and receivers
    senders = edge_index[0]
    receivers = edge_index[1]

    # Edge features (can be distances or just ones)
    # DIffUCO seems to use edge attributes, let's compute distances
    edge_features = np.ones((n_edges, 1), dtype=np.float32)

    # Create jraph GraphsTuple
    jraph_data = jraph.GraphsTuple(
        nodes=node_features,  # Component sizes
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=np.array([n_nodes]),
        n_edge=np.array([n_edges]),
        globals=None
    )

    # Convert positions to torch tensor
    if isinstance(positions, np.ndarray):
        positions_tensor = torch.from_numpy(positions).float()
    else:
        positions_tensor = positions

    return jraph_data, positions_tensor


def compute_hpwl(positions, jraph_data):
    """Compute Half-Perimeter Wirelength"""
    senders = jraph_data.senders
    receivers = jraph_data.receivers

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


def compute_density(node_features, canvas_area=4.0):
    """Compute placement density"""
    # node_features shape: (N, 2) - width, height
    total_component_area = np.sum(node_features[:, 0] * node_features[:, 1])
    density = total_component_area / canvas_area
    return density


def convert_all_benchmarks(input_dir, output_dir):
    """
    Convert all ChipDiffusion benchmarks to DIffUCO format

    Args:
        input_dir: Path to datasets/iccad04_chipdiffusion/iccad04_lefdef.cluster512.v1/
        output_dir: Path to save DIffUCO format files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all graph pickle files
    graph_files = sorted(input_path.glob("graph*.pickle"))

    print(f"Found {len(graph_files)} benchmarks in {input_dir}")

    # Prepare DIffUCO format dataset
    solutions = {
        "positions": [],
        "H_graphs": [],
        "sizes": [],
        "edge_attrs": [],
        "graph_sizes": [],
        "densities": [],
        "Energies": [],  # HPWL
        "compl_H_graphs": [],
        "gs_bins": [],
    }

    for graph_file in graph_files:
        # Extract index from filename (graph0.pickle -> 0)
        idx = int(graph_file.stem.replace("graph", ""))

        # Load ChipDiffusion format
        with open(graph_file, 'rb') as f:
            pyg_data = pickle.load(f)

        placement_file = input_path / f"output{idx}.pickle"
        with open(placement_file, 'rb') as f:
            positions = pickle.load(f)

        # Convert to numpy if needed
        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().numpy()

        # Convert to jraph format
        jraph_data, positions_tensor = convert_pyg_to_jraph(pyg_data, positions)

        # Compute metrics
        hpwl = compute_hpwl(positions, jraph_data)
        density = compute_density(jraph_data.nodes)

        # Add to solutions
        solutions["positions"].append(positions_tensor.numpy())
        solutions["H_graphs"].append(jraph_data)
        solutions["sizes"].append(jraph_data.nodes)
        solutions["edge_attrs"].append(jraph_data.edges)
        solutions["graph_sizes"].append(jraph_data.nodes.shape[0])
        solutions["densities"].append(density)
        solutions["Energies"].append(hpwl)
        solutions["compl_H_graphs"].append(jraph_data)
        solutions["gs_bins"].append(positions_tensor.numpy())

        print(f"  Converted {graph_file.name}: {jraph_data.nodes.shape[0]} nodes, "
              f"{jraph_data.edges.shape[0]} edges, HPWL={hpwl:.2f}, density={density:.3f}")

    # Save in DIffUCO format
    output_file = output_path / "test_ChipPlacement_seed_123_solutions.pickle"
    with open(output_file, 'wb') as f:
        pickle.dump(solutions, f)

    print(f"\n✓ Saved {len(graph_files)} benchmarks to {output_file}")
    print(f"  Format: DIffUCO jraph format")
    print(f"  Use with: python eval_and_visualize.py --checkpoint <your_checkpoint> --dataset ICCAD04_clustered")

    return solutions


def main():
    parser = argparse.ArgumentParser(description='Convert ChipDiffusion benchmarks to DIffUCO format')
    parser.add_argument('--input-dir', type=str,
                       default='datasets/iccad04_chipdiffusion/iccad04_lefdef.cluster512.v1',
                       help='Input directory with ChipDiffusion pickle files')
    parser.add_argument('--output-dir', type=str,
                       default='DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/ICCAD04_clustered',
                       help='Output directory for DIffUCO format')

    args = parser.parse_args()

    print("="*70)
    print("Converting ChipDiffusion Clustered Benchmarks to DIffUCO Format")
    print("="*70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print()

    solutions = convert_all_benchmarks(args.input_dir, args.output_dir)

    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Train DIffUCO model (or use existing checkpoint)")
    print("2. Run inference:")
    print("   python eval_and_visualize.py \\")
    print("       --checkpoint Checkpoints/YOUR_MODEL/best_*.pickle \\")
    print("       --dataset ICCAD04_clustered \\")
    print("       --n_samples 10")


if __name__ == '__main__':
    main()

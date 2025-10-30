"""
Convert ChipDiffusion clustered benchmarks to DIffUCO format

Converts PyTorch Geometric format (ChipDiffusion) to jraph format (DIffUCO)
while preserving the clustering (macros + 512 clusters)
"""

import pickle
import numpy as np
import jraph
import torch
from pathlib import Path
import argparse
from tqdm import tqdm


def convert_pyg_to_jraph(pyg_data, positions):
    """
    Convert PyTorch Geometric Data to jraph GraphsTuple

    Args:
        pyg_data: PyG Data object with x, edge_index, is_macros
        positions: numpy array of positions (N, 2)

    Returns:
        jraph_data: jraph.GraphsTuple
        positions_tensor: torch tensor of positions
    """
    # Extract data
    node_features = pyg_data.x.cpu().numpy()  # (N, 2) - component sizes
    edge_index = pyg_data.edge_index.cpu().numpy()  # (2, E)

    n_nodes = node_features.shape[0]
    n_edges = edge_index.shape[1]

    # Create jraph GraphsTuple
    senders = edge_index[0]
    receivers = edge_index[1]
    edge_features = np.ones((n_edges, 1), dtype=np.float32)

    jraph_data = jraph.GraphsTuple(
        nodes=node_features,  # Component sizes (width, height)
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

        bbox_width = max(pos1[0], pos2[0]) - min(pos1[0], pos2[0])
        bbox_height = max(pos1[1], pos2[1]) - min(pos1[1], pos2[1])

        total_hpwl += bbox_width + bbox_height

    return total_hpwl


def compute_density(node_features, canvas_area=4.0):
    """Compute placement density"""
    total_component_area = np.sum(node_features[:, 0] * node_features[:, 1])
    density = total_component_area / canvas_area
    return density


def convert_all_benchmarks(input_dir, output_dir, seed=123):
    """
    Convert all ChipDiffusion benchmarks to DIffUCO format

    Creates directory structure for both train and test:
    output_dir/train/seed/ChipPlacement/indexed/idx_*_solutions.pickle
    output_dir/test/seed/ChipPlacement/indexed/idx_*_solutions.pickle

    Args:
        input_dir: Path to datasets/iccad04_chipdiffusion/iccad04_lefdef.cluster512.v1/
        output_dir: Path to save DIffUCO format files
        seed: Random seed (default 123)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create directory structure for train/val/test (DIffUCO expects all three)
    train_indexed_dir = output_path / "train" / str(seed) / "ChipPlacement" / "indexed"
    val_indexed_dir = output_path / "val" / str(seed) / "ChipPlacement" / "indexed"
    test_indexed_dir = output_path / "test" / str(seed) / "ChipPlacement" / "indexed"
    train_indexed_dir.mkdir(parents=True, exist_ok=True)
    val_indexed_dir.mkdir(parents=True, exist_ok=True)
    test_indexed_dir.mkdir(parents=True, exist_ok=True)

    # Find all graph pickle files
    graph_files = sorted(input_path.glob("graph*.pickle"))

    print(f"\nFound {len(graph_files)} benchmarks in {input_dir}")
    print(f"Saving to: {output_path}")
    print()

    # Prepare full dataset dictionary for train/val/test
    train_solutions = {
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

    val_solutions = {
        "positions": [],
        "H_graphs": [],
        "sizes": [],
        "edge_attrs": [],
        "graph_sizes": [],
        "densities": [],
        "Energies": [],
        "compl_H_graphs": [],
        "gs_bins": [],
    }

    test_solutions = {
        "positions": [],
        "H_graphs": [],
        "sizes": [],
        "edge_attrs": [],
        "graph_sizes": [],
        "densities": [],
        "Energies": [],
        "compl_H_graphs": [],
        "gs_bins": [],
    }

    for graph_file in tqdm(graph_files, desc="Converting"):
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

        # Save individual indexed file (required by DIffUCO)
        indexed_solution = {
            "positions": positions_tensor.numpy(),
            "H_graphs": jraph_data,
            "sizes": jraph_data.nodes,
            "edge_attrs": jraph_data.edges,
            "graph_sizes": jraph_data.nodes.shape[0],
            "densities": density,
            "Energies": hpwl,
            "compl_H_graphs": jraph_data,
            "gs_bins": positions_tensor.numpy(),
        }

        # Save to train/val/test directories (same data for all)
        train_file = train_indexed_dir / f"idx_{idx}_solutions.pickle"
        val_file = val_indexed_dir / f"idx_{idx}_solutions.pickle"
        test_file = test_indexed_dir / f"idx_{idx}_solutions.pickle"

        for file_path in [train_file, val_file, test_file]:
            with open(file_path, 'wb') as f:
                pickle.dump(indexed_solution, f)

        # Add to all solutions lists
        for solutions in [train_solutions, val_solutions, test_solutions]:
            solutions["positions"].append(positions_tensor.numpy())
            solutions["H_graphs"].append(jraph_data)
            solutions["sizes"].append(jraph_data.nodes)
            solutions["edge_attrs"].append(jraph_data.edges)
            solutions["graph_sizes"].append(jraph_data.nodes.shape[0])
            solutions["densities"].append(density)
            solutions["Energies"].append(hpwl)
            solutions["compl_H_graphs"].append(jraph_data)
            solutions["gs_bins"].append(positions_tensor.numpy())

    # Save full dataset files for train/val/test
    train_file = output_path / f"train_ChipPlacement_seed_{seed}_solutions.pickle"
    val_file = output_path / f"val_ChipPlacement_seed_{seed}_solutions.pickle"
    test_file = output_path / f"test_ChipPlacement_seed_{seed}_solutions.pickle"

    with open(train_file, 'wb') as f:
        pickle.dump(train_solutions, f)
    with open(val_file, 'wb') as f:
        pickle.dump(val_solutions, f)
    with open(test_file, 'wb') as f:
        pickle.dump(test_solutions, f)

    print(f"\n✓ Conversion complete!")
    print(f"  Saved {len(graph_files)} benchmarks to train/val/test:")
    print(f"  - Train dataset: {train_file.name}")
    print(f"  - Val dataset: {val_file.name}")
    print(f"  - Test dataset: {test_file.name}")
    print(f"  - Train indexed: {train_indexed_dir}/idx_*_solutions.pickle")
    print(f"  - Val indexed: {val_indexed_dir}/idx_*_solutions.pickle")
    print(f"  - Test indexed: {test_indexed_dir}/idx_*_solutions.pickle")
    print(f"\n  Total nodes per benchmark:")
    for i, size in enumerate(test_solutions["graph_sizes"]):
        print(f"    ibm{i+1:02d}: {size} nodes (macros/IOs + clusters)")

    return test_solutions


def main():
    parser = argparse.ArgumentParser(description='Convert ChipDiffusion benchmarks to DIffUCO format')
    parser.add_argument('--input-dir', type=str,
                       default='datasets/iccad04_chipdiffusion/iccad04_lefdef.cluster512.v1',
                       help='Input directory with ChipDiffusion pickle files')
    parser.add_argument('--output-dir', type=str,
                       default='DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/ICCAD04_clustered',
                       help='Output directory for DIffUCO format')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed for dataset naming')

    args = parser.parse_args()

    print("="*70)
    print("Converting ChipDiffusion Clustered Benchmarks to DIffUCO Format")
    print("="*70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Seed:   {args.seed}")

    solutions = convert_all_benchmarks(args.input_dir, args.output_dir, args.seed)

    print("\n" + "="*70)
    print("Next Steps: Run DIffUCO Inference")
    print("="*70)
    print("\nOption 1: Use existing checkpoint")
    print("  python eval_and_visualize.py \\")
    print("      --checkpoint Checkpoints/YOUR_MODEL/best_*.pickle \\")
    print("      --dataset ICCAD04_clustered \\")
    print("      --n_samples 10")
    print("\nOption 2: Train new model on clustered benchmarks")
    print("  python train.py \\")
    print("      --config chip_placement_config.py \\")
    print("      --dataset ICCAD04_clustered")
    print("="*70)


if __name__ == '__main__':
    main()

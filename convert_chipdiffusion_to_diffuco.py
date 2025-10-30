"""
Convert ChipDiffusion benchmarks to DIffUCO format - MACROS AND I/Os ONLY

Filters out the 512 clusters and keeps only macros and I/O pads to match ChipDiffusion's actual format
"""

import pickle
import numpy as np
import jraph
import torch
from pathlib import Path
import argparse
from tqdm import tqdm


def filter_macros_ios(pyg_data, positions):
    """
    Filter to keep ONLY macros and I/Os, remove clusters

    Args:
        pyg_data: PyG Data with all nodes (macros + IOs + clusters)
        positions: Positions for all nodes

    Returns:
        filtered_data: PyG Data with only macros/IOs
        filtered_positions: Positions for only macros/IOs
    """
    # Get mask for macros/IOs
    is_macros = pyg_data.is_macros.cpu().numpy()
    macro_indices = np.where(is_macros)[0]

    # Filter node features
    filtered_x = pyg_data.x[is_macros]

    # Filter positions
    if isinstance(positions, torch.Tensor):
        filtered_positions = positions[is_macros]
    else:
        filtered_positions = positions[is_macros]

    # Create mapping from old indices to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(macro_indices)}

    # Filter edges - only keep edges between macros/IOs
    edge_index = pyg_data.edge_index.cpu().numpy()
    new_edges = []

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]

        # Only keep edge if both endpoints are macros/IOs
        if src in old_to_new and dst in old_to_new:
            new_src = old_to_new[src]
            new_dst = old_to_new[dst]
            new_edges.append([new_src, new_dst])

    if len(new_edges) > 0:
        filtered_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
    else:
        filtered_edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Create filtered PyG data
    filtered_data = type(pyg_data)()
    filtered_data.x = filtered_x
    filtered_data.edge_index = filtered_edge_index
    filtered_data.is_macros = torch.ones(filtered_x.shape[0], dtype=torch.bool)

    if hasattr(pyg_data, 'is_ports'):
        filtered_data.is_ports = pyg_data.is_ports[is_macros]

    return filtered_data, filtered_positions


def convert_pyg_to_jraph(pyg_data, positions):
    """Convert PyTorch Geometric Data to jraph GraphsTuple"""
    node_features = pyg_data.x.cpu().numpy()
    edge_index = pyg_data.edge_index.cpu().numpy()

    n_nodes = node_features.shape[0]
    n_edges = edge_index.shape[1]

    senders = edge_index[0]
    receivers = edge_index[1]
    edge_features = np.ones((n_edges, 1), dtype=np.float32)

    jraph_data = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=np.array([n_nodes]),
        n_edge=np.array([n_edges]),
        globals=None
    )

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
    """Convert ChipDiffusion benchmarks - MACROS AND IOs ONLY"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create directory structure
    train_indexed_dir = output_path / "train" / str(seed) / "ChipPlacement" / "indexed"
    val_indexed_dir = output_path / "val" / str(seed) / "ChipPlacement" / "indexed"
    test_indexed_dir = output_path / "test" / str(seed) / "ChipPlacement" / "indexed"
    train_indexed_dir.mkdir(parents=True, exist_ok=True)
    val_indexed_dir.mkdir(parents=True, exist_ok=True)
    test_indexed_dir.mkdir(parents=True, exist_ok=True)

    graph_files = sorted(input_path.glob("graph*.pickle"))

    print(f"\nFound {len(graph_files)} benchmarks in {input_dir}")
    print(f"Converting: MACROS AND I/Os ONLY (filtering out 512 clusters)")
    print(f"Saving to: {output_path}\n")

    # Prepare solutions
    train_solutions = {
        "positions": [], "H_graphs": [], "sizes": [], "edge_attrs": [],
        "graph_sizes": [], "densities": [], "Energies": [],
        "compl_H_graphs": [], "gs_bins": [], "legal_positions": [],
    }
    val_solutions = {k: [] for k in train_solutions.keys()}
    test_solutions = {k: [] for k in train_solutions.keys()}

    for graph_file in tqdm(graph_files, desc="Converting"):
        idx = int(graph_file.stem.replace("graph", ""))

        # Load ChipDiffusion format
        with open(graph_file, 'rb') as f:
            pyg_data = pickle.load(f)

        placement_file = input_path / f"output{idx}.pickle"
        with open(placement_file, 'rb') as f:
            positions = pickle.load(f)

        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().numpy()

        # FILTER: Keep only macros and IOs, remove clusters
        original_nodes = pyg_data.x.shape[0]
        filtered_pyg_data, filtered_positions = filter_macros_ios(pyg_data, positions)
        filtered_nodes = filtered_pyg_data.x.shape[0]

        # Convert to jraph format
        jraph_data, positions_tensor = convert_pyg_to_jraph(filtered_pyg_data, filtered_positions)

        # Compute metrics
        hpwl = compute_hpwl(filtered_positions, jraph_data)
        density = compute_density(jraph_data.nodes)

        # Save individual indexed file
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
            "legal_positions": positions_tensor.numpy(),
        }

        # Save to train/val/test
        for dir_path in [train_indexed_dir, val_indexed_dir, test_indexed_dir]:
            with open(dir_path / f"idx_{idx}_solutions.pickle", 'wb') as f:
                pickle.dump(indexed_solution, f)

        # Add to solutions lists
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
            solutions["legal_positions"].append(positions_tensor.numpy())

    # Save full dataset files
    for mode, solutions in [("train", train_solutions), ("val", val_solutions), ("test", test_solutions)]:
        with open(output_path / f"{mode}_ChipPlacement_seed_{seed}_solutions.pickle", 'wb') as f:
            pickle.dump(solutions, f)

    print(f"\n✓ Conversion complete! MACROS AND I/Os ONLY")
    print(f"\n  Nodes per benchmark:")
    for i, size in enumerate(test_solutions["graph_sizes"]):
        print(f"    IBM{i+1:02d}: {size} nodes (macros + I/Os)")

    return test_solutions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str,
                       default='datasets/iccad04_chipdiffusion/iccad04_lefdef.cluster512.v1')
    parser.add_argument('--output-dir', type=str,
                       default='DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/ICCAD04_clustered')
    parser.add_argument('--seed', type=int, default=123)

    args = parser.parse_args()

    print("="*70)
    print("Converting ChipDiffusion to DIffUCO - MACROS AND I/Os ONLY")
    print("="*70)

    convert_all_benchmarks(args.input_dir, args.output_dir, args.seed)

    print("\n" + "="*70)
    print("Next: Run inference")
    print("="*70)
    print("python eval_and_visualize.py \\")
    print("    --checkpoint Checkpoints/YOUR_MODEL/best_*.pickle \\")
    print("    --dataset ICCAD04_clustered \\")
    print("    --n_samples 10")


if __name__ == '__main__':
    main()

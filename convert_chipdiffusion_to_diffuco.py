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


def normalize_to_diffuco_format(component_sizes, positions, chip_size):
    """
    Normalize ChipDiffusion data to DIffUCO format following ChipDiffusion's preprocess_graph

    Args:
        component_sizes: (V, 2) tensor/array of component sizes in absolute units
        positions: (V, 2) tensor/array of positions in absolute coordinates
        chip_size: [x_min, y_min, x_max, y_max] canvas bounds

    Returns:
        normalized_sizes: (V, 2) component sizes normalized to 2x2 canvas
        normalized_positions: (V, 2) positions in [-1, 1] range
    """
    # Convert to tensors if needed
    if not isinstance(component_sizes, torch.Tensor):
        component_sizes = torch.from_numpy(component_sizes).float()
    if not isinstance(positions, torch.Tensor):
        positions = torch.from_numpy(positions).float()

    # Extract canvas dimensions
    if len(chip_size) == 4:  # [x_min, y_min, x_max, y_max]
        canvas_width = chip_size[2] - chip_size[0]
        canvas_height = chip_size[3] - chip_size[1]
        chip_offset = torch.tensor([chip_size[0], chip_size[1]], dtype=torch.float32)
    else:  # [width, height]
        canvas_width = chip_size[0]
        canvas_height = chip_size[1]
        chip_offset = torch.zeros(2, dtype=torch.float32)

    canvas_size = torch.tensor([canvas_width, canvas_height], dtype=torch.float32)

    # ChipDiffusion normalization (from preprocess_graph in utils.py)
    # 1. Normalize component sizes: cond.x = 2 * (cond.x / chip_size)
    normalized_sizes = 2.0 * (component_sizes / canvas_size)

    # 2. Normalize positions to [-1, 1]: x = 2 * (x / chip_size) - 1
    normalized_positions = 2.0 * ((positions - chip_offset) / canvas_size) - 1.0

    # 3. Adjust to center: x = x + cond.x/2
    normalized_positions = normalized_positions + normalized_sizes / 2.0

    return normalized_sizes, normalized_positions


def denormalize_from_diffuco_format(normalized_sizes, normalized_positions, chip_size):
    """
    Denormalize DIffUCO format back to ChipDiffusion's original scale

    This is the INVERSE of normalize_to_diffuco_format, following ChipDiffusion's postprocess_placement

    Args:
        normalized_sizes: (V, 2) component sizes in normalized 2x2 canvas
        normalized_positions: (V, 2) positions in [-1, 1] range
        chip_size: [x_min, y_min, x_max, y_max] original canvas bounds

    Returns:
        original_sizes: (V, 2) component sizes in original units
        original_positions: (V, 2) positions in original coordinate system
    """
    # Convert to tensors if needed
    if not isinstance(normalized_sizes, torch.Tensor):
        normalized_sizes = torch.from_numpy(normalized_sizes).float()
    if not isinstance(normalized_positions, torch.Tensor):
        normalized_positions = torch.from_numpy(normalized_positions).float()

    # Extract canvas dimensions
    if len(chip_size) == 4:  # [x_min, y_min, x_max, y_max]
        canvas_width = chip_size[2] - chip_size[0]
        canvas_height = chip_size[3] - chip_size[1]
        chip_offset = torch.tensor([chip_size[0], chip_size[1]], dtype=torch.float32)
    else:  # [width, height]
        canvas_width = chip_size[0]
        canvas_height = chip_size[1]
        chip_offset = torch.zeros(2, dtype=torch.float32)

    canvas_size = torch.tensor([canvas_width, canvas_height], dtype=torch.float32)

    # Inverse of ChipDiffusion normalization (from postprocess_placement in utils.py)
    # 1. Remove center adjustment: x = x - cond.x/2
    original_positions = normalized_positions - normalized_sizes / 2.0

    # 2. Denormalize positions: x = scale * (x+1)/2 + offset
    original_positions = canvas_size * (original_positions + 1.0) / 2.0 + chip_offset

    # 3. Denormalize sizes: cond.x = (cond.x * scale)/2
    original_sizes = (normalized_sizes * canvas_size) / 2.0

    return original_sizes, original_positions


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
    print(f"Normalizing: ChipDiffusion format → DIffUCO format ([-1,1] canvas)")
    print(f"Saving to: {output_path}\n")

    # Prepare solutions
    train_solutions = {
        "positions": [], "H_graphs": [], "sizes": [], "edge_attrs": [],
        "graph_sizes": [], "densities": [], "Energies": [],
        "compl_H_graphs": [], "gs_bins": [], "legal_positions": [],
        "chip_sizes": [],  # Store original chip_size for denormalization
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

        # NORMALIZE: Apply ChipDiffusion normalization to match DIffUCO training data format
        # Get chip_size from pyg_data
        if hasattr(pyg_data, 'chip_size'):
            chip_size = pyg_data.chip_size
        else:
            raise ValueError(f"Graph {idx} missing chip_size attribute!")

        normalized_sizes, normalized_positions = normalize_to_diffuco_format(
            filtered_pyg_data.x, filtered_positions, chip_size
        )

        # Update filtered data with normalized values
        filtered_pyg_data.x = normalized_sizes

        # Convert to jraph format
        jraph_data, positions_tensor = convert_pyg_to_jraph(filtered_pyg_data, normalized_positions)

        # Compute metrics using normalized values
        hpwl = compute_hpwl(normalized_positions.cpu().numpy() if isinstance(normalized_positions, torch.Tensor) else normalized_positions, jraph_data)
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
            "chip_size": chip_size,  # Store for denormalization during eval
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
            solutions["chip_sizes"].append(chip_size)

    # Save full dataset files
    for mode, solutions in [("train", train_solutions), ("val", val_solutions), ("test", test_solutions)]:
        with open(output_path / f"{mode}_ChipPlacement_seed_{seed}_solutions.pickle", 'wb') as f:
            pickle.dump(solutions, f)

    print(f"\n✓ Conversion complete! MACROS AND I/Os ONLY with NORMALIZATION")
    print(f"\n  Nodes per benchmark:")
    for i, size in enumerate(test_solutions["graph_sizes"]):
        print(f"    IBM{i+1:02d}: {size} nodes (macros + I/Os)")

    # Show normalization verification for first benchmark
    print(f"\n  Normalization verification (IBM01):")
    print(f"    Component sizes range: [{test_solutions['sizes'][0].min():.4f}, {test_solutions['sizes'][0].max():.4f}]")
    print(f"    Position X range: [{test_solutions['positions'][0][:, 0].min():.4f}, {test_solutions['positions'][0][:, 0].max():.4f}]")
    print(f"    Position Y range: [{test_solutions['positions'][0][:, 1].min():.4f}, {test_solutions['positions'][0][:, 1].max():.4f}]")
    print(f"    ✓ Data normalized to [-1, 1] canvas (matches DIffUCO training data)")

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
    print("Converting ChipDiffusion to DIffUCO")
    print("  - MACROS AND I/Os ONLY (filtering clusters)")
    print("  - Normalizing to [-1,1] canvas (matching DIffUCO training)")
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

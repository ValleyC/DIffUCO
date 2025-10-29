#!/usr/bin/env python3
"""
ChipDiffusion Benchmark Preparation Script
==========================================

This script downloads and converts chip placement benchmarks to ChipDiffusion format,
creating macro-only datasets that match ChipDiffusion's experiments exactly.

Usage:
    python prepare_chipdiffusion_benchmarks.py

Outputs:
    - datasets/graph/iccad04/          # Full IBM benchmarks
    - datasets/graph/ispd2005/         # Full ISPD2005 benchmarks
    - datasets/graph/macro-ibm/        # Macro-only IBM (terminals only)
    - datasets/graph/macro-ispd/       # Macro-only ISPD2005 (macros only)

Requirements:
    - Python 3.7+
    - PyTorch
    - torch_geometric
    - wget or curl (for downloads)
"""

import os
import re
import sys
import pickle
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import namedtuple

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    print("ERROR: PyTorch and torch_geometric are required!")
    print("Install with: pip install torch torch_geometric")
    sys.exit(1)

# Constants
ICCAD04_URL = "http://vlsicad.eecs.umich.edu/BK/ICCAD04bench/"
ISPD2005_URL = "http://www.ispd.cc/contests/05/ispd05.html"

# ChipDiffusion's hardcoded ISPD chip sizes (from their utils.py)
ISPD_CHIP_SIZES = {
    "adaptec1": [0.459, 0.459, 11.151, 11.139],
    "adaptec2": [0.609, 0.616, 14.663, 14.656],
    "adaptec3": [0.036, 0.058, 23.226, 23.386],
    "adaptec4": [0.036, 0.058, 23.226, 23.386],
    "bigblue1": [0.459, 0.459, 11.151, 11.139],
    "bigblue2": [0.036, 0.076, 18.726, 18.868],
    "bigblue3": [0.036, 0.076, 27.726, 27.868],
    "bigblue4": [0.036, 0.058, 32.226, 32.386],
}

# Pin structure
Pin = namedtuple("Pin", ["obj_name", "offset", "id"])


def download_benchmarks(output_dir: str = "benchmarks"):
    """Download ICCAD04 and ISPD2005 benchmarks"""

    print("="*80)
    print("DOWNLOADING BENCHMARKS")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # ICCAD04/IBM benchmarks
    iccad_dir = os.path.join(output_dir, "iccad04")
    os.makedirs(iccad_dir, exist_ok=True)

    ibm_circuits = [f"ibm{i:02d}" for i in range(1, 19)]

    print("\nDownloading ICCAD04 (IBM) benchmarks...")
    for circuit in ibm_circuits:
        circuit_dir = os.path.join(iccad_dir, circuit)
        if os.path.exists(circuit_dir):
            print(f"  {circuit}: Already exists, skipping")
            continue

        print(f"  Downloading {circuit}...")
        url = f"{ICCAD04_URL}{circuit}.tar.gz"

        try:
            # Try wget first, fallback to curl
            subprocess.run(
                ["wget", "-q", "-O", f"{circuit}.tar.gz", url],
                check=True,
                cwd=iccad_dir
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(
                    ["curl", "-s", "-o", f"{circuit}.tar.gz", url],
                    check=True,
                    cwd=iccad_dir
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"    ERROR: Could not download {circuit}. Install wget or curl.")
                continue

        # Extract
        tar_path = os.path.join(iccad_dir, f"{circuit}.tar.gz")
        if os.path.exists(tar_path):
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(iccad_dir)
            os.remove(tar_path)
            print(f"    Extracted to {circuit_dir}")

    # ISPD2005 benchmarks
    ispd_dir = os.path.join(output_dir, "ispd2005")
    os.makedirs(ispd_dir, exist_ok=True)

    print("\nDownloading ISPD2005 benchmarks...")
    print("NOTE: ISPD2005 benchmarks require manual download from:")
    print("  http://www.ispd.cc/contests/05/ispd05.html")
    print("Please download and extract to:", ispd_dir)
    print("Expected circuits: adaptec1-4, bigblue1-4")

    print("\nDownload complete!")


def parse_bookshelf(nodes_path: str, nets_path: str, pl_path: str,
                   verbose: bool = False) -> Tuple[Data, torch.Tensor]:
    """
    Parse Bookshelf format files into ChipDiffusion Data format

    Returns:
        data: torch_geometric.data.Data with graph structure
        positions: torch.Tensor (V, 2) with component positions
    """

    def print_fn(msg):
        if verbose:
            print(msg)

    # Parse placement file first to get object order
    with open(pl_path, 'r') as f:
        pl_lines = f.readlines()

    # Updated regex to handle negative coordinates and all orientation types
    pl_pattern = re.compile(
        r"^\s*(\S+)\s+(-?\d+(?:\.\d*)?)\s+(-?\d+(?:\.\d+)?)\s*:\s*(F?[NSEW])\s*(/FIXED)?\s*$"
    )

    placement_dict = {"obj_names": [], "positions": [], "is_fixed": []}

    for line in pl_lines:
        match = re.match(pl_pattern, line)
        if match:
            placement_dict["obj_names"].append(match[1])
            placement_dict["positions"].append((float(match[2]), float(match[3])))
            placement_dict["is_fixed"].append(match[5] is not None)

    print_fn(f"Parsed {len(placement_dict['positions'])} placed objects")

    # Parse nodes file
    with open(nodes_path, 'r') as f:
        nodes_lines = f.readlines()

    # Updated regex to handle trailing whitespace
    node_pattern = re.compile(
        r"^\s*(\S+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d*)?)\s+(terminal)?\s*$"
    )

    node_size_dict = {}  # name -> (width, height)
    node_is_macro_dict = {}  # name -> bool (True if terminal/macro)

    for line in nodes_lines:
        match = re.match(node_pattern, line)
        if match:
            name = match[1]
            width = float(match[2])
            height = float(match[3])
            is_terminal = match[4] is not None

            node_size_dict[name] = (width, height)
            # In Bookshelf: "terminal" keyword indicates macro/fixed block
            node_is_macro_dict[name] = is_terminal

    print_fn(f"Parsed {len(node_size_dict)} nodes")
    print_fn(f"  Macros/terminals: {sum(node_is_macro_dict.values())}")

    # Parse nets file
    with open(nets_path, 'r') as f:
        nets_lines = f.readlines()

    # Regex patterns for net parsing (handle scientific notation)
    net_line_with_coords = re.compile(
        r"^\s*(\S+)\s+(I|O|B)\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
    )
    net_line_no_coords = re.compile(r"^\s*(\S+)\s+(I|O|B)\s*$")

    nets = []
    current_net = None
    pin_id_counter = 0

    for line in nets_lines:
        line = line.strip()

        if line.startswith("NetDegree"):
            if current_net and "output" in current_net:
                nets.append(current_net)
            current_net = {"inputs": []}
        elif current_net is not None:
            # Try with coordinates first
            match = re.match(net_line_with_coords, line)
            if match:
                obj_name = match[1]
                pin_type = match[2]
                offset_x = float(match[3])
                offset_y = float(match[4])
            else:
                # Try without coordinates
                match = re.match(net_line_no_coords, line)
                if match:
                    obj_name = match[1]
                    pin_type = match[2]
                    offset_x = 0.0
                    offset_y = 0.0
                else:
                    continue

            pin = Pin(obj_name, (offset_x, offset_y), pin_id_counter)
            pin_id_counter += 1

            if pin_type == "I":
                current_net["inputs"].append(pin)
            elif pin_type in ["O", "B"]:  # B = bidirectional (IBM format)
                if "output" not in current_net:
                    current_net["output"] = pin

    if current_net and "output" in current_net:
        nets.append(current_net)

    print_fn(f"Parsed {len(nets)} nets")

    # Build graph structure
    is_macros = []
    cond_x = []  # Component sizes
    positions = []
    name_index_mapping = {}

    skipped_objects = 0
    for obj_name in placement_dict["obj_names"]:
        if obj_name not in node_size_dict:
            skipped_objects += 1
            continue

        obj_idx = len(name_index_mapping)
        name_index_mapping[obj_name] = obj_idx

        is_macros.append(node_is_macro_dict[obj_name])
        cond_x.append(node_size_dict[obj_name])

        # Get position from placement dict
        pl_idx = placement_dict["obj_names"].index(obj_name)
        positions.append(placement_dict["positions"][pl_idx])

    if skipped_objects > 0:
        print_fn(f"Skipped {skipped_objects} objects not in nodes file")

    # Build edges (clique expansion of hypergraph)
    edge_indices = []
    edge_attrs = []
    skipped_nets = 0

    for net in nets:
        src_pin = net["output"]

        # Skip nets involving unplaced objects
        if src_pin.obj_name not in name_index_mapping:
            skipped_nets += 1
            continue

        for sink_pin in net["inputs"]:
            if sink_pin.obj_name not in name_index_mapping:
                continue

            src_idx = name_index_mapping[src_pin.obj_name]
            sink_idx = name_index_mapping[sink_pin.obj_name]

            edge_indices.append((src_idx, sink_idx))
            edge_attrs.append((*src_pin.offset, *sink_pin.offset))

    if skipped_nets > 0:
        print_fn(f"Skipped {skipped_nets} nets involving unplaced objects")

    # Convert to tensors
    is_macros = torch.tensor(is_macros, dtype=torch.bool)
    is_ports = torch.zeros_like(is_macros)  # No ports in ICCAD/ISPD
    cond_x = torch.tensor(cond_x, dtype=torch.float32)
    positions = torch.tensor(positions, dtype=torch.float32)

    # Create undirected edges
    edge_index = torch.tensor(edge_indices, dtype=torch.int64).T  # (2, E)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    # Make undirected by adding reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr[:, [2, 3, 0, 1]]], dim=0)

    # Create Data object
    data = Data(
        x=cond_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        is_macros=is_macros,
        is_ports=is_ports,
    )

    print_fn(f"Created graph: {data.num_nodes} nodes, {data.num_edges} edges")
    print_fn(f"  Macros: {is_macros.sum().item()}")

    return data, positions


def extract_chip_dimensions(pl_path: str, scl_path: Optional[str] = None) -> List[float]:
    """Extract chip dimensions from .pl or .scl file"""

    # Try .scl file first if it exists
    if scl_path and os.path.exists(scl_path):
        with open(scl_path, 'r') as f:
            for line in f:
                if line.strip().startswith("Sitemap"):
                    parts = line.split()
                    if len(parts) >= 4:
                        width = float(parts[2]) / 1000  # Convert to ChipDiffusion units
                        height = float(parts[3]) / 1000
                        return [0, 0, width, height]

    # Fallback to .pl file
    with open(pl_path, 'r') as f:
        lines = f.readlines()

    coords = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            try:
                x = float(parts[1])
                y = float(parts[2].rstrip(':'))
                coords.append((x, y))
            except ValueError:
                continue

    if coords:
        xs, ys = zip(*coords)
        width = (max(xs) - min(xs)) / 1000  # Convert to ChipDiffusion units
        height = (max(ys) - min(ys)) / 1000
        return [0, 0, width, height]

    return [0, 0, 1, 1]  # Default


def extract_macro_only(data: Data, positions: torch.Tensor) -> Tuple[Data, torch.Tensor]:
    """
    Extract macro-only dataset matching ChipDiffusion's method exactly.

    This replicates ChipDiffusion's remove_non_macros() function:
    - Keep only nodes where is_macros == True
    - Keep only edges connecting two macros
    - Remap node indices
    """

    # Get macro mask
    macro_mask = data.is_macros
    num_macros = macro_mask.sum().item()

    if num_macros == 0:
        print("  WARNING: No macros found!")
        return None, None

    # Create index mapping: old_idx -> new_idx
    macro_indices = torch.where(macro_mask)[0]
    old_to_new = torch.full((data.num_nodes,), -1, dtype=torch.long)
    old_to_new[macro_indices] = torch.arange(num_macros)

    # Filter edges: keep only macro-to-macro connections
    edge_mask = macro_mask[data.edge_index[0]] & macro_mask[data.edge_index[1]]

    if edge_mask.sum() == 0:
        print(f"  WARNING: No edges between macros! (Macros: {num_macros})")
        # Create empty edge tensors
        new_edge_index = torch.zeros((2, 0), dtype=torch.long)
        new_edge_attr = torch.zeros((0, 4), dtype=torch.float32)
    else:
        # Remap edges to new indices
        old_edge_index = data.edge_index[:, edge_mask]
        new_edge_index = old_to_new[old_edge_index]
        new_edge_attr = data.edge_attr[edge_mask]

    # Create new data object
    macro_data = Data(
        x=data.x[macro_mask],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        is_macros=data.is_macros[macro_mask],
        is_ports=data.is_ports[macro_mask],
        chip_size=data.chip_size,
    )

    # Filter positions
    macro_positions = positions[macro_mask]

    return macro_data, macro_positions


def convert_benchmark(circuit_name: str, circuit_dir: str, output_dir: str,
                     file_idx: int, verbose: bool = False) -> bool:
    """Convert a single benchmark to ChipDiffusion format"""

    nodes_path = os.path.join(circuit_dir, f"{circuit_name}.nodes")
    nets_path = os.path.join(circuit_dir, f"{circuit_name}.nets")
    pl_path = os.path.join(circuit_dir, f"{circuit_name}.pl")
    scl_path = os.path.join(circuit_dir, f"{circuit_name}.scl")

    if not all(os.path.exists(p) for p in [nodes_path, nets_path, pl_path]):
        print(f"  ERROR: Missing required files for {circuit_name}")
        return False

    # Parse bookshelf files
    data, positions = parse_bookshelf(nodes_path, nets_path, pl_path, verbose=verbose)

    # Determine chip dimensions
    if circuit_name in ISPD_CHIP_SIZES:
        chip_size = ISPD_CHIP_SIZES[circuit_name]
    else:
        chip_size = extract_chip_dimensions(pl_path, scl_path)

    data.chip_size = torch.tensor(chip_size, dtype=torch.float32)

    # Save full benchmark
    graph_path = os.path.join(output_dir, f"graph{file_idx}.pickle")
    output_path = os.path.join(output_dir, f"output{file_idx}.pickle")

    with open(graph_path, 'wb') as f:
        pickle.dump(data, f)

    with open(output_path, 'wb') as f:
        pickle.dump(positions.numpy(), f)

    print(f"  ✓ {circuit_name}: {data.num_nodes} components, "
          f"{data.is_macros.sum().item()} macros, {data.num_edges} edges")

    return True


def main():
    """Main conversion pipeline"""

    print("\n" + "="*80)
    print("ChipDiffusion Benchmark Preparation")
    print("="*80)

    # Step 1: Download benchmarks
    download_benchmarks()

    # Step 2: Convert benchmarks
    print("\n" + "="*80)
    print("CONVERTING TO CHIPDIFFUSION FORMAT")
    print("="*80)

    # ICCAD04 (IBM) benchmarks
    print("\nConverting ICCAD04 (IBM) benchmarks...")
    iccad_output = "datasets/graph/iccad04"
    os.makedirs(iccad_output, exist_ok=True)

    ibm_circuits = [f"ibm{i:02d}" for i in range(1, 19)]
    for idx, circuit in enumerate(ibm_circuits):
        circuit_dir = os.path.join("benchmarks/iccad04", circuit)
        if os.path.exists(circuit_dir):
            convert_benchmark(circuit, circuit_dir, iccad_output, idx, verbose=False)

    # ISPD2005 benchmarks
    print("\nConverting ISPD2005 benchmarks...")
    ispd_output = "datasets/graph/ispd2005"
    os.makedirs(ispd_output, exist_ok=True)

    ispd_circuits = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
                     "bigblue1", "bigblue2", "bigblue3", "bigblue4"]
    for idx, circuit in enumerate(ispd_circuits):
        circuit_dir = os.path.join("benchmarks/ispd2005", circuit)
        if os.path.exists(circuit_dir):
            convert_benchmark(circuit, circuit_dir, ispd_output, idx, verbose=False)

    # Step 3: Create macro-only datasets
    print("\n" + "="*80)
    print("CREATING MACRO-ONLY DATASETS")
    print("="*80)

    # Macro-only IBM
    print("\nCreating macro-only IBM dataset...")
    macro_ibm_output = "datasets/graph/macro-ibm"
    os.makedirs(macro_ibm_output, exist_ok=True)

    macro_count = 0
    for idx in range(18):
        graph_path = os.path.join(iccad_output, f"graph{idx}.pickle")
        output_path = os.path.join(iccad_output, f"output{idx}.pickle")

        if not os.path.exists(graph_path):
            continue

        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
        with open(output_path, 'rb') as f:
            positions = torch.from_numpy(pickle.load(f))

        macro_data, macro_positions = extract_macro_only(data, positions)

        if macro_data is not None:
            macro_graph_path = os.path.join(macro_ibm_output, f"graph{idx}.pickle")
            macro_output_path = os.path.join(macro_ibm_output, f"output{idx}.pickle")

            with open(macro_graph_path, 'wb') as f:
                pickle.dump(macro_data, f)
            with open(macro_output_path, 'wb') as f:
                pickle.dump(macro_positions.numpy(), f)

            print(f"  ✓ IBM{idx+1:02d}: {macro_data.num_nodes} macros, "
                  f"{macro_data.num_edges} edges")
            macro_count += 1

    print(f"\nCreated macro-only IBM dataset: {macro_count} circuits")

    # Macro-only ISPD
    print("\nCreating macro-only ISPD2005 dataset...")
    macro_ispd_output = "datasets/graph/macro-ispd"
    os.makedirs(macro_ispd_output, exist_ok=True)

    macro_count = 0
    for idx in range(8):
        graph_path = os.path.join(ispd_output, f"graph{idx}.pickle")
        output_path = os.path.join(ispd_output, f"output{idx}.pickle")

        if not os.path.exists(graph_path):
            continue

        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
        with open(output_path, 'rb') as f:
            positions = torch.from_numpy(pickle.load(f))

        macro_data, macro_positions = extract_macro_only(data, positions)

        if macro_data is not None:
            macro_graph_path = os.path.join(macro_ispd_output, f"graph{idx}.pickle")
            macro_output_path = os.path.join(macro_ispd_output, f"output{idx}.pickle")

            with open(macro_graph_path, 'wb') as f:
                pickle.dump(macro_data, f)
            with open(macro_output_path, 'wb') as f:
                pickle.dump(macro_positions.numpy(), f)

            print(f"  ✓ {ispd_circuits[idx]}: {macro_data.num_nodes} macros, "
                  f"{macro_data.num_edges} edges")
            macro_count += 1

    print(f"\nCreated macro-only ISPD dataset: {macro_count} circuits")

    # Summary
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print("\nOutput directories:")
    print(f"  Full IBM:        {iccad_output}")
    print(f"  Full ISPD2005:   {ispd_output}")
    print(f"  Macro-only IBM:  {macro_ibm_output}")
    print(f"  Macro-only ISPD: {macro_ispd_output}")
    print("\nThese datasets match ChipDiffusion's format exactly.")


if __name__ == "__main__":
    main()

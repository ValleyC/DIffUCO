"""
Diagnostic script to verify overlap and boundary penalties are being computed correctly.

This directly tests the ChipPlacementEnergy class with a synthetic example where
all components are stacked at (1.0, 1.0) to verify penalties are calculated.
"""

import jax
import jax.numpy as jnp
import jraph
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EnergyFunctions.ChipPlacementEnergy import ChipPlacementEnergyClass


def create_test_graph_stacked(num_components=10):
    """
    Create a test graph with all components stacked at (1.0, 1.0).
    This should produce:
    - Very low HPWL (components close together)
    - High overlap penalty (all components overlapping)
    - High boundary penalty (all exceed canvas bounds)
    """

    # Component sizes (vary for realism)
    sizes = jnp.array([
        [0.15, 0.12],
        [0.18, 0.15],
        [0.10, 0.10],
        [0.20, 0.18],
        [0.12, 0.14],
        [0.16, 0.13],
        [0.14, 0.11],
        [0.17, 0.16],
        [0.11, 0.09],
        [0.19, 0.17],
    ])[:num_components]

    # All positions at (1.0, 1.0) - stacked!
    positions = jnp.ones((num_components, 2))  # All (1.0, 1.0)

    # Create simple netlist (star topology)
    # Center node (0) connected to all others
    senders = []
    receivers = []
    for i in range(1, num_components):
        senders.append(0)
        receivers.append(i)
        # Add reverse edges
        senders.append(i)
        receivers.append(0)

    senders = jnp.array(senders)
    receivers = jnp.array(receivers)

    # Create jraph graph
    graph = jraph.GraphsTuple(
        nodes=sizes,  # Node features = component sizes
        edges=jnp.zeros((len(senders), 4)),  # No terminal offsets for simplicity
        senders=senders,
        receivers=receivers,
        globals=None,
        n_node=jnp.array([num_components]),
        n_edge=jnp.array([len(senders)])
    )

    node_gr_idx = jnp.zeros(num_components, dtype=jnp.int32)  # All in graph 0

    return graph, positions, node_gr_idx, sizes


def create_test_graph_spread(num_components=10):
    """
    Create a test graph with components spread out (non-overlapping).
    This should produce:
    - Higher HPWL (components far apart)
    - Zero overlap penalty (no overlaps)
    - Zero boundary penalty (all within canvas)
    """

    # Component sizes
    sizes = jnp.array([
        [0.15, 0.12],
        [0.18, 0.15],
        [0.10, 0.10],
        [0.20, 0.18],
        [0.12, 0.14],
        [0.16, 0.13],
        [0.14, 0.11],
        [0.17, 0.16],
        [0.11, 0.09],
        [0.19, 0.17],
    ])[:num_components]

    # Positions spread out in a grid
    grid_size = int(jnp.ceil(jnp.sqrt(num_components)))
    positions = []
    for i in range(num_components):
        row = i // grid_size
        col = i % grid_size
        x = -0.8 + (col / grid_size) * 1.6
        y = -0.8 + (row / grid_size) * 1.6
        positions.append([x, y])
    positions = jnp.array(positions)

    # Same netlist as stacked
    senders = []
    receivers = []
    for i in range(1, num_components):
        senders.append(0)
        receivers.append(i)
        senders.append(i)
        receivers.append(0)

    senders = jnp.array(senders)
    receivers = jnp.array(receivers)

    graph = jraph.GraphsTuple(
        nodes=sizes,
        edges=jnp.zeros((len(senders), 4)),
        senders=senders,
        receivers=receivers,
        globals=None,
        n_node=jnp.array([num_components]),
        n_edge=jnp.array([len(senders)])
    )

    node_gr_idx = jnp.zeros(num_components, dtype=jnp.int32)

    return graph, positions, node_gr_idx, sizes


def main():
    print("=" * 80)
    print("CHIP PLACEMENT ENERGY DIAGNOSTIC")
    print("=" * 80)
    print("\nTesting overlap and boundary penalty computation...")
    print("\nThis diagnostic creates two scenarios:")
    print("1. STACKED: All components at (1.0, 1.0) - should have HIGH penalties")
    print("2. SPREAD: Components spread out in grid - should have LOW/ZERO penalties")
    print()

    # Create energy function
    config = {
        "continuous_dim": 2,
        "overlap_weight": 10.0,
        "boundary_weight": 10.0,
        "canvas_x_min": -1.0,
        "canvas_y_min": -1.0,
        "canvas_width": 2.0,
        "canvas_height": 2.0,
    }

    energy_fn = ChipPlacementEnergyClass(config)

    num_components = 10

    # Test 1: Stacked configuration
    print("\n" + "-" * 80)
    print("TEST 1: STACKED CONFIGURATION")
    print("-" * 80)
    graph, positions, node_gr_idx, sizes = create_test_graph_stacked(num_components)

    print(f"\nSetup:")
    print(f"  Number of components: {num_components}")
    print(f"  All positions: (1.0, 1.0)")
    print(f"  Component sizes (first 5):")
    for i in range(min(5, num_components)):
        print(f"    Component {i}: {sizes[i]}")

    # Compute energy
    energy, _, violations = energy_fn.calculate_Energy(graph, positions, node_gr_idx, sizes)

    # Compute individual components
    hpwl = energy_fn._compute_hpwl(graph, positions, node_gr_idx, 1)
    overlap = energy_fn._compute_overlap_penalty(positions, sizes, node_gr_idx, 1)
    boundary = energy_fn._compute_boundary_penalty(positions, sizes, node_gr_idx, 1)

    print(f"\nEnergy Components:")
    print(f"  HPWL:              {float(hpwl[0]):.6f}")
    print(f"  Overlap penalty:   {float(overlap[0]):.6f}")
    print(f"  Boundary penalty:  {float(boundary[0]):.6f}")
    print(f"  Total overlap term (weight * penalty): {float(config['overlap_weight'] * overlap[0]):.6f}")
    print(f"  Total boundary term (weight * penalty): {float(config['boundary_weight'] * boundary[0]):.6f}")
    print(f"\nTotal Energy:        {float(energy[0, 0]):.6f}")
    print(f"Constraint violations: {float(violations[0, 0]):.6f}")

    # Analyze
    print(f"\nAnalysis:")
    if overlap[0] > 1.0:
        print(f"  ✓ Overlap penalty is SIGNIFICANT ({float(overlap[0]):.2f})")
    else:
        print(f"  ✗ Overlap penalty is TOO SMALL ({float(overlap[0]):.6f}) - BUG!")

    if boundary[0] > 0.1:
        print(f"  ✓ Boundary penalty is SIGNIFICANT ({float(boundary[0]):.2f})")
    else:
        print(f"  ✗ Boundary penalty is TOO SMALL ({float(boundary[0]):.6f}) - BUG!")

    expected_min_energy = float(config['overlap_weight'] * overlap[0] + config['boundary_weight'] * boundary[0])
    if float(energy[0, 0]) >= expected_min_energy * 0.9:  # Allow 10% tolerance
        print(f"  ✓ Total energy includes penalties (expected >= {expected_min_energy:.2f})")
    else:
        print(f"  ✗ Total energy TOO LOW - penalties might not be included! (expected >= {expected_min_energy:.2f})")

    # Test 2: Spread configuration
    print("\n" + "-" * 80)
    print("TEST 2: SPREAD CONFIGURATION")
    print("-" * 80)
    graph, positions, node_gr_idx, sizes = create_test_graph_spread(num_components)

    print(f"\nSetup:")
    print(f"  Number of components: {num_components}")
    print(f"  Positions spread in grid:")
    for i in range(min(5, num_components)):
        print(f"    Component {i}: {positions[i]}")

    # Compute energy
    energy, _, violations = energy_fn.calculate_Energy(graph, positions, node_gr_idx, sizes)

    hpwl = energy_fn._compute_hpwl(graph, positions, node_gr_idx, 1)
    overlap = energy_fn._compute_overlap_penalty(positions, sizes, node_gr_idx, 1)
    boundary = energy_fn._compute_boundary_penalty(positions, sizes, node_gr_idx, 1)

    print(f"\nEnergy Components:")
    print(f"  HPWL:              {float(hpwl[0]):.6f}")
    print(f"  Overlap penalty:   {float(overlap[0]):.6f}")
    print(f"  Boundary penalty:  {float(boundary[0]):.6f}")
    print(f"\nTotal Energy:        {float(energy[0, 0]):.6f}")
    print(f"Constraint violations: {float(violations[0, 0]):.6f}")

    print(f"\nAnalysis:")
    if overlap[0] < 0.01:
        print(f"  ✓ Overlap penalty is NEAR ZERO ({float(overlap[0]):.6f}) - correct for spread layout")
    else:
        print(f"  ✗ Overlap penalty should be near zero ({float(overlap[0]):.6f})")

    if boundary[0] < 0.01:
        print(f"  ✓ Boundary penalty is NEAR ZERO ({float(boundary[0]):.6f}) - correct for in-bounds layout")
    else:
        print(f"  ✗ Boundary penalty should be near zero ({float(boundary[0]):.6f})")

    # Test 3: Verify penalty weights matter
    print("\n" + "-" * 80)
    print("TEST 3: PENALTY WEIGHT SENSITIVITY")
    print("-" * 80)

    graph, positions, node_gr_idx, sizes = create_test_graph_stacked(num_components)

    # Compute with different weights
    energy_w10, _, _ = energy_fn.calculate_Energy(graph, positions, node_gr_idx, sizes)

    config_w1 = config.copy()
    config_w1["overlap_weight"] = 1.0
    config_w1["boundary_weight"] = 1.0
    energy_fn_w1 = ChipPlacementEnergyClass(config_w1)
    energy_w1, _, _ = energy_fn_w1.calculate_Energy(graph, positions, node_gr_idx, sizes)

    config_w100 = config.copy()
    config_w100["overlap_weight"] = 100.0
    config_w100["boundary_weight"] = 100.0
    energy_fn_w100 = ChipPlacementEnergyClass(config_w100)
    energy_w100, _, _ = energy_fn_w100.calculate_Energy(graph, positions, node_gr_idx, sizes)

    print(f"\nSame stacked configuration with different weights:")
    print(f"  Weight = 1:    Energy = {float(energy_w1[0, 0]):.2f}")
    print(f"  Weight = 10:   Energy = {float(energy_w10[0, 0]):.2f}")
    print(f"  Weight = 100:  Energy = {float(energy_w100[0, 0]):.2f}")

    if energy_w100[0, 0] > energy_w10[0, 0] > energy_w1[0, 0]:
        print(f"\n  ✓ Energy increases with penalty weight - penalties ARE being applied!")
    else:
        print(f"\n  ✗ Energy doesn't scale with weight - penalties might NOT be applied!")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    graph, positions, node_gr_idx, sizes = create_test_graph_stacked(num_components)
    overlap = energy_fn._compute_overlap_penalty(positions, sizes, node_gr_idx, 1)
    boundary = energy_fn._compute_boundary_penalty(positions, sizes, node_gr_idx, 1)

    if overlap[0] > 1.0 and boundary[0] > 0.1:
        print("\n✓ Overlap and boundary penalties ARE being computed correctly!")
        print("✓ The energy function implementation is working.")
        print("\n⚠ The issue is likely in HOW the energy function is called during training:")
        print("  - Component sizes might not be passed correctly")
        print("  - Penalties might be computed but not backpropagated")
        print("  - Training might be using a different energy function")
    else:
        print("\n✗ Overlap or boundary penalties are NOT being computed correctly!")
        print("✗ There is a bug in the energy function implementation.")

    print("\nNext steps:")
    print("1. Run this diagnostic script on your system")
    print("2. If penalties are correct here, check training code")
    print("3. Add debug prints to training loop to verify energy components")
    print("=" * 80)


if __name__ == "__main__":
    main()

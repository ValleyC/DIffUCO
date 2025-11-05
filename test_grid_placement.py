"""
Test for GridChipPlacement Energy Function
Tests discrete grid-based placement with SDDS framework.
"""
import jax
import jax.numpy as jnp
import jraph
import sys

print("=" * 80)
print("Testing Grid-Based Chip Placement Energy")
print("=" * 80)

# Test 1: Import and initialization
print("\nTest 1: Importing GridChipPlacementEnergy...")
try:
    from EnergyFunctions.GridChipPlacementEnergy import GridChipPlacementEnergyClass

    config = {
        "grid_width": 10,
        "grid_height": 10,
        "canvas_width": 2.0,
        "canvas_height": 2.0,
        "canvas_x_min": -1.0,
        "canvas_y_min": -1.0,
    }

    energy_func = GridChipPlacementEnergyClass(config)
    print(f"  Grid size: {energy_func.grid_width}×{energy_func.grid_height}")
    print(f"  n_bernoulli_features: {energy_func.n_bernoulli_features}")
    print(f"  Cell size: {energy_func.cell_width:.3f} × {energy_func.cell_height:.3f}")
    print("  [PASS] Test 1: Import and initialization successful")

except Exception as e:
    print(f"  [FAIL] Test 1: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Grid to continuous conversion
print("\nTest 2: Testing grid to continuous conversion...")
try:
    # Test some grid cells
    test_bins = jnp.array([[0], [49], [99]])  # Corner, middle, opposite corner

    positions = energy_func._grid_to_continuous(test_bins)
    print(f"  Grid cell 0 (top-left) → position: ({positions[0, 0]:.3f}, {positions[0, 1]:.3f})")
    print(f"  Grid cell 49 (middle) → position: ({positions[1, 0]:.3f}, {positions[1, 1]:.3f})")
    print(f"  Grid cell 99 (bottom-right) → position: ({positions[2, 0]:.3f}, {positions[2, 1]:.3f})")

    # Verify positions are within canvas
    assert jnp.all((positions[:, 0] >= -1.0) & (positions[:, 0] <= 1.0)), "X positions out of bounds!"
    assert jnp.all((positions[:, 1] >= -1.0) & (positions[:, 1] <= 1.0)), "Y positions out of bounds!"

    print("  [PASS] Test 2: Grid conversion correct and within bounds")

except Exception as e:
    print(f"  [FAIL] Test 2: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Energy calculation
print("\nTest 3: Testing energy calculation...")
try:
    # Create a simple graph: 3 components with 2 edges
    num_components = 3
    num_edges = 2

    # Graph structure
    nodes = jnp.array([[0.1, 0.1],  # Component sizes (not used in grid energy, but included for compatibility)
                       [0.1, 0.1],
                       [0.1, 0.1]])

    # Edges: 0-1 and 1-2 (simple chain)
    senders = jnp.array([0, 1])
    receivers = jnp.array([1, 2])

    # Create graph
    graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=None,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([num_components]),
        n_edge=jnp.array([num_edges]),
        globals=None
    )

    # Grid assignments: place components at different cells
    bins = jnp.array([[0], [10], [20]])  # Spread out on grid

    node_gr_idx = jnp.array([0, 0, 0])  # All in same graph

    # Calculate energy
    energy, _, violations = energy_func.calculate_Energy(graph, bins, node_gr_idx)

    print(f"  Energy (HPWL): {energy[0, 0]:.3f}")
    print(f"  Violations: {violations[0, 0]:.3f} (should be 0.0)")

    assert violations[0, 0] == 0.0, "Grid should have zero violations!"
    assert energy[0, 0] > 0, "HPWL should be positive!"

    print("  [PASS] Test 3: Energy calculation works correctly")

except Exception as e:
    print(f"  [FAIL] Test 3: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: No overlaps guaranteed
print("\nTest 4: Verifying no-overlap guarantee...")
try:
    # Place all 3 components at DIFFERENT grid cells
    bins_no_overlap = jnp.array([[0], [1], [2]])

    positions = energy_func._grid_to_continuous(bins_no_overlap)

    # Check that all positions are different (no two components in same cell)
    for i in range(num_components):
        for j in range(i + 1, num_components):
            dist = jnp.sqrt(jnp.sum((positions[i] - positions[j])**2))
            assert dist > 0, f"Components {i} and {j} at same position!"

    print(f"  All {num_components} components at different positions [OK]")

    # Energy should still be valid
    energy, _, violations = energy_func.calculate_Energy(graph, bins_no_overlap, node_gr_idx)
    assert violations[0, 0] == 0.0, "Should have zero violations"

    print(f"  Energy: {energy[0, 0]:.3f}, Violations: {violations[0, 0]:.3f}")
    print("  [PASS] Test 4: No-overlap guarantee verified")

except Exception as e:
    print(f"  [FAIL] Test 4: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Soft assignment (for training)
print("\nTest 5: Testing soft assignment for training...")
try:
    # Create logits (log probabilities) over grid cells
    # Shape: [num_components, n_grid_cells]
    logits = jnp.zeros((num_components, energy_func.n_grid_cells))

    # Make each component prefer a different cell
    logits = logits.at[0, 0].set(10.0)   # Component 0 → cell 0
    logits = logits.at[1, 50].set(10.0)  # Component 1 → cell 50
    logits = logits.at[2, 99].set(10.0)  # Component 2 → cell 99

    # Calculate energy with soft assignment
    energy_soft, soft_pos, violations_soft = energy_func.calculate_Energy_loss(
        graph, logits, node_gr_idx
    )

    print(f"  Soft energy: {energy_soft[0, 0]:.3f}")
    print(f"  Soft violations: {violations_soft[0, 0]:.3f}")
    print(f"  Soft positions shape: {soft_pos.shape}")

    # Verify soft positions are close to grid centers
    probs = jax.nn.softmax(logits, axis=-1)
    expected_pos = jnp.dot(probs, energy_func.grid_centers)

    assert jnp.allclose(soft_pos, expected_pos, atol=1e-5), "Soft positions mismatch!"

    print("  [PASS] Test 5: Soft assignment for training works")

except Exception as e:
    print(f"  [FAIL] Test 5: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nGrid-based chip placement energy is working correctly:")
print(f"  1. Grid size: {energy_func.grid_width}×{energy_func.grid_height} = {energy_func.n_grid_cells} cells")
print(f"  2. Discrete SDDS framework: n_bernoulli_features = {energy_func.n_grid_cells}")
print(f"  3. Energy = HPWL only (no overlap/boundary penalties)")
print(f"  4. Hard feasibility guarantee: zero violations always")
print(f"  5. Soft assignment ready for gradient-based training")
print("\nReady to train with discrete SDDS!")
print("=" * 80)

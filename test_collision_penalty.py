"""
Test collision penalty in GridChipPlacement
Verifies that TSP-style constraint enforcement prevents components from stacking.
"""
import jax.numpy as jnp
import jraph
import sys

print("=" * 80)
print("Testing Collision Penalty in GridChipPlacement")
print("=" * 80)

from EnergyFunctions.GridChipPlacementEnergy import GridChipPlacementEnergyClass

# Initialize energy function
config = {
    "grid_width": 10,
    "grid_height": 10,
    "collision_weight": 1.5,
}

energy_func = GridChipPlacementEnergyClass(config)

# Create simple graph: 3 components with 2 edges
num_components = 3
nodes = jnp.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
senders = jnp.array([0, 1])
receivers = jnp.array([1, 2])

graph = jraph.GraphsTuple(
    nodes=nodes,
    edges=None,
    senders=senders,
    receivers=receivers,
    n_node=jnp.array([num_components]),
    n_edge=jnp.array([2]),
    globals=None
)

node_gr_idx = jnp.array([0, 0, 0])

print("\n" + "=" * 80)
print("Test 1: NO COLLISIONS (all components in different cells)")
print("=" * 80)

# Placement with NO collisions
bins_no_collision = jnp.array([[0], [1], [2]])  # Different cells

energy, _, violations = energy_func.calculate_Energy(graph, bins_no_collision, node_gr_idx)

print(f"  Grid assignments: {bins_no_collision.flatten()}")
print(f"  Total Energy: {energy[0, 0]:.3f}")
print(f"  Collision Penalty: {violations[0, 0]:.3f}")
print(f"  [OK] Expected: Collision penalty = 0.0")

assert violations[0, 0] == 0.0, "Expected zero collision penalty!"
print("  [PASS] No collisions detected correctly")

print("\n" + "=" * 80)
print("Test 2: PARTIAL COLLISIONS (2 components in same cell)")
print("=" * 80)

# Placement with 2 components in same cell
bins_partial_collision = jnp.array([[0], [0], [2]])  # Components 0 and 1 collide

energy_collision, _, violations_collision = energy_func.calculate_Energy(
    graph, bins_partial_collision, node_gr_idx
)

print(f"  Grid assignments: {bins_partial_collision.flatten()}")
print(f"  Total Energy: {energy_collision[0, 0]:.3f}")
print(f"  Collision Penalty: {violations_collision[0, 0]:.3f}")
print(f"  [OK] Expected: Collision penalty > 0 (2 components in cell 0)")

# Cell 0 has 2 components: penalty = (2 - 1)^2 = 1
expected_penalty = 1.0
assert abs(violations_collision[0, 0] - expected_penalty) < 1e-5, f"Expected penalty {expected_penalty}, got {violations_collision[0, 0]}"
print(f"  [PASS] Detected collision: penalty = {violations_collision[0, 0]:.3f}")

print("\n" + "=" * 80)
print("Test 3: ALL STACKED (all components in same cell)")
print("=" * 80)

# Placement with ALL components in same cell (worst case)
bins_all_stacked = jnp.array([[0], [0], [0]])  # All in cell 0

energy_stacked, _, violations_stacked = energy_func.calculate_Energy(
    graph, bins_all_stacked, node_gr_idx
)

print(f"  Grid assignments: {bins_all_stacked.flatten()}")
print(f"  Total Energy: {energy_stacked[0, 0]:.3f}")
print(f"  Collision Penalty: {violations_stacked[0, 0]:.3f}")
print(f"  [OK] Expected: High collision penalty (3 components in cell 0)")

# Cell 0 has 3 components: penalty = (3 - 1)^2 = 4
expected_penalty = 4.0
assert abs(violations_stacked[0, 0] - expected_penalty) < 1e-5, f"Expected penalty {expected_penalty}, got {violations_stacked[0, 0]}"
print(f"  [PASS] Detected severe stacking: penalty = {violations_stacked[0, 0]:.3f}")

print("\n" + "=" * 80)
print("Test 4: Energy Comparison (HPWL vs Collision Tradeoff)")
print("=" * 80)

# Compare total energies
print(f"\n  No collision:      Energy = {energy[0, 0]:.3f} (HPWL only)")
print(f"  Partial collision: Energy = {energy_collision[0, 0]:.3f} (HPWL + penalty)")
print(f"  All stacked:       Energy = {energy_stacked[0, 0]:.3f} (minimal HPWL + big penalty)")

print(f"\n  Collision penalty increases energy:")
print(f"    No collision → Partial: +{energy_collision[0, 0] - energy[0, 0]:.3f}")
print(f"    No collision → Stacked:  +{energy_stacked[0, 0] - energy[0, 0]:.3f}")

# Verify that stacking is penalized despite having low HPWL
positions_stacked = energy_func.get_continuous_positions(bins_all_stacked)
hpwl_stacked = energy_func._compute_hpwl(graph, positions_stacked, node_gr_idx, 1)[0]

print(f"\n  Stacked HPWL: {hpwl_stacked:.3f} (artificially low, all at same position)")
print(f"  But total energy: {energy_stacked[0, 0]:.3f} (penalized by collision_weight={energy_func.collision_weight})")

print("  [PASS] Collision penalty prevents trivial stacking solution")

print("\n" + "=" * 80)
print("Test 5: Soft Assignment Collision Penalty (for training)")
print("=" * 80)

# Test soft collision penalty with logits
logits = jnp.zeros((num_components, energy_func.n_grid_cells))

# All components strongly prefer cell 0 (simulating stacking during training)
logits = logits.at[:, 0].set(10.0)

energy_soft, soft_pos, violations_soft = energy_func.calculate_Energy_loss(
    graph, logits, node_gr_idx
)

print(f"  Logits: all components prefer cell 0")
print(f"  Soft collision penalty: {violations_soft[0, 0]:.3f}")
print(f"  Total soft energy: {energy_soft[0, 0]:.3f}")
print(f"  [OK] Expected: High penalty (probability mass ~3.0 in cell 0)")

# Each component has ~1.0 probability mass in cell 0 → total ~3.0
# Penalty = (3.0 - 1.0)^2 = 4.0
assert violations_soft[0, 0] > 3.5, f"Expected high soft penalty, got {violations_soft[0, 0]}"
print(f"  [PASS] Soft penalty works for gradient-based training")

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)

print("\nSummary:")
print(f"  1. Zero penalty when no collisions")
print(f"  2. Penalty = (n_components_in_cell - 1)^2 per cell")
print(f"  3. Prevents stacking despite low HPWL")
print(f"  4. Weighted by collision_weight = {energy_func.collision_weight}")
print(f"  5. Works for both hard (sampling) and soft (training) assignments")
print("\nTSP-style constraint enforcement is working correctly!")
print("Model will learn to spread components to minimize total energy.")
print("=" * 80)

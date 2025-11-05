from .BaseEnergy import BaseEnergyClass
from functools import partial
import jax
import jax.numpy as jnp
import jraph


class GridChipPlacementEnergyClass(BaseEnergyClass):
    """
    Grid-based chip placement energy function for discrete SDDS.

    Unlike continuous ChipPlacementEnergy, this uses a DISCRETE grid:
    - Each component chooses 1 grid cell from {0, 1, ..., grid_size-1}²
    - Grid guarantees no overlaps (different cells = no overlap)
    - Grid guarantees no boundary violations (all cells within canvas)
    - Energy = HPWL only (no penalty terms needed!)

    Framework:
    - Uses discrete SDDS (like MaxCut)
    - n_bernoulli_features = grid_width * grid_height (flattened grid)
    - bins[i] ∈ {0, 1, ..., n_bernoulli_features-1} = grid cell index

    Example: 10×10 grid
    - n_bernoulli_features = 100
    - bins[0] = 37 → grid position (3, 7) → continuous position (x, y)
    - Discrete diffusion: categorical distribution over 100 cells

    Advantages over continuous:
    - Hard feasibility guarantee (no overlaps/violations)
    - Simpler energy function (HPWL only)
    - Proven discrete SDDS framework

    Disadvantages:
    - Quantization error (~cell_size/2 per component)
    - Large grid → large action space (grid_size² choices)
    """

    def __init__(self, config):
        # Grid configuration
        self.grid_width = config.get("grid_width", 10)
        self.grid_height = config.get("grid_height", 10)
        self.n_grid_cells = self.grid_width * self.grid_height

        # Set n_bernoulli_features for discrete framework
        # Each component chooses 1 of n_grid_cells
        config["n_bernoulli_features"] = self.n_grid_cells

        super().__init__(config)

        # Canvas bounds
        self.canvas_width = config.get("canvas_width", 2.0)
        self.canvas_height = config.get("canvas_height", 2.0)
        self.canvas_x_min = config.get("canvas_x_min", -1.0)
        self.canvas_y_min = config.get("canvas_y_min", -1.0)

        # Collision penalty weight (TSP-style constraint enforcement)
        # Like TSP's A=1.45, this enforces unique cell assignments
        self.collision_weight = config.get("collision_weight", 1.5)

        # Grid cell dimensions
        self.cell_width = self.canvas_width / self.grid_width
        self.cell_height = self.canvas_height / self.grid_height

        # Precompute grid cell centers (for efficiency)
        grid_x = jnp.linspace(
            self.canvas_x_min + self.cell_width / 2,
            self.canvas_x_min + self.canvas_width - self.cell_width / 2,
            self.grid_width
        )
        grid_y = jnp.linspace(
            self.canvas_y_min + self.cell_height / 2,
            self.canvas_y_min + self.canvas_height - self.cell_height / 2,
            self.grid_height
        )

        # Create flattened grid: shape [grid_width * grid_height, 2]
        # grid_centers[i] = (x, y) continuous position of cell i
        xx, yy = jnp.meshgrid(grid_x, grid_y, indexing='ij')
        self.grid_centers = jnp.stack([xx.flatten(), yy.flatten()], axis=1)

        print("GridChipPlacementEnergy initialized")
        print(f"  Grid size: {self.grid_width}×{self.grid_height} = {self.n_grid_cells} cells")
        print(f"  Cell dimensions: {self.cell_width:.3f} × {self.cell_height:.3f}")
        print(f"  Canvas: [{self.canvas_x_min}, {self.canvas_x_min + self.canvas_width}] × "
              f"[{self.canvas_y_min}, {self.canvas_y_min + self.canvas_height}]")
        print(f"  n_bernoulli_features: {self.n_bernoulli_features}")
        print(f"  Collision weight: {self.collision_weight} (TSP-style constraint)")
        print(f"  Energy: HPWL + collision_penalty (like TSP assignment constraints)")
        print("______________")

    def _grid_to_continuous(self, grid_indices):
        """
        Convert grid cell indices to continuous positions.

        Args:
            grid_indices: component grid assignments [num_components, 1]
                         Each value in {0, 1, ..., n_grid_cells-1}

        Returns:
            positions: continuous positions [num_components, 2]
                      positions[i] = center of grid cell grid_indices[i]
        """
        # Flatten if needed
        if len(grid_indices.shape) > 1:
            grid_indices = jnp.squeeze(grid_indices, axis=-1)

        # Look up grid cell centers
        # grid_centers is precomputed: [n_grid_cells, 2]
        # grid_indices: [num_components]
        # Result: [num_components, 2]
        positions = self.grid_centers[grid_indices]

        return positions

    @partial(jax.jit, static_argnums=(0, 3))
    def _compute_collision_penalty(self, bins, node_gr_idx, n_graph):
        """
        Compute collision penalty (TSP-style constraint enforcement).

        Like TSP enforces each position visited exactly once:
          penalty = sum_over_cells (num_components_in_cell - 1)^2

        This penalizes multiple components in the same grid cell.

        Args:
            bins: grid cell assignments [num_components, 1]
            node_gr_idx: component to graph mapping
            n_graph: number of graphs

        Returns:
            collision_penalty_per_graph: [n_graphs,] collision penalty per graph
        """
        # Flatten bins if needed
        if len(bins.shape) > 1:
            bins_flat = jnp.squeeze(bins, axis=-1)  # [num_components]
        else:
            bins_flat = bins

        # Convert to one-hot: [num_components, n_grid_cells]
        x_mat = jax.nn.one_hot(bins_flat, num_classes=self.n_grid_cells, dtype=jnp.float32)

        # Count components per cell: sum over components
        # components_per_cell: [num_components, n_grid_cells] -> sum axis 0 -> [n_grid_cells]
        # But we need to do this per graph...

        # For each graph, compute collision penalty
        def compute_collision_for_graph(graph_id):
            # Get components belonging to this graph
            mask = (node_gr_idx == graph_id).astype(jnp.float32)  # [num_components]

            # Count components per cell for this graph
            # x_mat: [num_components, n_grid_cells]
            # mask: [num_components]
            components_per_cell = jnp.sum(x_mat * mask[:, jnp.newaxis], axis=0)  # [n_grid_cells]

            # Penalty: (count - 1)^2 for each cell (penalize > 1 component per cell)
            # If cell has 0 components: (0-1)^2 = 1 (undesirable, but okay)
            # If cell has 1 component: (1-1)^2 = 0 (perfect!)
            # If cell has 2 components: (2-1)^2 = 1 (collision!)
            # If cell has k components: (k-1)^2 (strong penalty)

            # Actually, let's only penalize cells with > 1 component:
            penalty = jnp.sum(jnp.maximum(0, components_per_cell - 1.0) ** 2)

            return penalty

        # Vectorize over graphs
        graph_ids = jnp.arange(n_graph)
        collision_penalty_per_graph = jax.vmap(compute_collision_for_graph)(graph_ids)

        return collision_penalty_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, H_graph, bins, node_gr_idx, component_sizes=None):
        """
        Calculate energy for grid-based placement with collision penalty.

        Args:
            H_graph: jraph graph structure
            bins: grid cell assignments [num_components, 1]
                  bins[i] ∈ {0, 1, ..., n_grid_cells-1}
            node_gr_idx: component to graph mapping
            component_sizes: NOT USED for grid (grid doesn't need sizes for overlap checking)
                           Included for compatibility with continuous ChipPlacement interface

        Returns:
            Energy_per_graph: (HPWL + collision_penalty) per graph [n_graphs, 1]
            bins: grid assignments (unchanged)
            violations: collision penalty (for monitoring) [n_graphs, 1]
        """
        # Note: component_sizes is ignored for grid placement
        n_graph = H_graph.n_node.shape[0]

        # Convert grid indices to continuous positions
        positions = self._grid_to_continuous(bins)

        # Compute HPWL
        hpwl_per_graph = self._compute_hpwl(H_graph, positions, node_gr_idx, n_graph)

        # Compute collision penalty (TSP-style constraint)
        collision_penalty_per_graph = self._compute_collision_penalty(bins, node_gr_idx, n_graph)

        # Energy = HPWL + weighted collision penalty
        # collision_weight acts like TSP's A parameter (typically 1.45-2.0)
        Energy_per_graph = hpwl_per_graph + self.collision_weight * collision_penalty_per_graph

        # Report collision penalty as "violations" for monitoring
        violations_per_graph = collision_penalty_per_graph

        # Ensure output shape [n_graphs, 1]
        if len(Energy_per_graph.shape) == 1:
            Energy_per_graph = jnp.expand_dims(Energy_per_graph, axis=-1)
        if len(violations_per_graph.shape) == 1:
            violations_per_graph = jnp.expand_dims(violations_per_graph, axis=-1)

        return Energy_per_graph, bins, violations_per_graph

    @partial(jax.jit, static_argnums=(0, 4))
    def _compute_hpwl(self, H_graph, positions, node_gr_idx, n_graph):
        """
        Compute Half-Perimeter Wirelength.

        Same as continuous version - HPWL is well-defined for any positions.

        Args:
            H_graph: graph structure
            positions: component positions [num_components, 2]
            node_gr_idx: component to graph mapping
            n_graph: number of graphs

        Returns:
            hpwl_per_graph: HPWL per graph [n_graphs,]
        """
        senders = H_graph.senders
        receivers = H_graph.receivers

        # Get positions of connected components
        sender_pos = positions[senders]  # [num_edges, 2]
        receiver_pos = positions[receivers]  # [num_edges, 2]

        # Bounding box for each edge (2-pin net)
        x_coords = jnp.stack([sender_pos[:, 0], receiver_pos[:, 0]], axis=1)
        y_coords = jnp.stack([sender_pos[:, 1], receiver_pos[:, 1]], axis=1)

        bbox_width = jnp.max(x_coords, axis=1) - jnp.min(x_coords, axis=1)
        bbox_height = jnp.max(y_coords, axis=1) - jnp.min(y_coords, axis=1)

        # HPWL per edge
        hpwl_per_edge = bbox_width + bbox_height

        # Aggregate to graph level
        edge_gr_idx = node_gr_idx[senders]
        hpwl_per_graph = jax.ops.segment_sum(hpwl_per_edge, edge_gr_idx, n_graph)

        return hpwl_per_graph

    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, component_sizes=None):
        """
        Calculate relaxed energy (same as regular for grid case).
        """
        return self.calculate_Energy(H_graph, bins, node_gr_idx, component_sizes)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx):
        """
        Calculate energy loss for training with collision penalty.

        During training with discrete SDDS:
        - logits: [num_components, n_grid_cells] probabilities
        - We use soft assignment for gradient flow
        - Collision penalty: penalize overlapping probability mass

        Args:
            H_graph: graph structure
            logits: log probabilities over grid cells [num_components, n_grid_cells]
            node_gr_idx: component to graph mapping

        Returns:
            Energy, soft_bins, violations
        """
        # Convert logits to probabilities
        probs = jax.nn.softmax(logits, axis=-1)  # [num_components, n_grid_cells]

        # Soft grid assignment: expected position under distribution
        # E[position] = sum_i p_i * grid_centers[i]
        # probs: [num_components, n_grid_cells]
        # grid_centers: [n_grid_cells, 2]
        # Result: [num_components, 2]
        soft_positions = jnp.dot(probs, self.grid_centers)

        # Compute HPWL with soft positions
        n_graph = H_graph.n_node.shape[0]
        hpwl_per_graph = self._compute_hpwl(H_graph, soft_positions, node_gr_idx, n_graph)

        # Compute SOFT collision penalty for training
        # For each graph, sum probability mass per cell and penalize if > 1
        def compute_soft_collision_for_graph(graph_id):
            # Get components belonging to this graph
            mask = (node_gr_idx == graph_id).astype(jnp.float32)  # [num_components]

            # Sum probability mass per cell for this graph
            # probs: [num_components, n_grid_cells]
            # mask: [num_components]
            prob_mass_per_cell = jnp.sum(probs * mask[:, jnp.newaxis], axis=0)  # [n_grid_cells]

            # Soft penalty: (prob_mass - 1)^2 if prob_mass > 1
            penalty = jnp.sum(jnp.maximum(0, prob_mass_per_cell - 1.0) ** 2)

            return penalty

        # Vectorize over graphs
        graph_ids = jnp.arange(n_graph)
        collision_penalty_per_graph = jax.vmap(compute_soft_collision_for_graph)(graph_ids)

        # Total energy with collision penalty
        Energy_per_graph = hpwl_per_graph + self.collision_weight * collision_penalty_per_graph
        violations_per_graph = collision_penalty_per_graph

        if len(Energy_per_graph.shape) == 1:
            Energy_per_graph = jnp.expand_dims(Energy_per_graph, axis=-1)
        if len(violations_per_graph.shape) == 1:
            violations_per_graph = jnp.expand_dims(violations_per_graph, axis=-1)

        return Energy_per_graph, soft_positions, violations_per_graph

    def get_HPWL_value(self, H_graph, bins, node_gr_idx, component_sizes=None):
        """
        Get HPWL value (same as energy for grid placement).

        Args:
            H_graph: graph structure
            bins: grid cell assignments
            node_gr_idx: component to graph mapping
            component_sizes: NOT USED (for compatibility)

        Returns:
            HPWL per graph
        """
        Energy, _, _ = self.calculate_Energy(H_graph, bins, node_gr_idx, component_sizes)
        return Energy

    def get_continuous_positions(self, bins):
        """
        Helper: Get continuous positions from grid assignments.

        Useful for visualization and debugging.

        Args:
            bins: grid cell assignments [num_components, 1]

        Returns:
            positions: continuous positions [num_components, 2]
        """
        return self._grid_to_continuous(bins)

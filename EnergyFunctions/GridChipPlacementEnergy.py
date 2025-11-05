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
        print(f"  Energy: HPWL only (no overlap/boundary penalties)")
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

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, H_graph, bins, node_gr_idx, component_sizes=None):
        """
        Calculate energy for grid-based placement.

        Args:
            H_graph: jraph graph structure
            bins: grid cell assignments [num_components, 1]
                  bins[i] ∈ {0, 1, ..., n_grid_cells-1}
            node_gr_idx: component to graph mapping
            component_sizes: NOT USED for grid (grid doesn't need sizes for overlap checking)
                           Included for compatibility with continuous ChipPlacement interface

        Returns:
            Energy_per_graph: HPWL per graph [n_graphs, 1]
            bins: grid assignments (unchanged)
            violations: always zero (grid guarantees feasibility)
        """
        # Note: component_sizes is ignored for grid placement
        # Grid cells guarantee no overlaps regardless of component size
        n_graph = H_graph.n_node.shape[0]

        # Convert grid indices to continuous positions
        positions = self._grid_to_continuous(bins)

        # Compute HPWL
        hpwl_per_graph = self._compute_hpwl(H_graph, positions, node_gr_idx, n_graph)

        # Energy = HPWL only (no penalties needed!)
        Energy_per_graph = hpwl_per_graph

        # No constraint violations (grid guarantees feasibility)
        violations_per_graph = jnp.zeros_like(Energy_per_graph)

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
        Calculate energy loss for training.

        During training with discrete SDDS:
        - logits: [num_components, n_grid_cells] probabilities
        - We use soft assignment (expected position) for gradient flow

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

        Energy_per_graph = hpwl_per_graph
        violations_per_graph = jnp.zeros_like(Energy_per_graph)

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

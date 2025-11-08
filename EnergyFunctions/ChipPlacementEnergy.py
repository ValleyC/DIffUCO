from .BaseEnergy import BaseEnergyClass
from functools import partial
import jax
import jax.numpy as jnp
import jraph


class ChipPlacementEnergyClass(BaseEnergyClass):
    """
    Energy function for chip placement problems.

    Energy = HPWL + overlap_weight * normalized_overlap + boundary_weight * normalized_boundary

    Where:
    - HPWL (Half-Perimeter Wirelength): sum of bounding box dimensions for all nets
    - normalized_overlap: overlap_area * HPWL (scales proportionally with circuit size)
    - normalized_boundary: boundary_area * HPWL (scales proportionally with circuit size)

    The normalization ensures that overlap_weight and boundary_weight represent
    "how many times HPWL when violation=1.0" rather than absolute penalty values.
    This makes penalties scale naturally with circuit size.

    For example, overlap_weight=5.0 means: when overlap_area=1.0, penalty = 5.0 * HPWL.
    This ensures the same weight values work across different circuit sizes (50 vs 400 components).

    This replaces MaxCutEnergyClass for continuous chip placement optimization.
    """

    def __init__(self, config):
        # Note: BaseEnergy expects n_bernoulli_features, but for continuous case we use continuous_dim
        # We'll set n_bernoulli_features = continuous_dim for compatibility
        if "continuous_dim" in config:
            config["n_bernoulli_features"] = config["continuous_dim"]
        elif "n_bernoulli_features" not in config:
            config["n_bernoulli_features"] = 2  # Default to 2D positions

        super().__init__(config)

        # Chip placement specific parameters
        self.continuous_dim = config.get("continuous_dim", 2)
        self.overlap_weight = config.get("overlap_weight", 10.0)
        self.boundary_weight = config.get("boundary_weight", 10.0)
        self.canvas_width = config.get("canvas_width", 2.0)  # Default: [-1, 1] -> width = 2
        self.canvas_height = config.get("canvas_height", 2.0)
        self.canvas_x_min = config.get("canvas_x_min", -1.0)
        self.canvas_y_min = config.get("canvas_y_min", -1.0)

        # Two-stage training approach: HPWL optimization + legalization
        self.use_constraints_in_training = config.get("use_constraints_in_training", True)
        self.spread_weight = config.get("spread_weight", 0.5)

        # Minimum distance constraint (alternative to spread)
        self.use_min_distance = config.get("use_min_distance", False)
        self.min_distance_weight = config.get("min_distance_weight", 0.1)
        self.min_distance_threshold = config.get("min_distance_threshold", 0.2)

        print("ChipPlacementEnergy initialized")
        print(f"  Continuous dim: {self.continuous_dim}")
        print(f"  Overlap weight: {self.overlap_weight} (absolute weight, no HPWL coupling)")
        print(f"  Boundary weight: {self.boundary_weight} (absolute weight, no HPWL coupling)")
        print(f"  Note: Weights represent absolute penalty importance (decoupled from HPWL)")
        print(f"  Canvas: [{self.canvas_x_min}, {self.canvas_x_min + self.canvas_width}] x [{self.canvas_y_min}, {self.canvas_y_min + self.canvas_height}]")
        print(f"\n  Two-Stage Training Configuration:")
        print(f"  Use constraints in training: {self.use_constraints_in_training}")
        if not self.use_constraints_in_training:
            print(f"  → Training mode: HPWL + spread + boundary (no overlap penalties)")
            print(f"  → Spread weight: {self.spread_weight} (prevents stacking, √N-normalized)")
            print(f"  → Boundary weight: {self.boundary_weight} (prevents out-of-bounds drift)")
            if self.use_min_distance:
                print(f"  → Min distance constraint: weight={self.min_distance_weight}, threshold={self.min_distance_threshold}")
            print(f"  → Evaluation mode: Full energy with all constraints")
            print(f"  → Post-processing: Legalization decoder can be applied after sampling")
        else:
            print(f"  → Training mode: Full energy with all constraints")
        print("______________")

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, H_graph, positions, node_gr_idx, component_sizes=None):
        """
        Calculate total energy for chip placement.

        Args:
            H_graph: jraph graph structure containing:
                - nodes: component features (if any)
                - edges: netlist connectivity
                - senders: source component indices for edges
                - receivers: sink component indices for edges
                - n_node: number of components per graph
                - n_edge: number of nets per graph
            positions: component positions (shape: [num_components, continuous_dim])
                       For 2D: positions[:, 0] = x, positions[:, 1] = y
            node_gr_idx: mapping from components to graphs
            component_sizes: component sizes (shape: [num_components, 2]) for (x_size, y_size)
                           If None, extracted from graph nodes or assumed small default

        Returns:
            Energy_per_graph: total energy per graph (shape: [n_graphs, 1])
            positions: positions (for interface compatibility)
            constraint_violations_per_graph: sum of overlap + boundary violations (shape: [n_graphs, 1])
        """
        n_graph = H_graph.n_node.shape[0]
        nodes = H_graph.nodes
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        # Extract or default component sizes
        if component_sizes is None:
            # Try to extract from graph nodes (assuming nodes contain size information)
            # If nodes shape is [num_components, features] and features >= 2, use first 2 as sizes
            if nodes.shape[-1] >= 2:
                component_sizes = nodes[:, :2]  # First 2 features are x_size, y_size
            else:
                # Default: small components (0.1 x 0.1)
                component_sizes = jnp.full((total_num_nodes, 2), 0.1)

        # Ensure positions has correct shape [num_components, continuous_dim]
        if len(positions.shape) == 3:  # [num_components, 1, continuous_dim]
            positions = positions[:, 0, :]  # Remove middle dimension
        elif len(positions.shape) == 1:  # [num_components * continuous_dim]
            positions = jnp.reshape(positions, (total_num_nodes, self.continuous_dim))

        # 1. Compute HPWL (Half-Perimeter Wirelength)
        hpwl_per_graph = self._compute_hpwl(H_graph, positions, node_gr_idx, n_graph)

        # 2. Compute overlap penalty
        overlap_per_graph = self._compute_overlap_penalty(
            positions, component_sizes, node_gr_idx, n_graph
        )

        # 3. Compute boundary penalty
        boundary_per_graph = self._compute_boundary_penalty(
            positions, component_sizes, node_gr_idx, n_graph
        )

        # FIX: Use raw penalties without HPWL coupling to prevent energy explosion
        # Old approach tried to scale penalties by HPWL, but this created dangerous coupling:
        # - Energy = HPWL + weight × (penalty × HPWL)
        # - This creates quadratic scaling with circuit size (O(N^1.5) for N components)
        # - For Chip_huge (N=400): HPWL≈5600, boundary≈40 → coupled energy = 2.2M explosion!
        #
        # New approach: Use raw penalties directly with absolute weights
        # - Energy = HPWL + weight × penalty
        # - No coupling, linear scaling with problem size
        # - Weights now represent absolute importance, not "X times HPWL"

        Energy_per_graph = (
            hpwl_per_graph +
            self.overlap_weight * overlap_per_graph +
            self.boundary_weight * boundary_per_graph
        )

        # Constraint violations (for monitoring)
        constraint_violations_per_graph = overlap_per_graph + boundary_per_graph

        # Ensure output shape matches expected format [n_graphs, 1]
        if len(Energy_per_graph.shape) == 1:
            Energy_per_graph = jnp.expand_dims(Energy_per_graph, axis=-1)
        if len(constraint_violations_per_graph.shape) == 1:
            constraint_violations_per_graph = jnp.expand_dims(constraint_violations_per_graph, axis=-1)

        return Energy_per_graph, positions, constraint_violations_per_graph

    @partial(jax.jit, static_argnums=(0, 4))
    def _compute_hpwl(self, H_graph, positions, node_gr_idx, n_graph):
        """
        Compute Half-Perimeter Wirelength.

        HPWL = sum over all nets of (bbox_width + bbox_height)

        For each net (edge in graph):
        - Find bounding box of all connected components
        - HPWL contribution = width + height of bbox

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

        # Get positions of sender and receiver components
        sender_pos = positions[senders]  # [num_edges, 2]
        receiver_pos = positions[receivers]  # [num_edges, 2]

        # For each edge, compute bounding box
        # Note: In chip placement, edges represent nets connecting two terminals
        # For 2-pin nets, bbox is simply the rectangle between the two points
        x_coords = jnp.stack([sender_pos[:, 0], receiver_pos[:, 0]], axis=1)  # [num_edges, 2]
        y_coords = jnp.stack([sender_pos[:, 1], receiver_pos[:, 1]], axis=1)  # [num_edges, 2]

        bbox_width = jnp.max(x_coords, axis=1) - jnp.min(x_coords, axis=1)  # [num_edges,]
        bbox_height = jnp.max(y_coords, axis=1) - jnp.min(y_coords, axis=1)  # [num_edges,]

        # HPWL per edge
        hpwl_per_edge = bbox_width + bbox_height  # [num_edges,]

        # Aggregate to graph level
        # Map edges to graphs via sender components
        edge_gr_idx = node_gr_idx[senders]
        hpwl_per_graph = jax.ops.segment_sum(hpwl_per_edge, edge_gr_idx, n_graph)

        return hpwl_per_graph

    @partial(jax.jit, static_argnums=(0, 4))
    def _compute_overlap_penalty(self, positions, component_sizes, node_gr_idx, n_graph):
        """
        Compute overlap penalty between components.

        For each pair of components, check if their bounding boxes overlap.
        Penalty = sum of overlap areas.

        Note: This is O(n^2) per graph, which can be expensive for large designs.
        For production, consider spatial hashing or other acceleration.

        Args:
            positions: component positions [num_components, 2]
            component_sizes: component sizes [num_components, 2] (width, height)
            node_gr_idx: component to graph mapping
            n_graph: number of graphs

        Returns:
            overlap_per_graph: total overlap area per graph [n_graphs,]
        """
        num_components = positions.shape[0]

        # Component bounding boxes: [x_min, y_min, x_max, y_max]
        # Position is center of component
        half_sizes = component_sizes / 2.0
        x_min = positions[:, 0] - half_sizes[:, 0]
        y_min = positions[:, 1] - half_sizes[:, 1]
        x_max = positions[:, 0] + half_sizes[:, 0]
        y_max = positions[:, 1] + half_sizes[:, 1]

        # Pairwise overlap computation (vectorized)
        # For components i and j, overlap exists if:
        # x_min[i] < x_max[j] AND x_max[i] > x_min[j] AND y_min[i] < y_max[j] AND y_max[i] > y_min[j]

        # Expand dimensions for broadcasting
        x_min_i = x_min[:, jnp.newaxis]  # [num_components, 1]
        x_max_i = x_max[:, jnp.newaxis]
        y_min_i = y_min[:, jnp.newaxis]
        y_max_i = y_max[:, jnp.newaxis]

        x_min_j = x_min[jnp.newaxis, :]  # [1, num_components]
        x_max_j = x_max[jnp.newaxis, :]
        y_min_j = y_min[jnp.newaxis, :]
        y_max_j = y_max[jnp.newaxis, :]

        # Overlap widths and heights (0 if no overlap)
        overlap_width = jnp.maximum(0.0, jnp.minimum(x_max_i, x_max_j) - jnp.maximum(x_min_i, x_min_j))
        overlap_height = jnp.maximum(0.0, jnp.minimum(y_max_i, y_max_j) - jnp.maximum(y_min_i, y_min_j))

        # Overlap area per pair
        overlap_area = overlap_width * overlap_height  # [num_components, num_components]

        # Only count each pair once (upper triangle, excluding diagonal)
        # Create mask: 1 for i < j, 0 otherwise
        i_indices = jnp.arange(num_components)[:, jnp.newaxis]
        j_indices = jnp.arange(num_components)[jnp.newaxis, :]
        upper_triangle_mask = (i_indices < j_indices).astype(jnp.float32)

        # Also mask to only consider pairs within same graph
        same_graph_mask = (node_gr_idx[:, jnp.newaxis] == node_gr_idx[jnp.newaxis, :]).astype(jnp.float32)

        # Combined mask
        valid_pairs_mask = upper_triangle_mask * same_graph_mask

        # Apply mask
        overlap_area_masked = overlap_area * valid_pairs_mask

        # Sum over all pairs and aggregate per graph
        # For each component i, sum overlaps where it's involved, then aggregate to graph
        overlap_per_component = jnp.sum(overlap_area_masked, axis=1)  # [num_components,]

        overlap_per_graph = jax.ops.segment_sum(overlap_per_component, node_gr_idx, n_graph)

        return overlap_per_graph

    @partial(jax.jit, static_argnums=(0, 4))
    def _compute_boundary_penalty(self, positions, component_sizes, node_gr_idx, n_graph):
        """
        Compute boundary violation penalty.

        Penalty = sum of out-of-bounds areas (how much component extends beyond canvas).

        Args:
            positions: component positions [num_components, 2]
            component_sizes: component sizes [num_components, 2]
            node_gr_idx: component to graph mapping
            n_graph: number of graphs

        Returns:
            boundary_penalty_per_graph: total out-of-bounds area per graph [n_graphs,]
        """
        # Component bounding boxes
        half_sizes = component_sizes / 2.0
        x_min = positions[:, 0] - half_sizes[:, 0]
        y_min = positions[:, 1] - half_sizes[:, 1]
        x_max = positions[:, 0] + half_sizes[:, 0]
        y_max = positions[:, 1] + half_sizes[:, 1]

        # Canvas boundaries
        canvas_x_max = self.canvas_x_min + self.canvas_width
        canvas_y_max = self.canvas_y_min + self.canvas_height

        # Out-of-bounds distances (0 if within bounds)
        left_violation = jnp.maximum(0.0, self.canvas_x_min - x_min)
        right_violation = jnp.maximum(0.0, x_max - canvas_x_max)
        bottom_violation = jnp.maximum(0.0, self.canvas_y_min - y_min)
        top_violation = jnp.maximum(0.0, y_max - canvas_y_max)

        # Approximate out-of-bounds "area" as sum of violations weighted by component dimension
        # This gives a differentiable penalty proportional to severity
        x_violation = (left_violation + right_violation) * component_sizes[:, 1]  # width violation * height
        y_violation = (bottom_violation + top_violation) * component_sizes[:, 0]  # height violation * width

        # Total boundary violation per component
        boundary_violation_per_component = x_violation + y_violation

        # Aggregate to graph level
        boundary_penalty_per_graph = jax.ops.segment_sum(
            boundary_violation_per_component, node_gr_idx, n_graph
        )

        return boundary_penalty_per_graph

    @partial(jax.jit, static_argnums=(0, 3))
    def _compute_spread_regularization(self, positions, node_gr_idx, n_graph):
        """
        Compute spread regularization to prevent stacking during HPWL-only training.

        Encourages components to spread out by rewarding high variance in positions.
        This prevents the trivial solution of stacking all components at one location.

        Spread = negative of inverse variance (higher variance = lower penalty)

        Approach:
        - Compute variance of component positions within each graph
        - Penalize LOW variance (clustered/stacked components)
        - Reward HIGH variance (well-distributed components)
        - Normalize by √N so spread_weight has consistent meaning across problem sizes

        Args:
            positions: component positions [num_components, 2]
            node_gr_idx: component to graph mapping
            n_graph: number of graphs

        Returns:
            spread_penalty_per_graph: penalty for clustering [n_graphs,]
                                     (lower is better = more spread)
                                     Normalized so same weight works across problem sizes
        """
        def compute_spread_for_graph(graph_id):
            # Get components belonging to this graph
            mask = (node_gr_idx == graph_id)
            graph_positions = positions[jnp.where(mask, size=positions.shape[0], fill_value=0)[0]]
            n_components_in_graph = jnp.sum(mask.astype(jnp.float32))

            # Compute mean position
            mean_pos = jnp.sum(graph_positions, axis=0) / jnp.maximum(n_components_in_graph, 1.0)

            # Compute variance (average squared distance from mean)
            # Higher variance = more spread = better
            squared_distances = jnp.sum((graph_positions - mean_pos) ** 2, axis=1)
            variance = jnp.sum(squared_distances * mask.astype(jnp.float32)) / jnp.maximum(n_components_in_graph, 1.0)

            # Penalty = 1 / (variance + epsilon)
            # Low variance → high penalty (stacked)
            # High variance → low penalty (spread)
            epsilon = 0.01  # Avoid division by zero
            spread_penalty = 1.0 / (variance + epsilon)

            # Normalize by √N to maintain consistent relative importance vs HPWL
            # HPWL scales roughly as O(√N), so this makes spread_weight scale-invariant
            # spread_weight = 0.5 now means "same relative importance" for all problem sizes
            normalization_factor = jnp.sqrt(jnp.maximum(n_components_in_graph, 1.0))
            normalized_spread_penalty = spread_penalty * normalization_factor

            return normalized_spread_penalty

        # Vectorize over graphs
        graph_ids = jnp.arange(n_graph)
        spread_penalty_per_graph = jax.vmap(compute_spread_for_graph)(graph_ids)

        return spread_penalty_per_graph

    @partial(jax.jit, static_argnums=(0, 3))
    def _compute_min_distance_penalty(self, positions, node_gr_idx, n_graph):
        """
        Compute minimum distance constraint penalty (alternative to spread).

        Penalizes pairs of components that are too close together.
        This directly prevents stacking without penalizing overlap area.

        For each pair i,j:
        - distance = ||pos_i - pos_j||
        - if distance < threshold: penalty += (threshold - distance)^2

        Args:
            positions: component positions [num_components, 2]
            node_gr_idx: component to graph mapping
            n_graph: number of graphs

        Returns:
            min_distance_penalty_per_graph: penalty for close pairs [n_graphs,]
        """
        num_components = positions.shape[0]

        # Pairwise distances (vectorized)
        pos_i = positions[:, jnp.newaxis, :]  # [num_components, 1, 2]
        pos_j = positions[jnp.newaxis, :, :]  # [1, num_components, 2]

        # Euclidean distance
        distances = jnp.sqrt(jnp.sum((pos_i - pos_j) ** 2, axis=2))  # [num_components, num_components]

        # Penalty for distances below threshold
        violations = jnp.maximum(0.0, self.min_distance_threshold - distances)
        penalty_per_pair = violations ** 2

        # Only count each pair once (upper triangle, excluding diagonal)
        i_indices = jnp.arange(num_components)[:, jnp.newaxis]
        j_indices = jnp.arange(num_components)[jnp.newaxis, :]
        upper_triangle_mask = (i_indices < j_indices).astype(jnp.float32)

        # Only consider pairs within same graph
        same_graph_mask = (node_gr_idx[:, jnp.newaxis] == node_gr_idx[jnp.newaxis, :]).astype(jnp.float32)

        # Combined mask
        valid_pairs_mask = upper_triangle_mask * same_graph_mask

        # Apply mask
        penalty_masked = penalty_per_pair * valid_pairs_mask

        # Sum per component and aggregate to graph
        penalty_per_component = jnp.sum(penalty_masked, axis=1)
        penalty_per_graph = jax.ops.segment_sum(penalty_per_component, node_gr_idx, n_graph)

        return penalty_per_graph

    def calculate_relaxed_Energy(self, H_graph, positions, node_gr_idx, component_sizes=None):
        """
        Calculate relaxed energy (same as regular energy for continuous case).

        In discrete problems, this might use soft assignments. For continuous chip placement,
        we don't need a separate relaxed version.
        """
        return self.calculate_Energy(H_graph, positions, node_gr_idx, component_sizes)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, mean, log_var, node_gr_idx, component_sizes=None):
        """
        Calculate energy loss for training.

        Two training modes:
        1. use_constraints_in_training=True (default):
           - Energy = HPWL + overlap_penalty + boundary_penalty
           - Direct constraint enforcement during training

        2. use_constraints_in_training=False (two-stage approach):
           - Training: Energy = HPWL + spread_regularization + boundary_penalty
           - Focus on learning HPWL optimization without overlap penalties
           - Spread regularization prevents trivial stacking solution
           - Boundary penalty prevents out-of-bounds drift (essential!)
           - Overlap constraints enforced via post-legalization or evaluation only

        Args:
            H_graph: graph structure
            mean: predicted mean positions [num_components, 1, continuous_dim] or [num_components, continuous_dim]
            log_var: predicted log variance (not used in energy, only for noise loss)
            node_gr_idx: component to graph mapping
            component_sizes: component sizes

        Returns:
            Energy, positions, constraint_violations
        """
        n_graph = H_graph.n_node.shape[0]
        nodes = H_graph.nodes
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        # Extract component sizes (same as calculate_Energy)
        if component_sizes is None:
            if nodes.shape[-1] >= 2:
                component_sizes = nodes[:, :2]
            else:
                component_sizes = jnp.full((total_num_nodes, 2), 0.1)

        # Ensure positions has correct shape
        positions = mean
        if len(positions.shape) == 3:
            positions = positions[:, 0, :]
        elif len(positions.shape) == 1:
            positions = jnp.reshape(positions, (total_num_nodes, self.continuous_dim))

        # Compute HPWL (always included)
        hpwl_per_graph = self._compute_hpwl(H_graph, positions, node_gr_idx, n_graph)

        if self.use_constraints_in_training:
            # Mode 1: Full constraint enforcement during training
            # Compute overlap and boundary penalties
            overlap_per_graph = self._compute_overlap_penalty(
                positions, component_sizes, node_gr_idx, n_graph
            )
            boundary_per_graph = self._compute_boundary_penalty(
                positions, component_sizes, node_gr_idx, n_graph
            )

            # FIX: Use raw penalties without HPWL coupling (same fix as calculate_Energy)
            # Old (buggy): normalized_overlap_penalty = overlap_per_graph * hpwl_per_graph
            # Old (buggy): normalized_boundary_penalty = boundary_per_graph * hpwl_per_graph

            Energy_per_graph = (
                hpwl_per_graph +
                self.overlap_weight * overlap_per_graph +
                self.boundary_weight * boundary_per_graph
            )

            constraint_violations_per_graph = overlap_per_graph + boundary_per_graph

        else:
            # Mode 2: Two-stage approach - HPWL optimization + spread regularization
            # No OVERLAP penalties during training (they couple with HPWL)
            # But KEEP boundary penalties (prevent out-of-bounds drift)
            # Use spread regularization to prevent stacking

            # Compute spread penalty
            spread_penalty_per_graph = self._compute_spread_regularization(
                positions, node_gr_idx, n_graph
            )

            # IMPORTANT: Keep boundary penalties to prevent out-of-bounds drift
            boundary_per_graph = self._compute_boundary_penalty(
                positions, component_sizes, node_gr_idx, n_graph
            )
            # FIX: Use raw boundary penalty without HPWL coupling to prevent explosion
            # Old (buggy): normalized_boundary_penalty = boundary_per_graph * hpwl_per_graph

            # Optionally add minimum distance constraint
            if self.use_min_distance:
                min_distance_penalty = self._compute_min_distance_penalty(
                    positions, node_gr_idx, n_graph
                )
                Energy_per_graph = (
                    hpwl_per_graph +
                    self.spread_weight * spread_penalty_per_graph +
                    self.boundary_weight * boundary_per_graph +
                    self.min_distance_weight * min_distance_penalty
                )
            else:
                Energy_per_graph = (
                    hpwl_per_graph +
                    self.spread_weight * spread_penalty_per_graph +
                    self.boundary_weight * boundary_per_graph
                )

            # For monitoring: report spread penalty + boundary as "violations" during training
            constraint_violations_per_graph = spread_penalty_per_graph + boundary_per_graph

        # Ensure output shape [n_graphs, 1]
        if len(Energy_per_graph.shape) == 1:
            Energy_per_graph = jnp.expand_dims(Energy_per_graph, axis=-1)
        if len(constraint_violations_per_graph.shape) == 1:
            constraint_violations_per_graph = jnp.expand_dims(constraint_violations_per_graph, axis=-1)

        return Energy_per_graph, positions, constraint_violations_per_graph

    def get_HPWL_value(self, H_graph, positions, node_gr_idx):
        """
        Get HPWL value only (no penalties).

        Useful for evaluation and reporting.

        Args:
            H_graph: graph structure
            positions: component positions
            node_gr_idx: component to graph mapping

        Returns:
            HPWL per graph
        """
        n_graph = H_graph.n_node.shape[0]

        # Ensure positions has correct shape
        if len(positions.shape) == 3:
            positions = positions[:, 0, :]
        elif len(positions.shape) == 1:
            total_num_nodes = jax.tree_util.tree_leaves(H_graph.nodes)[0].shape[0]
            positions = jnp.reshape(positions, (total_num_nodes, self.continuous_dim))

        hpwl_per_graph = self._compute_hpwl(H_graph, positions, node_gr_idx, n_graph)

        return jnp.expand_dims(hpwl_per_graph, axis=-1)

"""
Chip Placement Dataset Generator for DiffUCO
Adapted from ChipDiffusion V2 algorithm
"""
import torch
import torch.distributions as dist
import shapely
import numpy as np
from .BaseDatasetGenerator import BaseDatasetGenerator
from tqdm import tqdm
from torch_geometric.data import Data


class ChipDatasetGenerator(BaseDatasetGenerator):
    """
    Generates synthetic chip placement datasets compatible with DiffUCO
    Uses continuous positions for components on a 2D canvas
    """

    def __init__(self, config):
        super().__init__(config)

        # Default parameters (can be overridden in config)
        self.max_instance = config.get("max_instance", 400)
        self.max_attempts_per_instance = config.get("max_attempts_per_instance", 10)

        # Dataset sizes
        self.graph_config = {
            "n_train": config.get("n_train", 4000),
            "n_val": config.get("n_val", 500),
            "n_test": config.get("n_test", 1000),
        }

        if "huge" in self.dataset_name or "giant" in self.dataset_name:
            self.graph_config["n_test"] = 100

        if "dummy" in self.dataset_name:
            self.graph_config["n_train"] = 10
            self.graph_config["n_val"] = 5
            self.graph_config["n_test"] = 5

        print(f'\nGenerating Chip Placement {self.mode} dataset "{self.dataset_name}" '
              f'with {self.graph_config[f"n_{self.mode}"]} instances!\n')

    def generate_dataset(self):
        """
        Generate chip placement dataset instances
        """
        solutions = {
            "positions": [],          # Continuous (x,y) positions
            "H_graphs": [],          # PyG Data objects
            "sizes": [],             # Component sizes
            "edge_attrs": [],        # Terminal offsets
            "graph_sizes": [],       # Number of components
            "densities": [],         # Placement density
            "Energies": [],          # HPWL (for evaluation)
            "compl_H_graphs": [],    # Redundant (for compatibility)
        }

        for idx in tqdm(range(self.graph_config[f"n_{self.mode}"])):
            # Generate one chip placement instance
            positions, data, density, hpwl = self.sample_chip_instance()

            solutions["positions"].append(positions.numpy())
            solutions["H_graphs"].append(data)
            solutions["sizes"].append(data.x.numpy())
            solutions["edge_attrs"].append(data.edge_attr.numpy())
            solutions["graph_sizes"].append(positions.shape[0])
            solutions["densities"].append(density)
            solutions["Energies"].append(hpwl)
            solutions["compl_H_graphs"].append(data)  # Same as H_graphs for chip

            # Save individual instance
            indexed_solution_dict = {}
            for key in solutions.keys():
                if len(solutions[key]) > 0:
                    indexed_solution_dict[key] = solutions[key][idx]
            self.save_instance_solution(indexed_solution_dict, idx)

        # Save all solutions
        self.save_solutions(solutions)

    def sample_chip_instance(self):
        """
        Sample one chip placement instance using V2 algorithm

        Returns:
            positions: (V, 2) tensor of component center positions
            data: PyG Data object with graph structure
            density: float, placement density
            hpwl: float, Half-Perimeter Wirelength
        """

        # 1. Sample target density
        stop_density = self._sample_uniform(0.75, 0.9)

        # 2. Generate component sizes
        aspect_ratios = self._sample_uniform(0.25, 1.0, (self.max_instance,))
        long_sizes = self._sample_clipped_exp(scale=0.08, low=0.02, high=1.0,
                                               size=(self.max_instance,))
        short_sizes = aspect_ratios * long_sizes

        # Random orientation
        long_x = (torch.rand(self.max_instance) > 0.5).float()
        x_sizes = long_x * long_sizes + (1 - long_x) * short_sizes
        y_sizes = (1 - long_x) * long_sizes + long_x * short_sizes

        # Sort by area (largest first for better packing)
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)
        x_sizes = x_sizes[indices]
        y_sizes = y_sizes[indices]

        # 3. Place components sequentially
        placement = ChipPlacement()
        density = 0.0

        for x_size, y_size in zip(x_sizes, y_sizes):
            x_size_val = float(x_size)
            y_size_val = float(y_size)

            # Sample candidate positions (center coordinates)
            low = torch.tensor([(x_size_val/2) - 1.0, (y_size_val/2) - 1.0])
            high = torch.tensor([1.0 - (x_size_val/2), 1.0 - (y_size_val/2)])

            placed = False
            for attempt in range(self.max_attempts_per_instance):
                candidate_pos = torch.rand(2) * (high - low) + low

                if placement.check_legality(candidate_pos[0].item(),
                                           candidate_pos[1].item(),
                                           x_size_val, y_size_val):
                    placement.commit_instance(candidate_pos[0].item(),
                                            candidate_pos[1].item(),
                                            x_size_val, y_size_val)
                    placed = True
                    break

            if placed:
                density += (x_size_val * y_size_val) / 4.0

            if density >= stop_density:
                break

        # Extract placement data
        positions = placement.get_positions()  # (V, 2)
        sizes = placement.get_sizes()          # (V, 2)
        num_instances = positions.shape[0]

        # 4. Generate terminals
        instance_areas = sizes[:, 0] * sizes[:, 1]
        num_terminals = self._sample_conditional_binomial(instance_areas,
                                                           binom_p=0.5,
                                                           binom_min_n=4,
                                                           t=128, p=0.65)
        num_terminals = torch.clamp(num_terminals, min=1, max=256).int()
        max_num_terminals = torch.max(num_terminals)

        # Terminal offsets relative to component center
        terminal_offsets = self._get_terminal_offsets(sizes[:, 0], sizes[:, 1],
                                                       max_num_terminals)

        # 5. Generate netlist edges
        terminal_positions = positions.unsqueeze(1) + terminal_offsets  # (V, T, 2)
        terminal_distances = self._get_terminal_distances(terminal_positions)  # (V, T, V, T)

        # Sample edge probability based on distance
        scale_factor = np.exp(np.random.uniform(np.log(0.05), np.log(1.6)))
        edge_exists = self._sample_conditional_exp_bernoulli(terminal_distances,
                                                              scale=scale_factor)

        # Designate source and sink terminals
        is_source = (torch.rand(num_instances, max_num_terminals) < 0.3).float()

        # Process edges (remove invalid connections)
        edge_exists = self._process_edge_matrix(edge_exists, is_source, num_terminals)

        # Connect isolated instances
        self._connect_isolated_instances(edge_exists, terminal_distances)

        # 6. Convert to edge list format
        edge_index, edge_attr = self._generate_edge_list(edge_exists, terminal_offsets)

        # 7. Compute HPWL (for evaluation)
        hpwl = self._compute_hpwl(positions, sizes, edge_index, edge_attr)

        # 8. Create PyG Data object
        mask = placement.get_mask()
        data = Data(
            x=sizes,            # (V, 2) component sizes
            edge_index=edge_index,  # (2, E)
            edge_attr=edge_attr,    # (E, 4) terminal offsets
            is_ports=mask       # (V,) boolean mask
        )

        return positions, data, density, hpwl

    # ========== Helper Methods ==========

    def _sample_uniform(self, low, high, size=None):
        """Sample from uniform distribution"""
        if size is None:
            return torch.rand(1).item() * (high - low) + low
        return torch.rand(*size) * (high - low) + low

    def _sample_clipped_exp(self, scale, low, high, size):
        """Sample from clipped exponential distribution"""
        samples = torch.empty(size).exponential_(lambd=1.0/scale)
        return torch.clamp(samples, min=low, max=high)

    def _sample_conditional_binomial(self, areas, binom_p, binom_min_n, t, p):
        """
        Sample number of terminals based on component area
        n = clip(t * area^p, min=binom_min_n)
        """
        n = torch.clamp(torch.ceil(t * torch.pow(areas, p)), min=binom_min_n)
        distribution = dist.Binomial(n, torch.full(areas.shape, binom_p))
        return distribution.sample().int()

    def _sample_conditional_exp_bernoulli(self, distances, scale,
                                          prob_clip=0.9,
                                          prob_multiplier_factor=0.0212,
                                          prob_multiplier_exp=-1.42):
        """
        Sample edge existence based on distance
        prob = clip(multiplier * exp(-distance/scale), max=prob_clip)
        """
        prob_multiplier = prob_multiplier_factor * (scale ** prob_multiplier_exp)
        rate = distances / scale
        prob = torch.clamp(prob_multiplier * torch.exp(-rate), max=prob_clip)
        return (torch.rand_like(prob) < prob).float()

    def _get_terminal_offsets(self, x_sizes, y_sizes, max_num_terminals):
        """
        Generate terminal offsets on component boundaries

        Args:
            x_sizes: (V,) tensor
            y_sizes: (V,) tensor
            max_num_terminals: int

        Returns:
            terminal_offsets: (V, T, 2) tensor
        """
        half_perim = x_sizes + y_sizes

        # Sample locations along perimeter
        terminal_locs = torch.rand(max_num_terminals, x_sizes.shape[0]) * half_perim
        terminal_flip = (torch.rand(max_num_terminals, x_sizes.shape[0]) > 0.5).float()
        terminal_flip = 2 * terminal_flip - 1

        x_sizes_exp = x_sizes.unsqueeze(0)
        y_sizes_exp = y_sizes.unsqueeze(0)

        # Map to x/y offsets
        offset_x = torch.clamp(terminal_locs, torch.zeros_like(x_sizes_exp),
                              x_sizes_exp) - (x_sizes_exp / 2)
        offset_y = torch.clamp(terminal_locs - x_sizes_exp, torch.zeros_like(y_sizes_exp),
                              y_sizes_exp) - (y_sizes_exp / 2)

        offset_x = terminal_flip * offset_x
        offset_y = terminal_flip * offset_y

        terminal_offset = torch.stack((offset_x, offset_y), dim=-1)  # (T, V, 2)
        terminal_offset = terminal_offset.permute(1, 0, 2)  # (V, T, 2)

        return terminal_offset

    def _get_terminal_distances(self, terminal_positions):
        """
        Compute pairwise L1 distances between all terminals

        Args:
            terminal_positions: (V, T, 2) tensor

        Returns:
            distances: (V, T, V, T) tensor
        """
        V, T, _ = terminal_positions.shape
        t_pos_1 = terminal_positions.view(V, T, 1, 1, 2)
        t_pos_2 = terminal_positions.view(1, 1, V, T, 2)
        delta_pos = t_pos_1 - t_pos_2
        distances = torch.norm(delta_pos, p=1, dim=-1)  # L1 norm
        return distances

    def _process_edge_matrix(self, edge_exists, is_source, num_terminals):
        """
        Filter edge matrix to remove invalid connections
        - Remove edges ending in a source terminal
        - Remove edges starting from a sink terminal
        - Remove self-loops
        - Remove edges to/from non-existent terminals
        """
        V, T, _, _ = edge_exists.shape

        # Generate terminal filter
        terminal_filter = torch.zeros((V, T))
        for i, num_term in enumerate(num_terminals):
            terminal_filter[i, :num_term] = 1

        # Apply filters
        source_filter = (terminal_filter * is_source).view(V, T, 1, 1)
        sink_filter = (terminal_filter * (1 - is_source)).view(1, 1, V, T)
        self_edge_filter = (1 - torch.eye(V)).view(V, 1, V, 1)

        edges = edge_exists * source_filter * sink_filter * self_edge_filter
        return edges

    def _connect_isolated_instances(self, edge_matrix, terminal_distances):
        """
        Connect isolated components to prevent disconnected graphs (IN-PLACE)
        """
        V, T, _, _ = edge_matrix.shape
        out_degree = edge_matrix.sum(dim=(2, 3))  # per terminal
        in_degree = edge_matrix.sum(dim=(0, 1, 3))  # per instance
        degree = out_degree.sum(dim=-1) + in_degree
        max_dist = 10 + terminal_distances.max()

        for i in range(V):
            if degree[i] == 0:  # isolated instance
                distances = terminal_distances[i, 0, :, :]  # (V, T)
                distances = torch.where(out_degree > 0, distances, max_dist)

                min_idx = torch.argmin(distances)
                instance_idx = min_idx // T
                terminal_idx = min_idx % T

                # Connect to netlist
                edge_matrix[instance_idx, terminal_idx, i, 0] = 1

    def _generate_edge_list(self, edge_exists, terminal_offsets):
        """
        Convert edge matrix to edge list format

        Returns:
            edge_index: (2, E) tensor
            edge_attr: (E, 4) tensor of [src_x, src_y, sink_x, sink_y] offsets
        """
        V, T, _, _ = edge_exists.shape
        edges = torch.nonzero(edge_exists)  # (E, 4)

        edge_index_forward = edges[:, [0, 2]]  # (E, 2)
        edge_index_reverse = edges[:, [2, 0]]  # (E, 2)

        # Get terminal offsets for each edge
        edge_attr_source = terminal_offsets[edges[:, 0], edges[:, 1], :]  # (E, 2)
        edge_attr_sink = terminal_offsets[edges[:, 2], edges[:, 3], :]    # (E, 2)

        edge_attr_forward = torch.cat([edge_attr_source, edge_attr_sink], dim=-1)  # (E, 4)
        edge_attr_reverse = torch.cat([edge_attr_sink, edge_attr_source], dim=-1)  # (E, 4)

        # Create undirected graph
        edge_index = torch.cat([edge_index_forward, edge_index_reverse], dim=0).T  # (2, 2E)
        edge_attr = torch.cat([edge_attr_forward, edge_attr_reverse], dim=0)       # (2E, 4)

        return edge_index.clone(), edge_attr.clone()

    def _compute_hpwl(self, positions, sizes, edge_index, edge_attr):
        """
        Compute Half-Perimeter Wirelength

        HPWL = Σ_net (bbox_width + bbox_height)
        """
        # Get unique nets (undirected)
        num_edges = edge_index.shape[1] // 2
        edge_index_unique = edge_index[:, :num_edges]
        edge_attr_unique = edge_attr[:num_edges, :]

        # Compute terminal positions
        src_idx = edge_index_unique[0, :]
        sink_idx = edge_index_unique[1, :]

        src_term_pos = positions[src_idx] + edge_attr_unique[:, :2]
        sink_term_pos = positions[sink_idx] + edge_attr_unique[:, 2:4]

        # Compute bounding box per net
        # For multi-terminal nets, we need to group by net ID
        # Simplified: treat each 2-pin connection as a net
        x_coords = torch.stack([src_term_pos[:, 0], sink_term_pos[:, 0]], dim=1)
        y_coords = torch.stack([src_term_pos[:, 1], sink_term_pos[:, 1]], dim=1)

        bbox_width = x_coords.max(dim=1)[0] - x_coords.min(dim=1)[0]
        bbox_height = y_coords.max(dim=1)[0] - y_coords.min(dim=1)[0]

        hpwl = (bbox_width + bbox_height).sum().item()
        return hpwl


class ChipPlacement:
    """
    Helper class for managing chip placement with collision detection
    Uses shapely for efficient geometric operations
    """

    def __init__(self):
        self.instances = []
        self.x_coords = []
        self.y_coords = []
        self.x_sizes = []
        self.y_sizes = []
        self.is_port = []

        self.chip_bounds = shapely.box(-1, -1, 1, 1)
        self.eps = 1e-8

    def check_legality(self, x_pos, y_pos, x_size, y_size):
        """
        Check if placing component at (x_pos, y_pos) is legal
        Positions are center coordinates
        """
        candidate = shapely.box(
            x_pos - x_size/2, y_pos - y_size/2,
            x_pos + x_size/2, y_pos + y_size/2
        )

        # Check chip boundary
        if not self.chip_bounds.contains(candidate):
            return False

        # Check overlap with existing instances
        for inst in self.instances:
            if inst.intersects(candidate):
                return False

        return True

    def commit_instance(self, x_pos, y_pos, x_size, y_size, is_port=False):
        """Add instance to placement (assumes center coordinates)"""
        if not is_port:
            self.instances.append(shapely.box(
                x_pos - x_size/2, y_pos - y_size/2,
                x_pos + x_size/2, y_pos + y_size/2
            ))

        self.x_coords.append(x_pos)
        self.y_coords.append(y_pos)
        self.x_sizes.append(x_size)
        self.y_sizes.append(y_size)
        self.is_port.append(is_port)

    def get_positions(self):
        """Return (V, 2) tensor of center positions"""
        return torch.stack([
            torch.tensor(self.x_coords),
            torch.tensor(self.y_coords)
        ], dim=-1)

    def get_sizes(self):
        """Return (V, 2) tensor of sizes"""
        return torch.stack([
            torch.tensor(self.x_sizes),
            torch.tensor(self.y_sizes)
        ], dim=-1)

    def get_mask(self):
        """Return (V,) boolean tensor for ports"""
        return torch.tensor(self.is_port)

    def get_density(self):
        """Compute placement density"""
        total_area = sum([inst.area for inst in self.instances])
        return total_area / self.chip_bounds.area

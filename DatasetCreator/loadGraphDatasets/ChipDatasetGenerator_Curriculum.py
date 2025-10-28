"""
Chip Placement Dataset Generator for CURRICULUM LEARNING

Extended from ChipDatasetGenerator_Unsupervised to support:
- Target component count ranges (e.g., 90-110 components)
- Fixed-size datasets for staged training

Usage:
    Generate stage 1 (100 components):
        python generate_chip_curriculum.py --stage 1 --target_components 100 --component_variation 10

    Generate stage 2 (200 components):
        python generate_chip_curriculum.py --stage 2 --target_components 200 --component_variation 20
"""

from .ChipDatasetGenerator_Unsupervised import ChipDatasetGenerator as BaseGenerator
import torch
import numpy as np


class ChipDatasetGeneratorCurriculum(BaseGenerator):
    """
    Curriculum-aware chip placement dataset generator

    Extends base generator to control component count distribution
    """

    def __init__(self, config):
        super().__init__(config)

        # Curriculum parameters
        self.target_components = config.get("target_components", None)  # e.g., 100
        self.component_variation = config.get("component_variation", 10)  # ±10 variation

        if self.target_components is not None:
            self.component_min = max(10, self.target_components - self.component_variation)
            self.component_max = self.target_components + self.component_variation
            print(f'\nCURRICULUM MODE: Target {self.target_components} components '
                  f'(range: [{self.component_min}, {self.component_max}])')
        else:
            # Original v1 behavior
            self.component_min = None
            self.component_max = None
            print(f'\nSTANDARD MODE: Variable component count (original v1 behavior)')

    def sample_chip_instance_unsupervised(self):
        """
        Sample one chip placement instance with CONTROLLED component count

        Modified from base class to enforce target component range
        """

        # 1. Sample target density
        stop_density = self._sample_uniform(0.75, 0.9)

        # 2. Configure component generation based on target count
        if self.target_components is not None:
            # CURRICULUM MODE: Adjust parameters based on target
            component_count_target = np.random.randint(self.component_min, self.component_max + 1)
            max_instance, exp_scale, exp_min, exp_max = self._get_curriculum_params(
                component_count_target
            )
        else:
            # ORIGINAL v1 MODE
            max_instance = 400
            exp_scale = 0.08
            exp_min = 0.02
            exp_max = 1.0
            component_count_target = None

        aspect_ratio_range = (0.25, 1.0)

        # 3. Generate component pool
        aspect_ratios = self._sample_uniform(
            aspect_ratio_range[0], aspect_ratio_range[1], (max_instance,)
        )

        # Clipped Exponential size distribution (ChipDiffusion v1)
        long_sizes = self._sample_clipped_exp(exp_scale, exp_min, exp_max, (max_instance,))
        short_sizes = aspect_ratios * long_sizes

        # Random orientation
        long_x = (torch.rand(max_instance) > 0.5).float()
        x_sizes = long_x * long_sizes + (1 - long_x) * short_sizes
        y_sizes = (1 - long_x) * long_sizes + long_x * short_sizes

        # 4. Place components with target count enforcement
        if component_count_target is not None:
            initial_positions, placed_sizes, actual_density = self._place_components_with_target(
                x_sizes, y_sizes, stop_density, component_count_target
            )
        else:
            initial_positions, placed_sizes, actual_density = self._place_components_legal(
                x_sizes, y_sizes, stop_density
            )

        # 5. Generate netlist based on proximity
        edge_index, edge_attr = self._generate_netlist_proximity_based(
            initial_positions, placed_sizes
        )

        # 6. Randomize placement
        randomized_positions = torch.rand(len(placed_sizes), 2) * 2 - 1  # Uniform in [-1, 1]

        # 7. Compute HPWL
        hpwl = self._compute_hpwl(randomized_positions, placed_sizes, edge_index, edge_attr)

        # 8. Create jraph.GraphsTuple
        import jraph
        randomized_positions_np = randomized_positions.numpy() if isinstance(randomized_positions, torch.Tensor) else randomized_positions
        legal_positions_np = initial_positions.numpy() if isinstance(initial_positions, torch.Tensor) else initial_positions
        sizes_np = placed_sizes.numpy() if isinstance(placed_sizes, torch.Tensor) else placed_sizes
        edge_index_np = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
        edge_attr_np = edge_attr.numpy() if isinstance(edge_attr, torch.Tensor) else edge_attr

        num_nodes = sizes_np.shape[0]
        num_edges = edge_index_np.shape[1]

        H_graph = jraph.GraphsTuple(
            nodes=sizes_np,
            edges=edge_attr_np,
            senders=edge_index_np[0, :],
            receivers=edge_index_np[1, :],
            n_node=np.array([num_nodes]),
            n_edge=np.array([num_edges]),
            globals=None
        )

        return randomized_positions_np, legal_positions_np, H_graph, actual_density, hpwl

    def _get_curriculum_params(self, target_count):
        """
        Dynamically adjust component pool and size distribution for target count

        Key insight: Smaller target → smaller pool + slightly larger components
                     Larger target → larger pool + slightly smaller components

        Args:
            target_count: Desired number of components

        Returns:
            max_instance: Component pool size
            exp_scale: Exponential distribution scale
            exp_min: Min component size
            exp_max: Max component size
        """

        # Pool size: 2-3x target (ensures enough components even with large sizes)
        max_instance = int(target_count * 2.5)

        # Adjust size distribution to control density
        # More components → need smaller sizes to fit at same density
        if target_count < 100:
            # Few components: allow larger sizes
            exp_scale = 0.10  # More large components
            exp_min = 0.03
            exp_max = 1.2
        elif target_count < 200:
            # Medium components: balanced sizes
            exp_scale = 0.08  # Original v1
            exp_min = 0.02
            exp_max = 1.0
        elif target_count < 300:
            # Many components: smaller average size
            exp_scale = 0.06
            exp_min = 0.015
            exp_max = 0.8
        else:
            # Very many components: small sizes
            exp_scale = 0.05
            exp_min = 0.01
            exp_max = 0.6

        return max_instance, exp_scale, exp_min, exp_max

    def _place_components_with_target(self, x_sizes, y_sizes, stop_density, target_count):
        """
        Place components with DUAL stopping criteria:
        1. Density threshold (original)
        2. Component count threshold (NEW)

        Args:
            x_sizes: (V,) tensor
            y_sizes: (V,) tensor
            stop_density: float, target density
            target_count: int, target number of components

        Returns:
            positions: (V, 2) tensor
            sizes: (V, 2) tensor
            density: float
        """
        from .ChipDatasetGenerator_Unsupervised import ChipPlacement

        placement = ChipPlacement()
        density = 0.0

        # Sort by area (largest first)
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)
        x_sizes_sorted = x_sizes[indices]
        y_sizes_sorted = y_sizes[indices]

        placed_components = []

        # Allow some flexibility in target count (±5%)
        min_components = int(target_count * 0.8)
        max_components = int(target_count * 1.2)

        for idx, (x_size, y_size) in enumerate(zip(x_sizes_sorted, y_sizes_sorted)):
            x_size_val = float(x_size)
            y_size_val = float(y_size)

            # Calculate valid position range
            low = torch.tensor([(x_size_val/2) - 1.0, (y_size_val/2) - 1.0])
            high = torch.tensor([1.0 - (x_size_val/2), 1.0 - (y_size_val/2)])

            # Check if component fits
            if (low >= high).any():
                continue

            placed = False
            for attempt in range(self.max_attempts_per_instance):
                candidate_pos = torch.rand(2) * (high - low) + low

                if placement.check_legality(candidate_pos[0].item(),
                                           candidate_pos[1].item(),
                                           x_size_val, y_size_val):
                    placement.commit_instance(candidate_pos[0].item(),
                                            candidate_pos[1].item(),
                                            x_size_val, y_size_val)
                    placed_components.append(indices[idx].item())
                    placed = True
                    break

            if placed:
                density += (x_size_val * y_size_val) / 4.0

            num_placed = len(placed_components)

            # MODIFIED STOPPING CRITERIA
            # Stop if:
            # 1. Reached target count range AND achieved reasonable density (>60%)
            # 2. OR exceeded max count (safety)
            # 3. OR reached density target and have at least min_components

            if num_placed >= max_components:
                # Safety: don't exceed max
                break

            if num_placed >= min_components:
                # In target range - check density
                if density >= stop_density * 0.8:  # Allow 80% of target density
                    break

            # Original density stopping (with minimum component count)
            if density >= stop_density and num_placed >= min_components:
                break

        # Extract placement data
        positions = placement.get_positions()
        sizes = placement.get_sizes()

        return positions, sizes, density

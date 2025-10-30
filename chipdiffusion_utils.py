"""
Utility functions for ChipDiffusion benchmark normalization/denormalization

These functions match ChipDiffusion's preprocess_graph and postprocess_placement
"""

import numpy as np


def denormalize_positions_and_sizes(normalized_positions, normalized_sizes, chip_size):
    """
    Denormalize DIffUCO [-1,1] format back to original ChipDiffusion scale

    Args:
        normalized_positions: (V, 2) array, positions in [-1, 1] range
        normalized_sizes: (V, 2) array, component sizes normalized to 2x2 canvas
        chip_size: list/array [x_min, y_min, x_max, y_max] original canvas bounds

    Returns:
        original_positions: (V, 2) array, positions in original coordinate system
        original_sizes: (V, 2) array, component sizes in original units
    """
    # Extract canvas dimensions
    if len(chip_size) == 4:  # [x_min, y_min, x_max, y_max]
        canvas_width = chip_size[2] - chip_size[0]
        canvas_height = chip_size[3] - chip_size[1]
        chip_offset = np.array([chip_size[0], chip_size[1]], dtype=np.float32)
    else:  # [width, height]
        canvas_width = chip_size[0]
        canvas_height = chip_size[1]
        chip_offset = np.zeros(2, dtype=np.float32)

    canvas_size = np.array([canvas_width, canvas_height], dtype=np.float32)

    # Inverse of ChipDiffusion normalization (from postprocess_placement in utils.py)
    # 1. Remove center adjustment: x = x - cond.x/2
    original_positions = normalized_positions - normalized_sizes / 2.0

    # 2. Denormalize positions: x = scale * (x+1)/2 + offset
    original_positions = canvas_size * (original_positions + 1.0) / 2.0 + chip_offset

    # 3. Denormalize sizes: cond.x = (cond.x * scale)/2
    original_sizes = (normalized_sizes * canvas_size) / 2.0

    return original_positions, original_sizes

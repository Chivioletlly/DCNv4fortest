"""Spatial padding helpers shared by training and inference."""

from typing import Tuple

import torch
import torch.nn.functional as F


MODEL_SIZE_MULTIPLE = 8


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = MODEL_SIZE_MULTIPLE,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad the bottom/right edges so spatial dimensions divide ``multiple``."""
    if tensor.ndim < 2:
        raise ValueError(f"Expected an image tensor with at least 2 dimensions, got {tensor.shape}")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")

    height, width = tensor.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    if pad_height == 0 and pad_width == 0:
        return tensor, (height, width)

    # Reflection matches Restormer's test-time padding. Very small synthetic
    # images cannot satisfy reflect-padding constraints, so use replication.
    mode = "reflect" if pad_height < height and pad_width < width else "replicate"
    padded = F.pad(tensor, (0, pad_width, 0, pad_height), mode=mode)
    return padded, (height, width)


def crop_to_size(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Crop a bottom/right-padded tensor back to ``(height, width)``."""
    height, width = size
    if height <= 0 or width <= 0:
        raise ValueError(f"Crop size must be positive, got {size}")
    return tensor[..., :height, :width]

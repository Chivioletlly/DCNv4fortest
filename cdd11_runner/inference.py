"""Shared native or weighted tiled inference for all CDD-11 models."""

from __future__ import annotations

from typing import List

import torch


def _tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _blend_window(
    height: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    vertical = torch.hann_window(height, periodic=False, device=device, dtype=torch.float32)
    horizontal = torch.hann_window(width, periodic=False, device=device, dtype=torch.float32)
    return torch.outer(vertical, horizontal).clamp_min_(1.0e-3)[None, None]


def _forward_bf16(model: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    with torch.autocast(
        device_type=value.device.type,
        dtype=torch.bfloat16,
        enabled=value.device.type == "cuda",
    ):
        return model(value).float()


def tiled_inference(
    model: torch.nn.Module,
    degraded: torch.Tensor,
    *,
    tile_size: int = 512,
    overlap: int = 128,
) -> torch.Tensor:
    if degraded.ndim != 4 or degraded.shape[0] != 1:
        raise ValueError("Tiled CDD-11 inference requires a single NCHW image")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Require tile_size > overlap >= 0")
    height, width = degraded.shape[-2:]
    if height <= tile_size and width <= tile_size:
        return _forward_bf16(model, degraded)
    stride = tile_size - overlap
    top_values = _tile_starts(height, tile_size, stride)
    left_values = _tile_starts(width, tile_size, stride)
    result = torch.zeros(
        (1, 3, height, width),
        dtype=torch.float32,
        device=degraded.device,
    )
    weight_sum = torch.zeros(
        (1, 1, height, width),
        dtype=torch.float32,
        device=degraded.device,
    )
    for top in top_values:
        for left in left_values:
            tile = degraded[
                ...,
                top : min(top + tile_size, height),
                left : min(left + tile_size, width),
            ]
            restored = _forward_bf16(model, tile)
            window = _blend_window(
                restored.shape[-2],
                restored.shape[-1],
                device=degraded.device,
            )
            bottom = top + restored.shape[-2]
            right = left + restored.shape[-1]
            result[..., top:bottom, left:right].add_(restored * window)
            weight_sum[..., top:bottom, left:right].add_(window)
    if torch.any(weight_sum <= 0):
        raise RuntimeError("Tiled inference left uncovered pixels")
    return result / weight_sum


def restore_image(
    model: torch.nn.Module,
    degraded: torch.Tensor,
    *,
    mode: str,
    tile_size: int = 512,
    overlap: int = 128,
) -> torch.Tensor:
    if mode == "native":
        return _forward_bf16(model, degraded)
    if mode == "tiled":
        return tiled_inference(
            model,
            degraded,
            tile_size=tile_size,
            overlap=overlap,
        )
    raise ValueError(f"Unsupported inference mode: {mode!r}")

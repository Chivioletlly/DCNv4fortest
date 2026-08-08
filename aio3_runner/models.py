"""Frozen DCNv4 U-Net adapter for AIO3-v1."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from dcnv4_restoration_model import (
    DCNv4RestorationUNet,
    count_parameters,
    validate_restoration_checkpoint,
)


def build_dcnv4_unet(config: Dict[str, object]) -> DCNv4RestorationUNet:
    model_config = config["model"]
    model = DCNv4RestorationUNet(
        in_channels=int(model_config["in_channels"]),
        base_channels=int(model_config["base_channels"]),
        bottleneck_type=str(model_config["bottleneck_type"]),
        use_dcnv4=bool(model_config["use_dcnv4"]),
    )
    metadata = model.checkpoint_metadata()
    validate_restoration_checkpoint(metadata)
    return model


def model_parameter_counts(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = count_parameters(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    return total, trainable

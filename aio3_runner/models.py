"""DCNv4 U-Net model adapters for the frozen AIO3-v1 protocol."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from dcnv4_restoration_model import (
    DCNv4RestorationUNet,
    DegradationAwareDCNv4RestorationUNet,
    count_parameters,
    degradation_aware_checkpoint_metadata,
    restoration_checkpoint_metadata,
    validate_restoration_checkpoint,
)


def build_dcnv4_unet(config: Dict[str, object]) -> torch.nn.Module:
    model_config = config["model"]
    common_kwargs = {
        "in_channels": int(model_config["in_channels"]),
        "base_channels": int(model_config["base_channels"]),
        "bottleneck_type": str(model_config["bottleneck_type"]),
        "use_dcnv4": bool(model_config["use_dcnv4"]),
    }
    model_name = str(model_config["name"])
    if model_name == "dcnv4_restoration_unet":
        model = DCNv4RestorationUNet(**common_kwargs)
        expected_metadata = restoration_checkpoint_metadata()
    elif model_name == "degradation_aware_dcnv4_restoration_unet":
        model = DegradationAwareDCNv4RestorationUNet(
            **common_kwargs,
            context_scales=int(model_config["context_scales"]),
        )
        expected_metadata = degradation_aware_checkpoint_metadata()
    else:
        raise ValueError(f"Unsupported AIO3 DCNv4 model: {model_name!r}")
    metadata = model.checkpoint_metadata()
    validate_restoration_checkpoint(metadata, expected=expected_metadata)
    return model


def model_parameter_counts(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = count_parameters(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    return total, trainable

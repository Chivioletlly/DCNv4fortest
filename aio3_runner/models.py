"""DCNv4 U-Net model adapters for the frozen AIO3-v1 protocol."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from dcnv4_restoration_model import (
    AblationDCNv4RestorationUNet,
    DCNv4RestorationUNet,
    DegradationAwareDCNv4RestorationUNet,
    RegisteredDCNv4RestorationUNet,
    RegisteredDegradationAwareDCNv4RestorationUNet,
    count_parameters,
    degradation_aware_checkpoint_metadata,
    restoration_checkpoint_metadata,
    validate_restoration_checkpoint,
)

from .ablation import (
    ABLATION_REGISTRY_VERSION,
    get_ablation_variant,
    registry_snapshot,
)


def build_dcnv4_ablation_model(
    variant_id: str,
    *,
    in_channels: int = 3,
    base_channels: int = 64,
    bottleneck_type: str = "conv",
    use_dcnv4: bool = True,
    context_scales: int = 3,
    dcnv4_cls=None,
) -> torch.nn.Module:
    """Construct one registered ablation without consulting runner state."""

    variant = get_ablation_variant(variant_id)
    common_kwargs = {
        "in_channels": in_channels,
        "base_channels": base_channels,
        "bottleneck_type": bottleneck_type,
        "use_dcnv4": use_dcnv4,
        "dcnv4_cls": dcnv4_cls,
    }
    variant_metadata = variant.checkpoint_metadata()
    if variant.variant_id == "APG-000":
        return RegisteredDCNv4RestorationUNet(
            variant_metadata=variant_metadata,
            **common_kwargs,
        )
    if variant.variant_id == "APG-111":
        return RegisteredDegradationAwareDCNv4RestorationUNet(
            variant_metadata=variant_metadata,
            context_scales=context_scales,
            **common_kwargs,
        )
    return AblationDCNv4RestorationUNet(
        variant_metadata=variant_metadata,
        convolution_film=variant.convolution_film,
        dcnv4_input_affine=variant.dcnv4_input_affine,
        dcnv4_output_gate=variant.dcnv4_output_gate,
        skip_fusion_type=variant.skip_fusion,
        bottleneck_modulation_type=variant.bottleneck_modulation,
        degradation_context_type=variant.degradation_context,
        context_scales=context_scales,
        **common_kwargs,
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
    variant_id = model_config.get("variant_id")
    if variant_id is not None:
        variant = get_ablation_variant(str(variant_id))
        if config.get("ablation_registry_version") != ABLATION_REGISTRY_VERSION:
            raise RuntimeError("Run config has an incompatible ablation registry version")
        if config.get("ablation_registry") != registry_snapshot():
            raise RuntimeError("Run config differs from the frozen ablation registry snapshot")
        expected_registry = variant.model_config()
        mismatches = {
            key: (model_config.get(key), expected)
            for key, expected in expected_registry.items()
            if model_config.get(key) != expected
        }
        if mismatches:
            details = ", ".join(
                f"{key}={actual!r} (expected {expected!r})"
                for key, (actual, expected) in mismatches.items()
            )
            raise RuntimeError("Run config differs from the frozen ablation registry: " + details)
        model = build_dcnv4_ablation_model(
            variant.variant_id,
            **common_kwargs,
            context_scales=int(model_config.get("context_scales", 3)),
        )
        expected_metadata = variant.checkpoint_metadata()
    elif model_name == "dcnv4_restoration_unet":
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

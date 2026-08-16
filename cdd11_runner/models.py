"""Three frozen restoration-model adapters for the CDD-11 comparison."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch

from aio3_runner.models import build_dcnv4_ablation_model


MODEL_IDS: Tuple[str, ...] = (
    "unet",
    "degradation_aware_unet",
    "uformer",
)
MODEL_ALIASES = {
    "unet": "unet",
    "apg-000": "unet",
    "degradation_aware_unet": "degradation_aware_unet",
    "degradation-aware-unet": "degradation_aware_unet",
    "aware_unet": "degradation_aware_unet",
    "apg-111": "degradation_aware_unet",
    "uformer": "uformer",
    "uformer-b": "uformer",
    "uformer_b": "uformer",
}
EXPECTED_PARAMETERS = {
    "unet": 29_924_411,
    "degradation_aware_unet": 34_852_539,
    "uformer": 50_880_946,
}


def normalize_model_id(value: str) -> str:
    key = str(value).strip().casefold()
    try:
        return MODEL_ALIASES[key]
    except KeyError as error:
        raise ValueError(
            f"Unsupported model {value!r}; choose one of {MODEL_IDS}"
        ) from error


def frozen_model_config(model_id: str) -> Dict[str, object]:
    model_id = normalize_model_id(model_id)
    common = {
        "id": model_id,
        "pretrained": False,
        "autocast": "bf16",
        "expected_parameters": EXPECTED_PARAMETERS[model_id],
        "output_mode": "unbounded_restored_rgb",
    }
    if model_id == "unet":
        return {
            **common,
            "family": "dcnv4_unet",
            "variant_id": "APG-000",
            "in_channels": 3,
            "base_channels": 64,
            "bottleneck_type": "conv",
            "use_dcnv4": True,
            "context_scales": 3,
        }
    if model_id == "degradation_aware_unet":
        return {
            **common,
            "family": "dcnv4_unet",
            "variant_id": "APG-111",
            "in_channels": 3,
            "base_channels": 64,
            "bottleneck_type": "conv",
            "use_dcnv4": True,
            "context_scales": 3,
        }
    return {
        **common,
        "family": "uformer",
        "name": "uformer_b",
        "variant": "Uformer_B",
        "img_size": 128,
        "in_chans": 3,
        "dd_in": 3,
        "embed_dim": 32,
        "depths": [1, 2, 8, 8, 2, 8, 8, 2, 1],
        "num_heads": [1, 2, 4, 8, 16, 16, 8, 4, 2],
        "win_size": 8,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "patch_norm": True,
        "use_checkpoint": False,
        "token_projection": "linear",
        "token_mlp": "leff",
        "shift_flag": True,
        "modulator": True,
        "cross_modulator": False,
        "input_multiple": 128,
        "padding_mode": "zero_right_bottom_to_square_multiple_128",
        "initialization": "official_uformer_native_random_initialization",
    }


def validate_model_config(config: Mapping[str, object]) -> str:
    model_id = normalize_model_id(str(config.get("id", "")))
    expected = frozen_model_config(model_id)
    if dict(config) != expected:
        keys = sorted(set(config) | set(expected))
        mismatches = [
            f"{key}: {config.get(key)!r} != {expected.get(key)!r}"
            for key in keys
            if config.get(key) != expected.get(key)
        ]
        raise ValueError("Frozen CDD-11 model config mismatch: " + "; ".join(mismatches))
    return model_id


def _load_uformer_adapter(uformer_root: Path):
    root = Path(uformer_root).expanduser().resolve()
    adapter_path = root / "uformer_aio3_model.py"
    model_path = root / "model.py"
    if not adapter_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            "--uformer-root must contain uformer_aio3_model.py and model.py: "
            f"{root}"
        )
    existing_model = sys.modules.get("model")
    if existing_model is not None:
        existing_path = Path(str(getattr(existing_model, "__file__", ""))).resolve()
        if existing_path != model_path:
            raise RuntimeError(
                "Python module name 'model' is already loaded from a different path: "
                f"{existing_path}"
            )
    root_text = str(root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        module_name = "_cdd11_uformer_adapter"
        module = sys.modules.get(module_name)
        if module is None or Path(str(module.__file__)).resolve() != adapter_path:
            specification = importlib.util.spec_from_file_location(module_name, adapter_path)
            if specification is None or specification.loader is None:
                raise ImportError(f"Could not load Uformer adapter: {adapter_path}")
            module = importlib.util.module_from_spec(specification)
            sys.modules[module_name] = module
            specification.loader.exec_module(module)
        return module
    finally:
        if inserted:
            sys.path.remove(root_text)


def build_model(
    config: Mapping[str, object],
    *,
    uformer_root: Optional[Path] = None,
) -> torch.nn.Module:
    model_id = validate_model_config(config)
    if model_id in {"unet", "degradation_aware_unet"}:
        return build_dcnv4_ablation_model(
            str(config["variant_id"]),
            in_channels=int(config["in_channels"]),
            base_channels=int(config["base_channels"]),
            bottleneck_type=str(config["bottleneck_type"]),
            use_dcnv4=bool(config["use_dcnv4"]),
            context_scales=int(config["context_scales"]),
        )
    if uformer_root is None:
        raise ValueError("Uformer runs require an explicit --uformer-root")
    adapter = _load_uformer_adapter(Path(uformer_root))
    model = adapter.AIO3Uformer(dict(config))
    adapter.validate_uformer_checkpoint(model.checkpoint_metadata(), dict(config))
    return model


def architecture_metadata(model: torch.nn.Module) -> Dict[str, object]:
    callback = getattr(model, "checkpoint_metadata", None)
    if not callable(callback):
        raise TypeError("CDD-11 models must expose checkpoint_metadata()")
    metadata = callback()
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint_metadata() must return a mapping")
    return dict(metadata)


def validate_architecture_metadata(
    model: torch.nn.Module,
    metadata: Mapping[str, object],
) -> None:
    expected = architecture_metadata(model)
    if dict(metadata) != expected:
        raise RuntimeError("Checkpoint architecture metadata differs from the model")


def model_parameter_counts(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable

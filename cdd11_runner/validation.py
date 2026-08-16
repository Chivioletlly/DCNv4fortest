"""Frozen CDD-11 validation with shared inference and metric definitions."""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from .inference import restore_image
from .metrics import CDD11MetricAccumulator
from .protocol import DEGRADATIONS
from .runtime import atomic_write_json


@dataclass(frozen=True)
class ValidationResult:
    global_step: int
    summary: Mapping[str, float]
    per_image: List[Mapping[str, object]]
    visuals: List[Mapping[str, object]]


def safe_sample_name(sample_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("_")
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:80]}-{digest}"


def save_display_tensor(tensor: torch.Tensor, path: Path) -> None:
    value = tensor.detach().float().clamp(0.0, 1.0)
    array = (
        value.mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def save_visual_sample(
    *,
    visual_root: Path,
    sample_id: str,
    degradation: str,
    arity: int,
    degraded: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    psnr: float,
    ssim: float,
) -> Mapping[str, object]:
    sample_dir = visual_root / safe_sample_name(sample_id)
    residual = prediction.detach().float() - degraded.detach().float()
    absolute_error = (prediction.detach().float() - target.detach().float()).abs()
    paths = {
        "input_path": sample_dir / "input.png",
        "prediction_path": sample_dir / "prediction.png",
        "target_path": sample_dir / "target.png",
        "absolute_error_path": sample_dir / "absolute_error_range_0_0.25.png",
        "signed_residual_path": sample_dir / "signed_residual_range_-0.25_0.25.png",
    }
    save_display_tensor(degraded, paths["input_path"])
    save_display_tensor(prediction, paths["prediction_path"])
    save_display_tensor(target, paths["target_path"])
    save_display_tensor(absolute_error / 0.25, paths["absolute_error_path"])
    save_display_tensor((residual + 0.25) / 0.5, paths["signed_residual_path"])
    value: Dict[str, object] = {
        "sample_id": sample_id,
        "degradation": degradation,
        "arity": int(arity),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }
    value.update({key: str(path) for key, path in paths.items()})
    return value


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    global_step: int,
    inference_mode: str,
    tile_size: int = 512,
    tile_overlap: int = 128,
    visual_sample_ids: Optional[Sequence[str]] = None,
    visual_dir: Optional[Path] = None,
) -> ValidationResult:
    requested = tuple(str(value) for value in (visual_sample_ids or ()))
    if requested and visual_dir is None:
        raise ValueError("visual_dir is required for requested validation visuals")
    if len(set(requested)) != len(requested):
        raise ValueError("visual_sample_ids must be unique")
    requested_set = set(requested)
    visuals_by_id: Dict[str, Mapping[str, object]] = {}
    accumulator = CDD11MetricAccumulator()
    raw_l1: Dict[str, List[float]] = {value: [] for value in DEGRADATIONS}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in dataloader:
                degraded = batch["degraded"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                prediction = restore_image(
                    model,
                    degraded,
                    mode=inference_mode,
                    tile_size=tile_size,
                    overlap=tile_overlap,
                )
                degradation = str(batch["degradation"][0])
                arity = int(batch["arity"][0])
                sample_id = str(batch["sample_id"][0])
                raw_l1[degradation].append(
                    float(F.l1_loss(prediction, target.float()).item())
                )
                accumulator.add_batch(
                    sample_ids=[sample_id],
                    degradations=[degradation],
                    arities=[arity],
                    prediction=prediction,
                    target=target,
                )
                if sample_id in requested_set:
                    metric = accumulator.rows[-1]
                    visuals_by_id[sample_id] = save_visual_sample(
                        visual_root=Path(visual_dir),
                        sample_id=sample_id,
                        degradation=degradation,
                        arity=arity,
                        degraded=degraded[0],
                        prediction=prediction[0],
                        target=target[0],
                        psnr=metric.psnr,
                        ssim=metric.ssim,
                    )
    finally:
        model.train(was_training)

    summary = accumulator.summarize()
    for degradation, values in raw_l1.items():
        if not values:
            raise RuntimeError(f"Validation contains no {degradation!r} samples")
        summary[f"{degradation}/raw_l1"] = sum(values) / len(values)
    missing = [sample_id for sample_id in requested if sample_id not in visuals_by_id]
    if missing:
        raise RuntimeError(f"Fixed validation visual IDs were not found: {missing}")
    return ValidationResult(
        global_step=int(global_step),
        summary=summary,
        per_image=accumulator.per_image_dicts(),
        visuals=[visuals_by_id[value] for value in requested],
    )


def _atomic_write_csv(path: Path, fieldnames, rows) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_validation_result(result: ValidationResult, validation_dir: Path) -> None:
    validation_dir = Path(validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)
    stem = f"step_{result.global_step:06d}"
    atomic_write_json(
        validation_dir / f"metrics_{stem}.json",
        {
            "global_step": result.global_step,
            "summary": dict(result.summary),
            "per_image_count": len(result.per_image),
        },
    )
    _atomic_write_csv(
        validation_dir / f"per_image_metrics_{stem}.csv",
        ("sample_id", "degradation", "arity", "psnr", "ssim"),
        result.per_image,
    )
    if result.visuals:
        atomic_write_json(
            validation_dir / f"visuals_{stem}.json",
            {
                "global_step": result.global_step,
                "samples": result.visuals,
            },
        )

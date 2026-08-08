"""Native-resolution validation for the frozen AIO3-v1 protocol."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from .metrics import AIO3MetricAccumulator
from .runtime import atomic_write_json


@dataclass(frozen=True)
class ValidationResult:
    global_step: int
    summary: Mapping[str, float]
    per_image: List[Mapping[str, object]]
    visuals: List[Mapping[str, object]]
    residual_histogram: List[float]


def _safe_sample_directory(sample_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("_")
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:80]}-{digest}"


def _save_display_tensor(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.detach().float().clamp(0.0, 1.0)
    array = (
        tensor.mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _save_visual_sample(
    *,
    visual_root: Path,
    sample_id: str,
    task: str,
    sigma: Optional[int],
    degraded: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    psnr: float,
    ssim: float,
) -> tuple:
    sample_dir = visual_root / _safe_sample_directory(sample_id)
    residual = prediction.detach().float() - degraded.detach().float()
    absolute_error = (prediction.detach().float() - target.detach().float()).abs()
    paths = {
        "input_path": sample_dir / "input.png",
        "prediction_path": sample_dir / "prediction.png",
        "target_path": sample_dir / "target.png",
        "absolute_error_path": sample_dir / "absolute_error_range_0_0.25.png",
        "signed_residual_path": sample_dir / "signed_residual_range_-0.25_0.25.png",
    }
    _save_display_tensor(degraded, paths["input_path"])
    _save_display_tensor(prediction, paths["prediction_path"])
    _save_display_tensor(target, paths["target_path"])
    _save_display_tensor(absolute_error / 0.25, paths["absolute_error_path"])
    _save_display_tensor((residual + 0.25) / 0.5, paths["signed_residual_path"])

    flattened = residual.flatten()
    stride = max(1, flattened.numel() // 4096)
    histogram_values = flattened[::stride][:4096].cpu().tolist()
    visual = {
        "sample_id": sample_id,
        "task": task,
        "sigma": sigma,
        "psnr": float(psnr),
        "ssim": float(ssim),
        "residual_mean": float(residual.mean().item()),
        "residual_negative_fraction": float((residual < 0.0).float().mean().item()),
    }
    visual.update({key: str(path) for key, path in paths.items()})
    return visual, histogram_values


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    global_step: int,
    visual_sample_ids: Optional[Sequence[str]] = None,
    visual_dir: Optional[Path] = None,
) -> ValidationResult:
    was_training = model.training
    model.eval()
    accumulator = AIO3MetricAccumulator()
    raw_l1_by_task: Dict[str, List[float]] = {
        "denoise": [],
        "derain": [],
        "dehaze": [],
    }
    residual_negative_by_task: Dict[str, List[float]] = {
        "denoise": [],
        "derain": [],
        "dehaze": [],
    }
    requested_visual_ids = tuple(visual_sample_ids or ())
    if requested_visual_ids and visual_dir is None:
        raise ValueError("visual_dir is required when visual_sample_ids are requested")
    if len(set(requested_visual_ids)) != len(requested_visual_ids):
        raise ValueError("visual_sample_ids must be unique")
    visual_id_set = set(requested_visual_ids)
    visuals_by_id: Dict[str, Mapping[str, object]] = {}
    residual_histogram: List[float] = []
    try:
        with torch.inference_mode():
            for batch in dataloader:
                degraded = batch["degraded"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    restored_raw = model(degraded)
                restored_float = restored_raw.float()
                task = str(batch["task"][0])
                raw_l1_by_task[task].append(
                    float(F.l1_loss(restored_float, target.float()).item())
                )
                residual = restored_float - degraded.float()
                residual_negative_by_task[task].append(
                    float((residual < 0.0).float().mean().item())
                )
                accumulator.add_batch(
                    sample_ids=batch["sample_id"],
                    tasks=batch["task"],
                    sigmas=[int(value) for value in batch["sigma"]],
                    prediction=restored_float,
                    target=target,
                )
                sample_id = str(batch["sample_id"][0])
                if sample_id in visual_id_set:
                    row = accumulator.rows[-1]
                    visual, histogram_values = _save_visual_sample(
                        visual_root=Path(visual_dir),
                        sample_id=sample_id,
                        task=task,
                        sigma=row.sigma,
                        degraded=degraded[0],
                        prediction=restored_float[0],
                        target=target[0],
                        psnr=row.psnr,
                        ssim=row.ssim,
                    )
                    visuals_by_id[sample_id] = visual
                    residual_histogram.extend(histogram_values)
    finally:
        model.train(was_training)

    summary = accumulator.summarize()
    for task, values in raw_l1_by_task.items():
        if not values:
            raise RuntimeError(f"Validation contains no {task} samples")
        summary[f"{task}/raw_l1"] = sum(values) / len(values)
        summary[f"{task}/residual_negative_fraction"] = (
            sum(residual_negative_by_task[task]) / len(values)
        )
    missing_visuals = [
        sample_id for sample_id in requested_visual_ids if sample_id not in visuals_by_id
    ]
    if missing_visuals:
        raise RuntimeError(f"Fixed validation visual IDs were not found: {missing_visuals}")
    return ValidationResult(
        global_step=int(global_step),
        summary=summary,
        per_image=accumulator.per_image_dicts(),
        visuals=[visuals_by_id[sample_id] for sample_id in requested_visual_ids],
        residual_histogram=residual_histogram,
    )


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
    if result.visuals:
        atomic_write_json(
            validation_dir / f"visuals_{stem}.json",
            {
                "global_step": result.global_step,
                "absolute_error_display_range": [0.0, 0.25],
                "signed_residual_display_range": [-0.25, 0.25],
                "samples": result.visuals,
            },
        )
    csv_path = validation_dir / f"per_image_metrics_{stem}.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "task", "sigma", "psnr", "ssim"),
        )
        writer.writeheader()
        writer.writerows(result.per_image)
    temporary.replace(csv_path)

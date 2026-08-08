"""Native-resolution validation for the frozen AIO3-v1 protocol."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import torch
import torch.nn.functional as F

from .metrics import AIO3MetricAccumulator
from .runtime import atomic_write_json


@dataclass(frozen=True)
class ValidationResult:
    global_step: int
    summary: Mapping[str, float]
    per_image: List[Mapping[str, object]]


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    global_step: int,
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
    return ValidationResult(
        global_step=int(global_step),
        summary=summary,
        per_image=accumulator.per_image_dicts(),
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

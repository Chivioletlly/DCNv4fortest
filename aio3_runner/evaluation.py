"""Frozen native-resolution AIO3-v1 formal test evaluation."""

from __future__ import annotations

import csv
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch

from .data import AIO3ManifestDataset
from .metrics import AIO3MetricAccumulator
from .protocol import AIO3_PROTOCOL_VERSION, NOISE_SIGMAS
from .runtime import atomic_write_json
from .validation import (
    _safe_sample_directory,
    _save_display_tensor,
    _save_visual_sample,
)


@dataclass(frozen=True)
class TestResult:
    global_step: int
    summary: Mapping[str, float]
    per_image: List[Mapping[str, object]]
    visuals: List[Mapping[str, object]]
    total_runtime_seconds: float


def _gallery_rank(sample_id: str) -> bytes:
    value = f"{AIO3_PROTOCOL_VERSION}:test-gallery:{sample_id}"
    return hashlib.sha256(value.encode("utf-8")).digest()


def select_fixed_test_gallery(dataset: AIO3ManifestDataset) -> Dict[str, object]:
    """Select 14 test samples deterministically without consulting predictions."""

    if dataset.split != "test":
        raise ValueError("Fixed test gallery selection requires the test split")
    selected: Dict[str, List[str]] = {}
    for sigma in NOISE_SIGMAS:
        candidates = [
            record
            for record in dataset.records
            if record.task == "denoise" and int(record.metadata["sigma"]) == sigma
        ]
        candidates.sort(key=lambda record: _gallery_rank(record.sample_id))
        selected[f"denoise_sigma{sigma}"] = [record.sample_id for record in candidates[:2]]
    for task, count in (("derain", 4), ("dehaze", 4)):
        candidates = [record for record in dataset.records if record.task == task]
        candidates.sort(key=lambda record: _gallery_rank(record.sample_id))
        selected[task] = [record.sample_id for record in candidates[:count]]

    expected_counts = {
        "denoise_sigma15": 2,
        "denoise_sigma25": 2,
        "denoise_sigma50": 2,
        "derain": 4,
        "dehaze": 4,
    }
    for group, expected_count in expected_counts.items():
        if len(selected[group]) != expected_count:
            raise RuntimeError(
                f"Test gallery group {group} requires {expected_count} samples, "
                f"found {len(selected[group])}"
            )
    ordered = [sample_id for group in expected_counts for sample_id in selected[group]]
    if len(set(ordered)) != 14:
        raise RuntimeError("Fixed test gallery sample IDs must be unique")
    return {
        "protocol": AIO3_PROTOCOL_VERSION,
        "selection_rule": "lowest SHA256(aio3-v1:test-gallery:<sample_id>)",
        "samples": selected,
        "ordered_sample_ids": ordered,
    }


def _prediction_group(task: str, sigma: Optional[int]) -> str:
    if task == "denoise":
        if sigma not in NOISE_SIGMAS:
            raise ValueError(f"Invalid denoise sigma: {sigma}")
        return f"BSD68_sigma{sigma}"
    if task == "derain":
        return "Rain100L"
    if task == "dehaze":
        return "SOTS-outdoor"
    raise ValueError(f"Unsupported test task: {task!r}")


def _dataset_label(task: str) -> str:
    return {
        "denoise": "BSD68",
        "derain": "Rain100L",
        "dehaze": "SOTS-outdoor",
    }[task]


def _test_summary(metric_summary: Mapping[str, float], timings: Sequence[float], total: float):
    summary: Dict[str, float] = {}
    for sigma in NOISE_SIGMAS:
        for metric in ("psnr", "ssim", "images"):
            summary[f"bsd68/sigma{sigma}/{metric}"] = float(
                metric_summary[f"denoise/sigma{sigma}/{metric}"]
            )
    for metric in ("psnr", "ssim"):
        summary[f"bsd68/mean/{metric}"] = float(metric_summary[f"denoise/mean/{metric}"])
        summary[f"rain100l/{metric}"] = float(metric_summary[f"derain/{metric}"])
        summary[f"sots_outdoor/{metric}"] = float(metric_summary[f"dehaze/{metric}"])
        summary[f"macro/{metric}"] = float(metric_summary[f"macro/{metric}"])
    summary["rain100l/images"] = float(metric_summary["derain/images"])
    summary["sots_outdoor/images"] = float(metric_summary["dehaze/images"])
    summary["images"] = float(metric_summary["images"])
    summary["total_runtime_seconds"] = float(total)
    summary["mean_inference_seconds"] = float(sum(timings) / len(timings))
    return summary


def evaluate_test_model(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    global_step: int,
    prediction_dir: Path,
    gallery_dir: Path,
    gallery_sample_ids: Sequence[str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> TestResult:
    """Evaluate all test images once, saving PNGs but measuring FP32 tensors."""

    prediction_dir = Path(prediction_dir)
    gallery_dir = Path(gallery_dir)
    if prediction_dir.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {prediction_dir}")
    if gallery_dir.exists():
        raise FileExistsError(f"Refusing to overwrite test gallery: {gallery_dir}")
    prediction_dir.mkdir(parents=True)
    gallery_dir.mkdir(parents=True)

    requested_gallery_ids = tuple(str(value) for value in gallery_sample_ids)
    if len(requested_gallery_ids) != 14 or len(set(requested_gallery_ids)) != 14:
        raise ValueError("Formal test gallery must contain exactly 14 unique sample IDs")
    gallery_id_set = set(requested_gallery_ids)
    visuals_by_id: Dict[str, Mapping[str, object]] = {}
    accumulator = AIO3MetricAccumulator()
    per_image: List[Mapping[str, object]] = []
    timings: List[float] = []
    total_images = len(dataloader.dataset)
    was_training = model.training
    model.eval()
    total_started = time.perf_counter()
    try:
        with torch.inference_mode():
            for processed, batch in enumerate(dataloader, start=1):
                degraded = batch["degraded"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_started = time.perf_counter()
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    restored_raw = model(degraded)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds = time.perf_counter() - inference_started
                timings.append(inference_seconds)

                restored_float = restored_raw.float()
                task = str(batch["task"][0])
                sigma_value = int(batch["sigma"][0])
                sigma = sigma_value if task == "denoise" else None
                sample_id = str(batch["sample_id"][0])
                accumulator.add_batch(
                    sample_ids=[sample_id],
                    tasks=[task],
                    sigmas=[sigma_value],
                    prediction=restored_float,
                    target=target,
                )
                metric_row = accumulator.rows[-1]
                prediction_group = _prediction_group(task, sigma)
                filename = _safe_sample_directory(sample_id) + ".png"
                prediction_path = prediction_dir / prediction_group / filename
                if prediction_path.exists():
                    raise FileExistsError(f"Refusing to overwrite prediction: {prediction_path}")
                _save_display_tensor(restored_float[0], prediction_path)
                per_image.append(
                    {
                        "dataset": _dataset_label(task),
                        "task": task,
                        "sigma": sigma,
                        "sample_id": sample_id,
                        "psnr": metric_row.psnr,
                        "ssim": metric_row.ssim,
                        "inference_time_seconds": inference_seconds,
                        "prediction_path": str(prediction_path),
                    }
                )

                if sample_id in gallery_id_set:
                    visual, _ = _save_visual_sample(
                        visual_root=gallery_dir,
                        sample_id=sample_id,
                        task=task,
                        sigma=sigma,
                        degraded=degraded[0],
                        prediction=restored_float[0],
                        target=target[0],
                        psnr=metric_row.psnr,
                        ssim=metric_row.ssim,
                    )
                    visuals_by_id[sample_id] = visual

                if progress_callback is not None and (
                    processed % 25 == 0 or processed == total_images
                ):
                    progress_callback(processed, total_images)
    finally:
        model.train(was_training)

    total_runtime_seconds = time.perf_counter() - total_started
    missing_gallery = [
        sample_id for sample_id in requested_gallery_ids if sample_id not in visuals_by_id
    ]
    if missing_gallery:
        raise RuntimeError(f"Fixed test gallery IDs were not found: {missing_gallery}")
    summary = _test_summary(
        accumulator.summarize(),
        timings,
        total_runtime_seconds,
    )
    return TestResult(
        global_step=int(global_step),
        summary=summary,
        per_image=per_image,
        visuals=[visuals_by_id[value] for value in requested_gallery_ids],
        total_runtime_seconds=total_runtime_seconds,
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


def write_test_result(
    result: TestResult,
    test_dir: Path,
    *,
    metadata: Mapping[str, object],
) -> None:
    test_dir = Path(test_dir)
    metrics_path = test_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite formal test metrics: {metrics_path}")
    atomic_write_json(
        metrics_path,
        {
            "protocol": AIO3_PROTOCOL_VERSION,
            "global_step": result.global_step,
            "metadata": dict(metadata),
            "summary": dict(result.summary),
            "per_image_count": len(result.per_image),
        },
    )
    summary_rows = []
    for dataset, condition, prefix, images in (
        (
            "BSD68",
            "sigma15",
            "bsd68/sigma15",
            int(result.summary["bsd68/sigma15/images"]),
        ),
        (
            "BSD68",
            "sigma25",
            "bsd68/sigma25",
            int(result.summary["bsd68/sigma25/images"]),
        ),
        (
            "BSD68",
            "sigma50",
            "bsd68/sigma50",
            int(result.summary["bsd68/sigma50/images"]),
        ),
        (
            "BSD68",
            "equal_sigma_mean",
            "bsd68/mean",
            int(sum(result.summary[f"bsd68/sigma{sigma}/images"] for sigma in NOISE_SIGMAS)),
        ),
        (
            "Rain100L",
            "default",
            "rain100l",
            int(result.summary["rain100l/images"]),
        ),
        (
            "SOTS-outdoor",
            "default",
            "sots_outdoor",
            int(result.summary["sots_outdoor/images"]),
        ),
        ("AIO3", "task_macro", "macro", int(result.summary["images"])),
    ):
        summary_rows.append(
            {
                "protocol": AIO3_PROTOCOL_VERSION,
                "dataset": dataset,
                "condition": condition,
                "images": images,
                "psnr": result.summary[f"{prefix}/psnr"],
                "ssim": result.summary[f"{prefix}/ssim"],
            }
        )
    _atomic_write_csv(
        test_dir / "metrics.csv",
        ("protocol", "dataset", "condition", "images", "psnr", "ssim"),
        summary_rows,
    )
    _atomic_write_csv(
        test_dir / "per_image_metrics.csv",
        (
            "dataset",
            "task",
            "sigma",
            "sample_id",
            "psnr",
            "ssim",
            "inference_time_seconds",
            "prediction_path",
        ),
        result.per_image,
    )
    atomic_write_json(
        test_dir / "gallery.json",
        {
            "global_step": result.global_step,
            "absolute_error_display_range": [0.0, 0.25],
            "signed_residual_display_range": [-0.25, 0.25],
            "samples": result.visuals,
        },
    )

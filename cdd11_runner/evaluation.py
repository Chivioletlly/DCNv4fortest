"""Formal CDD-11 test evaluation and immutable result artifacts."""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch

from .data import CDD11ManifestDataset
from .inference import restore_image
from .metrics import CDD11MetricAccumulator
from .protocol import ARITY_GROUPS, CDD11_PROTOCOL_VERSION, DEGRADATIONS, sample_sort_key
from .runtime import atomic_write_json
from .validation import safe_sample_name, save_display_tensor, save_visual_sample


@dataclass(frozen=True)
class TestResult:
    global_step: int
    summary: Mapping[str, float]
    per_image: List[Mapping[str, object]]
    visuals: List[Mapping[str, object]]
    total_runtime_seconds: float


def select_fixed_test_gallery(dataset: CDD11ManifestDataset) -> Dict[str, object]:
    if dataset.split != "test":
        raise ValueError("Test gallery selection requires the test split")
    scenes = sorted(
        {record.scene_id for record in dataset.records},
        key=lambda value: sample_sort_key("test-gallery-scene", value),
    )[:2]
    lookup = {
        (record.scene_id, record.degradation): record.sample_id
        for record in dataset.records
    }
    ordered = [
        lookup[(scene_id, degradation)]
        for scene_id in scenes
        for degradation in DEGRADATIONS
    ]
    if len(scenes) != 2 or len(ordered) != 22 or len(set(ordered)) != 22:
        raise RuntimeError("CDD-11 test gallery requires 2 complete scenes x 11 categories")
    return {
        "protocol": CDD11_PROTOCOL_VERSION,
        "selection_rule": (
            "two lowest SHA256(cdd11-v1:test-gallery-scene:<scene_id>), "
            "then frozen degradation order"
        ),
        "scene_ids": scenes,
        "ordered_sample_ids": ordered,
    }


def evaluate_test_model(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    global_step: int,
    prediction_dir: Path,
    gallery_dir: Path,
    gallery_sample_ids: Sequence[str],
    inference_mode: str,
    tile_size: int = 512,
    tile_overlap: int = 128,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> TestResult:
    prediction_dir = Path(prediction_dir)
    gallery_dir = Path(gallery_dir)
    if prediction_dir.exists() or gallery_dir.exists():
        raise FileExistsError("Refusing to overwrite predictions or gallery")
    prediction_dir.mkdir(parents=True)
    gallery_dir.mkdir(parents=True)
    requested = tuple(str(value) for value in gallery_sample_ids)
    if len(requested) != 22 or len(set(requested)) != 22:
        raise ValueError("CDD-11 formal gallery requires exactly 22 unique samples")
    requested_set = set(requested)
    visuals_by_id: Dict[str, Mapping[str, object]] = {}
    accumulator = CDD11MetricAccumulator()
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
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                prediction = restore_image(
                    model,
                    degraded,
                    mode=inference_mode,
                    tile_size=tile_size,
                    overlap=tile_overlap,
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                timings.append(elapsed)
                sample_id = str(batch["sample_id"][0])
                degradation = str(batch["degradation"][0])
                arity = int(batch["arity"][0])
                accumulator.add_batch(
                    sample_ids=[sample_id],
                    degradations=[degradation],
                    arities=[arity],
                    prediction=prediction,
                    target=target,
                )
                metric = accumulator.rows[-1]
                prediction_path = (
                    prediction_dir / degradation / f"{safe_sample_name(sample_id)}.png"
                )
                if prediction_path.exists():
                    raise FileExistsError(f"Duplicate prediction path: {prediction_path}")
                save_display_tensor(prediction[0], prediction_path)
                per_image.append(
                    {
                        "sample_id": sample_id,
                        "degradation": degradation,
                        "arity": arity,
                        "psnr": metric.psnr,
                        "ssim": metric.ssim,
                        "inference_time_seconds": elapsed,
                        "prediction_path": str(prediction_path),
                    }
                )
                if sample_id in requested_set:
                    visuals_by_id[sample_id] = save_visual_sample(
                        visual_root=gallery_dir,
                        sample_id=sample_id,
                        degradation=degradation,
                        arity=arity,
                        degraded=degraded[0],
                        prediction=prediction[0],
                        target=target[0],
                        psnr=metric.psnr,
                        ssim=metric.ssim,
                    )
                if progress_callback is not None and (
                    processed % 25 == 0 or processed == total_images
                ):
                    progress_callback(processed, total_images)
    finally:
        model.train(was_training)
    total_runtime = time.perf_counter() - total_started
    missing = [sample_id for sample_id in requested if sample_id not in visuals_by_id]
    if missing:
        raise RuntimeError(f"Test gallery IDs not found: {missing}")
    summary = accumulator.summarize()
    summary["total_runtime_seconds"] = float(total_runtime)
    summary["mean_inference_seconds"] = float(sum(timings) / len(timings))
    return TestResult(
        global_step=int(global_step),
        summary=summary,
        per_image=per_image,
        visuals=[visuals_by_id[value] for value in requested],
        total_runtime_seconds=total_runtime,
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
        raise FileExistsError(f"Refusing to overwrite formal metrics: {metrics_path}")
    atomic_write_json(
        metrics_path,
        {
            "protocol": CDD11_PROTOCOL_VERSION,
            "global_step": result.global_step,
            "metadata": dict(metadata),
            "summary": dict(result.summary),
            "per_image_count": len(result.per_image),
        },
    )
    summary_rows = []
    for degradation in DEGRADATIONS:
        summary_rows.append(
            {
                "protocol": CDD11_PROTOCOL_VERSION,
                "group": "degradation",
                "condition": degradation,
                "images": int(result.summary[f"{degradation}/images"]),
                "psnr": result.summary[f"{degradation}/psnr"],
                "ssim": result.summary[f"{degradation}/ssim"],
            }
        )
    for group in ("single", "double", "triple", "macro"):
        images = (
            int(result.summary["images"])
            if group == "macro"
            else sum(
                int(result.summary[f"{degradation}/images"])
                for degradation in ARITY_GROUPS[group]
            )
        )
        summary_rows.append(
            {
                "protocol": CDD11_PROTOCOL_VERSION,
                "group": "aggregate",
                "condition": group,
                "images": images,
                "psnr": result.summary[f"{group}/psnr"],
                "ssim": result.summary[f"{group}/ssim"],
            }
        )
    _atomic_write_csv(
        test_dir / "metrics.csv",
        ("protocol", "group", "condition", "images", "psnr", "ssim"),
        summary_rows,
    )
    _atomic_write_csv(
        test_dir / "per_image_metrics.csv",
        (
            "sample_id",
            "degradation",
            "arity",
            "psnr",
            "ssim",
            "inference_time_seconds",
            "prediction_path",
        ),
        result.per_image,
    )
    atomic_write_json(
        test_dir / "gallery.json",
        {"global_step": result.global_step, "samples": result.visuals},
    )

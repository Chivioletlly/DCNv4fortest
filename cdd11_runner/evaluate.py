"""Command-line entry point for completed formal CDD-11-v1 test runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

import torch

from .checkpoint import load_checkpoint, validate_checkpoint_identity
from .data import build_eval_dataloader
from .evaluation import evaluate_test_model, select_fixed_test_gallery, write_test_result
from .models import (
    architecture_metadata,
    build_model,
    model_parameter_counts,
    validate_architecture_metadata,
)
from .runtime import (
    atomic_write_json,
    file_sha256,
    git_state,
    load_run_config,
    seed_everything,
    verify_manifest_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write_state(
    test_dir: Path,
    *,
    status: str,
    global_step: int,
    processed_images: int,
    total_images: int,
    message: Optional[str] = None,
    summary: Optional[Mapping[str, float]] = None,
) -> None:
    value = {
        "status": status,
        "global_step": int(global_step),
        "processed_images": int(processed_images),
        "total_images": int(total_images),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if message is not None:
        value["message"] = message
    if summary is not None:
        value["macro_psnr"] = float(summary["macro/psnr"])
        value["macro_ssim"] = float(summary["macro/ssim"])
    atomic_write_json(test_dir / "state.json", value)


def _resolve(checkpoint_path: Path):
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if checkpoint_path.name != "best_macro_psnr.pth":
        raise RuntimeError("Formal test requires checkpoints/best_macro_psnr.pth")
    checkpoint = load_checkpoint(checkpoint_path)
    run_dir = Path(str(checkpoint["run_dir"])).expanduser().resolve()
    if checkpoint_path != run_dir / "checkpoints" / "best_macro_psnr.pth":
        raise RuntimeError("Checkpoint is outside its frozen run directory")
    config = load_run_config(run_dir / "config.yaml")
    if config.get("run_kind") != "formal":
        raise RuntimeError("CDD-11 test data may only be used by a formal run")
    run_state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    if run_state.get("status") != "completed" or int(run_state["global_step"]) != 200_000:
        raise RuntimeError(f"Formal training is not complete: {run_state}")
    repository_state = git_state(REPOSITORY_ROOT)
    if repository_state["dirty"]:
        raise RuntimeError("Refusing formal evaluation from a dirty worktree")
    uformer_commit = None
    if config["model"]["id"] == "uformer":
        uformer_state = git_state(Path(str(config["paths"]["uformer_root"])))
        if uformer_state["dirty"]:
            raise RuntimeError("Refusing formal evaluation from dirty Uformer source")
        uformer_commit = str(uformer_state["commit"])
    validate_checkpoint_identity(
        checkpoint,
        config=config,
        repository_commit=str(repository_state["commit"]),
        uformer_commit=uformer_commit,
    )
    verified = verify_manifest_bundle(Path(str(config["paths"]["manifest_dir"])))
    if verified["hashes"] != config["data"]["manifest_sha256"]:
        raise RuntimeError("Formal evaluation manifest hash mismatch")
    return checkpoint_path, checkpoint, run_dir, config, repository_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a formal CDD-11-v1 run")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("CDD-11 formal evaluation requires a CUDA GPU")
    checkpoint_path, checkpoint, run_dir, config, repository_state = _resolve(
        args.checkpoint
    )
    test_dir = run_dir / "test"
    conflicts = [
        path
        for path in (
            test_dir / "metrics.json",
            test_dir / "metrics.csv",
            test_dir / "per_image_metrics.csv",
            test_dir / "predictions",
            test_dir / "gallery",
        )
        if path.exists()
    ]
    if conflicts:
        raise RuntimeError(f"Refusing to overwrite formal outputs: {conflicts}")
    device = torch.device("cuda", 0)
    seed_everything(int(config["seed"]))
    uformer_root = config["paths"].get("uformer_root")
    model = build_model(
        config["model"],
        uformer_root=Path(str(uformer_root)) if uformer_root else None,
    ).to(device)
    total, trainable = model_parameter_counts(model)
    if total != trainable or total != int(config["model"]["expected_parameters"]):
        raise RuntimeError("Formal evaluation model parameter count mismatch")
    validate_architecture_metadata(model, checkpoint["architecture"])
    model.load_state_dict(checkpoint["model"], strict=True)
    global_step = int(checkpoint["global_step"])
    checkpoint_metadata = {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "global_step": global_step,
        "best_metrics": dict(checkpoint["best_metrics"]),
        "architecture": architecture_metadata(model),
    }
    manifest_dir = Path(str(config["paths"]["manifest_dir"]))
    test_loader, dataset = build_eval_dataloader(
        manifest_dir / "test.jsonl",
        split="test",
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if len(dataset) != 2_200:
        raise RuntimeError(f"Frozen CDD-11 test split requires 2200 rows, got {len(dataset)}")
    gallery = select_fixed_test_gallery(dataset)
    atomic_write_json(test_dir / "gallery_selection.json", gallery)
    processed = 0
    _write_state(
        test_dir,
        status="evaluating",
        global_step=global_step,
        processed_images=0,
        total_images=len(dataset),
    )
    try:
        def update_progress(value: int, total_images: int) -> None:
            nonlocal processed
            processed = value
            _write_state(
                test_dir,
                status="evaluating",
                global_step=global_step,
                processed_images=value,
                total_images=total_images,
            )
            print(f"CDD-11 formal test: {value}/{total_images}", flush=True)

        result = evaluate_test_model(
            model,
            test_loader,
            device=device,
            global_step=global_step,
            prediction_dir=test_dir / "predictions",
            gallery_dir=test_dir / "gallery",
            gallery_sample_ids=gallery["ordered_sample_ids"],
            inference_mode=str(config["validation"]["inference_mode"]),
            tile_size=int(config["validation"]["tile_size"]),
            tile_overlap=int(config["validation"]["tile_overlap"]),
            progress_callback=update_progress,
        )
        if len(result.per_image) != 2_200:
            raise RuntimeError(
                f"Formal test produced {len(result.per_image)} metric rows, expected 2200"
            )
        prediction_count = sum(
            1 for path in (test_dir / "predictions").rglob("*.png") if path.is_file()
        )
        if prediction_count != 2_200:
            raise RuntimeError(
                f"Formal test produced {prediction_count} predictions, expected 2200"
            )
        metadata = {
            "checkpoint": checkpoint_metadata,
            "manifest_sha256": dict(config["data"]["manifest_sha256"]),
            "training_repository_commit": config["source"]["repository_commit"],
            "evaluation_repository_commit": repository_state["commit"],
            "uformer_commit": config["source"]["uformer_commit"],
            "precision": "bf16",
            "inference_mode": config["validation"]["inference_mode"],
            "tile_size": config["validation"]["tile_size"],
            "tile_overlap": config["validation"]["tile_overlap"],
            "batch_size": 1,
            "test_time_augmentation": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_test_result(result, test_dir, metadata=metadata)
        _write_state(
            test_dir,
            status="completed",
            global_step=global_step,
            processed_images=len(dataset),
            total_images=len(dataset),
            summary=result.summary,
        )
    except Exception as error:
        _write_state(
            test_dir,
            status="failed",
            global_step=global_step,
            processed_images=processed,
            total_images=len(dataset),
            message=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()

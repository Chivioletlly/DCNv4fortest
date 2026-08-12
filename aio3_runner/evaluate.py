"""Command-line entry point for the frozen AIO3-v1 formal test."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

import torch

from dcnv4_restoration_model import validate_restoration_checkpoint

from .checkpoint import load_checkpoint, validate_checkpoint_identity
from .data import build_eval_dataloader
from .evaluation import (
    evaluate_test_model,
    select_fixed_test_gallery,
    write_test_result,
)
from .models import build_dcnv4_unet, model_parameter_counts
from .monitoring import WandbMonitor
from .runtime import (
    atomic_write_json,
    file_sha256,
    git_state,
    load_run_config,
    seed_everything,
    verify_manifest_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a completed formal AIO3-v1 run on the frozen test split."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the formal run's checkpoints/best_macro_psnr.pth.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_test_state(
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
        "updated_at_utc": _utc_now(),
    }
    if message is not None:
        value["message"] = message
    if summary is not None:
        value["macro_psnr"] = float(summary["macro/psnr"])
        value["macro_ssim"] = float(summary["macro/ssim"])
    atomic_write_json(test_dir / "state.json", value)


def validate_formal_evaluation_config(config: Mapping[str, object]) -> None:
    if config.get("run_kind") != "formal":
        raise RuntimeError(
            "AIO3-v1 test data may only be evaluated by a completed formal run; "
            f"got run_kind={config.get('run_kind')!r}"
        )


def _resolve_formal_evaluation(checkpoint_path: Path):
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if checkpoint_path.name != "best_macro_psnr.pth":
        raise RuntimeError("Formal test requires checkpoints/best_macro_psnr.pth")
    checkpoint = load_checkpoint(checkpoint_path)
    run_dir = Path(str(checkpoint["run_dir"])).expanduser().resolve()
    expected_checkpoint = run_dir / "checkpoints" / "best_macro_psnr.pth"
    if checkpoint_path != expected_checkpoint:
        raise RuntimeError(
            f"Checkpoint path is outside its frozen run directory: {checkpoint_path}"
        )
    config = load_run_config(run_dir / "config.yaml")
    validate_formal_evaluation_config(config)
    repository_state = git_state(REPOSITORY_ROOT)
    if repository_state["dirty"]:
        raise RuntimeError("Refusing formal evaluation from a dirty worktree")
    validate_checkpoint_identity(
        checkpoint,
        config=config,
        current_repository_commit=str(repository_state["commit"]),
    )
    run_state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    max_steps = int(config["training"]["max_steps"])
    if run_state.get("status") != "completed" or int(run_state["global_step"]) != max_steps:
        raise RuntimeError(f"Formal training run is not complete: {run_state}")
    if int(checkpoint["best_metrics"]["global_step"]) != int(checkpoint["global_step"]):
        raise RuntimeError("Best checkpoint global_step differs from best_metrics")
    verified = verify_manifest_bundle(Path(str(config["paths"]["manifest_dir"])))
    if verified["hashes"] != config["data"]["manifest_sha256"]:
        raise RuntimeError("Formal evaluation manifest hashes differ from frozen config")
    return checkpoint_path, checkpoint, run_dir, config, repository_state


def main() -> None:
    args = build_parser().parse_args()
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("AIO3 DCNv4 formal evaluation requires an available CUDA GPU")

    (
        checkpoint_path,
        checkpoint,
        run_dir,
        config,
        repository_state,
    ) = _resolve_formal_evaluation(args.checkpoint)
    test_dir = run_dir / "test"
    terminal_outputs = (
        test_dir / "metrics.json",
        test_dir / "metrics.csv",
        test_dir / "per_image_metrics.csv",
        test_dir / "predictions",
        test_dir / "gallery",
    )
    conflicts = [str(path) for path in terminal_outputs if path.exists()]
    if conflicts:
        raise RuntimeError(
            "Refusing to overwrite existing formal test outputs: " + ", ".join(conflicts)
        )

    device = torch.device("cuda", 0)
    seed_everything(int(config["seed"]))
    torch.backends.cudnn.benchmark = bool(config["training"]["cudnn_benchmark"])
    torch.use_deterministic_algorithms(bool(config["training"]["deterministic_algorithms"]))
    global_step = int(checkpoint["global_step"])
    total_images = 804
    processed_images = 0
    _write_test_state(
        test_dir,
        status="initializing",
        global_step=global_step,
        processed_images=0,
        total_images=total_images,
    )

    monitor: Optional[WandbMonitor] = None
    try:
        model = build_dcnv4_unet(dict(config)).to(device)
        validate_restoration_checkpoint(
            dict(checkpoint["architecture"]),
            expected=model.checkpoint_metadata(),
        )
        monitor = WandbMonitor(config=config, run_dir=run_dir, resume=True)
        total_parameters, trainable_parameters = model_parameter_counts(model)
        if total_parameters != trainable_parameters or total_parameters != int(
            config["model"]["expected_parameters"]
        ):
            raise RuntimeError("Formal evaluation model parameter count is incompatible")
        model.load_state_dict(checkpoint["model"], strict=True)
        checkpoint_sha256 = file_sha256(checkpoint_path)
        checkpoint_metadata = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "global_step": global_step,
            "best_metrics": dict(checkpoint["best_metrics"]),
            "architecture": dict(checkpoint["architecture"]),
        }
        del checkpoint
        gc.collect()

        manifest_dir = Path(str(config["paths"]["manifest_dir"]))
        test_loader, test_dataset = build_eval_dataloader(
            manifest_dir / "test.jsonl",
            split="test",
            num_workers=args.num_workers,
            pin_memory=True,
            validate_paths=False,
        )
        total_images = len(test_dataset)
        if total_images != 804:
            raise RuntimeError(f"Frozen AIO3-v1 test split requires 804 rows, got {total_images}")
        gallery_selection = select_fixed_test_gallery(test_dataset)
        atomic_write_json(test_dir / "gallery_selection.json", gallery_selection)

        _write_test_state(
            test_dir,
            status="evaluating",
            global_step=global_step,
            processed_images=0,
            total_images=total_images,
        )

        def update_progress(processed: int, total: int) -> None:
            nonlocal processed_images
            processed_images = processed
            _write_test_state(
                test_dir,
                status="evaluating",
                global_step=global_step,
                processed_images=processed,
                total_images=total,
            )
            print(f"formal test: {processed}/{total}", flush=True)

        result = evaluate_test_model(
            model,
            test_loader,
            device=device,
            global_step=global_step,
            prediction_dir=test_dir / "predictions",
            gallery_dir=test_dir / "gallery",
            gallery_sample_ids=gallery_selection["ordered_sample_ids"],
            progress_callback=update_progress,
        )
        metadata = {
            "checkpoint": checkpoint_metadata,
            "manifest_sha256": dict(config["data"]["manifest_sha256"]),
            "training_repository_commit": config["source"]["repository_commit"],
            "evaluation_repository_commit": repository_state["commit"],
            "precision": config["model"]["autocast"],
            "native_resolution": True,
            "batch_size": 1,
            "test_time_augmentation": False,
            "tiled_inference": False,
            "num_workers": args.num_workers,
            "created_at_utc": _utc_now(),
        }
        write_test_result(result, test_dir, metadata=metadata)
        monitor.log_test_evaluation(
            global_step=global_step,
            summary=result.summary,
            per_image=result.per_image,
            visuals=result.visuals,
        )
        monitor.log_evaluation_artifact(test_dir, metadata=metadata)
        _write_test_state(
            test_dir,
            status="completed",
            global_step=global_step,
            processed_images=total_images,
            total_images=total_images,
            summary=result.summary,
        )
        print(
            f"formal test completed: macro_psnr={result.summary['macro/psnr']:.4f} "
            f"macro_ssim={result.summary['macro/ssim']:.6f}",
            flush=True,
        )
    except Exception as error:
        _write_test_state(
            test_dir,
            status="failed",
            global_step=global_step,
            processed_images=processed_images,
            total_images=total_images,
            message=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        if monitor is not None:
            monitor.finish()


if __name__ == "__main__":
    main()

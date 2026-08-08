"""DCNv4 U-Net training core for AIO3-v1."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import torch
import torch.nn.functional as F

from dcnv4_restoration_model import validate_restoration_checkpoint

from .checkpoint import (
    atomic_torch_save,
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    validate_checkpoint_identity,
)
from .data import TASKS, build_eval_dataloader, build_train_dataloader
from .models import build_dcnv4_unet, model_parameter_counts
from .monitoring import WandbMonitor
from .runtime import (
    append_jsonl,
    atomic_write_json,
    git_state,
    load_fixed_visual_sample_ids,
    verify_manifest_bundle,
)
from .schedule import WarmupCosineScheduler
from .validation import evaluate_model, write_validation_result


class TrainingMetricWindow:
    def __init__(self) -> None:
        self.scalar_sums: MutableMapping[str, float] = defaultdict(float)
        self.steps = 0
        self.samples = Counter()
        self.elapsed_seconds = 0.0
        self.residual_min = float("inf")
        self.residual_max = float("-inf")

    def update(
        self,
        *,
        per_sample_l1: torch.Tensor,
        tasks,
        residual: torch.Tensor,
        prediction: torch.Tensor,
        learning_rate: float,
        grad_norm: float,
        step_time_seconds: float,
    ) -> None:
        batch_size = len(tasks)
        if per_sample_l1.shape != (batch_size,):
            raise ValueError("per_sample_l1 must contain one value per training sample")
        self.steps += 1
        self.elapsed_seconds += step_time_seconds
        self.scalar_sums["loss"] += float(per_sample_l1.mean().item())
        self.scalar_sums["learning_rate"] += float(learning_rate)
        self.scalar_sums["grad_norm"] += float(grad_norm)

        residual_float = residual.detach().float()
        self.scalar_sums["residual_mean"] += float(residual_float.mean().item())
        self.scalar_sums["residual_std"] += float(residual_float.std(unbiased=False).item())
        self.residual_min = min(self.residual_min, float(residual_float.min().item()))
        self.residual_max = max(self.residual_max, float(residual_float.max().item()))
        self.scalar_sums["residual_negative_fraction"] += float(
            (residual_float < 0.0).float().mean().item()
        )
        self.scalar_sums["residual_positive_fraction"] += float(
            (residual_float > 0.0).float().mean().item()
        )
        self.scalar_sums["residual_near_zero_fraction"] += float(
            (residual_float.abs() <= 1e-6).float().mean().item()
        )
        restored_float = prediction.detach().float()
        self.scalar_sums["prediction_below_zero_fraction"] += float(
            (restored_float < 0.0).float().mean().item()
        )
        self.scalar_sums["prediction_above_one_fraction"] += float(
            (restored_float > 1.0).float().mean().item()
        )

        for task in TASKS:
            indices = [index for index, value in enumerate(tasks) if value == task]
            if len(indices) != 4:
                raise RuntimeError(
                    f"Balanced training batch requires 4 {task} samples, got {len(indices)}"
                )
            self.samples[task] += len(indices)
            self.scalar_sums[f"{task}_l1"] += float(per_sample_l1[indices].mean().item())
            task_residual = residual_float[indices]
            self.scalar_sums[f"{task}_residual_negative_fraction"] += float(
                (task_residual < 0.0).float().mean().item()
            )

    def finish(self, *, global_step: int, device: torch.device) -> Dict[str, object]:
        if self.steps == 0:
            raise RuntimeError("Cannot finish an empty training metric window")
        mean = {key: value / self.steps for key, value in self.scalar_sums.items()}
        total_images = sum(self.samples.values())
        metrics: Dict[str, object] = {
            "global_step": int(global_step),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "train/loss": mean["loss"],
            "train/denoise_l1": mean["denoise_l1"],
            "train/derain_l1": mean["derain_l1"],
            "train/dehaze_l1": mean["dehaze_l1"],
            "train/learning_rate": mean["learning_rate"],
            "train/grad_norm": mean["grad_norm"],
            "train/step_time_seconds": self.elapsed_seconds / self.steps,
            "train/images_per_second": total_images / self.elapsed_seconds,
            "train/samples_denoise": self.samples["denoise"],
            "train/samples_derain": self.samples["derain"],
            "train/samples_dehaze": self.samples["dehaze"],
            "diagnostics/residual_mean": mean["residual_mean"],
            "diagnostics/residual_std": mean["residual_std"],
            "diagnostics/residual_min": self.residual_min,
            "diagnostics/residual_max": self.residual_max,
            "diagnostics/residual_negative_fraction": mean[
                "residual_negative_fraction"
            ],
            "diagnostics/residual_positive_fraction": mean[
                "residual_positive_fraction"
            ],
            "diagnostics/residual_near_zero_fraction": mean[
                "residual_near_zero_fraction"
            ],
            "diagnostics/prediction_below_zero_fraction": mean[
                "prediction_below_zero_fraction"
            ],
            "diagnostics/prediction_above_one_fraction": mean[
                "prediction_above_one_fraction"
            ],
        }
        for task in TASKS:
            metrics[f"diagnostics/{task}/residual_negative_fraction"] = mean[
                f"{task}_residual_negative_fraction"
            ]
        if device.type == "cuda":
            metrics["system/gpu_memory_allocated_gib"] = (
                torch.cuda.memory_allocated(device) / 1024**3
            )
            metrics["system/gpu_memory_reserved_gib"] = (
                torch.cuda.memory_reserved(device) / 1024**3
            )
        return metrics


def _update_run_state(
    run_dir: Path,
    *,
    status: str,
    global_step: int,
    best_metrics: Mapping[str, object],
    message: Optional[str] = None,
) -> None:
    state = {
        "status": status,
        "global_step": int(global_step),
        "best_macro_psnr": best_metrics.get("macro_psnr"),
        "best_macro_ssim": best_metrics.get("macro_ssim"),
        "best_global_step": best_metrics.get("global_step"),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if message is not None:
        state["message"] = message
    atomic_write_json(run_dir / "run_state.json", state)


def _save_training_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    global_step: int,
    best_metrics: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    checkpoint = build_checkpoint(
        model=model,
        architecture=model.checkpoint_metadata(),
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        best_metrics=best_metrics,
        config=config,
    )
    atomic_torch_save(checkpoint, path)


def run_training(
    *,
    repository_root: Path,
    run_dir: Path,
    config: Mapping[str, object],
    resume_checkpoint: Optional[Path] = None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("AIO3 DCNv4 training requires a CUDA GPU")
    device = torch.device("cuda", 0)
    torch.backends.cudnn.benchmark = bool(config["training"]["cudnn_benchmark"])
    torch.use_deterministic_algorithms(
        bool(config["training"]["deterministic_algorithms"])
    )
    repository_state = git_state(repository_root)
    if repository_state["dirty"]:
        raise RuntimeError("Refusing to train from a dirty worktree")
    if repository_state["commit"] != config["source"]["repository_commit"]:
        raise RuntimeError(
            "Run config Git commit differs from current checkout: "
            f"{config['source']['repository_commit']} != {repository_state['commit']}"
        )
    manifest_dir = Path(config["paths"]["manifest_dir"])
    verified = verify_manifest_bundle(manifest_dir)
    if verified["hashes"] != config["data"]["manifest_sha256"]:
        raise RuntimeError("Run manifest hashes differ from frozen config")

    model = build_dcnv4_unet(dict(config)).to(device)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Model parameters must remain FP32; use autocast for BF16")
    total_parameters, trainable_parameters = model_parameter_counts(model)
    if total_parameters != trainable_parameters:
        raise RuntimeError("Frozen baseline expects every model parameter to be trainable")
    if total_parameters != int(config["model"]["expected_parameters"]):
        raise RuntimeError(
            "Model parameter count differs from frozen config: "
            f"{total_parameters} != {config['model']['expected_parameters']}"
        )

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler_config = config["scheduler"]
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=float(optimizer_config["learning_rate"]),
        min_lr=float(scheduler_config["min_learning_rate"]),
        warmup_steps=int(scheduler_config["warmup_steps"]),
        max_steps=int(config["training"]["max_steps"]),
    )

    global_step = 0
    best_metrics: Dict[str, object] = {
        "macro_psnr": None,
        "macro_ssim": None,
        "global_step": None,
    }
    if resume_checkpoint is not None:
        checkpoint = load_checkpoint(resume_checkpoint)
        validate_checkpoint_identity(
            checkpoint,
            config=config,
            current_repository_commit=str(repository_state["commit"]),
        )
        validate_restoration_checkpoint(dict(checkpoint["architecture"]))
        restore_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        global_step = int(checkpoint["global_step"])
        best_metrics = dict(checkpoint["best_metrics"])

    max_steps = int(config["training"]["max_steps"])
    if global_step >= max_steps:
        raise RuntimeError(f"Run is already complete at global_step={global_step}")
    remaining_steps = max_steps - global_step
    workers = int(config["data"]["num_workers"])
    train_loader, _, sampler = build_train_dataloader(
        manifest_dir / "train.jsonl",
        patch_size=int(config["data"]["patch_size"]),
        start_step=global_step,
        num_batches=remaining_steps,
        seed=int(config["seed"]),
        num_workers=workers,
        pin_memory=True,
        validate_paths=False,
    )
    if sampler.batch_size != int(config["data"]["batch_size"]):
        raise RuntimeError("Balanced sampler batch size differs from frozen config")
    validation_loader, _ = build_eval_dataloader(
        manifest_dir / "val.jsonl",
        split="val",
        num_workers=max(0, min(workers, 4)),
        pin_memory=True,
        validate_paths=False,
    )

    scalar_interval = int(config["monitoring"]["scalar_interval_steps"])
    validation_interval = int(config["validation"]["interval_steps"])
    checkpoint_interval = int(config["checkpoint"]["interval_steps"])
    milestone_interval = int(config["checkpoint"]["milestone_interval_steps"])
    media_interval = int(config["monitoring"]["media_interval_steps"])
    grad_clip_norm = float(config["training"]["grad_clip_norm"])
    train_log_path = run_dir / "train_metrics.jsonl"
    checkpoints_dir = run_dir / "checkpoints"
    metric_window = TrainingMetricWindow()
    fixed_visual_sample_ids = load_fixed_visual_sample_ids(manifest_dir)
    _update_run_state(
        run_dir,
        status="running",
        global_step=global_step,
        best_metrics=best_metrics,
    )
    model.train()
    safe_to_checkpoint = True
    monitor: Optional[WandbMonitor] = None

    try:
        monitor = WandbMonitor(
            config=config,
            run_dir=run_dir,
            resume=resume_checkpoint is not None,
        )
        for batch in train_loader:
            safe_to_checkpoint = False
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            degraded = batch["degraded"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            tasks = list(batch["task"])
            counts = Counter(tasks)
            if counts != Counter({"denoise": 4, "derain": 4, "dehaze": 4}):
                raise RuntimeError(f"Unbalanced AIO3 training batch: {dict(counts)}")

            optimizer.zero_grad(set_to_none=True)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                restored_raw = model(degraded)
                per_sample_l1 = (restored_raw - target).abs().flatten(1).mean(dim=1)
                loss = per_sample_l1.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {global_step + 1}")
            loss.backward()
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            safe_to_checkpoint = True
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started

            metric_window.update(
                per_sample_l1=per_sample_l1.detach().float(),
                tasks=tasks,
                residual=(restored_raw.detach().float() - degraded.float()),
                prediction=restored_raw,
                learning_rate=learning_rate,
                grad_norm=float(grad_norm_tensor.item()),
                step_time_seconds=elapsed,
            )
            if global_step % scalar_interval == 0 or global_step == max_steps:
                metrics = metric_window.finish(global_step=global_step, device=device)
                append_jsonl(train_log_path, metrics)
                monitor.log_scalars(metrics)
                print(
                    f"step={global_step}/{max_steps} "
                    f"loss={metrics['train/loss']:.6f} "
                    f"lr={metrics['train/learning_rate']:.8g} "
                    f"images/s={metrics['train/images_per_second']:.2f}",
                    flush=True,
                )
                metric_window = TrainingMetricWindow()

            validation_improved = False
            if global_step % validation_interval == 0 or global_step == max_steps:
                capture_media = global_step % media_interval == 0
                result = evaluate_model(
                    model,
                    validation_loader,
                    device=device,
                    global_step=global_step,
                    visual_sample_ids=(
                        fixed_visual_sample_ids if capture_media else None
                    ),
                    visual_dir=(
                        run_dir / "validation" / "media" / f"step_{global_step:06d}"
                        if capture_media
                        else None
                    ),
                )
                write_validation_result(result, run_dir / "validation")
                validation_log = {"global_step": global_step}
                validation_log.update(
                    {f"val/{key}": value for key, value in result.summary.items()}
                )
                append_jsonl(run_dir / "validation_metrics.jsonl", validation_log)
                monitor.log_validation(
                    global_step=global_step,
                    summary=result.summary,
                    visuals=result.visuals,
                    residual_histogram=result.residual_histogram,
                )
                macro_psnr = float(result.summary["macro/psnr"])
                previous_best = best_metrics.get("macro_psnr")
                if previous_best is None or macro_psnr > float(previous_best):
                    best_metrics = {
                        "macro_psnr": macro_psnr,
                        "macro_ssim": float(result.summary["macro/ssim"]),
                        "global_step": global_step,
                    }
                    validation_improved = True
                    monitor.update_best_summary(best_metrics)
                print(
                    f"validation step={global_step} "
                    f"macro_psnr={result.summary['macro/psnr']:.4f} "
                    f"macro_ssim={result.summary['macro/ssim']:.6f}",
                    flush=True,
                )

            should_checkpoint = (
                global_step % checkpoint_interval == 0 or global_step == max_steps
            )
            if should_checkpoint or validation_improved:
                latest_path = checkpoints_dir / "latest.pth"
                _save_training_checkpoint(
                    path=latest_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    best_metrics=best_metrics,
                    config=config,
                )
                loaded_latest = load_checkpoint(latest_path)
                if int(loaded_latest["global_step"]) != global_step:
                    raise RuntimeError("Checkpoint round-trip changed global_step")
                if validation_improved:
                    _save_training_checkpoint(
                        path=checkpoints_dir / "best_macro_psnr.pth",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        best_metrics=best_metrics,
                        config=config,
                    )
                if global_step % milestone_interval == 0:
                    _save_training_checkpoint(
                        path=checkpoints_dir / f"step_{global_step:06d}.pth",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        best_metrics=best_metrics,
                        config=config,
                    )
                _update_run_state(
                    run_dir,
                    status="running" if global_step < max_steps else "completed",
                    global_step=global_step,
                    best_metrics=best_metrics,
                )
        best_checkpoint_path = checkpoints_dir / "best_macro_psnr.pth"
        if best_checkpoint_path.is_file() and best_metrics.get("macro_psnr") is not None:
            monitor.log_best_checkpoint(best_checkpoint_path, best_metrics)
    except KeyboardInterrupt:
        if safe_to_checkpoint and global_step > 0:
            _save_training_checkpoint(
                path=checkpoints_dir / "latest.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                best_metrics=best_metrics,
                config=config,
            )
        _update_run_state(
            run_dir,
            status="interrupted",
            global_step=global_step,
            best_metrics=best_metrics,
            message="KeyboardInterrupt",
        )
        raise
    except Exception as error:
        _update_run_state(
            run_dir,
            status="failed",
            global_step=global_step,
            best_metrics=best_metrics,
            message=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        if monitor is not None:
            monitor.finish()

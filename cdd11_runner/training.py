"""Shared CDD-11 training loop with exact 11-category gradient accumulation."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import torch

from aio3_runner.schedule import WarmupCosineScheduler

from .checkpoint import (
    atomic_torch_save,
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    validate_checkpoint_identity,
)
from .data import build_eval_dataloader, build_train_dataloader
from .models import (
    architecture_metadata,
    build_model,
    model_parameter_counts,
    validate_architecture_metadata,
)
from .monitoring import CDD11WandbMonitor
from .protocol import DEGRADATIONS
from .recover import validate_completed_run_artifacts
from .runtime import (
    append_jsonl,
    atomic_write_json,
    git_state,
    load_fixed_visual_sample_ids,
    verify_manifest_bundle,
)
from .validation import evaluate_model, write_validation_result


class TrainingMetricWindow:
    def __init__(self) -> None:
        self.steps = 0
        self.elapsed_seconds = 0.0
        self.sums: MutableMapping[str, float] = defaultdict(float)

    def update(
        self,
        *,
        per_sample_l1: Sequence[float],
        degradations: Sequence[str],
        learning_rate: float,
        grad_norm: float,
        elapsed_seconds: float,
    ) -> None:
        if len(per_sample_l1) != len(DEGRADATIONS) or len(degradations) != len(DEGRADATIONS):
            raise ValueError("CDD-11 metric update requires exactly 11 samples")
        counts = Counter(degradations)
        expected = Counter({value: 1 for value in DEGRADATIONS})
        if counts != expected:
            raise RuntimeError(f"Unbalanced CDD-11 training batch: {dict(counts)}")
        self.steps += 1
        self.elapsed_seconds += float(elapsed_seconds)
        self.sums["loss"] += sum(per_sample_l1) / len(per_sample_l1)
        self.sums["learning_rate"] += float(learning_rate)
        self.sums["grad_norm"] += float(grad_norm)
        for degradation, loss in zip(degradations, per_sample_l1):
            self.sums[f"{degradation}_l1"] += float(loss)

    def finish(self, *, global_step: int, device: torch.device) -> Dict[str, object]:
        if self.steps <= 0:
            raise RuntimeError("Cannot finish an empty training metric window")
        value: Dict[str, object] = {
            "global_step": int(global_step),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "train/loss": self.sums["loss"] / self.steps,
            "train/learning_rate": self.sums["learning_rate"] / self.steps,
            "train/grad_norm": self.sums["grad_norm"] / self.steps,
            "train/step_time_seconds": self.elapsed_seconds / self.steps,
            "train/images_per_second": (
                self.steps * len(DEGRADATIONS) / self.elapsed_seconds
            ),
        }
        for degradation in DEGRADATIONS:
            value[f"train/{degradation}_l1"] = (
                self.sums[f"{degradation}_l1"] / self.steps
            )
        if device.type == "cuda":
            value["system/gpu_memory_allocated_gib"] = (
                torch.cuda.memory_allocated(device) / 1024**3
            )
            value["system/gpu_memory_reserved_gib"] = (
                torch.cuda.memory_reserved(device) / 1024**3
            )
        return value


def backward_effective_batch(
    model: torch.nn.Module,
    degraded: torch.Tensor,
    target: torch.Tensor,
    *,
    microbatch_size: int,
    device: Optional[torch.device] = None,
) -> Sequence[float]:
    """Backpropagate the exact mean L1 over one 11-sample effective batch.

    The optimizer must be zeroed by the caller. This function intentionally
    does not step or clip so those operations can happen exactly once outside
    the microbatch loop.
    """

    effective_batch_size = len(DEGRADATIONS)
    if degraded.shape != target.shape or degraded.ndim != 4:
        raise ValueError("degraded and target must be same-shaped NCHW tensors")
    if degraded.shape[0] != effective_batch_size:
        raise ValueError("CDD-11 effective batch must contain exactly 11 samples")
    if not 1 <= microbatch_size <= effective_batch_size:
        raise ValueError("microbatch_size must be between 1 and 11")
    target_device = degraded.device if device is None else torch.device(device)
    values = []
    for start in range(0, effective_batch_size, microbatch_size):
        end = min(start + microbatch_size, effective_batch_size)
        non_blocking = target_device.type == "cuda"
        input_chunk = degraded[start:end].to(target_device, non_blocking=non_blocking)
        target_chunk = target[start:end].to(target_device, non_blocking=non_blocking)
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.bfloat16,
            enabled=target_device.type == "cuda",
        ):
            prediction = model(input_chunk)
            per_sample_l1 = (prediction - target_chunk).abs().flatten(1).mean(dim=1)
            weighted_loss = per_sample_l1.sum() / effective_batch_size
        if not torch.isfinite(weighted_loss):
            raise FloatingPointError("Non-finite CDD-11 effective-batch loss")
        weighted_loss.backward()
        values.extend(per_sample_l1.detach().float().cpu().tolist())
    return values


def resolve_training_target_step(
    *,
    global_step: int,
    max_steps: int,
    scalar_interval: int,
    pause_at_step: Optional[int],
) -> int:
    if pause_at_step is None:
        return max_steps
    pause = int(pause_at_step)
    if not global_step < pause < max_steps:
        raise ValueError(
            "--pause-at-step must be greater than the restored step and smaller "
            f"than max_steps: {global_step} < {pause} < {max_steps}"
        )
    if pause % scalar_interval != 0:
        raise ValueError("--pause-at-step must align with scalar_interval_steps")
    return pause


def _update_run_state(
    run_dir: Path,
    *,
    status: str,
    global_step: int,
    best_metrics: Mapping[str, object],
    message: Optional[str] = None,
) -> None:
    value = {
        "status": status,
        "global_step": int(global_step),
        "best_macro_psnr": best_metrics.get("macro_psnr"),
        "best_macro_ssim": best_metrics.get("macro_ssim"),
        "best_global_step": best_metrics.get("global_step"),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if message is not None:
        value["message"] = message
    atomic_write_json(run_dir / "run_state.json", value)


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    global_step: int,
    best_metrics: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    atomic_torch_save(
        build_checkpoint(
            model=model,
            architecture=architecture_metadata(model),
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            best_metrics=best_metrics,
            config=config,
        ),
        path,
    )


def _current_source_state(repository_root: Path, config: Mapping[str, object]):
    repository_state = git_state(repository_root)
    if repository_state["dirty"]:
        raise RuntimeError("Refusing CDD-11 training from a dirty main worktree")
    if repository_state["commit"] != config["source"]["repository_commit"]:
        raise RuntimeError("Main repository commit differs from frozen run config")
    uformer_commit = None
    if config["model"]["id"] == "uformer":
        uformer_root = Path(str(config["paths"]["uformer_root"]))
        uformer_state = git_state(uformer_root)
        if uformer_state["dirty"]:
            raise RuntimeError("Refusing Uformer training from a dirty Uformer worktree")
        uformer_commit = str(uformer_state["commit"])
        if uformer_commit != config["source"]["uformer_commit"]:
            raise RuntimeError("Uformer repository commit differs from frozen run config")
    return repository_state, uformer_commit


def run_training(
    *,
    repository_root: Path,
    run_dir: Path,
    config: Mapping[str, object],
    resume_checkpoint: Optional[Path] = None,
    pause_at_step: Optional[int] = None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CDD-11 training requires a CUDA GPU")
    device = torch.device("cuda", 0)
    torch.backends.cudnn.benchmark = bool(config["training"]["cudnn_benchmark"])
    torch.use_deterministic_algorithms(
        bool(config["training"]["deterministic_algorithms"])
    )
    repository_state, uformer_commit = _current_source_state(repository_root, config)
    manifest_dir = Path(str(config["paths"]["manifest_dir"]))
    verified = verify_manifest_bundle(manifest_dir)
    if verified["hashes"] != config["data"]["manifest_sha256"]:
        raise RuntimeError("Manifest hashes differ from the frozen run config")

    uformer_root = config["paths"].get("uformer_root")
    model = build_model(
        config["model"],
        uformer_root=Path(str(uformer_root)) if uformer_root else None,
    ).to(device)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Model parameters must remain FP32; BF16 is autocast-only")
    total_parameters, trainable_parameters = model_parameter_counts(model)
    expected_parameters = int(config["model"]["expected_parameters"])
    if total_parameters != trainable_parameters or total_parameters != expected_parameters:
        raise RuntimeError(
            "Model parameter count mismatch: "
            f"total={total_parameters}, trainable={trainable_parameters}, "
            f"expected={expected_parameters}"
        )

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=float(optimizer_config["learning_rate"]),
        min_lr=float(config["scheduler"]["min_learning_rate"]),
        warmup_steps=int(config["scheduler"]["warmup_steps"]),
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
            repository_commit=str(repository_state["commit"]),
            uformer_commit=uformer_commit,
        )
        validate_architecture_metadata(model, checkpoint["architecture"])
        restore_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        global_step = int(checkpoint["global_step"])
        best_metrics = dict(checkpoint["best_metrics"])

    max_steps = int(config["training"]["max_steps"])
    if resume_checkpoint is not None and global_step == max_steps:
        evidence = validate_completed_run_artifacts(run_dir, config)
        _update_run_state(
            run_dir,
            status="completed",
            global_step=global_step,
            best_metrics=best_metrics,
            message="Idempotent completion recovery from a fully evaluated checkpoint",
        )
        print(
            "CDD-11 completed checkpoint already has full validation artifacts: "
            f"model={evidence['model']} step={global_step}",
            flush=True,
        )
        return
    scalar_interval = int(config["monitoring"]["scalar_interval_steps"])
    target_step = resolve_training_target_step(
        global_step=global_step,
        max_steps=max_steps,
        scalar_interval=scalar_interval,
        pause_at_step=pause_at_step,
    )
    workers = int(config["data"]["num_workers"])
    train_loader, _, sampler = build_train_dataloader(
        manifest_dir / "train.jsonl",
        patch_size=int(config["data"]["patch_size"]),
        start_step=global_step,
        num_batches=target_step - global_step,
        seed=int(config["seed"]),
        num_workers=workers,
        pin_memory=True,
    )
    effective_batch_size = int(config["data"]["effective_batch_size"])
    if sampler.batch_size != effective_batch_size or effective_batch_size != 11:
        raise RuntimeError("CDD-11 effective batch must contain exactly 11 samples")
    validation_loader, _ = build_eval_dataloader(
        manifest_dir / "val.jsonl",
        split="val",
        num_workers=max(0, min(workers, 4)),
        pin_memory=True,
    )
    fixed_visual_ids = load_fixed_visual_sample_ids(manifest_dir)
    validation_interval = int(config["validation"]["interval_steps"])
    checkpoint_interval = int(config["checkpoint"]["interval_steps"])
    milestone_interval = int(config["checkpoint"]["milestone_interval_steps"])
    microbatch_size = int(config["data"]["microbatch_size"])
    grad_clip_norm = float(config["training"]["grad_clip_norm"])
    checkpoints_dir = run_dir / "checkpoints"
    latest_path = checkpoints_dir / "latest.pth"
    metric_window = TrainingMetricWindow()
    _update_run_state(
        run_dir,
        status="running",
        global_step=global_step,
        best_metrics=best_metrics,
    )
    model.train()
    safe_to_checkpoint = True
    monitor: Optional[CDD11WandbMonitor] = None
    try:
        monitor = CDD11WandbMonitor(
            config=config,
            run_dir=run_dir,
            resume=resume_checkpoint is not None,
        )
        for batch in train_loader:
            safe_to_checkpoint = False
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            degradations = [str(value) for value in batch["degradation"]]
            counts = Counter(degradations)
            if counts != Counter({value: 1 for value in DEGRADATIONS}):
                raise RuntimeError(f"Unbalanced CDD-11 batch: {dict(counts)}")
            optimizer.zero_grad(set_to_none=True)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            per_sample_values = backward_effective_batch(
                model,
                batch["degraded"],
                batch["target"],
                microbatch_size=microbatch_size,
                device=device,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            safe_to_checkpoint = True
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            metric_window.update(
                per_sample_l1=per_sample_values,
                degradations=degradations,
                learning_rate=learning_rate,
                grad_norm=float(grad_norm.item()),
                elapsed_seconds=elapsed,
            )
            if global_step % scalar_interval == 0 or global_step == target_step:
                metrics = metric_window.finish(global_step=global_step, device=device)
                append_jsonl(run_dir / "train_metrics.jsonl", metrics)
                monitor.log_scalars(metrics)
                print(
                    f"step={global_step}/{max_steps} "
                    f"loss={metrics['train/loss']:.6f} "
                    f"lr={metrics['train/learning_rate']:.8g}",
                    flush=True,
                )
                metric_window = TrainingMetricWindow()

            is_complete = global_step == max_steps
            should_validate = global_step % validation_interval == 0 or is_complete
            should_checkpoint = global_step % checkpoint_interval == 0 or is_complete
            if should_checkpoint or should_validate:
                _save_checkpoint(
                    latest_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    best_metrics=best_metrics,
                    config=config,
                )
            improved = False
            if should_validate:
                _update_run_state(
                    run_dir,
                    status="validating",
                    global_step=global_step,
                    best_metrics=best_metrics,
                )
                result = evaluate_model(
                    model,
                    validation_loader,
                    device=device,
                    global_step=global_step,
                    inference_mode=str(config["validation"]["inference_mode"]),
                    tile_size=int(config["validation"]["tile_size"]),
                    tile_overlap=int(config["validation"]["tile_overlap"]),
                    visual_sample_ids=fixed_visual_ids,
                    visual_dir=run_dir / "validation" / "media" / f"step_{global_step:06d}",
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
                )
                macro_psnr = float(result.summary["macro/psnr"])
                if best_metrics["macro_psnr"] is None or macro_psnr > float(
                    best_metrics["macro_psnr"]
                ):
                    best_metrics = {
                        "macro_psnr": macro_psnr,
                        "macro_ssim": float(result.summary["macro/ssim"]),
                        "global_step": global_step,
                    }
                    improved = True
                    monitor.update_best_summary(best_metrics)
                print(
                    f"validation step={global_step} macro_psnr={macro_psnr:.4f}",
                    flush=True,
                )
            if improved:
                _save_checkpoint(
                    latest_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    best_metrics=best_metrics,
                    config=config,
                )
                _save_checkpoint(
                    checkpoints_dir / "best_macro_psnr.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    best_metrics=best_metrics,
                    config=config,
                )
            if global_step % milestone_interval == 0:
                _save_checkpoint(
                    checkpoints_dir / f"step_{global_step:06d}.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    best_metrics=best_metrics,
                    config=config,
                )
            if (
                global_step % scalar_interval == 0
                or should_checkpoint
                or should_validate
            ):
                _update_run_state(
                    run_dir,
                    status="completed" if is_complete else "running",
                    global_step=global_step,
                    best_metrics=best_metrics,
                )

        if pause_at_step is not None:
            _save_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                best_metrics=best_metrics,
                config=config,
            )
            if int(load_checkpoint(latest_path)["global_step"]) != global_step:
                raise RuntimeError("Safe-pause checkpoint round-trip changed global_step")
            _update_run_state(
                run_dir,
                status="paused",
                global_step=global_step,
                best_metrics=best_metrics,
                message="Requested safe pause at optimizer-step boundary",
            )
    except KeyboardInterrupt:
        if safe_to_checkpoint and global_step > 0:
            _save_checkpoint(
                latest_path,
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

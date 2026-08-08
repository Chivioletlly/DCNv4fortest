"""Atomic checkpointing and exact RNG restoration for AIO3-v1."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
from torch.optim import Optimizer

from .protocol import AIO3_PROTOCOL_VERSION
from .schedule import WarmupCosineScheduler


REQUIRED_CHECKPOINT_KEYS = {
    "protocol",
    "global_step",
    "model",
    "architecture",
    "optimizer",
    "scheduler",
    "best_metrics",
    "rng_state",
    "config",
    "manifest_sha256",
    "repository_commit",
    "wandb_run_id",
}


def capture_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from checkpoint RNG state: "
                f"{torch.cuda.device_count()} != {len(cuda_state)}"
            )
        torch.cuda.set_rng_state_all(cuda_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "Checkpoint contains NumPy RNG state but NumPy is unavailable"
            ) from error
        np.random.set_state(numpy_state)


@contextmanager
def preserve_rng_state():
    """Prevent monitoring or serialization helpers from perturbing training RNG."""

    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def build_checkpoint(
    *,
    model: torch.nn.Module,
    architecture: Mapping[str, object],
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    global_step: int,
    best_metrics: Mapping[str, object],
    config: Mapping[str, object],
) -> Dict[str, object]:
    if scheduler.completed_steps != global_step:
        raise RuntimeError(
            "Scheduler/global-step mismatch before checkpoint: "
            f"{scheduler.completed_steps} != {global_step}"
        )
    return {
        "checkpoint_version": 1,
        "protocol": AIO3_PROTOCOL_VERSION,
        "global_step": int(global_step),
        "model": model.state_dict(),
        "architecture": dict(architecture),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metrics": dict(best_metrics),
        "rng_state": capture_rng_state(),
        "config": dict(config),
        "manifest_sha256": dict(config["data"]["manifest_sha256"]),
        "repository_commit": config["source"]["repository_commit"],
        "wandb_run_id": config["monitoring"]["wandb_run_id"],
        "run_name": config["run_name"],
        "run_dir": config["paths"]["run_dir"],
    }


def atomic_torch_save(checkpoint: Mapping[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(checkpoint), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(checkpoint))
    if missing:
        raise RuntimeError(f"Checkpoint is missing required keys: {missing}")
    if checkpoint["protocol"] != AIO3_PROTOCOL_VERSION:
        raise RuntimeError(f"Incompatible checkpoint protocol: {checkpoint['protocol']!r}")
    return checkpoint


def validate_checkpoint_identity(
    checkpoint: Mapping[str, object],
    *,
    config: Mapping[str, object],
    current_repository_commit: str,
) -> None:
    checks = {
        "config": (checkpoint["config"], config),
        "manifest_sha256": (
            checkpoint["manifest_sha256"],
            config["data"]["manifest_sha256"],
        ),
        "repository_commit": (
            checkpoint["repository_commit"],
            current_repository_commit,
        ),
        "wandb_run_id": (
            checkpoint["wandb_run_id"],
            config["monitoring"]["wandb_run_id"],
        ),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        details = "; ".join(
            f"{name}: checkpoint={checks[name][0]!r}, current={checks[name][1]!r}"
            for name in mismatches
        )
        raise RuntimeError("Refusing incompatible checkpoint resume: " + details)


def restore_training_state(
    checkpoint: Mapping[str, object],
    *,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
) -> None:
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    global_step = int(checkpoint["global_step"])
    if scheduler.completed_steps != global_step:
        raise RuntimeError(
            "Restored scheduler/global-step mismatch: "
            f"{scheduler.completed_steps} != {global_step}"
        )
    restore_rng_state(checkpoint["rng_state"])

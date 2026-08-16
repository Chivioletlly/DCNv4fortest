"""Atomic optimizer-boundary checkpoints for CDD-11-v1."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch

from aio3_runner.checkpoint import capture_rng_state, restore_rng_state
from aio3_runner.schedule import WarmupCosineScheduler

from .protocol import CDD11_PROTOCOL_VERSION


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
    "uformer_commit",
    "run_dir",
}


def build_checkpoint(
    *,
    model: torch.nn.Module,
    architecture: Mapping[str, object],
    optimizer: torch.optim.Optimizer,
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
        "protocol": CDD11_PROTOCOL_VERSION,
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
        "uformer_commit": config["source"]["uformer_commit"],
        "run_dir": config["paths"]["run_dir"],
        "run_name": config["run_name"],
    }


def atomic_torch_save(checkpoint: Mapping[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(checkpoint), temporary)
        with temporary.open("rb+") as stream:
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
        value = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, Mapping):
        raise RuntimeError("Checkpoint must contain a mapping")
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(value))
    if missing:
        raise RuntimeError(f"Checkpoint is missing required keys: {missing}")
    if value["protocol"] != CDD11_PROTOCOL_VERSION:
        raise RuntimeError(f"Incompatible checkpoint protocol: {value['protocol']!r}")
    return dict(value)


def validate_checkpoint_identity(
    checkpoint: Mapping[str, object],
    *,
    config: Mapping[str, object],
    repository_commit: str,
    uformer_commit: Optional[str],
) -> None:
    checks = {
        "config": (checkpoint["config"], config),
        "manifest_sha256": (
            checkpoint["manifest_sha256"],
            config["data"]["manifest_sha256"],
        ),
        "repository_commit": (checkpoint["repository_commit"], repository_commit),
        "uformer_commit": (checkpoint["uformer_commit"], uformer_commit),
    }
    mismatches = [name for name, pair in checks.items() if pair[0] != pair[1]]
    if mismatches:
        details = "; ".join(
            f"{name}: checkpoint={checks[name][0]!r}, current={checks[name][1]!r}"
            for name in mismatches
        )
        raise RuntimeError("Refusing incompatible CDD-11 checkpoint: " + details)


def restore_training_state(
    checkpoint: Mapping[str, object],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
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

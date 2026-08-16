"""Run preparation and immutable configuration for CDD-11-v1."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import yaml

from aio3_runner.runtime import (
    WANDB_VERSION,
    append_jsonl,
    atomic_write_json,
    file_sha256,
    git_state,
    seed_everything,
)

from .models import frozen_model_config, normalize_model_id
from .protocol import CDD11_PROTOCOL_VERSION, DEGRADATIONS


MANIFEST_FILES = ("train.jsonl", "val.jsonl", "test.jsonl")
AUXILIARY_DATA_FILES = ("data_audit.json", "visual_samples.json")
RUN_PROFILES: Mapping[str, Mapping[str, int]] = {
    "smoke": {
        "max_steps": 100,
        "scalar_interval_steps": 10,
        "validation_interval_steps": 100,
        "checkpoint_interval_steps": 100,
    },
    "pilot": {
        "max_steps": 5_000,
        "scalar_interval_steps": 50,
        "validation_interval_steps": 5_000,
        "checkpoint_interval_steps": 5_000,
    },
    "formal": {
        "max_steps": 200_000,
        "scalar_interval_steps": 50,
        "validation_interval_steps": 5_000,
        "checkpoint_interval_steps": 5_000,
    },
}


def verify_manifest_bundle(manifest_dir: Path) -> Dict[str, object]:
    manifest_dir = Path(manifest_dir).expanduser().resolve()
    audit_path = manifest_dir / "data_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing CDD-11 audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("protocol") != CDD11_PROTOCOL_VERSION or audit.get("status") != "pass":
        raise RuntimeError(
            "CDD-11 data audit did not pass the frozen protocol: "
            f"protocol={audit.get('protocol')!r}, status={audit.get('status')!r}"
        )
    hashes: Dict[str, str] = {}
    audit_manifests = audit.get("manifests", {})
    for filename in MANIFEST_FILES:
        path = manifest_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen manifest: {path}")
        actual = file_sha256(path)
        expected = audit_manifests.get(filename, {}).get("sha256")
        if actual != expected:
            raise RuntimeError(
                f"Manifest SHA256 mismatch for {filename}: {actual} != {expected}"
            )
        hashes[filename] = actual
    visual_path = manifest_dir / "visual_samples.json"
    if not visual_path.is_file():
        raise FileNotFoundError(f"Missing fixed visual sample selection: {visual_path}")
    actual_visual = file_sha256(visual_path)
    expected_visual = audit.get("visual_samples", {}).get("sha256")
    if actual_visual != expected_visual:
        raise RuntimeError(
            f"visual_samples.json SHA256 mismatch: {actual_visual} != {expected_visual}"
        )
    hashes["visual_samples.json"] = actual_visual
    hashes["data_audit.json"] = file_sha256(audit_path)
    return {"directory": str(manifest_dir), "hashes": hashes, "audit": audit}


def load_fixed_visual_sample_ids(manifest_dir: Path) -> Tuple[str, ...]:
    value = json.loads(
        (Path(manifest_dir) / "visual_samples.json").read_text(encoding="utf-8")
    )
    if value.get("protocol") != CDD11_PROTOCOL_VERSION:
        raise RuntimeError("Fixed visual sample protocol mismatch")
    sample_ids = tuple(str(item) for item in value.get("ordered_sample_ids", ()))
    if len(sample_ids) != 2 * len(DEGRADATIONS) or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("CDD-11 visuals require exactly 22 unique sample IDs")
    return sample_ids


def _write_yaml_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(dict(value), stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_run_config(path: Path) -> Dict[str, object]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Run config must be a mapping: {path}")
    if value.get("protocol") != CDD11_PROTOCOL_VERSION:
        raise RuntimeError(f"Run config protocol mismatch: {value.get('protocol')!r}")
    return value


def build_run_config(
    *,
    model_id: str,
    run_kind: str,
    seed: int,
    run_name: str,
    run_dir: Path,
    manifest_hashes: Mapping[str, str],
    repository_state: Mapping[str, object],
    uformer_state: Optional[Mapping[str, object]],
    uformer_root: Optional[Path],
    num_workers: int,
    microbatch_size: int,
    inference_mode: str,
    output_root: Optional[Path] = None,
    wandb_mode: str = "disabled",
    wandb_entity: Optional[str] = None,
    wandb_project: str = "cdd11-restoration",
    wandb_run_id: Optional[str] = None,
) -> Dict[str, object]:
    model_id = normalize_model_id(model_id)
    if run_kind not in RUN_PROFILES:
        raise ValueError(f"run_kind must be one of {tuple(RUN_PROFILES)}")
    if not 1 <= microbatch_size <= len(DEGRADATIONS):
        raise ValueError("microbatch_size must be between 1 and 11")
    if inference_mode not in {"native", "tiled"}:
        raise ValueError("inference_mode must be 'native' or 'tiled'")
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be 'online', 'offline', or 'disabled'")
    if not wandb_project:
        raise ValueError("wandb_project must be non-empty")
    if wandb_mode != "disabled" and not wandb_run_id:
        raise ValueError("Enabled W&B monitoring requires a frozen run ID")
    if model_id == "uformer" and (uformer_state is None or uformer_root is None):
        raise ValueError("Uformer config requires repository state and root")
    profile = RUN_PROFILES[run_kind]
    return {
        "protocol": CDD11_PROTOCOL_VERSION,
        "run_kind": run_kind,
        "run_name": run_name,
        "seed": int(seed),
        "model": frozen_model_config(model_id),
        "data": {
            "patch_size": 256,
            "effective_batch_size": len(DEGRADATIONS),
            "samples_per_degradation": {
                degradation: 1 for degradation in DEGRADATIONS
            },
            "microbatch_size": int(microbatch_size),
            "num_workers": int(num_workers),
            "manifest_sha256": dict(manifest_hashes),
        },
        "training": {
            "max_steps": int(profile["max_steps"]),
            "precision": "bf16",
            "loss": "l1",
            "grad_clip_norm": 1.0,
            "cudnn_benchmark": True,
            "deterministic_algorithms": False,
            "pretrained": False,
            "ema": False,
            "labels_as_model_input": False,
            "clamp_before_loss": False,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 2.0e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 1.0e-4,
        },
        "scheduler": {
            "name": "linear_warmup_cosine",
            "warmup_steps": min(2_000, int(profile["max_steps"])),
            "min_learning_rate": 1.0e-6,
            "unit": "optimizer_step",
        },
        "validation": {
            "interval_steps": int(profile["validation_interval_steps"]),
            "selection_metric": "macro/psnr",
            "inference_mode": inference_mode,
            "tile_size": 512,
            "tile_overlap": 128,
            "tta": False,
        },
        "checkpoint": {
            "interval_steps": int(profile["checkpoint_interval_steps"]),
            "milestone_interval_steps": 50_000,
        },
        "monitoring": {
            "provider": "wandb",
            "version": WANDB_VERSION,
            "mode": wandb_mode,
            "entity": wandb_entity,
            "project": wandb_project,
            "group": (
                f"{CDD11_PROTOCOL_VERSION}-{run_kind}-{inference_mode}-seed{seed}"
            ),
            "wandb_run_id": wandb_run_id,
            "tags": [
                CDD11_PROTOCOL_VERSION,
                str(run_kind),
                str(model_id),
                str(inference_mode),
                f"seed-{seed}",
            ],
            "local_backend": "jsonl",
            "scalar_interval_steps": int(profile["scalar_interval_steps"]),
        },
        "paths": {
            "output_root": str(
                Path(output_root).expanduser().resolve()
                if output_root is not None
                else Path(run_dir).resolve().parents[1]
            ),
            "run_dir": str(Path(run_dir).resolve()),
            "manifest_dir": str((Path(run_dir) / "manifests").resolve()),
            "uformer_root": str(Path(uformer_root).resolve()) if uformer_root else None,
        },
        "source": {
            "repository_commit": str(repository_state["commit"]),
            "repository_dirty_at_creation": bool(repository_state["dirty"]),
            "uformer_commit": str(uformer_state["commit"]) if uformer_state else None,
            "uformer_dirty_at_creation": bool(uformer_state["dirty"]) if uformer_state else None,
        },
    }


def prepare_new_run(
    *,
    repository_root: Path,
    manifest_dir: Path,
    output_root: Path,
    model_id: str,
    run_kind: str,
    seed: int,
    num_workers: int,
    microbatch_size: int,
    inference_mode: str,
    uformer_root: Optional[Path] = None,
    run_name: Optional[str] = None,
    wandb_mode: str = "online",
    wandb_entity: Optional[str] = None,
    wandb_project: str = "cdd11-restoration",
) -> Tuple[Path, Dict[str, object]]:
    model_id = normalize_model_id(model_id)
    if run_kind not in RUN_PROFILES:
        raise ValueError(f"run_kind must be one of {tuple(RUN_PROFILES)}")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if not 1 <= microbatch_size <= len(DEGRADATIONS):
        raise ValueError("microbatch_size must be between 1 and 11")
    if inference_mode not in {"native", "tiled"}:
        raise ValueError("inference_mode must be 'native' or 'tiled'")
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be 'online', 'offline', or 'disabled'")
    if wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B is enabled but the wandb SDK is not installed"
            ) from error
        if wandb.__version__ != WANDB_VERSION:
            raise RuntimeError(
                f"CDD-11-v1 requires wandb=={WANDB_VERSION}, got {wandb.__version__}"
            )
    repository_state = git_state(repository_root)
    if repository_state["dirty"]:
        raise RuntimeError(
            "Commit or stash the CDD-11 implementation before creating a run"
        )
    resolved_uformer_root = (
        Path(uformer_root).expanduser().resolve() if uformer_root is not None else None
    )
    if model_id == "uformer" and resolved_uformer_root is None:
        raise ValueError("Uformer runs require --uformer-root")
    uformer_state = git_state(resolved_uformer_root) if model_id == "uformer" else None
    if uformer_state is not None and uformer_state["dirty"]:
        raise RuntimeError("Uformer source worktree must be clean before creating a run")
    verified = verify_manifest_bundle(manifest_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if run_name is None:
        run_name = f"{model_id}-{run_kind}-seed{seed}-{timestamp}"
    if any(character in run_name for character in ("/", "\\")):
        raise ValueError("run_name must be a single directory name")
    run_dir = Path(output_root).expanduser().resolve() / model_id / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    destination = run_dir / "manifests"
    destination.mkdir()
    for filename in (*MANIFEST_FILES, *AUXILIARY_DATA_FILES):
        shutil.copy2(Path(manifest_dir) / filename, destination / filename)
    copied = verify_manifest_bundle(destination)
    if copied["hashes"] != verified["hashes"]:
        raise RuntimeError("Copied CDD-11 manifest bundle changed content")
    config = build_run_config(
        model_id=model_id,
        run_kind=run_kind,
        seed=seed,
        run_name=run_name,
        run_dir=run_dir,
        manifest_hashes=copied["hashes"],
        repository_state=repository_state,
        uformer_state=uformer_state,
        uformer_root=resolved_uformer_root,
        num_workers=num_workers,
        microbatch_size=microbatch_size,
        inference_mode=inference_mode,
        output_root=output_root,
        wandb_mode=wandb_mode,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_run_id=(uuid.uuid4().hex[:16] if wandb_mode != "disabled" else None),
    )
    _write_yaml_atomic(run_dir / "config.yaml", config)
    atomic_write_json(
        run_dir / "run_state.json",
        {
            "status": "created",
            "global_step": 0,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    for directory in ("checkpoints", "logs", "validation", "wandb"):
        (run_dir / directory).mkdir()
    if config["monitoring"]["wandb_run_id"] is not None:
        (run_dir / "wandb_run_id.txt").write_text(
            str(config["monitoring"]["wandb_run_id"]) + "\n",
            encoding="utf-8",
        )
    return run_dir, config


__all__ = [
    "RUN_PROFILES",
    "WANDB_VERSION",
    "append_jsonl",
    "atomic_write_json",
    "build_run_config",
    "file_sha256",
    "git_state",
    "load_fixed_visual_sample_ids",
    "load_run_config",
    "prepare_new_run",
    "seed_everything",
    "verify_manifest_bundle",
]

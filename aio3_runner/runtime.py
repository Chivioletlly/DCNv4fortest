"""Run preparation and frozen configuration for AIO3-v1 training."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import PIL
import torch
import torchvision

from .ablation import (
    ABLATION_REGISTRY,
    ABLATION_REGISTRY_VERSION,
    MODEL_VARIANT_CHOICES,
    get_ablation_variant,
    normalize_variant_id,
    registry_snapshot,
)
from .protocol import AIO3_PROTOCOL_VERSION


MODEL_NAME = "dcnv4_unet"
MODEL_VARIANTS = MODEL_VARIANT_CHOICES
MODEL_DIRECTORIES = {
    variant_id: variant.output_directory for variant_id, variant in ABLATION_REGISTRY.items()
}
MODEL_DIRECTORIES.update(
    {
        "baseline": ABLATION_REGISTRY["APG-000"].output_directory,
        "degradation-aware": ABLATION_REGISTRY["APG-111"].output_directory,
    }
)
BASELINE_EXPECTED_PARAMETERS = 29924411
# Kept as a frozen constant once the architecture is defined. The model tests
# independently recompute it so accidental topology changes fail loudly.
DEGRADATION_AWARE_EXPECTED_PARAMETERS = 34852539
WANDB_VERSION = "0.25.1"
MANIFEST_FILES = ("train.jsonl", "val.jsonl", "test.jsonl")
AUXILIARY_DATA_FILES = ("data_audit.json", "visual_samples.json")
RUN_PROFILES: Mapping[str, Mapping[str, int]] = {
    "smoke": {
        "max_steps": 100,
        "warmup_steps": 100,
        "scalar_interval_steps": 10,
        "validation_interval_steps": 100,
        "checkpoint_interval_steps": 100,
        "media_interval_steps": 100,
    },
    "pilot": {
        "max_steps": 5000,
        "warmup_steps": 2000,
        "scalar_interval_steps": 50,
        "validation_interval_steps": 5000,
        "checkpoint_interval_steps": 5000,
        "media_interval_steps": 5000,
    },
    "formal": {
        "max_steps": 200000,
        "warmup_steps": 2000,
        "scalar_interval_steps": 50,
        "validation_interval_steps": 5000,
        "checkpoint_interval_steps": 5000,
        "media_interval_steps": 10000,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def git_state(repository_root: Path) -> Dict[str, object]:
    repository_root = Path(repository_root)

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Could not resolve Git state in {repository_root}") from error
    return {"commit": commit, "dirty": bool(status), "status_porcelain": status}


def verify_manifest_bundle(manifest_dir: Path) -> Dict[str, object]:
    manifest_dir = Path(manifest_dir).expanduser().resolve()
    audit_path = manifest_dir / "data_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing AIO3 audit: {audit_path}")
    with audit_path.open("r", encoding="utf-8") as stream:
        audit = json.load(stream)
    if audit.get("protocol") != AIO3_PROTOCOL_VERSION:
        raise RuntimeError(
            f"Audit protocol is {audit.get('protocol')!r}, expected {AIO3_PROTOCOL_VERSION!r}"
        )
    if audit.get("status") not in {"pass", "pass_with_warnings"}:
        raise RuntimeError(f"AIO3 data audit did not pass: {audit.get('status')!r}")

    hashes: Dict[str, str] = {}
    audit_manifests = audit.get("manifests", {})
    for filename in MANIFEST_FILES:
        path = manifest_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen manifest: {path}")
        actual = file_sha256(path)
        expected = audit_manifests.get(filename, {}).get("sha256")
        if actual != expected:
            raise RuntimeError(f"Manifest SHA256 mismatch for {filename}: {actual} != {expected}")
        hashes[filename] = actual

    visual_path = manifest_dir / "visual_samples.json"
    if not visual_path.is_file():
        raise FileNotFoundError(f"Missing fixed visual sample selection: {visual_path}")
    visual_sha = file_sha256(visual_path)
    expected_visual_sha = audit.get("visual_samples", {}).get("sha256")
    if visual_sha != expected_visual_sha:
        raise RuntimeError(
            "visual_samples.json SHA256 mismatch: " f"{visual_sha} != {expected_visual_sha}"
        )
    hashes["visual_samples.json"] = visual_sha
    hashes["data_audit.json"] = file_sha256(audit_path)
    return {"directory": str(manifest_dir), "hashes": hashes, "audit": audit}


def load_fixed_visual_sample_ids(manifest_dir: Path) -> Tuple[str, ...]:
    path = Path(manifest_dir) / "visual_samples.json"
    with path.open("r", encoding="utf-8") as stream:
        selection = json.load(stream)
    if selection.get("protocol") != AIO3_PROTOCOL_VERSION:
        raise RuntimeError(f"Invalid visual sample protocol: {selection.get('protocol')!r}")
    samples = selection.get("samples", {})
    expected_counts = {
        "denoise_sigma15": 2,
        "denoise_sigma25": 2,
        "denoise_sigma50": 2,
        "derain": 4,
        "dehaze": 4,
    }
    unexpected = sorted(set(samples) - set(expected_counts))
    if unexpected:
        raise RuntimeError(f"Unexpected fixed visual sample groups: {unexpected}")
    ordered = []
    for group, expected_count in expected_counts.items():
        values = samples.get(group)
        if not isinstance(values, list) or len(values) != expected_count:
            raise RuntimeError(
                f"Fixed visual group {group} requires {expected_count} IDs, got {values!r}"
            )
        ordered.extend(str(value) for value in values)
    if len(set(ordered)) != len(ordered):
        raise RuntimeError("Fixed visual sample IDs must be unique")
    return tuple(ordered)


def _copy_manifest_bundle(source_dir: Path, destination_dir: Path) -> Dict[str, str]:
    destination_dir.mkdir(parents=True, exist_ok=False)
    for filename in (*MANIFEST_FILES, *AUXILIARY_DATA_FILES):
        shutil.copy2(source_dir / filename, destination_dir / filename)
    verified = verify_manifest_bundle(destination_dir)
    return dict(verified["hashes"])


def build_run_config(
    *,
    run_kind: str,
    seed: int,
    run_name: str,
    run_dir: Path,
    manifest_hashes: Mapping[str, str],
    repository_state: Mapping[str, object],
    num_workers: int,
    wandb_run_id: str,
    wandb_mode: str,
    wandb_entity: Optional[str],
    model_variant: str = "baseline",
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    if run_kind not in RUN_PROFILES:
        raise ValueError(f"run_kind must be one of {tuple(RUN_PROFILES)}, got {run_kind!r}")
    variant = get_ablation_variant(model_variant)
    profile = RUN_PROFILES[run_kind]
    model_config = variant.model_config()
    model_config.update(
        {
            "in_channels": 3,
            "base_channels": 64,
            "bottleneck_type": "conv",
            "use_dcnv4": True,
            "output_mode": "signed_residual",
            "normalization": "groupnorm",
            "initialization": "exact_identity_zero_initialized_signed_residual_head",
        }
    )
    if variant.degradation_context != "none":
        model_config["context_scales"] = 3
    return {
        "protocol": AIO3_PROTOCOL_VERSION,
        "run_kind": run_kind,
        "run_name": run_name,
        "seed": int(seed),
        "model": model_config,
        "ablation_registry_version": ABLATION_REGISTRY_VERSION,
        "ablation_registry": registry_snapshot(),
        "data": {
            "patch_size": 128,
            "batch_size": 12,
            "samples_per_task": {"denoise": 4, "derain": 4, "dehaze": 4},
            "num_workers": int(num_workers),
            "pin_memory": True,
            "multiprocessing_context": "spawn",
            "manifest_sha256": dict(manifest_hashes),
        },
        "training": {
            "max_steps": profile["max_steps"],
            "precision": "bf16",
            "loss": "l1",
            "grad_clip_norm": 1.0,
            "cudnn_benchmark": True,
            "deterministic_algorithms": False,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 2.0e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 1.0e-4,
        },
        "scheduler": {
            "name": "warmup_cosine",
            "warmup_steps": profile["warmup_steps"],
            "min_learning_rate": 1.0e-6,
        },
        "validation": {
            "interval_steps": profile["validation_interval_steps"],
            "batch_size": 1,
        },
        "checkpoint": {
            "interval_steps": profile["checkpoint_interval_steps"],
            "milestone_interval_steps": 50000,
        },
        "monitoring": {
            "provider": "wandb",
            "version": WANDB_VERSION,
            "mode": wandb_mode,
            "entity": wandb_entity,
            "project": "aio3-restoration",
            "group": AIO3_PROTOCOL_VERSION,
            "scalar_interval_steps": profile["scalar_interval_steps"],
            "media_interval_steps": profile["media_interval_steps"],
            "wandb_run_id": wandb_run_id,
            "upload_manifest_artifact": True,
            "upload_best_checkpoint_artifact": True,
            "tags": [*variant.wandb_tags, run_kind],
        },
        "source": {
            "repository_commit": repository_state["commit"],
            "repository_dirty": repository_state["dirty"],
            "protocol_document_sha256": repository_state["protocol_document_sha256"],
            "ablation_plan_sha256": repository_state.get("ablation_plan_sha256"),
        },
        "paths": {
            "output_root": str(
                Path(output_root).expanduser().resolve()
                if output_root is not None
                else run_dir.parents[1]
            ),
            "run_dir": str(run_dir),
            "manifest_dir": str(run_dir / "manifests"),
            "protocol_document": str(run_dir / "AIO3_TRAINING_EVALUATION_PROTOCOL.md"),
            "ablation_plan": str(run_dir / "AIO3_DCNV4_ABLATION_PLAN.md"),
        },
    }


def environment_info(repository_state: Mapping[str, object]) -> Dict[str, object]:
    cuda_device: Optional[Dict[str, object]] = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        cuda_device = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "device_count": torch.cuda.device_count(),
        }
    try:
        import DCNv4

        dcnv4_location = str(Path(DCNv4.__file__).resolve())
    except Exception as error:  # The environment report must preserve the import failure.
        dcnv4_location = f"IMPORT_ERROR: {type(error).__name__}: {error}"
    try:
        import wandb

        wandb_version: Optional[str] = wandb.__version__
    except ImportError:
        wandb_version = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "command": list(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_device,
        "dcnv4_location": dcnv4_location,
        "wandb": wandb_version,
        "repository": dict(repository_state),
    }


def prepare_new_run(
    *,
    repository_root: Path,
    manifest_dir: Path,
    output_root: Path,
    run_kind: str,
    seed: int,
    num_workers: int,
    wandb_mode: str,
    wandb_entity: Optional[str],
    run_name: Optional[str] = None,
    model_variant: str = "baseline",
) -> Tuple[Path, Dict[str, object]]:
    repository_state = git_state(repository_root)
    canonical_variant_id = normalize_variant_id(model_variant)
    variant = get_ablation_variant(canonical_variant_id)
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"Unsupported W&B mode: {wandb_mode!r}")
    if run_kind == "formal" and wandb_mode == "disabled":
        raise ValueError("Formal AIO3-v1 training requires W&B online or offline mode")
    if wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B is enabled but the wandb SDK is not installed in this environment"
            ) from error
        if wandb.__version__ != WANDB_VERSION:
            raise RuntimeError(f"AIO3-v1 requires wandb=={WANDB_VERSION}, got {wandb.__version__}")
    if repository_state["dirty"]:
        raise RuntimeError(
            "Refusing to start a frozen AIO3 run from a dirty worktree:\n"
            + str(repository_state["status_porcelain"])
        )
    protocol_document = Path(repository_root) / "docs" / "AIO3_TRAINING_EVALUATION_PROTOCOL.md"
    if not protocol_document.is_file():
        raise FileNotFoundError(f"Missing AIO3 protocol document: {protocol_document}")
    repository_state["protocol_document_sha256"] = file_sha256(protocol_document)
    ablation_plan = Path(repository_root) / "docs" / "AIO3_DCNV4_ABLATION_PLAN.md"
    if not ablation_plan.is_file():
        raise FileNotFoundError(f"Missing DCNv4 ablation plan: {ablation_plan}")
    repository_state["ablation_plan_sha256"] = file_sha256(ablation_plan)
    verified_source = verify_manifest_bundle(manifest_dir)
    if run_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        model_slug = canonical_variant_id.lower()
        run_name = f"dcnv4-ablation-{model_slug}-{run_kind}-seed{seed}-{timestamp}"
    run_dir = Path(output_root).expanduser().resolve() / variant.output_directory / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    for directory in ("checkpoints", "logs", "validation", "test", "wandb"):
        (run_dir / directory).mkdir()
    shutil.copy2(
        protocol_document,
        run_dir / "AIO3_TRAINING_EVALUATION_PROTOCOL.md",
    )
    shutil.copy2(
        ablation_plan,
        run_dir / "AIO3_DCNV4_ABLATION_PLAN.md",
    )

    copied_hashes = _copy_manifest_bundle(
        Path(str(verified_source["directory"])),
        run_dir / "manifests",
    )
    wandb_run_id = uuid.uuid4().hex[:16]
    config = build_run_config(
        run_kind=run_kind,
        seed=seed,
        run_name=run_name,
        run_dir=run_dir,
        manifest_hashes=copied_hashes,
        repository_state=repository_state,
        num_workers=num_workers,
        wandb_run_id=wandb_run_id,
        wandb_mode=wandb_mode,
        wandb_entity=wandb_entity,
        model_variant=canonical_variant_id,
        output_root=output_root,
    )
    atomic_write_json(run_dir / "config.yaml", config)
    atomic_write_json(run_dir / "environment.json", environment_info(repository_state))
    (run_dir / "wandb_run_id.txt").write_text(wandb_run_id + "\n", encoding="utf-8")
    atomic_write_json(
        run_dir / "run_state.json",
        {
            "status": "initialized",
            "global_step": 0,
            "best_macro_psnr": None,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return run_dir, config


def load_run_config(path: Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("protocol") != AIO3_PROTOCOL_VERSION:
        raise RuntimeError(f"Incompatible run protocol: {config.get('protocol')!r}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

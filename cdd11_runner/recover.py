"""Audited completion recovery for fully evaluated interrupted smoke runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

from .checkpoint import load_checkpoint, validate_checkpoint_identity
from .runtime import (
    atomic_write_json,
    file_sha256,
    git_state,
    load_fixed_visual_sample_ids,
    load_run_config,
    verify_manifest_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_FILENAME = "completion_recovery.json"


def _read_json(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> Sequence[Mapping[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Expected JSON objects in {path}")
    return rows


def _close(left: object, right: object) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def validate_completion_values(
    *,
    state: Mapping[str, object],
    latest: Mapping[str, object],
    best: Mapping[str, object],
    metrics: Mapping[str, object],
    expected_validation_pairs: Set[Tuple[str, str]],
    actual_validation_pairs: Set[Tuple[str, str]],
    expected_visual_ids: Set[str],
    actual_visual_ids: Set[str],
    validation_csv_rows: int,
    validation_visual_rows: int,
    train_metric_step: int,
    validation_metric_step: int,
    max_steps: int,
) -> Dict[str, object]:
    if int(state.get("global_step", -1)) != max_steps:
        raise RuntimeError("Interrupted run state is not at max_steps")
    if int(latest["global_step"]) != max_steps:
        raise RuntimeError("latest.pth is not at max_steps")
    scheduler = latest.get("scheduler", {})
    if int(scheduler.get("completed_steps", -1)) != max_steps:
        raise RuntimeError("latest.pth scheduler is not at max_steps")
    best_metrics = dict(latest["best_metrics"])
    best_step = best_metrics.get("global_step")
    if best_step is None or int(best_step) != int(best["global_step"]):
        raise RuntimeError("Best checkpoint step differs from latest best_metrics")
    if dict(best["best_metrics"]) != best_metrics:
        raise RuntimeError("Best checkpoint metrics differ from latest.pth")
    if int(metrics.get("global_step", -1)) != max_steps:
        raise RuntimeError("Final validation metrics are not at max_steps")
    if int(metrics.get("per_image_count", -1)) != len(expected_validation_pairs):
        raise RuntimeError("Final validation per-image count differs from val manifest")
    if validation_csv_rows != len(expected_validation_pairs):
        raise RuntimeError("Final validation CSV row count differs from val manifest")
    if actual_validation_pairs != expected_validation_pairs:
        raise RuntimeError("Final validation CSV does not match the val manifest")
    if actual_visual_ids != expected_visual_ids:
        raise RuntimeError("Final validation visuals differ from the frozen selection")
    if validation_visual_rows != len(expected_visual_ids):
        raise RuntimeError("Final validation visual count differs from the frozen selection")
    if train_metric_step != max_steps or validation_metric_step != max_steps:
        raise RuntimeError("Training or validation metric logs do not end at max_steps")
    if int(best_step) == max_steps:
        summary = metrics.get("summary", {})
        if not _close(summary.get("macro/psnr"), best_metrics.get("macro_psnr")):
            raise RuntimeError("Final macro PSNR differs from best checkpoint")
        if not _close(summary.get("macro/ssim"), best_metrics.get("macro_ssim")):
            raise RuntimeError("Final macro SSIM differs from best checkpoint")
    return best_metrics


def validate_completed_run_artifacts(
    run_dir: Path,
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    run_dir = Path(run_dir).expanduser().resolve()
    config_value = dict(config) if config is not None else load_run_config(run_dir / "config.yaml")
    if Path(str(config_value["paths"]["run_dir"])).resolve() != run_dir:
        raise RuntimeError("Run directory differs from config.yaml")
    state_path = run_dir / "run_state.json"
    state = _read_json(state_path)
    if state.get("status") not in {"interrupted", "running", "completed"}:
        raise RuntimeError(f"Run state is not recoverable: {state}")
    max_steps = int(config_value["training"]["max_steps"])
    manifest_dir = Path(str(config_value["paths"]["manifest_dir"])).resolve()
    verified = verify_manifest_bundle(manifest_dir)
    if verified["hashes"] != config_value["data"]["manifest_sha256"]:
        raise RuntimeError("Run manifest hashes differ from config.yaml")

    latest_path = run_dir / "checkpoints" / "latest.pth"
    best_path = run_dir / "checkpoints" / "best_macro_psnr.pth"
    latest = load_checkpoint(latest_path)
    best = load_checkpoint(best_path)
    for checkpoint in (latest, best):
        validate_checkpoint_identity(
            checkpoint,
            config=config_value,
            repository_commit=str(config_value["source"]["repository_commit"]),
            uformer_commit=config_value["source"].get("uformer_commit"),
        )
        if Path(str(checkpoint["run_dir"])).resolve() != run_dir:
            raise RuntimeError("Checkpoint run_dir differs from the recovered run")
    if dict(latest["architecture"]) != dict(best["architecture"]):
        raise RuntimeError("latest and best checkpoint architectures differ")

    stem = f"step_{max_steps:06d}"
    validation_dir = run_dir / "validation"
    metrics_path = validation_dir / f"metrics_{stem}.json"
    csv_path = validation_dir / f"per_image_metrics_{stem}.csv"
    visuals_path = validation_dir / f"visuals_{stem}.json"
    metrics = _read_json(metrics_path)
    visuals = _read_json(visuals_path)

    val_rows = _read_jsonl(manifest_dir / "val.jsonl")
    expected_pairs = {
        (str(row["id"]), str(row["degradation"])) for row in val_rows
    }
    if len(expected_pairs) != len(val_rows):
        raise RuntimeError("Validation manifest contains duplicate sample/degradation pairs")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    actual_pairs = {
        (str(row["sample_id"]), str(row["degradation"])) for row in csv_rows
    }
    expected_visual_ids = set(load_fixed_visual_sample_ids(manifest_dir))
    visual_samples = list(visuals.get("samples", ()))
    actual_visual_ids = {str(sample["sample_id"]) for sample in visual_samples}
    train_metrics = _read_jsonl(run_dir / "train_metrics.jsonl")
    validation_metrics = _read_jsonl(run_dir / "validation_metrics.jsonl")
    if not train_metrics or not validation_metrics:
        raise RuntimeError("Training or validation metric log is empty")

    best_metrics = validate_completion_values(
        state=state,
        latest=latest,
        best=best,
        metrics=metrics,
        expected_validation_pairs=expected_pairs,
        actual_validation_pairs=actual_pairs,
        expected_visual_ids=expected_visual_ids,
        actual_visual_ids=actual_visual_ids,
        validation_csv_rows=len(csv_rows),
        validation_visual_rows=len(visual_samples),
        train_metric_step=int(train_metrics[-1]["global_step"]),
        validation_metric_step=int(validation_metrics[-1]["global_step"]),
        max_steps=max_steps,
    )
    artifact_paths = {
        "config": run_dir / "config.yaml",
        "state_before": state_path,
        "latest_checkpoint": latest_path,
        "best_checkpoint": best_path,
        "validation_metrics": metrics_path,
        "validation_per_image": csv_path,
        "validation_visuals": visuals_path,
        "train_metric_log": run_dir / "train_metrics.jsonl",
        "validation_metric_log": run_dir / "validation_metrics.jsonl",
    }
    return {
        "run_dir": str(run_dir),
        "model": config_value["model"]["id"],
        "max_steps": max_steps,
        "state_before": state,
        "best_metrics": best_metrics,
        "validation_images": len(expected_pairs),
        "visual_samples": len(expected_visual_ids),
        "source": dict(config_value["source"]),
        "artifact_sha256": {
            name: file_sha256(path) for name, path in artifact_paths.items()
        },
    }


def _is_ancestor(repository_root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def recover_completed_smoke(repository_root: Path, run_dir: Path) -> Dict[str, object]:
    repository_root = Path(repository_root).resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    config = load_run_config(run_dir / "config.yaml")
    if config.get("run_kind") != "smoke":
        raise RuntimeError("Out-of-commit completion recovery is restricted to smoke runs")
    current = git_state(repository_root)
    if current["dirty"]:
        raise RuntimeError("Recovery tool requires a clean main worktree")
    source_commit = str(config["source"]["repository_commit"])
    if not _is_ancestor(repository_root, source_commit, str(current["commit"])):
        raise RuntimeError("Run source commit is not an ancestor of the recovery tool")
    if config["model"]["id"] == "uformer":
        uformer_root = Path(str(config["paths"]["uformer_root"]))
        uformer_state = git_state(uformer_root)
        if uformer_state["dirty"] or uformer_state["commit"] != config["source"]["uformer_commit"]:
            raise RuntimeError("Uformer source changed since the smoke run")

    evidence = validate_completed_run_artifacts(run_dir, config)
    before = dict(evidence["state_before"])
    if before.get("status") != "interrupted" or before.get("message") != "KeyboardInterrupt":
        raise RuntimeError("Only a KeyboardInterrupt state may be recovered")
    recovery_path = run_dir / RECOVERY_FILENAME
    if recovery_path.exists():
        raise RuntimeError(f"Refusing to overwrite recovery audit: {recovery_path}")

    now = datetime.now(timezone.utc).isoformat()
    best_metrics = evidence["best_metrics"]
    completed_state = {
        "status": "completed",
        "global_step": int(evidence["max_steps"]),
        "best_macro_psnr": best_metrics.get("macro_psnr"),
        "best_macro_ssim": best_metrics.get("macro_ssim"),
        "best_global_step": best_metrics.get("global_step"),
        "updated_at_utc": now,
        "message": "Audited completion recovery after post-validation KeyboardInterrupt",
        "completion_recovery": str(recovery_path),
    }
    audit = {
        "protocol": config["protocol"],
        "status": "validated_pending_state_update",
        "recovery_tool_commit": current["commit"],
        "recovered_at_utc": now,
        "evidence": evidence,
        "state_after": completed_state,
    }
    atomic_write_json(recovery_path, audit)
    atomic_write_json(run_dir / "run_state.json", completed_state)
    audit["status"] = "completed"
    atomic_write_json(recovery_path, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and recover a fully evaluated interrupted CDD-11 smoke run"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = recover_completed_smoke(REPOSITORY_ROOT, args.run_dir)
    print(
        "CDD-11 completion recovery passed: "
        f"model={audit['evidence']['model']} "
        f"step={audit['evidence']['max_steps']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

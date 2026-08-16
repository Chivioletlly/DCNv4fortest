import copy
import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from aio3_runner.schedule import WarmupCosineScheduler
from cdd11_runner.checkpoint import atomic_torch_save, build_checkpoint
from cdd11_runner.protocol import CDD11_PROTOCOL_VERSION
from cdd11_runner.recover import (
    validate_completed_run_artifacts,
    validate_completion_values,
)
from cdd11_runner.runtime import (
    atomic_write_json,
    build_run_config,
    file_sha256,
    verify_manifest_bundle,
)


@contextmanager
def _temporary_directory():
    parent = REPOSITORY_ROOT / ".tmp_cdd11_recovery_tests"
    parent.mkdir(exist_ok=True)
    path = parent / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)
        try:
            parent.rmdir()
        except OSError:
            pass


def _evidence():
    best_metrics = {
        "macro_psnr": 15.8,
        "macro_ssim": 0.51,
        "global_step": 100,
    }
    return {
        "state": {"status": "interrupted", "global_step": 100},
        "latest": {
            "global_step": 100,
            "scheduler": {"completed_steps": 100},
            "best_metrics": best_metrics,
        },
        "best": {"global_step": 100, "best_metrics": best_metrics},
        "metrics": {
            "global_step": 100,
            "per_image_count": 2,
            "summary": {"macro/psnr": 15.8, "macro/ssim": 0.51},
        },
        "expected_validation_pairs": {("a", "low"), ("b", "haze")},
        "actual_validation_pairs": {("a", "low"), ("b", "haze")},
        "expected_visual_ids": {"a", "b"},
        "actual_visual_ids": {"a", "b"},
        "validation_csv_rows": 2,
        "validation_visual_rows": 2,
        "train_metric_step": 100,
        "validation_metric_step": 100,
        "max_steps": 100,
    }


def test_completion_recovery_accepts_complete_interrupted_evidence():
    evidence = _evidence()
    result = validate_completion_values(**evidence)
    assert result == evidence["latest"]["best_metrics"]


def test_completion_recovery_rejects_incomplete_validation_csv():
    evidence = copy.deepcopy(_evidence())
    evidence["validation_csv_rows"] = 1
    try:
        validate_completion_values(**evidence)
    except RuntimeError as error:
        assert "CSV row count" in str(error)
    else:
        raise AssertionError("Expected incomplete validation artifacts to be rejected")


def test_completed_run_artifact_validator_checks_frozen_files_end_to_end():
    with _temporary_directory() as run_dir:
        manifest_dir = run_dir / "manifests"
        manifest_dir.mkdir()
        visual_ids = [f"val:{index:02d}" for index in range(22)]
        validation_rows = [
            {"id": sample_id, "degradation": f"class-{index:02d}"}
            for index, sample_id in enumerate(visual_ids)
        ]
        manifest_values = {
            "train.jsonl": [],
            "val.jsonl": validation_rows,
            "test.jsonl": [],
        }
        for filename, rows in manifest_values.items():
            (manifest_dir / filename).write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
        visual_path = manifest_dir / "visual_samples.json"
        visual_path.write_text(
            json.dumps(
                {
                    "protocol": CDD11_PROTOCOL_VERSION,
                    "ordered_sample_ids": visual_ids,
                }
            ),
            encoding="utf-8",
        )
        audit_path = manifest_dir / "data_audit.json"
        atomic_write_json(
            audit_path,
            {
                "protocol": CDD11_PROTOCOL_VERSION,
                "status": "pass",
                "manifests": {
                    filename: {"sha256": file_sha256(manifest_dir / filename)}
                    for filename in manifest_values
                },
                "visual_samples": {"sha256": file_sha256(visual_path)},
            },
        )
        verified = verify_manifest_bundle(manifest_dir)
        config = build_run_config(
            model_id="unet",
            run_kind="smoke",
            seed=3407,
            run_name="recovery-test",
            run_dir=run_dir,
            manifest_hashes=verified["hashes"],
            repository_state={"commit": "source-commit", "dirty": False},
            uformer_state=None,
            uformer_root=None,
            num_workers=0,
            microbatch_size=1,
            inference_mode="native",
        )
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir()
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = WarmupCosineScheduler(
            optimizer,
            base_lr=2e-4,
            min_lr=1e-6,
            warmup_steps=100,
            max_steps=100,
        )
        for _ in range(100):
            scheduler.step()
        best_metrics = {
            "macro_psnr": 15.8,
            "macro_ssim": 0.51,
            "global_step": 100,
        }
        checkpoint = build_checkpoint(
            model=model,
            architecture={"name": "test"},
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=100,
            best_metrics=best_metrics,
            config=config,
        )
        atomic_torch_save(checkpoint, checkpoints / "latest.pth")
        atomic_torch_save(checkpoint, checkpoints / "best_macro_psnr.pth")
        atomic_write_json(
            run_dir / "run_state.json",
            {
                "status": "interrupted",
                "global_step": 100,
                "message": "KeyboardInterrupt",
            },
        )
        validation_dir = run_dir / "validation"
        validation_dir.mkdir()
        atomic_write_json(
            validation_dir / "metrics_step_000100.json",
            {
                "global_step": 100,
                "per_image_count": len(validation_rows),
                "summary": {"macro/psnr": 15.8, "macro/ssim": 0.51},
            },
        )
        with (validation_dir / "per_image_metrics_step_000100.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=("sample_id", "degradation"))
            writer.writeheader()
            for row in validation_rows:
                writer.writerow(
                    {
                        "sample_id": row["id"],
                        "degradation": row["degradation"],
                    }
                )
        atomic_write_json(
            validation_dir / "visuals_step_000100.json",
            {"global_step": 100, "samples": [{"sample_id": value} for value in visual_ids]},
        )
        (run_dir / "train_metrics.jsonl").write_text(
            json.dumps({"global_step": 100}) + "\n", encoding="utf-8"
        )
        (run_dir / "validation_metrics.jsonl").write_text(
            json.dumps({"global_step": 100}) + "\n", encoding="utf-8"
        )

        evidence = validate_completed_run_artifacts(run_dir)

        assert evidence["max_steps"] == 100
        assert evidence["validation_images"] == 22
        assert evidence["visual_samples"] == 22

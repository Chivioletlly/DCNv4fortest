import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from aio3_runner.evaluation import (
    evaluate_test_model,
    select_fixed_test_gallery,
    write_test_result,
)
from aio3_runner.evaluate import validate_formal_evaluation_config


@contextmanager
def _workspace_temporary_directory():
    parent = REPOSITORY_ROOT / ".tmp_aio3_evaluation_tests"
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


class _AddConstant(torch.nn.Module):
    def forward(self, image):
        return image + 0.1


class _ListLoader(list):
    def __init__(self, batches):
        super().__init__(batches)
        self.dataset = list(range(len(batches)))


def _test_records():
    records = []
    for sigma in (15, 25, 50):
        for index in range(2):
            records.append(
                SimpleNamespace(
                    sample_id=f"bsd68:sigma{sigma}:{index}",
                    task="denoise",
                    metadata={"sigma": sigma},
                )
            )
    for task in ("derain", "dehaze"):
        for index in range(4):
            records.append(
                SimpleNamespace(
                    sample_id=f"{task}:{index}",
                    task=task,
                    metadata={},
                )
            )
    return records


def _batches(records):
    batches = []
    for record in records:
        sigma = int(record.metadata.get("sigma", -1))
        batches.append(
            {
                "degraded": torch.full((1, 3, 16, 17), 0.1),
                "target": torch.zeros((1, 3, 16, 17)),
                "sample_id": [record.sample_id],
                "task": [record.task],
                "sigma": torch.tensor([sigma]),
            }
        )
    return batches


def test_fixed_test_gallery_is_deterministic_and_outcome_independent():
    dataset = SimpleNamespace(split="test", records=_test_records())
    first = select_fixed_test_gallery(dataset)
    second = select_fixed_test_gallery(dataset)
    assert first == second
    assert len(first["ordered_sample_ids"]) == 14
    assert len(set(first["ordered_sample_ids"])) == 14


def test_formal_test_rejects_smoke_and_pilot_configs():
    validate_formal_evaluation_config({"run_kind": "formal"})
    for run_kind in ("smoke", "pilot"):
        try:
            validate_formal_evaluation_config({"run_kind": run_kind})
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Formal test accepted run_kind={run_kind}")


def test_test_evaluation_saves_predictions_metrics_and_fixed_gallery():
    records = _test_records()
    dataset = SimpleNamespace(split="test", records=records)
    selection = select_fixed_test_gallery(dataset)
    loader = _ListLoader(_batches(records))
    model = _AddConstant()
    model.train()
    progress = []

    with _workspace_temporary_directory() as root:
        test_dir = root / "test"
        test_dir.mkdir()
        result = evaluate_test_model(
            model,
            loader,
            device=torch.device("cpu"),
            global_step=5000,
            prediction_dir=test_dir / "predictions",
            gallery_dir=test_dir / "gallery",
            gallery_sample_ids=selection["ordered_sample_ids"],
            progress_callback=lambda processed, total: progress.append((processed, total)),
        )
        write_test_result(
            result,
            test_dir,
            metadata={"checkpoint": {"sha256": "unit"}},
        )

        assert model.training
        assert len(result.per_image) == 14
        assert result.summary["images"] == 14.0
        assert result.summary["bsd68/sigma15/images"] == 2.0
        assert result.summary["rain100l/images"] == 4.0
        assert result.summary["sots_outdoor/images"] == 4.0
        assert progress[-1] == (14, 14)
        assert len(list((test_dir / "predictions").rglob("*.png"))) == 14
        assert len(list((test_dir / "gallery").rglob("*.png"))) == 70
        assert {row["dataset"] for row in result.per_image} == {
            "BSD68",
            "Rain100L",
            "SOTS-outdoor",
        }

        try:
            evaluate_test_model(
                model,
                loader,
                device=torch.device("cpu"),
                global_step=5000,
                prediction_dir=test_dir / "predictions",
                gallery_dir=test_dir / "gallery-second-attempt",
                gallery_sample_ids=selection["ordered_sample_ids"],
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("Formal test overwrote an existing prediction directory")

        metrics = json.loads((test_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["per_image_count"] == 14
        with (test_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 7
        with (test_dir / "per_image_metrics.csv").open("r", encoding="utf-8", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 14


if __name__ == "__main__":
    tests = [
        test_fixed_test_gallery_is_deterministic_and_outcome_independent,
        test_formal_test_rejects_smoke_and_pilot_configs,
        test_test_evaluation_saves_predictions_metrics_and_fixed_gallery,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")

import json
import math
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from aio3_runner.training import (
    TrainingMetricWindow,
    _update_run_state,
    resolve_training_target_step,
)
from aio3_runner.validation import evaluate_model


@contextmanager
def _workspace_temporary_directory():
    parent = REPOSITORY_ROOT / ".tmp_aio3_training_core_tests"
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


def test_training_window_tracks_balanced_tasks_and_raw_prediction_range():
    tasks = ["denoise"] * 4 + ["derain"] * 4 + ["dehaze"] * 4
    per_sample_l1 = torch.arange(1, 13, dtype=torch.float32) / 100.0
    residual = torch.zeros((12, 3, 2, 2), dtype=torch.float32)
    residual[:6].fill_(-0.25)
    residual[6:].fill_(0.5)
    prediction = torch.full_like(residual, 0.5)
    prediction[0].fill_(-0.1)
    prediction[1].fill_(1.1)

    window = TrainingMetricWindow()
    window.update(
        per_sample_l1=per_sample_l1,
        tasks=tasks,
        residual=residual,
        prediction=prediction,
        learning_rate=1e-4,
        grad_norm=2.0,
        step_time_seconds=0.5,
    )
    metrics = window.finish(global_step=1, device=torch.device("cpu"))

    assert metrics["train/samples_denoise"] == 4
    assert metrics["train/samples_derain"] == 4
    assert metrics["train/samples_dehaze"] == 4
    assert metrics["diagnostics/residual_negative_fraction"] == 0.5
    assert metrics["diagnostics/residual_positive_fraction"] == 0.5
    assert math.isclose(
        metrics["diagnostics/prediction_below_zero_fraction"],
        1.0 / 12.0,
        abs_tol=1e-7,
    )
    assert math.isclose(
        metrics["diagnostics/prediction_above_one_fraction"],
        1.0 / 12.0,
        abs_tol=1e-7,
    )


class _AddConstant(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, image):
        return image + self.value


def _validation_batch(sample_id: str, task: str, sigma: int):
    return {
        "degraded": torch.full((1, 3, 16, 17), 0.1),
        "target": torch.zeros((1, 3, 16, 17)),
        "sample_id": [sample_id],
        "task": [task],
        "sigma": torch.tensor([sigma]),
    }


def test_native_validation_runs_all_required_metric_groups_and_restores_mode():
    model = _AddConstant(0.1)
    model.train()
    dataloader = [
        _validation_batch("noise-15", "denoise", 15),
        _validation_batch("noise-25", "denoise", 25),
        _validation_batch("noise-50", "denoise", 50),
        _validation_batch("rain", "derain", -1),
        _validation_batch("haze", "dehaze", -1),
    ]
    visual_ids = [batch["sample_id"][0] for batch in dataloader]
    with _workspace_temporary_directory() as root:
        result = evaluate_model(
            model,
            dataloader,
            device=torch.device("cpu"),
            global_step=100,
            visual_sample_ids=visual_ids,
            visual_dir=root / "media",
        )

        assert len(result.visuals) == 5
        assert result.residual_histogram
        for visual in result.visuals:
            for key in (
                "input_path",
                "prediction_path",
                "target_path",
                "absolute_error_path",
                "signed_residual_path",
            ):
                assert Path(visual[key]).is_file()

    assert model.training
    assert result.global_step == 100
    assert len(result.per_image) == 5
    assert result.summary["images"] == 5.0
    assert result.summary["denoise/sigma15/images"] == 1.0
    assert result.summary["denoise/sigma25/images"] == 1.0
    assert result.summary["denoise/sigma50/images"] == 1.0
    assert result.summary["derain/images"] == 1.0
    assert result.summary["dehaze/images"] == 1.0
    assert math.isfinite(result.summary["macro/psnr"])
    assert math.isfinite(result.summary["macro/ssim"])


def test_safe_pause_target_is_strict_and_scalar_aligned():
    assert (
        resolve_training_target_step(
            global_step=0,
            max_steps=100,
            scalar_interval=10,
            pause_at_step=None,
        )
        == 100
    )
    assert (
        resolve_training_target_step(
            global_step=0,
            max_steps=100,
            scalar_interval=10,
            pause_at_step=50,
        )
        == 50
    )

    invalid_cases = (
        {"global_step": 50, "pause_at_step": 50},
        {"global_step": 50, "pause_at_step": 40},
        {"global_step": 0, "pause_at_step": 100},
        {"global_step": 0, "pause_at_step": 55},
    )
    for case in invalid_cases:
        try:
            resolve_training_target_step(
                global_step=case["global_step"],
                max_steps=100,
                scalar_interval=10,
                pause_at_step=case["pause_at_step"],
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid safe-pause request was accepted: {case}")


def test_run_state_records_validation_boundary_at_current_step():
    with _workspace_temporary_directory() as root:
        _update_run_state(
            root,
            status="validating",
            global_step=5000,
            best_metrics={
                "macro_psnr": 25.0,
                "macro_ssim": 0.8,
                "global_step": 4500,
            },
        )
        state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
        assert state["status"] == "validating"
        assert state["global_step"] == 5000
        assert state["best_global_step"] == 4500


if __name__ == "__main__":
    tests = [
        test_training_window_tracks_balanced_tasks_and_raw_prediction_range,
        test_native_validation_runs_all_required_metric_groups_and_restores_mode,
        test_safe_pause_target_is_strict_and_scalar_aligned,
        test_run_state_records_validation_boundary_at_current_step,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")

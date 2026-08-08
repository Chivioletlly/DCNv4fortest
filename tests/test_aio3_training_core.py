import math
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from aio3_runner.training import TrainingMetricWindow
from aio3_runner.validation import evaluate_model


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
    result = evaluate_model(
        model,
        dataloader,
        device=torch.device("cpu"),
        global_step=100,
    )

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


if __name__ == "__main__":
    tests = [
        test_training_window_tracks_balanced_tasks_and_raw_prediction_range,
        test_native_validation_runs_all_required_metric_groups_and_restores_mode,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")

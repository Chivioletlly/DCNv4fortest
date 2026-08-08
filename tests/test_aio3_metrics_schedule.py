import math
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from aio3_runner.metrics import (
    AIO3MetricAccumulator,
    rgb_psnr_per_image,
    rgb_ssim_per_image,
)
from aio3_runner.schedule import WarmupCosineScheduler


def test_metrics_identity_and_prediction_only_clamp():
    generator = torch.Generator(device="cpu").manual_seed(123)
    target = torch.rand((2, 3, 17, 19), generator=generator)

    psnr = rgb_psnr_per_image(target, target)
    ssim = rgb_ssim_per_image(target, target)
    assert torch.isinf(psnr).all()
    torch.testing.assert_close(ssim, torch.ones_like(ssim), rtol=0.0, atol=1e-5)

    white = torch.ones((1, 3, 16, 16))
    above_white = torch.full_like(white, 1.25)
    assert torch.isinf(rgb_psnr_per_image(above_white, white)).all()


def test_psnr_is_per_image_rgb_mean():
    target = torch.zeros((2, 3, 16, 16))
    prediction = target.clone()
    prediction[0].fill_(0.5)
    prediction[1].fill_(0.25)
    values = rgb_psnr_per_image(prediction, target)
    torch.testing.assert_close(
        values,
        torch.tensor([6.0206003, 12.0412006]),
        rtol=0.0,
        atol=1e-5,
    )


def test_task_summary_uses_equal_sigma_and_task_macro_weights():
    accumulator = AIO3MetricAccumulator()
    for sigma, psnr, ssim in ((15, 10.0, 0.1), (25, 20.0, 0.2), (50, 30.0, 0.3)):
        accumulator.add_values(
            sample_id=f"noise-{sigma}",
            task="denoise",
            sigma=sigma,
            psnr=psnr,
            ssim=ssim,
        )
    accumulator.add_values(
        sample_id="rain-1", task="derain", sigma=None, psnr=40.0, ssim=0.4
    )
    accumulator.add_values(
        sample_id="rain-2", task="derain", sigma=None, psnr=60.0, ssim=0.6
    )
    accumulator.add_values(
        sample_id="haze-1", task="dehaze", sigma=None, psnr=80.0, ssim=0.8
    )

    summary = accumulator.summarize()
    assert summary["denoise/mean/psnr"] == 20.0
    assert summary["derain/psnr"] == 50.0
    assert summary["dehaze/psnr"] == 80.0
    assert summary["macro/psnr"] == 50.0
    assert math.isclose(summary["macro/ssim"], 0.5, rel_tol=0.0, abs_tol=1e-15)
    assert summary["images"] == 6.0


def _optimizer():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    return torch.optim.AdamW([parameter], lr=9.0)


def test_warmup_cosine_has_exact_update_endpoints():
    optimizer = _optimizer()
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=2e-4,
        min_lr=1e-6,
        warmup_steps=2,
        max_steps=6,
    )
    expected = [1e-4, 2e-4]
    expected.extend(
        1e-6 + (2e-4 - 1e-6) * 0.5 * (1.0 + math.cos(math.pi * index / 4.0))
        for index in (1, 2, 3, 4)
    )
    actual = []
    for _ in range(6):
        actual.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    torch.testing.assert_close(
        torch.tensor(actual, dtype=torch.float64),
        torch.tensor(expected, dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )
    assert scheduler.completed_steps == 6
    assert optimizer.param_groups[0]["lr"] == 1e-6


def test_scheduler_resume_restores_exact_next_update_lr():
    uninterrupted_optimizer = _optimizer()
    uninterrupted = WarmupCosineScheduler(
        uninterrupted_optimizer,
        base_lr=2e-4,
        min_lr=1e-6,
        warmup_steps=3,
        max_steps=10,
    )
    for _ in range(4):
        uninterrupted_optimizer.step()
        uninterrupted.step()
    state = uninterrupted.state_dict()
    expected_next_lr = uninterrupted_optimizer.param_groups[0]["lr"]

    resumed_optimizer = _optimizer()
    resumed = WarmupCosineScheduler(
        resumed_optimizer,
        base_lr=2e-4,
        min_lr=1e-6,
        warmup_steps=3,
        max_steps=10,
    )
    resumed.load_state_dict(state)

    assert resumed.completed_steps == 4
    assert resumed_optimizer.param_groups[0]["lr"] == expected_next_lr


if __name__ == "__main__":
    tests = [
        test_metrics_identity_and_prediction_only_clamp,
        test_psnr_is_per_image_rgb_mean,
        test_task_summary_uses_equal_sigma_and_task_macro_weights,
        test_warmup_cosine_has_exact_update_endpoints,
        test_scheduler_resume_restores_exact_next_update_lr,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")

import pytest
import torch

from general_decomposition_model import (
    DecompositionLoss,
    frequency_high_freq_loss,
    gradient_edge_loss,
)
from train_general_decomp import GeneralDecompositionTrainer


def test_decomposition_loss_exposes_consistent_per_sample_values():
    criterion = DecompositionLoss()
    pattern = torch.rand(3, 3, 16, 16, requires_grad=True)
    background = torch.rand(3, 3, 16, 16, requires_grad=True)
    input_image = torch.rand(3, 3, 16, 16)
    pattern_gt = torch.rand_like(input_image)
    background_gt = torch.rand_like(input_image)
    orthogonal = torch.tensor(0.25, requires_grad=True)

    total, losses, per_sample = criterion(
        pattern,
        background,
        orthogonal,
        input_image,
        pattern_gt,
        background_gt,
        return_per_sample=True,
    )

    for name, values in per_sample.items():
        assert values.shape == (3,)
        if name != "image_total":
            assert torch.allclose(values.mean(), losses[name])
    assert torch.allclose(
        total,
        per_sample["image_total"].mean()
        + criterion.w_orthogonal * orthogonal,
    )

    reconstruction = (pattern + background).clamp(0, 1)
    expected = criterion.w_orthogonal * orthogonal
    for prediction, target, weight in (
        (pattern, pattern_gt, criterion.w_pattern),
        (background, background_gt, criterion.w_bg),
        (reconstruction, input_image, criterion.w_recon),
    ):
        expected = expected + weight * torch.nn.functional.l1_loss(
            prediction, target
        )
        expected = expected + weight * criterion.w_ssim * criterion.ssim_loss(
            prediction, target
        )
        expected = expected + weight * criterion.w_frequency * (
            frequency_high_freq_loss(prediction, target)
        )
        expected = expected + weight * criterion.w_edge * gradient_edge_loss(
            prediction, target
        )
    assert torch.allclose(total, expected, rtol=1e-5, atol=1e-6)
    total.backward()
    assert pattern.grad is not None
    assert background.grad is not None


def test_frequency_loss_promotes_half_inputs_to_float32():
    prediction = torch.rand(1, 3, 64, 64, dtype=torch.float16, requires_grad=True)
    target = torch.rand_like(prediction)

    loss = frequency_high_freq_loss(prediction, target)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_frequency_loss_uses_orthonormal_fft_scaling():
    prediction = torch.rand(1, 3, 16, 16)
    target = torch.rand_like(prediction)
    mask = torch.ones(16, 16)
    low_frequency_size = 5
    center = 8
    mask[
        center - low_frequency_size // 2:center + low_frequency_size // 2 + 1,
        center - low_frequency_size // 2:center + low_frequency_size // 2 + 1,
    ] = 0

    prediction_dft = torch.fft.fftshift(
        torch.fft.fft2(prediction, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
    )
    target_dft = torch.fft.fftshift(
        torch.fft.fft2(target, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
    )
    expected = (
        torch.abs(torch.abs(prediction_dft * mask) - torch.abs(target_dft * mask))
        .mean()
    )

    assert torch.allclose(frequency_high_freq_loss(prediction, target), expected)


def test_degradation_types_are_normalized_per_sample():
    normalize = GeneralDecompositionTrainer._normalize_degradation_types
    assert normalize("rain", 2) == ["rain", "rain"]
    assert normalize(["rain", "fog"], 2) == ["rain", "fog"]
    with pytest.raises(ValueError, match="does not match batch size"):
        normalize(["rain"], 2)


def test_mixed_batch_metrics_create_stable_per_type_keys():
    stats = {}
    GeneralDecompositionTrainer._accumulate_loss_stats(
        stats,
        {"total": torch.tensor(4.0)},
        {"image_total": torch.tensor([1.0, 3.0, 5.0])},
        ["rain", "fog", "rain"],
    )
    averages = GeneralDecompositionTrainer._average_stats(stats)

    assert averages["overall/total"] == pytest.approx(4.0)
    assert averages["by_type/rain/image_total"] == pytest.approx(3.0)
    assert averages["by_type/fog/image_total"] == pytest.approx(3.0)
    assert not any("['rain'" in key for key in averages)


def test_epoch_overall_average_is_weighted_by_sample_count():
    stats = {}
    GeneralDecompositionTrainer._accumulate_loss_stats(
        stats,
        {"total": torch.tensor(1.0)},
        {"image_total": torch.tensor([1.0, 1.0])},
        ["rain", "fog"],
    )
    GeneralDecompositionTrainer._accumulate_loss_stats(
        stats,
        {"total": torch.tensor(4.0)},
        {"image_total": torch.tensor([4.0])},
        ["rain"],
    )
    averages = GeneralDecompositionTrainer._average_stats(stats)
    assert averages["overall/total"] == pytest.approx(2.0)

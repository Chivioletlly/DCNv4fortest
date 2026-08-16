import copy
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from cdd11_runner.inference import tiled_inference
from cdd11_runner.training import backward_effective_batch, resolve_training_target_step


def _gradients(model):
    return [parameter.grad.detach().clone() for parameter in model.parameters()]


def test_microbatch_accumulation_matches_full_effective_batch_gradient():
    torch.manual_seed(3407)
    reference = torch.nn.Sequential(
        torch.nn.Conv2d(3, 5, kernel_size=3, padding=1),
        torch.nn.GELU(),
        torch.nn.Conv2d(5, 3, kernel_size=1),
    )
    degraded = torch.rand(11, 3, 8, 8)
    target = torch.rand(11, 3, 8, 8)
    gradients = {}
    losses = {}
    for microbatch_size in (11, 4, 1):
        model = copy.deepcopy(reference)
        model.zero_grad(set_to_none=True)
        losses[microbatch_size] = backward_effective_batch(
            model,
            degraded,
            target,
            microbatch_size=microbatch_size,
        )
        gradients[microbatch_size] = _gradients(model)

    torch.testing.assert_close(
        torch.tensor(losses[11]),
        torch.tensor(losses[4]),
        rtol=0.0,
        atol=1e-7,
    )
    for full, chunked, single in zip(gradients[11], gradients[4], gradients[1]):
        torch.testing.assert_close(full, chunked, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(full, single, rtol=1e-5, atol=1e-7)


def test_weighted_tiling_covers_image_and_preserves_identity_model():
    value = torch.rand(1, 3, 23, 31)
    restored = tiled_inference(
        torch.nn.Identity(),
        value,
        tile_size=16,
        overlap=5,
    )
    assert torch.isfinite(restored).all()
    torch.testing.assert_close(restored, value, rtol=1e-6, atol=1e-6)


def test_resume_pause_boundary_must_align_with_scalar_window():
    assert resolve_training_target_step(
        global_step=0,
        max_steps=100,
        scalar_interval=10,
        pause_at_step=50,
    ) == 50
    try:
        resolve_training_target_step(
            global_step=0,
            max_steps=100,
            scalar_interval=10,
            pause_at_step=51,
        )
    except ValueError as error:
        assert "align" in str(error)
    else:
        raise AssertionError("Expected an unaligned safe pause to be rejected")

"""Linux GPU smoke tests for the vendored DCNv4 integration."""

import argparse
import math
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from general_decomposition_model import (  # noqa: E402
    DCNv4FeatureBlock,
    DecompositionLoss,
    GeneralDecompositionNet,
)


def assert_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf values")


def test_blocks(device):
    for dtype in (torch.float32, torch.float16):
        for channels in (32, 128, 256, 512):
            block = DCNv4FeatureBlock(channels).to(device=device, dtype=dtype).train()
            x = torch.randn(1, channels, 16, 16, device=device, dtype=dtype, requires_grad=True)
            y = block(x)
            if y.shape != x.shape:
                raise RuntimeError(f"shape mismatch for C={channels}: {y.shape} != {x.shape}")
            assert_finite(y, f"block output C={channels} dtype={dtype}")
            y.float().square().mean().backward()
            assert_finite(x.grad, f"block input gradient C={channels} dtype={dtype}")
            del block, x, y
    torch.cuda.empty_cache()


def test_model(device, resolution, steps):
    model = GeneralDecompositionNet(
        base_channels=64,
        bottleneck_type="conv",
        use_orient_block=True,
    ).to(device).train()
    criterion = DecompositionLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    initial_offset = model.encoder.orient2.dcn.offset_mask.weight.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        x = torch.rand(1, 3, resolution, resolution, device=device)
        pattern_gt = torch.rand_like(x) * 0.3
        bg_gt = torch.rand_like(x)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pattern, background, orth_loss = model(x)
            total, losses = criterion(
                pattern, background, orth_loss, x, pattern_gt=pattern_gt, bg_gt=bg_gt
            )
        assert_finite(total, "total loss")
        total.backward()
        for name, parameter in model.named_parameters():
            if "offset_mask" in name and parameter.grad is not None:
                assert_finite(parameter.grad, f"gradient {name}")
        optimizer.step()

    offset_change = (
        model.encoder.orient2.dcn.offset_mask.weight.detach() - initial_offset
    ).abs().max().item()
    if not math.isfinite(offset_change) or offset_change == 0.0:
        raise RuntimeError("DCNv4 offset/mask parameters were not updated")

    expected_shape = (1, 3, resolution, resolution)
    if pattern.shape != expected_shape or background.shape != expected_shape:
        raise RuntimeError("full-model output shape mismatch")
    print(f"Full model: resolution={resolution}, steps={steps}, offset_change={offset_change:.6g}")
    print("Losses:", ", ".join(sorted(losses)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for DCNv4 tests")
    device = torch.device("cuda")
    test_blocks(device)
    test_model(device, args.resolution, args.steps)
    print("All DCNv4 integration tests passed.")


if __name__ == "__main__":
    main()

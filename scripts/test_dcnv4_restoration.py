"""CUDA/BF16 smoke test for DCNv4RestorationUNet."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from dcnv4_restoration_model import DCNv4RestorationUNet, count_parameters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This integration test requires a CUDA GPU")

    device = torch.device("cuda")
    model = DCNv4RestorationUNet(
        base_channels=args.base_channels,
        bottleneck_type="conv",
        use_dcnv4=True,
    ).to(device)
    model.train()

    degraded = torch.rand(
        args.batch_size,
        3,
        args.height,
        args.width,
        device=device,
    )
    target = torch.rand_like(degraded)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        restored = model(degraded)
        loss = F.l1_loss(restored, target)
    loss.backward()
    torch.cuda.synchronize(device)

    if restored.shape != degraded.shape:
        raise AssertionError(
            f"Output shape {tuple(restored.shape)} != input shape {tuple(degraded.shape)}"
        )
    if not torch.isfinite(restored).all() or not torch.isfinite(loss):
        raise AssertionError("Forward/backward produced non-finite values")
    output_gradient = model.residual_head.out_conv.weight.grad
    if output_gradient is None or not torch.isfinite(output_gradient).all():
        raise AssertionError("Signed residual output head has no finite gradient")

    print("architecture metadata:", model.checkpoint_metadata())
    print("parameters:", count_parameters(model))
    print("input/output:", degraded.shape, restored.shape)
    print("output dtype:", restored.dtype)
    print("output range:", restored.float().min().item(), restored.float().max().item())
    print("loss:", loss.item())
    print(
        "peak memory GiB:",
        torch.cuda.max_memory_allocated(device) / 1024**3,
    )
    print("BF16 DCNv4 restoration forward/backward: PASS")


if __name__ == "__main__":
    main()

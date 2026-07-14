"""Compare the removed directional block with the DCNv4 replacement on Linux GPU."""

import argparse
import os
import statistics
import sys

import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from general_decomposition_model import DCNv4FeatureBlock, GeneralDecompositionNet  # noqa: E402


class LegacyOrientationAwareBlock(nn.Module):
    """Exact pre-DCNv4 block retained only for benchmarking."""

    def __init__(self, channels, num_orientations=4):
        super().__init__()
        per_orient = channels // num_orientations
        kernel_sizes = ((1, 7), (1, 11), (1, 15), (1, 21))
        self.per_orient = per_orient
        self.directional_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    per_orient,
                    per_orient,
                    kernel,
                    padding=(kernel[0] // 2, kernel[1] // 2),
                    bias=False,
                ),
                nn.BatchNorm2d(per_orient),
                nn.ReLU(inplace=True),
            )
            for kernel in kernel_sizes
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        parts = x.split(self.per_orient, dim=1)
        out = torch.cat(
            [conv(part) for conv, part in zip(self.directional_convs, parts)], dim=1
        )
        return x + self.fusion(out)


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def benchmark_forward(module, x, warmup, iterations):
    module.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize()

        timings = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            module(x)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    return statistics.mean(timings), statistics.median(timings)


def training_peak_memory(module, x):
    module.train()
    module.zero_grad(set_to_none=True)
    x.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = module(x)
        if isinstance(output, tuple):
            loss = sum(item.float().mean() for item in output)
        else:
            loss = output.float().mean()
    loss.backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def make_legacy_model(device):
    model = GeneralDecompositionNet(
        base_channels=64, bottleneck_type="conv", use_orient_block=False
    )
    model.encoder.use_orient_block = True
    model.encoder.orient2 = LegacyOrientationAwareBlock(128)
    model.encoder.orient3 = LegacyOrientationAwareBlock(256)
    model.encoder.orient4 = LegacyOrientationAwareBlock(512)
    model.disentangle.use_orient_block = True
    model.disentangle.orient_pattern = LegacyOrientationAwareBlock(32)
    return model.to(device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for benchmarking")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    feature_shapes = ((32, 512), (128, 256), (256, 128), (512, 64))
    print("\nSingle blocks (FP16 inference, batch=1)")
    for channels, size_at_512 in feature_shapes:
        size = max(1, size_at_512 * args.resolution // 512)
        x = torch.randn(
            1, channels, size, size, device=device, requires_grad=True
        )
        for label, block_factory in (
            ("legacy", lambda: LegacyOrientationAwareBlock(channels)),
            ("dcnv4", lambda: DCNv4FeatureBlock(channels)),
        ):
            block = block_factory().to(device)
            mean_ms, median_ms = benchmark_forward(
                block, x, args.warmup, args.iterations
            )
            peak_mb = training_peak_memory(block, x)
            print(
                f"C={channels:3d} H=W={size:3d} {label:6s}: "
                f"params={parameter_count(block):,}, mean={mean_ms:.3f} ms, "
                f"median={median_ms:.3f} ms, train_peak={peak_mb:.1f} MiB"
            )
            del block
        del x
        torch.cuda.empty_cache()

    x = torch.randn(
        1, 3, args.resolution, args.resolution, device=device, requires_grad=True
    )
    model_factories = (
        ("legacy", lambda: make_legacy_model(device)),
        (
            "dcnv4",
            lambda: GeneralDecompositionNet(
                base_channels=64, bottleneck_type="conv", use_orient_block=True
            ).to(device),
        ),
    )
    print("\nFull models (FP16 inference, batch=1)")
    for label, model_factory in model_factories:
        model = model_factory()
        mean_ms, median_ms = benchmark_forward(model, x, args.warmup, args.iterations)
        peak_mb = training_peak_memory(model, x)
        print(
            f"{label:6s}: params={parameter_count(model):,}, mean={mean_ms:.3f} ms, "
            f"median={median_ms:.3f} ms, train_peak={peak_mb:.1f} MiB"
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

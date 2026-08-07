"""CUDA/BF16 smoke test with an FP32 DCNv4 operator fallback."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from dcnv4_restoration_model import (
    DCNv4FeatureBlock,
    DCNv4RestorationUNet,
    count_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    return parser.parse_args()


def register_dcn_dtype_hooks(model):
    """Record feature and operator dtypes at every DCNv4 autocast boundary."""

    records = {}
    handles = []
    blocks = []

    for name, module in model.named_modules():
        if not isinstance(module, DCNv4FeatureBlock):
            continue

        records[name] = {}
        blocks.append((name, module))

        def record_feature_input(_module, inputs, block_name=name):
            records[block_name]["feature_input"] = inputs[0].dtype

        def record_operator_input(_module, inputs, block_name=name):
            records[block_name]["operator_input"] = inputs[0].dtype

        def record_operator_output(_module, _inputs, output, block_name=name):
            records[block_name]["operator_output"] = output.dtype

        handles.append(module.register_forward_pre_hook(record_feature_input))
        handles.append(module.dcn.register_forward_pre_hook(record_operator_input))
        handles.append(module.dcn.register_forward_hook(record_operator_output))

    return blocks, records, handles


def validate_dcn_amp_boundary(blocks, records):
    if len(blocks) != 4:
        raise AssertionError(f"Expected four DCNv4 blocks, found {len(blocks)}")

    for name, block in blocks:
        record = records.get(name, {})
        operator_input = record.get("operator_input")
        operator_output = record.get("operator_output")
        if operator_input != torch.float32 or operator_output != torch.float32:
            raise AssertionError(
                f"{name} escaped the FP32 DCNv4 boundary: "
                f"input={operator_input}, output={operator_output}"
            )

        gradients = [
            parameter.grad
            for parameter in block.dcn.parameters()
            if parameter.requires_grad
        ]
        if not gradients or any(gradient is None for gradient in gradients):
            raise AssertionError(f"{name} did not participate in backward")
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError(f"{name} produced non-finite gradients")


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

    dcn_blocks, dcn_dtype_records, hook_handles = register_dcn_dtype_hooks(model)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            restored = model(degraded)
            loss = F.l1_loss(restored, target)
    finally:
        for handle in hook_handles:
            handle.remove()
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
    validate_dcn_amp_boundary(dcn_blocks, dcn_dtype_records)

    print("architecture metadata:", model.checkpoint_metadata())
    print("parameters:", count_parameters(model))
    print("input/output:", degraded.shape, restored.shape)
    print("output dtype:", restored.dtype)
    print("output range:", restored.float().min().item(), restored.float().max().item())
    print("loss:", loss.item())
    print("DCNv4 autocast boundaries:")
    for block_name, _block in dcn_blocks:
        record = dcn_dtype_records[block_name]
        print(
            f"  {block_name}: feature={record['feature_input']}, "
            f"operator_in={record['operator_input']}, "
            f"operator_out={record['operator_output']}"
        )
    print(
        "peak memory GiB:",
        torch.cuda.max_memory_allocated(device) / 1024**3,
    )
    print("BF16 network / FP32 DCNv4 restoration forward/backward: PASS")


if __name__ == "__main__":
    main()

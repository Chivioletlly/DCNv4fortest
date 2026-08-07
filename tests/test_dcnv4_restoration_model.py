import sys
from pathlib import Path

import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from dcnv4_restoration_model import (
    ARCHITECTURE_VERSION,
    DCNv4FeatureBlock,
    DCNv4RestorationUNet,
    NORMALIZATION_TYPE,
    OUTPUT_MODE,
    validate_restoration_checkpoint,
)


class FakeDCNv4(nn.Module):
    """CPU stand-in that preserves the official operator interface."""

    def __init__(self, channels, group, **_kwargs):
        super().__init__()
        self.offset_mask = nn.Linear(channels, group * 27)
        self.value_proj = nn.Linear(channels, channels)
        self.last_input_dtype = None
        self.last_output_dtype = None

    def forward(self, x, shape):
        height, width = shape
        assert x.shape[1] == height * width
        self.last_input_dtype = x.dtype
        output = self.value_proj(x)
        self.last_output_dtype = output.dtype
        return output


def build_model():
    return DCNv4RestorationUNet(
        base_channels=32,
        bottleneck_type="conv",
        use_dcnv4=True,
        dcnv4_cls=FakeDCNv4,
    )


def test_zero_initialized_model_is_identity_at_arbitrary_size():
    model = build_model().eval()
    degraded = torch.rand(1, 3, 17, 23)

    with torch.no_grad():
        restored = model(degraded)

    assert restored.shape == degraded.shape
    torch.testing.assert_close(restored, degraded, rtol=0.0, atol=0.0)


def test_pixel_residual_is_signed_and_unclamped():
    model = build_model().eval()
    with torch.no_grad():
        model.residual_head.out_conv.weight.zero_()
        model.residual_head.out_conv.bias.copy_(torch.tensor([0.25, -0.25, 1.25]))

    degraded = torch.zeros(1, 3, 16, 16)
    with torch.no_grad():
        restored = model(degraded)

    torch.testing.assert_close(restored[:, 0], torch.full_like(restored[:, 0], 0.25))
    torch.testing.assert_close(restored[:, 1], torch.full_like(restored[:, 1], -0.25))
    torch.testing.assert_close(restored[:, 2], torch.full_like(restored[:, 2], 1.25))
    assert restored.min().item() < 0.0
    assert restored.max().item() > 1.0


def test_model_uses_four_dcnv4_blocks_and_no_batch_norm():
    model = build_model()

    dcn_blocks = [module for module in model.modules() if isinstance(module, DCNv4FeatureBlock)]
    batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    group_norms = [module for module in model.modules() if isinstance(module, nn.GroupNorm)]

    assert len(dcn_blocks) == 4
    assert not batch_norms
    assert group_norms


def test_output_head_receives_gradient_from_l1_loss():
    model = build_model().train()
    degraded = torch.rand(1, 3, 16, 16)
    target = torch.rand_like(degraded)

    loss = (model(degraded) - target).abs().mean()
    loss.backward()

    gradient = model.residual_head.out_conv.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_bfloat16_features_use_float32_inside_dcn_and_keep_gradients():
    block = DCNv4FeatureBlock(32, dcnv4_cls=FakeDCNv4).train()
    sequence = torch.randn(
        1,
        8 * 8,
        32,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    output = block._apply_dcn(sequence, height=8, width=8)
    loss = output.float().square().mean()
    loss.backward()

    assert block.dcn.last_input_dtype == torch.float32
    assert output.dtype == torch.bfloat16
    assert sequence.grad is not None
    assert torch.isfinite(sequence.grad).all()


def test_float32_features_do_not_inherit_outer_bfloat16_autocast():
    block = DCNv4FeatureBlock(32, dcnv4_cls=FakeDCNv4).train()
    sequence = torch.randn(1, 8 * 8, 32, requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = block._apply_dcn(sequence, height=8, width=8)
        loss = output.square().mean()
    loss.backward()

    assert block.dcn.last_input_dtype == torch.float32
    assert block.dcn.last_output_dtype == torch.float32
    assert output.dtype == torch.float32
    assert sequence.grad is not None
    assert torch.isfinite(sequence.grad).all()


def test_checkpoint_metadata_identifies_restoration_architecture():
    model = build_model()
    metadata = model.checkpoint_metadata()

    assert metadata["architecture_version"] == ARCHITECTURE_VERSION
    assert metadata["output_mode"] == OUTPUT_MODE
    assert metadata["normalization"] == NORMALIZATION_TYPE
    validate_restoration_checkpoint(metadata)


def test_legacy_checkpoint_is_rejected():
    try:
        validate_restoration_checkpoint(
            {
                "architecture_version": 2,
                "model_name": "general_decomposition",
                "output_mode": "pattern_background",
                "normalization": "batchnorm",
                "direction_block_type": "dcnv4",
            }
        )
    except RuntimeError as error:
        assert "not compatible" in str(error)
    else:
        raise AssertionError("Legacy checkpoint metadata was accepted")


if __name__ == "__main__":
    tests = [
        value
        for name, value in list(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")

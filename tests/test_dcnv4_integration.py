import pytest
import torch
import torch.nn as nn

import general_decomposition_model as model_module


class FakeDCNv4(nn.Module):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.projection = nn.Linear(channels, channels)
        self.offset_mask = nn.Linear(channels, kwargs["group"] * 27)

    def forward(self, x, shape=None):
        return self.projection(x)


def test_baseline_does_not_require_dcnv4(monkeypatch):
    monkeypatch.setattr(model_module, "_DCNv4", None)
    model = model_module.GeneralDecompositionNet(
        base_channels=16, bottleneck_type="conv", use_orient_block=False
    )
    x = torch.rand(1, 3, 32, 32)
    pattern, background, orth_loss = model(x)
    assert pattern.shape == x.shape
    assert background.shape == x.shape
    assert orth_loss.ndim == 0


def test_enabled_block_reports_linux_build_command_when_dcnv4_is_missing(
    monkeypatch,
):
    monkeypatch.setattr(model_module, "_DCNv4", None)
    monkeypatch.setattr(
        model_module, "_DCNV4_IMPORT_ERROR", ImportError("extension unavailable")
    )
    with pytest.raises(RuntimeError, match="bash scripts/build_dcnv4.sh"):
        model_module.DCNv4FeatureBlock(32)


@pytest.mark.parametrize(
    ("channels", "expected_group"), ((32, 1), (128, 4), (256, 8), (512, 16), (48, 3))
)
def test_group_selection(channels, expected_group):
    assert model_module.DCNv4FeatureBlock._select_group(channels) == expected_group


def test_invalid_group_selection():
    with pytest.raises(ValueError, match="divisible by 32 or 16"):
        model_module.DCNv4FeatureBlock._select_group(24)


def test_dcnv4_block_preserves_shape_and_has_residual(monkeypatch):
    monkeypatch.setattr(model_module, "_DCNv4", FakeDCNv4)
    block = model_module.DCNv4FeatureBlock(32).eval()
    x = torch.randn(2, 32, 8, 12, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_default_model_uses_expected_groups_at_all_four_sites(monkeypatch):
    monkeypatch.setattr(model_module, "_DCNv4", FakeDCNv4)
    model = model_module.GeneralDecompositionNet(
        base_channels=64, bottleneck_type="conv", use_orient_block=True
    )
    blocks = (
        model.encoder.orient2,
        model.encoder.orient3,
        model.encoder.orient4,
        model.disentangle.orient_pattern,
    )
    assert [block.group for block in blocks] == [4, 8, 16, 1]


def test_dcnv4_starts_with_zero_offsets_and_uniform_weights(monkeypatch):
    monkeypatch.setattr(model_module, "_DCNv4", FakeDCNv4)
    block = model_module.DCNv4FeatureBlock(32)
    bias = block.dcn.offset_mask.bias.detach()
    assert torch.count_nonzero(bias[:18]) == 0
    assert torch.allclose(bias[18:27], torch.full((9,), 1.0 / 9.0))


def test_legacy_direction_checkpoint_is_rejected():
    with pytest.raises(RuntimeError, match="legacy OrientationAwareBlock"):
        model_module.validate_checkpoint_architecture({"use_orient_block": True})


def test_dcnv4_checkpoint_is_accepted():
    model_module.validate_checkpoint_architecture(
        {
            "use_orient_block": True,
            "architecture_version": 2,
            "direction_block_type": "dcnv4",
        }
    )

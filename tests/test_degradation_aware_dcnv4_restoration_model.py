import sys
from pathlib import Path

import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from aio3_runner.runtime import (
    BASELINE_EXPECTED_PARAMETERS,
    DEGRADATION_AWARE_EXPECTED_PARAMETERS,
    build_run_config,
)
from aio3_runner.models import build_dcnv4_unet
import dcnv4_restoration_model as restoration_module
from dcnv4_restoration_model import (
    AdaptiveGatedFusion,
    ContextGatedDCNv4FeatureBlock,
    ContextGatedDualDomainModulation,
    DCNv4FeatureBlock,
    DCNv4RestorationUNet,
    DEGRADATION_AWARE_ARCHITECTURE_VERSION,
    DEGRADATION_AWARE_MODEL_NAME,
    DegradationAwareContext,
    DegradationAwareDCNv4RestorationUNet,
    count_parameters,
    validate_restoration_checkpoint,
)


class FakeDCNv4(nn.Module):
    """CPU stand-in that preserves the official DCNv4 interface."""

    def __init__(self, channels, group, **_kwargs):
        super().__init__()
        self.offset_mask = nn.Linear(channels, group * 27)
        self.value_proj = nn.Linear(channels, channels)

    def forward(self, x, shape):
        height, width = shape
        assert x.shape[1] == height * width
        return self.value_proj(x)


def build_model(base_channels=32):
    return DegradationAwareDCNv4RestorationUNet(
        base_channels=base_channels,
        bottleneck_type="conv",
        use_dcnv4=True,
        context_scales=3,
        dcnv4_cls=FakeDCNv4,
    )


def test_degradation_aware_model_is_identity_at_arbitrary_size():
    model = build_model().eval()
    degraded = torch.rand(2, 3, 17, 19)

    with torch.no_grad():
        restored = model(degraded)

    assert restored.shape == degraded.shape
    torch.testing.assert_close(restored, degraded, rtol=0.0, atol=0.0)


def test_context_encoder_produces_stage_aligned_input_dependent_prompts():
    model = build_model().eval()
    dark = torch.zeros(2, 3, 16, 24)
    bright = torch.ones_like(dark)

    with torch.no_grad():
        dark_prompts, dark_global = model.extract_degradation_context(dark)
        bright_prompts, bright_global = model.extract_degradation_context(bright)

    assert [tuple(prompt.shape) for prompt in dark_prompts] == [
        (2, 32),
        (2, 64),
        (2, 128),
        (2, 256),
    ]
    assert dark_global.shape == (2, 64)
    assert not torch.allclose(dark_global, bright_global)
    assert any(
        not torch.allclose(dark_prompt, bright_prompt)
        for dark_prompt, bright_prompt in zip(dark_prompts, bright_prompts)
    )


def test_model_contains_all_degradation_aware_paths_without_batch_norm():
    model = build_model()

    assert len(
        [
            module
            for module in model.modules()
            if isinstance(module, ContextGatedDCNv4FeatureBlock)
        ]
    ) == 4
    assert len(
        [module for module in model.modules() if isinstance(module, AdaptiveGatedFusion)]
    ) == 3
    assert len(
        [
            module
            for module in model.modules()
            if isinstance(module, ContextGatedDualDomainModulation)
        ]
    ) == 1
    assert len(
        [module for module in model.modules() if isinstance(module, DegradationAwareContext)]
    ) == 1
    assert not [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]


def test_context_gated_dcnv4_is_identity_compatible_at_initialization():
    baseline = DCNv4FeatureBlock(32, dcnv4_cls=FakeDCNv4).eval()
    aware = ContextGatedDCNv4FeatureBlock(
        32,
        context_channels=16,
        dcnv4_cls=FakeDCNv4,
    ).eval()
    aware.load_state_dict(baseline.state_dict(), strict=False)
    feature = torch.randn(2, 32, 8, 8)
    context = torch.randn(2, 16)

    with torch.no_grad():
        baseline_output = baseline(feature)
        aware_output = aware(feature, context)

    torch.testing.assert_close(aware_output, baseline_output, rtol=0.0, atol=0.0)


def test_reconstruction_loss_reaches_degradation_context_encoder():
    model = build_model().train()
    with torch.no_grad():
        nn.init.normal_(model.residual_head.out_conv.weight, std=1.0e-3)
    degraded = torch.rand(1, 3, 16, 16)
    target = torch.rand_like(degraded)

    loss = (model(degraded) - target).abs().mean()
    loss.backward()

    gradient = model.degradation_context.stem[0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_dual_domain_module_keeps_fft_in_a_differentiable_fp32_path():
    module = ContextGatedDualDomainModulation(16, 8).train()
    feature = torch.randn(2, 16, 8, 10, requires_grad=True)
    context = torch.randn(2, 8, requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = module(feature, context)
        loss = output.float().square().mean()
    loss.backward()

    assert output.shape == feature.shape
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()


def test_checkpoint_metadata_separates_baseline_and_degradation_aware_models():
    baseline = DCNv4RestorationUNet(base_channels=32, dcnv4_cls=FakeDCNv4)
    aware = build_model()
    baseline_metadata = baseline.checkpoint_metadata()
    aware_metadata = aware.checkpoint_metadata()

    assert aware_metadata["architecture_version"] == DEGRADATION_AWARE_ARCHITECTURE_VERSION
    assert aware_metadata["model_name"] == DEGRADATION_AWARE_MODEL_NAME
    validate_restoration_checkpoint(aware_metadata, expected=aware_metadata)

    try:
        validate_restoration_checkpoint(
            baseline_metadata,
            expected=aware_metadata,
        )
    except RuntimeError as error:
        assert "not compatible" in str(error)
    else:
        raise AssertionError("Baseline checkpoint was accepted by the aware model")


def test_frozen_parameter_count_and_runner_variant_config():
    baseline = DCNv4RestorationUNet(base_channels=64, dcnv4_cls=FakeDCNv4)
    aware = build_model(base_channels=64)
    fake_operator_delta = count_parameters(aware) - count_parameters(baseline)
    frozen_delta = (
        DEGRADATION_AWARE_EXPECTED_PARAMETERS - BASELINE_EXPECTED_PARAMETERS
    )
    assert fake_operator_delta == frozen_delta

    common = {
        "run_kind": "smoke",
        "seed": 3407,
        "run_name": "unit",
        "run_dir": Path("output") / "model" / "run",
        "manifest_hashes": {},
        "repository_state": {
            "commit": "unit",
            "dirty": False,
            "protocol_document_sha256": "unit",
        },
        "num_workers": 0,
        "wandb_run_id": "unit",
        "wandb_mode": "disabled",
        "wandb_entity": None,
    }
    baseline_config = build_run_config(**common)
    aware_config = build_run_config(**common, model_variant="degradation-aware")
    assert baseline_config["model"]["name"] == "dcnv4_restoration_unet"
    assert aware_config["model"]["name"] == DEGRADATION_AWARE_MODEL_NAME
    assert (
        aware_config["model"]["expected_parameters"]
        == DEGRADATION_AWARE_EXPECTED_PARAMETERS
    )

    original_operator = restoration_module._DCNv4
    restoration_module._DCNv4 = FakeDCNv4
    try:
        built = build_dcnv4_unet(aware_config)
    finally:
        restoration_module._DCNv4 = original_operator
    assert isinstance(built, DegradationAwareDCNv4RestorationUNet)


if __name__ == "__main__":
    tests = [
        value
        for name, value in list(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")

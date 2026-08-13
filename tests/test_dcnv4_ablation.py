import io
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import dcnv4_restoration_model as restoration_module
from aio3_runner.ablation import (
    ABLATION_REGISTRY,
    ABLATION_VARIANT_IDS,
    get_ablation_variant,
    registry_snapshot,
)
from aio3_runner.models import build_dcnv4_ablation_model, build_dcnv4_unet
from aio3_runner.runtime import BASELINE_EXPECTED_PARAMETERS, build_run_config
from dcnv4_restoration_model import (
    AdaptiveGatedFusion,
    ContextGatedDCNv4FeatureBlock,
    ContextGatedDualDomainModulation,
    ContextGatedSpatialModulation,
    DCNv4RestorationUNet,
    DegradationAwareContext,
    DegradationAwareDCNv4RestorationUNet,
    ProjectedSkipFusion,
    PromptFeatureModulation,
    RegisteredDCNv4RestorationUNet,
    RegisteredDegradationAwareDCNv4RestorationUNet,
    StaticDualDomainModulation,
    count_parameters,
)


class FakeDCNv4(nn.Module):
    """CPU stand-in with the same parameter delta at every DCNv4 site."""

    def __init__(self, channels, group, **_kwargs):
        super().__init__()
        self.offset_mask = nn.Linear(channels, group * 27)
        self.value_proj = nn.Linear(channels, channels)
        self.last_input_dtype = None

    def forward(self, x, shape):
        height, width = shape
        assert x.shape[1] == height * width
        self.last_input_dtype = x.dtype
        return self.value_proj(x)


def build_variant(variant_id, base_channels=32):
    return build_dcnv4_ablation_model(
        variant_id,
        base_channels=base_channels,
        bottleneck_type="conv",
        use_dcnv4=True,
        context_scales=3,
        dcnv4_cls=FakeDCNv4,
    )


def run_config(variant_id):
    return build_run_config(
        run_kind="smoke",
        seed=3407,
        run_name="unit",
        run_dir=Path("output") / variant_id / "unit",
        manifest_hashes={},
        repository_state={
            "commit": "unit",
            "dirty": False,
            "protocol_document_sha256": "unit",
            "ablation_plan_sha256": "unit",
        },
        num_workers=0,
        wandb_run_id="unit",
        wandb_mode="disabled",
        wandb_entity=None,
        model_variant=variant_id,
    )


def _assert_module_gradient(module):
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def _cpu_bfloat16_model_autocast_supported():
    try:
        convolution = nn.Conv2d(2, 2, 3, padding=2, dilation=2)
        normalization = nn.LayerNorm(2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            feature = convolution(torch.rand(1, 2, 4, 4))
            normalization(feature.mean(dim=(2, 3)))
    except RuntimeError:
        return False
    return True


CPU_BFLOAT16_MODEL_AUTOCAST = _cpu_bfloat16_model_autocast_supported()


def test_registry_has_exactly_the_sixteen_preregistered_ids():
    assert ABLATION_VARIANT_IDS == (
        "APG-000",
        "APG-100",
        "APG-010",
        "APG-001",
        "APG-110",
        "APG-101",
        "APG-011",
        "APG-111",
        "P-FILM",
        "P-DIN",
        "P-DOUT",
        "P-DIO",
        "SKIP-PROJ-000",
        "SKIP-PROJ-011",
        "CGDM-STATIC",
        "CGDM-SPATIAL",
    )
    assert len(registry_snapshot()) == 16
    assert all(variant.expected_parameters > 0 for variant in ABLATION_REGISTRY.values())


@pytest.mark.parametrize("variant_id", ABLATION_VARIANT_IDS)
def test_run_config_freezes_full_registry_metadata_and_output_identity(variant_id):
    variant = get_ablation_variant(variant_id)
    config = run_config(variant_id)
    assert config["model"]["variant_id"] == variant_id
    assert config["model"]["expected_parameters"] == variant.expected_parameters
    assert config["model"]["output_directory"].endswith(variant_id)
    assert config["model"] == {
        **variant.model_config(),
        "in_channels": 3,
        "base_channels": 64,
        "bottleneck_type": "conv",
        "use_dcnv4": True,
        "output_mode": "signed_residual",
        "normalization": "groupnorm",
        "initialization": "exact_identity_zero_initialized_signed_residual_head",
        **({"context_scales": 3} if variant.degradation_context != "none" else {}),
    }
    assert config["ablation_registry"] == registry_snapshot()
    assert variant_id.lower() in config["monitoring"]["tags"]


@pytest.mark.parametrize("variant_id", ABLATION_VARIANT_IDS)
def test_variant_structure_identity_and_checkpoint_round_trip(variant_id):
    variant = get_ablation_variant(variant_id)
    model = build_variant(variant_id).eval()
    modules = tuple(model.modules())
    assert sum(isinstance(module, AdaptiveGatedFusion) for module in modules) == (
        3 if variant.skip_fusion == "adaptive_spatial_channel_gate" else 0
    )
    assert sum(isinstance(module, ProjectedSkipFusion) for module in modules) == (
        3 if variant.skip_fusion == "concat_projection" else 0
    )
    assert sum(isinstance(module, PromptFeatureModulation) for module in modules) == (
        4 if variant.convolution_film else 0
    )
    assert sum(isinstance(module, ContextGatedDCNv4FeatureBlock) for module in modules) == (
        4 if variant.dcnv4_conditioning != "none" else 0
    )
    assert sum(isinstance(module, DegradationAwareContext) for module in modules) == (
        0 if variant.degradation_context == "none" else 1
    )
    assert sum(isinstance(module, ContextGatedDualDomainModulation) for module in modules) == (
        1 if variant.bottleneck_modulation == "context_gated_dual_domain" else 0
    )
    assert sum(isinstance(module, StaticDualDomainModulation) for module in modules) == (
        1 if variant.bottleneck_modulation == "static_dual_domain" else 0
    )
    assert sum(isinstance(module, ContextGatedSpatialModulation) for module in modules) == (
        1 if variant.bottleneck_modulation == "context_gated_spatial" else 0
    )

    degraded = torch.rand(1, 3, 17, 19)
    with torch.no_grad():
        restored = model(degraded)
    assert restored.shape == degraded.shape
    torch.testing.assert_close(restored, degraded, rtol=0.0, atol=0.0)
    metadata = model.checkpoint_metadata()
    for key, expected in variant.checkpoint_metadata().items():
        assert metadata[key] == expected

    with torch.no_grad():
        nn.init.normal_(model.residual_head.out_conv.weight, std=1.0e-3)
    fixed_input = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        expected_output = model(fixed_input)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored_model = build_variant(variant_id).eval()
    restored_model.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    with torch.no_grad():
        actual_output = restored_model(fixed_input)
    torch.testing.assert_close(actual_output, expected_output, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("variant_id", ABLATION_VARIANT_IDS)
def test_public_runner_builds_every_registered_variant(monkeypatch, variant_id):
    monkeypatch.setattr(restoration_module, "_DCNv4", FakeDCNv4)
    model = build_dcnv4_unet(run_config(variant_id))
    assert model.checkpoint_metadata()["variant_id"] == variant_id


@pytest.mark.parametrize("variant_id", ABLATION_VARIANT_IDS)
def test_variant_forward_backward_has_finite_active_module_gradients(variant_id):
    variant = get_ablation_variant(variant_id)
    model = build_variant(variant_id).train()
    with torch.no_grad():
        nn.init.normal_(model.residual_head.out_conv.weight, std=1.0e-3)
    degraded = torch.rand(1, 3, 16, 16)
    target = torch.rand_like(degraded)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    # Prompt consumers are deliberately zero-initialized.  The first update
    # opens the conditioning path; the second backward verifies that gradients
    # reach the DAM rather than misclassifying safe initialization as a dead path.
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        restored = model(degraded)
        loss = (restored - target).abs().mean()
        loss.backward()
        if step == 0:
            optimizer.step()
    assert restored.shape == degraded.shape
    assert torch.isfinite(restored).all()
    for module in model.modules():
        if isinstance(module, FakeDCNv4):
            assert module.last_input_dtype == torch.float32

    active_types = [type(model.residual_head)]
    if variant.degradation_context != "none":
        active_types.append(DegradationAwareContext)
    if variant.skip_fusion == "adaptive_spatial_channel_gate":
        active_types.append(AdaptiveGatedFusion)
    if variant.skip_fusion == "concat_projection":
        active_types.append(ProjectedSkipFusion)
    if variant.bottleneck_modulation == "context_gated_dual_domain":
        active_types.append(ContextGatedDualDomainModulation)
    if variant.bottleneck_modulation == "static_dual_domain":
        active_types.append(StaticDualDomainModulation)
    if variant.bottleneck_modulation == "context_gated_spatial":
        active_types.append(ContextGatedSpatialModulation)
    for module_type in active_types:
        matching = [module for module in model.modules() if type(module) is module_type]
        assert matching
        for module in matching:
            _assert_module_gradient(module)


@pytest.mark.skipif(
    not CPU_BFLOAT16_MODEL_AUTOCAST,
    reason="installed CPU PyTorch lacks BF16 dilated-convolution/LayerNorm support",
)
@pytest.mark.parametrize("variant_id", ABLATION_VARIANT_IDS)
def test_variant_bfloat16_autocast_keeps_dcnv4_and_fft_paths_float32(variant_id):
    model = build_variant(variant_id).train()
    with torch.no_grad():
        nn.init.normal_(model.residual_head.out_conv.weight, std=1.0e-3)
    degraded = torch.rand(1, 3, 16, 16)
    target = torch.rand_like(degraded)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        restored = model(degraded)
        loss = (restored.float() - target).abs().mean()
    loss.backward()
    assert torch.isfinite(restored).all()
    dcn_modules = [module for module in model.modules() if isinstance(module, FakeDCNv4)]
    assert len(dcn_modules) == 4
    assert all(module.last_input_dtype == torch.float32 for module in dcn_modules)


def test_frozen_parameter_counts_match_registry_and_spatial_control_is_exactly_matched():
    baseline = build_variant("APG-000", base_channels=64)
    baseline_fake_parameters = count_parameters(baseline)
    for variant_id in ABLATION_VARIANT_IDS:
        variant = get_ablation_variant(variant_id)
        model = build_variant(variant_id, base_channels=64)
        expected_real_parameters = (
            BASELINE_EXPECTED_PARAMETERS + count_parameters(model) - baseline_fake_parameters
        )
        assert expected_real_parameters == variant.expected_parameters
    assert (
        get_ablation_variant("CGDM-SPATIAL").expected_parameters
        == get_ablation_variant("APG-001").expected_parameters
    )


def test_anchor_wrappers_preserve_historical_state_dicts_and_forward_math():
    cases = (
        (
            DCNv4RestorationUNet,
            RegisteredDCNv4RestorationUNet,
            "APG-000",
        ),
        (
            DegradationAwareDCNv4RestorationUNet,
            RegisteredDegradationAwareDCNv4RestorationUNet,
            "APG-111",
        ),
    )
    fixed_input = torch.rand(1, 3, 16, 16)
    for legacy_class, registered_class, variant_id in cases:
        common = {
            "base_channels": 32,
            "bottleneck_type": "conv",
            "use_dcnv4": True,
            "dcnv4_cls": FakeDCNv4,
        }
        legacy = legacy_class(**common).eval()
        with torch.no_grad():
            nn.init.normal_(legacy.residual_head.out_conv.weight, std=1.0e-3)
        registered = registered_class(
            variant_metadata=get_ablation_variant(variant_id).checkpoint_metadata(),
            **common,
        ).eval()
        assert tuple(legacy.state_dict()) == tuple(registered.state_dict())
        registered.load_state_dict(legacy.state_dict(), strict=True)
        with torch.no_grad():
            legacy_output = legacy(fixed_input)
            registered_output = registered(fixed_input)
        torch.testing.assert_close(
            registered_output,
            legacy_output,
            rtol=0.0,
            atol=0.0,
        )


def test_prompt_context_injection_and_neutral_intervention_are_available():
    model = build_variant("APG-010").eval()
    with torch.no_grad():
        nn.init.normal_(model.residual_head.out_conv.weight, std=1.0e-3)
        for module in model.modules():
            if isinstance(module, PromptFeatureModulation):
                module.affine.bias.fill_(0.25)
            if isinstance(module, ContextGatedDCNv4FeatureBlock):
                if module.use_input_affine:
                    module.input_affine.bias.fill_(0.25)
                if module.use_output_gate:
                    module.output_gate.bias.fill_(0.25)
    degraded = torch.rand(1, 3, 16, 16)
    prompts, global_context = model.extract_degradation_context(degraded)
    with torch.no_grad():
        correct = model.forward_with_degradation_context(
            degraded,
            prompts=prompts,
            global_context=global_context,
        )
        neutral = model.forward_with_degradation_context(
            degraded,
            prompts=prompts,
            global_context=global_context,
            neutral_conditioning=True,
        )
    assert not torch.allclose(correct, neutral)


def test_builder_rejects_registry_tampering():
    config = run_config("APG-100")
    config["model"]["skip_fusion"] = "concat"
    with pytest.raises(RuntimeError, match="frozen ablation registry"):
        build_dcnv4_unet(config)

    config = run_config("APG-100")
    config["ablation_registry"]["APG-100"]["skip_fusion"] = "concat"
    with pytest.raises(RuntimeError, match="registry snapshot"):
        build_dcnv4_unet(config)

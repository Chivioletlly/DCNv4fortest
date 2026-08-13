"""Frozen AIO3-v1 DCNv4 ablation registry.

The registry is the single source of truth for model topology, checkpoint
identity, output layout, and experiment tracking tags.  The public IDs match
``docs/AIO3_DCNV4_ABLATION_PLAN.md`` exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple


ABLATION_ARCHITECTURE_VERSION = 5
ABLATION_REGISTRY_VERSION = "aio3_dcnv4_ablation_plan_v1"
BASELINE_ARCHITECTURE_VERSION = 3
DEGRADATION_AWARE_ARCHITECTURE_VERSION = 4


@dataclass(frozen=True)
class AblationVariant:
    variant_id: str
    factor_a: bool
    factor_p: bool
    factor_g: bool
    convolution_film: bool
    dcnv4_input_affine: bool
    dcnv4_output_gate: bool
    skip_fusion: str
    bottleneck_modulation: str
    degradation_context: str
    model_name: str
    architecture_version: int
    expected_parameters: int
    autocast: str

    @property
    def output_directory(self) -> str:
        return f"dcnv4_ablation/{self.variant_id}"

    @property
    def wandb_tags(self) -> Tuple[str, ...]:
        return (
            "aio3-v1",
            "dcnv4",
            "ablation",
            self.variant_id.lower(),
            "signed-residual",
        )

    @property
    def uses_stage_prompts(self) -> bool:
        return self.degradation_context in {
            "stage_prompts",
            "stage_prompts_and_global_context",
        }

    @property
    def uses_global_context(self) -> bool:
        return self.degradation_context in {
            "global_context",
            "stage_prompts_and_global_context",
        }

    @property
    def dcnv4_conditioning(self) -> str:
        if self.dcnv4_input_affine and self.dcnv4_output_gate:
            return "input_affine_and_output_gate"
        if self.dcnv4_input_affine:
            return "input_affine"
        if self.dcnv4_output_gate:
            return "output_gate"
        return "none"

    def model_config(self) -> Dict[str, object]:
        value = asdict(self)
        value.update(
            {
                "name": self.model_name,
                "ablation_registry_version": ABLATION_REGISTRY_VERSION,
                "a": self.factor_a,
                "p": self.factor_p,
                "g": self.factor_g,
                "dcnv4_conditioning": self.dcnv4_conditioning,
                "output_directory": self.output_directory,
                "wandb_tags": list(self.wandb_tags),
            }
        )
        value.pop("factor_a")
        value.pop("factor_p")
        value.pop("factor_g")
        value.pop("model_name")
        return value

    def checkpoint_metadata(self) -> Dict[str, object]:
        return {
            "architecture_version": self.architecture_version,
            "ablation_registry_version": ABLATION_REGISTRY_VERSION,
            "model_name": self.model_name,
            "variant_id": self.variant_id,
            "a": self.factor_a,
            "p": self.factor_p,
            "g": self.factor_g,
            "convolution_film": self.convolution_film,
            "dcnv4_conditioning": self.dcnv4_conditioning,
            "skip_fusion_type": self.skip_fusion,
            "bottleneck_modulation_type": self.bottleneck_modulation,
            "degradation_context_type": self.degradation_context,
            "expected_trainable_parameters": self.expected_parameters,
            "autocast": self.autocast,
        }


def _variant(
    variant_id: str,
    *,
    a: bool = False,
    p: bool = False,
    g: bool = False,
    film: bool = False,
    dcn_in: bool = False,
    dcn_out: bool = False,
    skip: str = "concat",
    bottleneck: str = "none",
    context: str = "none",
    model_name: str = "dcnv4_ablation_unet",
    architecture_version: int = ABLATION_ARCHITECTURE_VERSION,
    expected_parameters: int = 0,
) -> AblationVariant:
    autocast = "bf16_network_fp32_dcnv4"
    if bottleneck in {"context_gated_dual_domain", "static_dual_domain"}:
        autocast += "_fft"
    return AblationVariant(
        variant_id=variant_id,
        factor_a=a,
        factor_p=p,
        factor_g=g,
        convolution_film=film,
        dcnv4_input_affine=dcn_in,
        dcnv4_output_gate=dcn_out,
        skip_fusion=skip,
        bottleneck_modulation=bottleneck,
        degradation_context=context,
        model_name=model_name,
        architecture_version=architecture_version,
        expected_parameters=expected_parameters,
        autocast=autocast,
    )


# Expected parameter counts are frozen after constructing every variant with
# base_channels=64 and the official DCNv4 operator.  They are deliberately
# constants rather than values computed at runtime.
_VARIANTS = (
    _variant(
        "APG-000",
        model_name="dcnv4_restoration_unet",
        architecture_version=BASELINE_ARCHITECTURE_VERSION,
        expected_parameters=29_924_411,
    ),
    _variant(
        "APG-100",
        a=True,
        skip="adaptive_spatial_channel_gate",
        expected_parameters=29_716_763,
    ),
    _variant(
        "APG-010",
        p=True,
        film=True,
        dcn_in=True,
        dcn_out=True,
        context="stage_prompts",
        expected_parameters=31_577_051,
    ),
    _variant(
        "APG-001",
        g=True,
        bottleneck="context_gated_dual_domain",
        context="global_context",
        expected_parameters=33_552_507,
    ),
    _variant(
        "APG-110",
        a=True,
        p=True,
        film=True,
        dcn_in=True,
        dcn_out=True,
        skip="adaptive_spatial_channel_gate",
        context="stage_prompts",
        expected_parameters=31_369_403,
    ),
    _variant(
        "APG-101",
        a=True,
        g=True,
        skip="adaptive_spatial_channel_gate",
        bottleneck="context_gated_dual_domain",
        context="global_context",
        expected_parameters=33_344_859,
    ),
    _variant(
        "APG-011",
        p=True,
        g=True,
        film=True,
        dcn_in=True,
        dcn_out=True,
        bottleneck="context_gated_dual_domain",
        context="stage_prompts_and_global_context",
        expected_parameters=35_060_187,
    ),
    _variant(
        "APG-111",
        a=True,
        p=True,
        g=True,
        film=True,
        dcn_in=True,
        dcn_out=True,
        skip="adaptive_spatial_channel_gate",
        bottleneck="context_gated_dual_domain",
        context="stage_prompts_and_global_context",
        model_name="degradation_aware_dcnv4_restoration_unet",
        architecture_version=DEGRADATION_AWARE_ARCHITECTURE_VERSION,
        expected_parameters=34_852_539,
    ),
    _variant(
        "P-FILM",
        p=True,
        film=True,
        context="stage_prompts",
        expected_parameters=30_535_931,
    ),
    _variant(
        "P-DIN",
        p=True,
        dcn_in=True,
        context="stage_prompts",
        expected_parameters=31_048_763,
    ),
    _variant(
        "P-DOUT",
        p=True,
        dcn_out=True,
        context="stage_prompts",
        expected_parameters=30_701_723,
    ),
    _variant(
        "P-DIO",
        p=True,
        dcn_in=True,
        dcn_out=True,
        context="stage_prompts",
        expected_parameters=31_395_803,
    ),
    _variant(
        "SKIP-PROJ-000",
        skip="concat_projection",
        expected_parameters=29_322_747,
    ),
    _variant(
        "SKIP-PROJ-011",
        p=True,
        g=True,
        film=True,
        dcn_in=True,
        dcn_out=True,
        skip="concat_projection",
        bottleneck="context_gated_dual_domain",
        context="stage_prompts_and_global_context",
        expected_parameters=34_458_523,
    ),
    _variant(
        "CGDM-STATIC",
        g=True,
        bottleneck="static_dual_domain",
        expected_parameters=32_817_211,
    ),
    _variant(
        "CGDM-SPATIAL",
        g=True,
        bottleneck="context_gated_spatial",
        context="global_context",
        expected_parameters=33_552_507,
    ),
)

ABLATION_REGISTRY: Mapping[str, AblationVariant] = {
    variant.variant_id: variant for variant in _VARIANTS
}
ABLATION_VARIANT_IDS: Tuple[str, ...] = tuple(ABLATION_REGISTRY)
LEGACY_VARIANT_ALIASES: Mapping[str, str] = {
    "baseline": "APG-000",
    "degradation-aware": "APG-111",
}
MODEL_VARIANT_CHOICES: Tuple[str, ...] = (
    *ABLATION_VARIANT_IDS,
    *LEGACY_VARIANT_ALIASES,
)


if len(ABLATION_REGISTRY) != 16:
    raise RuntimeError("The frozen AIO3-v1 ablation registry must contain 16 variants")


def normalize_variant_id(variant_id: str) -> str:
    canonical = LEGACY_VARIANT_ALIASES.get(variant_id, variant_id)
    if canonical not in ABLATION_REGISTRY:
        raise ValueError(
            f"model_variant must be one of {MODEL_VARIANT_CHOICES}, got {variant_id!r}"
        )
    return canonical


def get_ablation_variant(variant_id: str) -> AblationVariant:
    return ABLATION_REGISTRY[normalize_variant_id(variant_id)]


def registry_snapshot() -> Dict[str, Dict[str, object]]:
    return {variant_id: variant.model_config() for variant_id, variant in ABLATION_REGISTRY.items()}

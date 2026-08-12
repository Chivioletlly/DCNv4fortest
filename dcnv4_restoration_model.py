"""DCNv4 U-Net models for all-in-one image restoration.

The frozen baseline intentionally lives beside the legacy decomposition model
instead of replacing it. The legacy network predicts an independently constrained
degradation pattern and background; the restoration models predict one
unconstrained, signed RGB residual and add it to the degraded input. A separate
degradation-aware variant adapts the baseline without changing its behavior or
checkpoint contract.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
from einops import rearrange

try:
    from .image_geometry import crop_to_size, pad_to_multiple
except ImportError:  # Support running this file directly from the repository root.
    from image_geometry import crop_to_size, pad_to_multiple


ARCHITECTURE_VERSION = 3
MODEL_NAME = "dcnv4_restoration_unet"
OUTPUT_MODE = "signed_residual"
NORMALIZATION_TYPE = "groupnorm"
DCNV4_BLOCK_TYPE = "dcnv4"

DEGRADATION_AWARE_ARCHITECTURE_VERSION = 4
DEGRADATION_AWARE_MODEL_NAME = "degradation_aware_dcnv4_restoration_unet"
DEGRADATION_CONTEXT_TYPE = "multi_scale_mean_std_prompt"
SKIP_FUSION_TYPE = "adaptive_spatial_channel_gate"
BOTTLENECK_MODULATION_TYPE = "context_gated_dual_domain"

try:
    from DCNv4.modules.dcnv4 import DCNv4 as _DCNv4
# Some DCNv4 releases query CUDA device properties while importing. Defer
# missing-driver and binary-runtime failures so CPU structure tests can still
# inject a stand-in operator; constructing a real DCNv4 block will re-raise the
# original failure with the build guidance below.
except (ImportError, OSError, RuntimeError) as exc:
    _DCNV4_IMPORT_ERROR = exc
    _DCNv4 = None
else:
    _DCNV4_IMPORT_ERROR = None


def group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Build GroupNorm without assuming a fixed channel count."""

    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise ValueError(f"Could not select GroupNorm groups for {channels} channels")


def restoration_checkpoint_metadata() -> Dict[str, object]:
    """Metadata that must accompany checkpoints from this architecture."""

    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "model_name": MODEL_NAME,
        "output_mode": OUTPUT_MODE,
        "normalization": NORMALIZATION_TYPE,
        "direction_block_type": DCNV4_BLOCK_TYPE,
    }


def degradation_aware_checkpoint_metadata() -> Dict[str, object]:
    """Metadata shared by degradation-aware DCNv4 restoration checkpoints."""

    return {
        "architecture_version": DEGRADATION_AWARE_ARCHITECTURE_VERSION,
        "model_name": DEGRADATION_AWARE_MODEL_NAME,
        "output_mode": OUTPUT_MODE,
        "normalization": NORMALIZATION_TYPE,
        "direction_block_type": DCNV4_BLOCK_TYPE,
        "degradation_context_type": DEGRADATION_CONTEXT_TYPE,
        "skip_fusion_type": SKIP_FUSION_TYPE,
        "bottleneck_modulation_type": BOTTLENECK_MODULATION_TYPE,
    }


def validate_restoration_checkpoint(
    config: Dict[str, object],
    expected: Optional[Dict[str, object]] = None,
) -> None:
    """Reject decomposition or otherwise incompatible checkpoints."""

    expected = dict(expected or restoration_checkpoint_metadata())
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in mismatches.items()
        )
        raise RuntimeError(
            f"Checkpoint is not compatible with {expected['model_name']}: " + details
        )


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attention, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        x = x + attention
        return x + self.mlp(self.norm2(x))


class TransformerBottleneck(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_layers: int = 4,
        max_height: int = 128,
        max_width: int = 128,
    ):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.to_patch = nn.Conv2d(channels, channels, 1)
        self.layers = nn.ModuleList(
            [TransformerBlock(channels, num_heads) for _ in range(num_layers)]
        )
        self.pos_h = nn.Parameter(torch.randn(1, max_height, 1, channels))
        self.pos_w = nn.Parameter(torch.randn(1, 1, max_width, channels))
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height > self.max_height or width > self.max_width:
            raise ValueError(
                f"Feature map {height}x{width} exceeds transformer maximum "
                f"{self.max_height}x{self.max_width}"
            )
        x = self.to_patch(x)
        sequence = rearrange(x, "b c h w -> b (h w) c")
        position = self.pos_h[:, :height] + self.pos_w[:, :, :width]
        sequence = sequence + rearrange(position, "1 h w c -> 1 (h w) c")
        for layer in self.layers:
            sequence = layer(sequence)
        x = rearrange(sequence, "b (h w) c -> b c h w", h=height, w=width)
        return self.out_proj(x)


class DCNv4FeatureBlock(nn.Module):
    """Residual DCNv4 feature block using GroupNorm.

    The official DCNv4 CUDA backward kernel supports FP32 and FP16, but not
    BF16. The DCNv4 operator therefore always runs locally with autocast
    disabled and FP32 inputs, then its output is cast back to the surrounding
    feature dtype. This is required even when the incoming feature is FP32:
    an outer BF16 autocast context would otherwise cast DCNv4's internal
    Linear layers to BF16. Parameters remain FP32 as required by normal AMP
    training; callers should not convert the whole model to BF16.
    """

    def __init__(
        self,
        channels: int,
        dcnv4_cls: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        operator = dcnv4_cls or _DCNv4
        if operator is None:
            raise RuntimeError(
                "DCNv4 is required when use_dcnv4=True. Build the vendored CUDA "
                "extension with: bash scripts/build_dcnv4.sh"
            ) from _DCNV4_IMPORT_ERROR

        self.channels = channels
        self.group = self._select_group(channels)
        self.dcn = operator(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=self.group,
            offset_scale=1.0,
            dw_kernel_size=3,
            center_feature_scale=False,
            remove_center=False,
            output_bias=True,
            without_pointwise=False,
        )
        self._initialize_aggregation_bias()
        self.norm = group_norm(channels)
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def _select_group(channels: int) -> int:
        for channels_per_group in (32, 16):
            if channels % channels_per_group == 0:
                return channels // channels_per_group
        raise ValueError(
            "DCNv4 feature channels must be divisible by 32 or 16, "
            f"got {channels}"
        )

    def _initialize_aggregation_bias(self) -> None:
        """Match the stable regular-grid initialization used by the legacy model."""

        kernel_points = 9
        values_per_group = kernel_points * 3
        offset_mask = getattr(self.dcn, "offset_mask", None)
        bias = getattr(offset_mask, "bias", None)
        if bias is None:
            raise AttributeError("DCNv4 operator must expose offset_mask.bias")
        with torch.no_grad():
            bias.zero_()
            for group_index in range(self.group):
                mask_start = group_index * values_per_group + kernel_points * 2
                bias[mask_start : mask_start + kernel_points].fill_(
                    1.0 / kernel_points
                )

    def _apply_dcn(
        self,
        sequence: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Run DCNv4 in FP32 without inheriting the surrounding AMP context."""

        parameter = next(self.dcn.parameters(), None)
        if parameter is not None and parameter.dtype != torch.float32:
            raise RuntimeError(
                "DCNv4 AMP compatibility requires FP32 master parameters. "
                "Keep the model in FP32 and use torch.autocast instead of "
                "casting the model parameters to a reduced precision dtype."
            )

        feature_dtype = sequence.dtype
        with torch.autocast(device_type=sequence.device.type, enabled=False):
            output = self.dcn(sequence.float(), shape=(height, width))
        return output.to(dtype=feature_dtype)

    def _dcn_feature(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected NCHW input with {self.channels} channels, "
                f"got {tuple(x.shape)}"
            )
        batch, channels, height, width = x.shape
        sequence = (
            x.permute(0, 2, 3, 1)
            .reshape(batch, height * width, channels)
            .contiguous()
        )
        output = self._apply_dcn(sequence, height, width)
        output = (
            output.reshape(batch, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._dcn_feature(x)
        return x + self.activation(self.norm(output))


class ContextGatedDCNv4FeatureBlock(DCNv4FeatureBlock):
    """Condition DCNv4 sampling and residual propagation on an image prompt.

    DCNv4 predicts offsets and aggregation weights from its input features. A
    prompt-conditioned affine transform therefore makes that prediction
    degradation-aware without changing the CUDA operator. A second channel gate
    controls how much of the resulting DCNv4 response is propagated. Both
    projections are zero-initialized, so the block initially behaves exactly like
    :class:`DCNv4FeatureBlock` and can be introduced without an abrupt scale shift.
    """

    def __init__(
        self,
        channels: int,
        context_channels: int,
        dcnv4_cls: Optional[Type[nn.Module]] = None,
    ):
        super().__init__(channels=channels, dcnv4_cls=dcnv4_cls)
        if context_channels <= 0:
            raise ValueError("context_channels must be positive")
        self.context_channels = context_channels
        self.input_affine = nn.Linear(context_channels, channels * 2)
        self.output_gate = nn.Linear(context_channels, channels)
        self.reset_context_parameters()

    def reset_context_parameters(self) -> None:
        for layer in (self.input_affine, self.output_gate):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape != (
            x.shape[0],
            self.context_channels,
        ):
            raise ValueError(
                "Expected degradation context shaped "
                f"({x.shape[0]}, {self.context_channels}), got {tuple(context.shape)}"
            )
        scale, shift = self.input_affine(context).chunk(2, dim=1)
        scale = torch.tanh(scale).unsqueeze(-1).unsqueeze(-1)
        shift = torch.tanh(shift).unsqueeze(-1).unsqueeze(-1)
        conditioned = x * (1.0 + scale) + shift
        response = self.activation(self.norm(self._dcn_feature(conditioned)))
        gate = 2.0 * torch.sigmoid(self.output_gate(context))
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        return x + response * gate


class PromptFeatureModulation(nn.Module):
    """Identity-initialized channel-wise affine modulation from a prompt."""

    def __init__(self, channels: int, context_channels: int):
        super().__init__()
        self.channels = channels
        self.context_channels = context_channels
        self.affine = nn.Linear(context_channels, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected NCHW input with {self.channels} channels, got {tuple(x.shape)}"
            )
        if context.ndim != 2 or context.shape != (
            x.shape[0],
            self.context_channels,
        ):
            raise ValueError(
                "Expected degradation context shaped "
                f"({x.shape[0]}, {self.context_channels}), got {tuple(context.shape)}"
            )
        scale, shift = self.affine(context).chunk(2, dim=1)
        scale = torch.tanh(scale).unsqueeze(-1).unsqueeze(-1)
        shift = torch.tanh(shift).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + scale) + shift


class ConvBottleneck(nn.Module):
    def __init__(self, channels: int, num_layers: int = 4):
        super().__init__()
        dilation_rates = [1, 2, 4, 2][:num_layers]
        self.layers = nn.ModuleList()
        for dilation in dilation_rates:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    group_norm(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        channels,
                        channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    group_norm(channels),
                )
            )
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return self.out_proj(x)


def convolution_block(
    in_channels: int,
    out_channels: int,
    dilation: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        ),
        group_norm(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        ),
        group_norm(out_channels),
        nn.ReLU(inplace=True),
    )


class DegradationAwareContext(nn.Module):
    """Infer label-free global and stage-wise degradation representations.

    The design follows the degradation-aware module in DACG-IR: parallel
    depth-wise 3/5/7 convolutions collect degradation evidence at different
    spatial scales, a learned spatial gate filters content-irrelevant responses,
    and mean/std pooling produces a compact global descriptor. Successive prompt
    projections align that descriptor with the four U-Net channel widths.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        prompt_channels: Sequence[int],
        num_scales: int = 3,
    ):
        super().__init__()
        if num_scales <= 0:
            raise ValueError("num_scales must be positive")
        if len(prompt_channels) != 4:
            raise ValueError("Exactly four prompt channel widths are required")
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.prompt_channels = tuple(int(value) for value in prompt_channels)
        self.num_scales = num_scales
        context_channels = base_channels * 2
        self.context_channels = context_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GELU(),
        )
        self.scale_branches = nn.ModuleList()
        for scale_index in range(num_scales):
            kernel_size = 2 * scale_index + 3
            self.scale_branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        base_channels,
                        base_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        groups=base_channels,
                    ),
                    nn.Conv2d(base_channels, base_channels, 1),
                )
            )
        self.fusion = nn.Conv2d(
            base_channels * num_scales,
            context_channels,
            1,
        )
        # The paper specifies a depth-wise 3x3 gate; this preserves local
        # degradation evidence that a point-wise gate cannot observe.
        self.spatial_gate = nn.Conv2d(
            context_channels,
            context_channels,
            3,
            padding=1,
            groups=context_channels,
        )
        self.global_process = nn.Sequential(
            nn.Linear(context_channels * 2, context_channels * 2),
            nn.LayerNorm(context_channels * 2),
            nn.GELU(),
            nn.Linear(context_channels * 2, context_channels),
        )

        self.prompt_layers = nn.ModuleList()
        current_channels = context_channels
        for output_channels in self.prompt_channels:
            self.prompt_layers.append(
                nn.Sequential(
                    nn.Linear(current_channels, current_channels),
                    nn.LayerNorm(current_channels),
                    nn.GELU(),
                    nn.Linear(current_channels, output_channels),
                )
            )
            current_channels = output_channels

    def forward(
        self,
        degraded: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        if degraded.ndim != 4 or degraded.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected NCHW input with {self.in_channels} channels, "
                f"got {tuple(degraded.shape)}"
            )
        stem = self.stem(degraded)
        multi_scale = torch.cat(
            [branch(stem) for branch in self.scale_branches],
            dim=1,
        )
        fused = self.fusion(multi_scale)
        gated = fused * torch.sigmoid(self.spatial_gate(fused))

        # Pool in FP32 so the variance descriptor stays stable under BF16 AMP.
        pooled = gated.float()
        mean = pooled.mean(dim=(2, 3))
        std = pooled.var(dim=(2, 3), correction=0).clamp_min(0.0).sqrt()
        global_context = self.global_process(torch.cat([mean, std], dim=1))

        prompts = []
        prompt = global_context
        for layer in self.prompt_layers:
            prompt = layer(prompt)
            prompts.append(prompt)
        return tuple(prompts), global_context


class AdaptiveGatedFusion(nn.Module):
    """Spatial-channel dual gate for filtering a U-Net encoder skip."""

    def __init__(self, channels: int):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            group_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        hidden_channels = max(channels // 2, 1)
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Linear(channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
        )

    def forward(
        self,
        encoder_feature: torch.Tensor,
        decoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        if encoder_feature.shape != decoder_feature.shape:
            raise ValueError(
                "Adaptive gated fusion requires matching encoder/decoder shapes, "
                f"got {tuple(encoder_feature.shape)} and {tuple(decoder_feature.shape)}"
            )
        if encoder_feature.ndim != 4 or encoder_feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected NCHW features with {self.channels} channels, "
                f"got {tuple(encoder_feature.shape)}"
            )
        combined = torch.cat([encoder_feature, decoder_feature], dim=1)
        spatial_logit = self.spatial_gate(combined)
        pooled = self.channel_pool(combined).flatten(1)
        channel_logit = self.channel_gate(pooled).unsqueeze(-1).unsqueeze(-1)
        skip_gate = torch.sigmoid(spatial_logit + channel_logit)
        filtered_encoder = encoder_feature * skip_gate
        return self.fusion(torch.cat([filtered_encoder, decoder_feature], dim=1))


class ContextGatedDualDomainModulation(nn.Module):
    """Prompt-gated spatial/frequency refinement at the low-resolution latent."""

    def __init__(self, channels: int, context_channels: int):
        super().__init__()
        self.channels = channels
        self.context_channels = context_channels
        self.frequency_mixer = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 1),
        )
        self.context_mapper = nn.Sequential(
            nn.Linear(context_channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels * 2),
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.fusion = nn.Conv2d(channels * 2, channels, 1)

    def forward(
        self,
        feature: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected NCHW input with {self.channels} channels, "
                f"got {tuple(feature.shape)}"
            )
        if global_context.ndim != 2 or global_context.shape != (
            feature.shape[0],
            self.context_channels,
        ):
            raise ValueError(
                "Expected global context shaped "
                f"({feature.shape[0]}, {self.context_channels}), "
                f"got {tuple(global_context.shape)}"
            )
        spatial_feature = self.spatial_branch(feature)
        height, width = feature.shape[-2:]

        # CUDA FFT and complex reconstruction are kept in FP32 even when the
        # surrounding U-Net runs under BF16 autocast.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            spectrum = torch.fft.rfft2(feature.float(), norm="ortho")
            frequency = torch.cat([spectrum.real, spectrum.imag], dim=1)
            frequency = self.frequency_mixer(frequency)
            frequency_gate = torch.sigmoid(
                self.context_mapper(global_context.float())
            ).unsqueeze(-1).unsqueeze(-1)
            frequency = frequency * frequency_gate
            real, imaginary = frequency.chunk(2, dim=1)
            frequency_feature = torch.fft.irfft2(
                torch.complex(real, imaginary),
                s=(height, width),
                norm="ortho",
            )

        frequency_feature = frequency_feature.to(dtype=spatial_feature.dtype)
        fused = self.fusion(
            torch.cat([spatial_feature, frequency_feature], dim=1)
        )
        return feature + fused


class UNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        use_dcnv4: bool,
        dcnv4_cls: Optional[Type[nn.Module]],
    ):
        super().__init__()
        channels = base_channels
        self.use_dcnv4 = use_dcnv4
        self.enc1 = convolution_block(in_channels, channels, dilation=1)
        self.enc2 = convolution_block(channels, channels * 2, dilation=2)
        self.enc3 = convolution_block(channels * 2, channels * 4, dilation=4)
        self.enc4 = convolution_block(channels * 4, channels * 8, dilation=2)
        self.pool = nn.MaxPool2d(2)

        if use_dcnv4:
            self.dcn2 = DCNv4FeatureBlock(channels * 2, dcnv4_cls=dcnv4_cls)
            self.dcn3 = DCNv4FeatureBlock(channels * 4, dcnv4_cls=dcnv4_cls)
            self.dcn4 = DCNv4FeatureBlock(channels * 8, dcnv4_cls=dcnv4_cls)

    def forward(self, x: torch.Tensor):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        if self.use_dcnv4:
            enc2 = self.dcn2(enc2)
        enc3 = self.enc3(self.pool(enc2))
        if self.use_dcnv4:
            enc3 = self.dcn3(enc3)
        enc4 = self.enc4(self.pool(enc3))
        if self.use_dcnv4:
            enc4 = self.dcn4(enc4)
        return enc1, enc2, enc3, enc4


class UNetDecoder(nn.Module):
    def __init__(self, base_channels: int):
        super().__init__()
        channels = base_channels
        self.up4 = nn.ConvTranspose2d(
            channels * 8,
            channels * 4,
            4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.up3 = nn.ConvTranspose2d(
            channels * 4,
            channels * 2,
            4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.up2 = nn.ConvTranspose2d(
            channels * 2,
            channels,
            4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.dec4 = convolution_block(channels * 8, channels * 4)
        self.dec3 = convolution_block(channels * 4, channels * 2)
        self.dec2 = convolution_block(channels * 2, channels)

    def forward(self, bottleneck: torch.Tensor, encoder_features) -> torch.Tensor:
        enc1, enc2, enc3 = encoder_features
        decoded = self.dec4(torch.cat([self.up4(bottleneck), enc3], dim=1))
        decoded = self.dec3(torch.cat([self.up3(decoded), enc2], dim=1))
        return self.dec2(torch.cat([self.up2(decoded), enc1], dim=1))


class SignedResidualHead(nn.Module):
    """Map decoder features to an unconstrained RGB correction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dcnv4: bool,
        dcnv4_cls: Optional[Type[nn.Module]],
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            group_norm(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.dcn = (
            DCNv4FeatureBlock(hidden_channels, dcnv4_cls=dcnv4_cls)
            if use_dcnv4
            else nn.Identity()
        )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        self.reset_output_parameters()

    def reset_output_parameters(self) -> None:
        """Start the full network as an exact identity mapping."""

        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_conv(self.dcn(self.pre(x)))


class DegradationAwareUNetEncoder(nn.Module):
    """U-Net encoder whose DCNv4 responses are conditioned by stage prompts."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        use_dcnv4: bool,
        dcnv4_cls: Optional[Type[nn.Module]],
    ):
        super().__init__()
        channels = base_channels
        self.use_dcnv4 = use_dcnv4
        self.enc1 = convolution_block(in_channels, channels, dilation=1)
        self.enc2 = convolution_block(channels, channels * 2, dilation=2)
        self.enc3 = convolution_block(channels * 2, channels * 4, dilation=4)
        self.enc4 = convolution_block(channels * 4, channels * 8, dilation=2)
        self.pool = nn.MaxPool2d(2)
        self.mod1 = PromptFeatureModulation(channels, channels)

        block_class = (
            ContextGatedDCNv4FeatureBlock if use_dcnv4 else PromptFeatureModulation
        )
        if use_dcnv4:
            self.mod2 = block_class(
                channels * 2,
                channels * 2,
                dcnv4_cls=dcnv4_cls,
            )
            self.mod3 = block_class(
                channels * 4,
                channels * 4,
                dcnv4_cls=dcnv4_cls,
            )
            self.mod4 = block_class(
                channels * 8,
                channels * 8,
                dcnv4_cls=dcnv4_cls,
            )
        else:
            self.mod2 = block_class(channels * 2, channels * 2)
            self.mod3 = block_class(channels * 4, channels * 4)
            self.mod4 = block_class(channels * 8, channels * 8)

    def forward(
        self,
        x: torch.Tensor,
        prompts: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(prompts) != 4:
            raise ValueError("The degradation-aware encoder requires four prompts")
        prompt1, prompt2, prompt3, prompt4 = prompts
        enc1 = self.mod1(self.enc1(x), prompt1)
        enc2 = self.mod2(self.enc2(self.pool(enc1)), prompt2)
        enc3 = self.mod3(self.enc3(self.pool(enc2)), prompt3)
        enc4 = self.mod4(self.enc4(self.pool(enc3)), prompt4)
        return enc1, enc2, enc3, enc4


class DegradationAwareUNetDecoder(nn.Module):
    """Decoder with degradation-filtering skip fusion and layer prompts."""

    def __init__(self, base_channels: int):
        super().__init__()
        channels = base_channels
        self.up4 = nn.ConvTranspose2d(
            channels * 8,
            channels * 4,
            4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.up3 = nn.ConvTranspose2d(
            channels * 4,
            channels * 2,
            4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.up2 = nn.ConvTranspose2d(
            channels * 2,
            channels,
            4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.skip4 = AdaptiveGatedFusion(channels * 4)
        self.skip3 = AdaptiveGatedFusion(channels * 2)
        self.skip2 = AdaptiveGatedFusion(channels)
        self.dec4 = convolution_block(channels * 4, channels * 4)
        self.dec3 = convolution_block(channels * 2, channels * 2)
        self.dec2 = convolution_block(channels, channels)
        self.mod4 = PromptFeatureModulation(channels * 4, channels * 4)
        self.mod3 = PromptFeatureModulation(channels * 2, channels * 2)
        self.mod2 = PromptFeatureModulation(channels, channels)

    def forward(
        self,
        bottleneck: torch.Tensor,
        encoder_features: Sequence[torch.Tensor],
        prompts: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(encoder_features) != 3 or len(prompts) != 3:
            raise ValueError(
                "The degradation-aware decoder requires three skips and prompts"
            )
        enc1, enc2, enc3 = encoder_features
        prompt1, prompt2, prompt3 = prompts
        decoded = self.skip4(enc3, self.up4(bottleneck))
        decoded = self.mod4(self.dec4(decoded), prompt3)
        decoded = self.skip3(enc2, self.up3(decoded))
        decoded = self.mod3(self.dec3(decoded), prompt2)
        decoded = self.skip2(enc1, self.up2(decoded))
        return self.mod2(self.dec2(decoded), prompt1)


class DegradationAwareSignedResidualHead(nn.Module):
    """Identity-initialized signed residual head conditioned by a prompt."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_channels: int,
        use_dcnv4: bool,
        dcnv4_cls: Optional[Type[nn.Module]],
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.use_dcnv4 = use_dcnv4
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            group_norm(hidden_channels),
            nn.ReLU(inplace=True),
        )
        if use_dcnv4:
            self.context_block = ContextGatedDCNv4FeatureBlock(
                hidden_channels,
                context_channels,
                dcnv4_cls=dcnv4_cls,
            )
        else:
            self.context_block = PromptFeatureModulation(
                hidden_channels,
                context_channels,
            )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        self.reset_output_parameters()

    def reset_output_parameters(self) -> None:
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.out_conv(self.context_block(self.pre(x), context))


class DCNv4RestorationUNet(nn.Module):
    """Degradation-agnostic U-Net with four optional DCNv4 feature blocks."""

    architecture_version = ARCHITECTURE_VERSION
    output_mode = OUTPUT_MODE
    normalization_type = NORMALIZATION_TYPE

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        bottleneck_type: str = "conv",
        use_dcnv4: bool = True,
        dcnv4_cls: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        if base_channels % 32 != 0:
            raise ValueError(
                "base_channels must be divisible by 32 so all four DCNv4 blocks "
                f"have valid group widths, got {base_channels}"
            )
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.bottleneck_type = bottleneck_type
        self.use_dcnv4 = use_dcnv4

        self.encoder = UNetEncoder(
            in_channels,
            base_channels,
            use_dcnv4=use_dcnv4,
            dcnv4_cls=dcnv4_cls,
        )
        if bottleneck_type == "conv":
            self.bottleneck = ConvBottleneck(base_channels * 8)
        elif bottleneck_type == "transformer":
            self.bottleneck = TransformerBottleneck(base_channels * 8)
        else:
            raise ValueError(f"Unknown bottleneck_type: {bottleneck_type}")
        self.decoder = UNetDecoder(base_channels)
        self.residual_head = SignedResidualHead(
            base_channels,
            in_channels,
            use_dcnv4=use_dcnv4,
            dcnv4_cls=dcnv4_cls,
        )

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        if degraded.ndim != 4 or degraded.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected NCHW input with {self.in_channels} channels, "
                f"got {tuple(degraded.shape)}"
            )
        padded, original_size = pad_to_multiple(degraded)
        encoder_features = self.encoder(padded)
        bottleneck = self.bottleneck(encoder_features[-1])
        decoded = self.decoder(bottleneck, encoder_features[:-1])
        signed_residual = self.residual_head(decoded)
        restored = padded + signed_residual
        return crop_to_size(restored, original_size)

    def checkpoint_metadata(self) -> Dict[str, object]:
        metadata = restoration_checkpoint_metadata()
        metadata.update(
            {
                "in_channels": self.in_channels,
                "base_channels": self.base_channels,
                "bottleneck_type": self.bottleneck_type,
                "use_dcnv4": self.use_dcnv4,
            }
        )
        return metadata


class DegradationAwareDCNv4RestorationUNet(nn.Module):
    """DCNv4 U-Net with label-free degradation-aware context gating.

    The original signed-residual model is retained as a frozen baseline. This
    variant adds four complementary adaptations inspired by DACG-IR:

    * a multi-scale degradation context encoder;
    * prompt-conditioned DCNv4 sampling/response blocks;
    * spatial-channel gated U-Net skip connections; and
    * context-gated spatial/frequency modulation at the bottleneck.

    It still predicts one unconstrained signed RGB residual and starts as an
    exact identity mapping, preserving the AIO3 output and loss contract.
    """

    architecture_version = DEGRADATION_AWARE_ARCHITECTURE_VERSION
    output_mode = OUTPUT_MODE
    normalization_type = NORMALIZATION_TYPE

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        bottleneck_type: str = "conv",
        use_dcnv4: bool = True,
        context_scales: int = 3,
        dcnv4_cls: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        if base_channels % 32 != 0:
            raise ValueError(
                "base_channels must be divisible by 32 so all four DCNv4 blocks "
                f"have valid group widths, got {base_channels}"
            )
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.bottleneck_type = bottleneck_type
        self.use_dcnv4 = use_dcnv4
        self.context_scales = context_scales
        prompt_channels = tuple(base_channels * 2**index for index in range(4))

        self.degradation_context = DegradationAwareContext(
            in_channels=in_channels,
            base_channels=base_channels,
            prompt_channels=prompt_channels,
            num_scales=context_scales,
        )
        self.encoder = DegradationAwareUNetEncoder(
            in_channels,
            base_channels,
            use_dcnv4=use_dcnv4,
            dcnv4_cls=dcnv4_cls,
        )
        if bottleneck_type == "conv":
            self.bottleneck = ConvBottleneck(base_channels * 8)
        elif bottleneck_type == "transformer":
            self.bottleneck = TransformerBottleneck(base_channels * 8)
        else:
            raise ValueError(f"Unknown bottleneck_type: {bottleneck_type}")
        self.dual_domain_modulation = ContextGatedDualDomainModulation(
            channels=base_channels * 8,
            context_channels=base_channels * 2,
        )
        self.decoder = DegradationAwareUNetDecoder(base_channels)
        self.residual_head = DegradationAwareSignedResidualHead(
            base_channels,
            in_channels,
            context_channels=base_channels,
            use_dcnv4=use_dcnv4,
            dcnv4_cls=dcnv4_cls,
        )

    def _validate_input(self, degraded: torch.Tensor) -> None:
        if degraded.ndim != 4 or degraded.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected NCHW input with {self.in_channels} channels, "
                f"got {tuple(degraded.shape)}"
            )

    def extract_degradation_context(
        self,
        degraded: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Expose learned prompts for diagnostics without requiring task labels."""

        self._validate_input(degraded)
        padded, _ = pad_to_multiple(degraded)
        return self.degradation_context(padded)

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        self._validate_input(degraded)
        padded, original_size = pad_to_multiple(degraded)
        prompts, global_context = self.degradation_context(padded)
        encoder_features = self.encoder(padded, prompts)
        bottleneck = self.bottleneck(encoder_features[-1])
        bottleneck = self.dual_domain_modulation(bottleneck, global_context)
        decoded = self.decoder(
            bottleneck,
            encoder_features[:-1],
            prompts[:-1],
        )
        signed_residual = self.residual_head(decoded, prompts[0])
        return crop_to_size(padded + signed_residual, original_size)

    def checkpoint_metadata(self) -> Dict[str, object]:
        metadata = degradation_aware_checkpoint_metadata()
        metadata.update(
            {
                "in_channels": self.in_channels,
                "base_channels": self.base_channels,
                "bottleneck_type": self.bottleneck_type,
                "use_dcnv4": self.use_dcnv4,
                "context_scales": self.context_scales,
                "dcnv4_conditioning": "prompt_affine_and_output_gate",
            }
        )
        return metadata


def create_dcnv4_restoration_model(**kwargs) -> DCNv4RestorationUNet:
    return DCNv4RestorationUNet(**kwargs)


def create_degradation_aware_dcnv4_restoration_model(
    **kwargs,
) -> DegradationAwareDCNv4RestorationUNet:
    return DegradationAwareDCNv4RestorationUNet(**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

"""DCNv4 U-Net baseline for all-in-one image restoration.

This module intentionally lives beside the legacy decomposition model instead of
replacing it.  The legacy network predicts an independently constrained
degradation pattern and background; this baseline predicts one unconstrained,
signed RGB residual and adds it to the degraded input.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

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


def validate_restoration_checkpoint(config: Dict[str, object]) -> None:
    """Reject decomposition or otherwise incompatible checkpoints."""

    expected = restoration_checkpoint_metadata()
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
            "Checkpoint is not compatible with DCNv4RestorationUNet: " + details
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return x + self.activation(self.norm(output))


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


def create_dcnv4_restoration_model(**kwargs) -> DCNv4RestorationUNet:
    return DCNv4RestorationUNet(**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

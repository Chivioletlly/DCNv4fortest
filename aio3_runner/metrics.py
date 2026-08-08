"""Exact image-quality metrics and task aggregation for AIO3-v1."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .data import TASKS
from .protocol import NOISE_SIGMAS


SSIM_WINDOW_SIZE = 11
SSIM_WINDOW_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03
METRIC_DATA_RANGE = 1.0


def _validate_image_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError(
            "prediction and target must be NCHW tensors, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction.shape} != {target.shape}"
        )
    if prediction.shape[1] != 3:
        raise ValueError(f"AIO3-v1 metrics require RGB input, got {prediction.shape[1]} channels")
    prediction = prediction.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
    target = target.detach().to(device=prediction.device, dtype=torch.float32)
    return prediction, target


def rgb_psnr_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return RGB PSNR for each image after clamping prediction to [0, 1]."""

    prediction, target = _validate_image_pair(prediction, target)
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    data_range_squared = METRIC_DATA_RANGE * METRIC_DATA_RANGE
    return 10.0 * torch.log10(data_range_squared / mse)


def _gaussian_window(
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(
        SSIM_WINDOW_SIZE,
        device=device,
        dtype=dtype,
    )
    coordinates = coordinates - (SSIM_WINDOW_SIZE - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * SSIM_WINDOW_SIGMA**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, -1, -1).contiguous()


def rgb_ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return valid-window RGB SSIM for each image using the AIO3-v1 definition."""

    prediction, target = _validate_image_pair(prediction, target)
    height, width = prediction.shape[-2:]
    if height < SSIM_WINDOW_SIZE or width < SSIM_WINDOW_SIZE:
        raise ValueError(
            "SSIM requires both image dimensions to be at least "
            f"{SSIM_WINDOW_SIZE}, got {height}x{width}"
        )

    channels = prediction.shape[1]
    window = _gaussian_window(
        channels,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    mu_prediction = F.conv2d(prediction, window, groups=channels, padding=0)
    mu_target = F.conv2d(target, window, groups=channels, padding=0)

    mu_prediction_squared = mu_prediction.square()
    mu_target_squared = mu_target.square()
    mu_product = mu_prediction * mu_target

    prediction_variance = (
        F.conv2d(prediction.square(), window, groups=channels, padding=0)
        - mu_prediction_squared
    )
    target_variance = (
        F.conv2d(target.square(), window, groups=channels, padding=0)
        - mu_target_squared
    )
    covariance = (
        F.conv2d(prediction * target, window, groups=channels, padding=0)
        - mu_product
    )

    c1 = (SSIM_K1 * METRIC_DATA_RANGE) ** 2
    c2 = (SSIM_K2 * METRIC_DATA_RANGE) ** 2
    numerator = (2.0 * mu_product + c1) * (2.0 * covariance + c2)
    denominator = (
        (mu_prediction_squared + mu_target_squared + c1)
        * (prediction_variance + target_variance + c2)
    )
    ssim_map = numerator / denominator
    return ssim_map.flatten(1).mean(dim=1)


def rgb_metrics_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-image ``(PSNR, SSIM)`` tensors under the frozen protocol."""

    return rgb_psnr_per_image(prediction, target), rgb_ssim_per_image(
        prediction,
        target,
    )


@dataclass(frozen=True)
class PerImageMetric:
    sample_id: str
    task: str
    sigma: Optional[int]
    psnr: float
    ssim: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty metric group")
    return math.fsum(values) / len(values)


class AIO3MetricAccumulator:
    """Collect per-image metrics and produce the frozen task-balanced summary."""

    def __init__(self) -> None:
        self.rows: List[PerImageMetric] = []

    def add_values(
        self,
        *,
        sample_id: str,
        task: str,
        sigma: Optional[int],
        psnr: float,
        ssim: float,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"Unsupported AIO3 task: {task!r}")
        if task == "denoise":
            if sigma not in NOISE_SIGMAS:
                raise ValueError(f"Denoise sigma must be one of {NOISE_SIGMAS}, got {sigma}")
        elif sigma not in (None, -1):
            raise ValueError(f"{task} must not have a noise sigma, got {sigma}")
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        if math.isnan(psnr) or math.isnan(ssim):
            raise ValueError(f"NaN metric for {sample_id}")
        self.rows.append(
            PerImageMetric(
                sample_id=sample_id,
                task=task,
                sigma=int(sigma) if task == "denoise" else None,
                psnr=float(psnr),
                ssim=float(ssim),
            )
        )

    def add_batch(
        self,
        *,
        sample_ids: Sequence[str],
        tasks: Sequence[str],
        sigmas: Sequence[int],
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        batch_size = prediction.shape[0]
        if not (
            len(sample_ids) == len(tasks) == len(sigmas) == batch_size
        ):
            raise ValueError("Metric metadata lengths must equal prediction batch size")
        psnr, ssim = rgb_metrics_per_image(prediction, target)
        for index in range(batch_size):
            self.add_values(
                sample_id=str(sample_ids[index]),
                task=str(tasks[index]),
                sigma=int(sigmas[index]),
                psnr=float(psnr[index].item()),
                ssim=float(ssim[index].item()),
            )

    def summarize(self) -> Dict[str, float]:
        """Return flat metric keys suitable for JSON and W&B ``val/*`` logging."""

        grouped: Dict[str, List[PerImageMetric]] = {}
        for sigma in NOISE_SIGMAS:
            key = f"denoise/sigma{sigma}"
            grouped[key] = [
                row
                for row in self.rows
                if row.task == "denoise" and row.sigma == sigma
            ]
        grouped["derain"] = [row for row in self.rows if row.task == "derain"]
        grouped["dehaze"] = [row for row in self.rows if row.task == "dehaze"]

        missing = [key for key, rows in grouped.items() if not rows]
        if missing:
            raise ValueError(f"Metric summary is missing required groups: {missing}")

        summary: Dict[str, float] = {}
        for key, rows in grouped.items():
            summary[f"{key}/psnr"] = _mean([row.psnr for row in rows])
            summary[f"{key}/ssim"] = _mean([row.ssim for row in rows])
            summary[f"{key}/images"] = float(len(rows))

        for metric in ("psnr", "ssim"):
            denoise_value = _mean(
                [summary[f"denoise/sigma{sigma}/{metric}"] for sigma in NOISE_SIGMAS]
            )
            summary[f"denoise/mean/{metric}"] = denoise_value
            summary[f"macro/{metric}"] = _mean(
                [
                    denoise_value,
                    summary[f"derain/{metric}"],
                    summary[f"dehaze/{metric}"],
                ]
            )
        summary["images"] = float(len(self.rows))
        return summary

    def per_image_dicts(self) -> List[Mapping[str, object]]:
        return [row.to_dict() for row in self.rows]

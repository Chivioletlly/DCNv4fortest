"""Frozen RGB metrics and category-balanced aggregation for CDD-11."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Sequence

import torch

from aio3_runner.metrics import rgb_metrics_per_image, rgb_psnr_per_image, rgb_ssim_per_image

from .protocol import ARITY_GROUPS, DEGRADATION_ARITY, DEGRADATIONS


@dataclass(frozen=True)
class PerImageMetric:
    sample_id: str
    degradation: str
    arity: int
    psnr: float
    ssim: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty metric group")
    return math.fsum(values) / len(values)


class CDD11MetricAccumulator:
    """Collect per-image values and compute all frozen CDD-11 summaries."""

    def __init__(self) -> None:
        self.rows: List[PerImageMetric] = []

    def add_values(
        self,
        *,
        sample_id: str,
        degradation: str,
        arity: int,
        psnr: float,
        ssim: float,
    ) -> None:
        if degradation not in DEGRADATIONS:
            raise ValueError(f"Unsupported CDD-11 degradation: {degradation!r}")
        expected_arity = DEGRADATION_ARITY[degradation]
        if int(arity) != expected_arity:
            raise ValueError(
                f"Arity for {degradation!r} must be {expected_arity}, got {arity}"
            )
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        if math.isnan(psnr) or math.isnan(ssim):
            raise ValueError(f"NaN metric for {sample_id}")
        self.rows.append(
            PerImageMetric(
                sample_id=sample_id,
                degradation=degradation,
                arity=expected_arity,
                psnr=float(psnr),
                ssim=float(ssim),
            )
        )

    def add_batch(
        self,
        *,
        sample_ids: Sequence[str],
        degradations: Sequence[str],
        arities: Sequence[int],
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        batch_size = prediction.shape[0]
        if not (
            len(sample_ids) == len(degradations) == len(arities) == batch_size
        ):
            raise ValueError("Metric metadata lengths must equal prediction batch size")
        psnr, ssim = rgb_metrics_per_image(prediction, target)
        for index in range(batch_size):
            self.add_values(
                sample_id=str(sample_ids[index]),
                degradation=str(degradations[index]),
                arity=int(arities[index]),
                psnr=float(psnr[index].item()),
                ssim=float(ssim[index].item()),
            )

    def summarize(self) -> Dict[str, float]:
        grouped = {
            degradation: [
                row for row in self.rows if row.degradation == degradation
            ]
            for degradation in DEGRADATIONS
        }
        missing = [key for key, rows in grouped.items() if not rows]
        if missing:
            raise ValueError(f"Metric summary is missing categories: {missing}")

        summary: Dict[str, float] = {}
        for degradation, rows in grouped.items():
            summary[f"{degradation}/psnr"] = _mean([row.psnr for row in rows])
            summary[f"{degradation}/ssim"] = _mean([row.ssim for row in rows])
            summary[f"{degradation}/images"] = float(len(rows))

        for group_name, categories in ARITY_GROUPS.items():
            for metric in ("psnr", "ssim"):
                summary[f"{group_name}/{metric}"] = _mean(
                    [summary[f"{category}/{metric}"] for category in categories]
                )
            summary[f"{group_name}/categories"] = float(len(categories))

        for metric in ("psnr", "ssim"):
            summary[f"macro/{metric}"] = _mean(
                [summary[f"{degradation}/{metric}"] for degradation in DEGRADATIONS]
            )
        summary["images"] = float(len(self.rows))
        return summary

    def per_image_dicts(self) -> List[Mapping[str, object]]:
        return [row.to_dict() for row in self.rows]


__all__ = [
    "CDD11MetricAccumulator",
    "PerImageMetric",
    "rgb_metrics_per_image",
    "rgb_psnr_per_image",
    "rgb_ssim_per_image",
]

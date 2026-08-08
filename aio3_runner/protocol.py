"""Frozen constants for the AIO3-v1 experimental protocol."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Dict, Tuple


AIO3_PROTOCOL_VERSION = "aio3-v1"
NOISE_SIGMAS: Tuple[int, ...] = (15, 25, 50)


@dataclass(frozen=True)
class ProtocolExpectations:
    """Expected source counts and deterministic validation sizes.

    Production uses the defaults below. Tests may inject a smaller instance,
    but a production manifest must always be generated with the defaults.
    """

    bsd400_images: int = 400
    wed_gt_images: int = 4744
    wed_noisy_images: int = 4744
    bsd68_images: int = 68
    raintrain_input_images: int = 200
    raintrain_target_images: int = 200
    rain100_input_images: int = 100
    rain100_target_images: int = 100
    ots_clear_images: int = 2061
    ots_haze_images: int = 72135
    ots_haze_excluded_files: int = 4
    sots_input_images: int = 500
    sots_target_images: int = 492
    sots_paired_images: int = 500
    denoise_validation_scenes: int = 100
    derain_validation_scenes: int = 20
    dehaze_validation_scenes: int = 100

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


DEFAULT_EXPECTATIONS = ProtocolExpectations()


def stable_digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def deterministic_seed(text: str) -> int:
    """Return a stable non-negative seed accepted by torch.Generator."""

    return int.from_bytes(stable_digest(text)[:8], "big") & ((1 << 63) - 1)


def split_sort_key(task: str, scene_id: str) -> str:
    value = f"{AIO3_PROTOCOL_VERSION}:{task}:{scene_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def noise_seed(split: str, image_id: str, sigma: int) -> int:
    value = f"{AIO3_PROTOCOL_VERSION}:{split}:{image_id}:sigma{sigma}"
    return deterministic_seed(value)

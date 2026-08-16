"""Frozen constants and deterministic helpers for CDD-11-v1."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple


CDD11_PROTOCOL_VERSION = "cdd11-v1"
DEGRADATIONS: Tuple[str, ...] = (
    "low",
    "haze",
    "rain",
    "snow",
    "low_haze",
    "low_rain",
    "low_snow",
    "haze_rain",
    "haze_snow",
    "low_haze_rain",
    "low_haze_snow",
)
DEGRADATION_ARITY: Mapping[str, int] = {
    degradation: degradation.count("_") + 1 for degradation in DEGRADATIONS
}
ARITY_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "single": tuple(value for value in DEGRADATIONS if DEGRADATION_ARITY[value] == 1),
    "double": tuple(value for value in DEGRADATIONS if DEGRADATION_ARITY[value] == 2),
    "triple": tuple(value for value in DEGRADATIONS if DEGRADATION_ARITY[value] == 3),
}


@dataclass(frozen=True)
class ProtocolExpectations:
    """Expected official source counts and frozen validation size."""

    official_train_scenes: int = 1183
    official_test_scenes: int = 200
    validation_scenes: int = 100
    image_width: int = 1080
    image_height: int = 720
    allow_transposed_orientation: bool = True
    require_rgb: bool = True
    require_png: bool = True

    @property
    def training_scenes(self) -> int:
        return self.official_train_scenes - self.validation_scenes

    @property
    def allowed_image_sizes(self) -> Tuple[Tuple[int, int], ...]:
        sizes = [(self.image_width, self.image_height)]
        if self.allow_transposed_orientation and self.image_width != self.image_height:
            sizes.append((self.image_height, self.image_width))
        return tuple(sizes)

    def to_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["training_scenes"] = self.training_scenes
        value["allowed_image_sizes"] = [
            {"width": width, "height": height}
            for width, height in self.allowed_image_sizes
        ]
        return value


DEFAULT_EXPECTATIONS = ProtocolExpectations()


def stable_digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def deterministic_seed(text: str) -> int:
    return int.from_bytes(stable_digest(text)[:8], "big") & ((1 << 63) - 1)


def split_sort_key(scene_id: str) -> str:
    value = f"{CDD11_PROTOCOL_VERSION}:split:3407:{scene_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sample_sort_key(namespace: str, sample_id: str) -> str:
    value = f"{CDD11_PROTOCOL_VERSION}:{namespace}:{sample_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

"""CDD-11 manifest datasets and deterministic 1x11 balanced sampling."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import functional as TF

from .protocol import (
    CDD11_PROTOCOL_VERSION,
    DEGRADATION_ARITY,
    DEGRADATIONS,
    deterministic_seed,
)


MAX_TORCH_SEED = (1 << 63) - 1
SampleRequest = Tuple[int, int]


class ManifestFormatError(ValueError):
    """Raised when a CDD-11 JSONL row violates the frozen schema."""


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    degradation: str
    split: str
    input_path: Path
    target_path: Path
    scene_id: str
    metadata: Mapping[str, object]

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, object],
        *,
        source: Path,
        line_number: int,
    ) -> "ManifestRecord":
        required = {
            "id",
            "degradation",
            "split",
            "input",
            "target",
            "scene_id",
            "metadata",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ManifestFormatError(
                f"{source}:{line_number} is missing required keys: {missing}"
            )
        sample_id = str(row["id"]).strip()
        scene_id = str(row["scene_id"]).strip()
        if not sample_id or not scene_id:
            raise ManifestFormatError(
                f"{source}:{line_number} id and scene_id must be non-empty"
            )
        degradation = str(row["degradation"])
        if degradation not in DEGRADATIONS:
            raise ManifestFormatError(
                f"{source}:{line_number} has unsupported degradation {degradation!r}"
            )
        split = str(row["split"])
        if split not in {"train", "val", "test"}:
            raise ManifestFormatError(
                f"{source}:{line_number} has unsupported split {split!r}"
            )
        metadata = row["metadata"]
        if not isinstance(metadata, Mapping):
            raise ManifestFormatError(
                f"{source}:{line_number} metadata must be an object"
            )
        expected_arity = DEGRADATION_ARITY[degradation]
        if metadata.get("arity") != expected_arity:
            raise ManifestFormatError(
                f"{source}:{line_number} arity must be {expected_arity} for "
                f"{degradation!r}"
            )
        input_path = Path(str(row["input"]))
        target_path = Path(str(row["target"]))
        if not input_path.is_absolute() or not target_path.is_absolute():
            raise ManifestFormatError(
                f"{source}:{line_number} input and target paths must be absolute"
            )
        return cls(
            sample_id=sample_id,
            degradation=degradation,
            split=split,
            input_path=input_path,
            target_path=target_path,
            scene_id=scene_id,
            metadata=dict(metadata),
        )


def load_manifest(
    path: Path,
    expected_split: Optional[str] = None,
) -> List[ManifestRecord]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    records: List[ManifestRecord] = []
    sample_ids = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ManifestFormatError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, Mapping):
                raise ManifestFormatError(f"{path}:{line_number} must contain an object")
            record = ManifestRecord.from_mapping(
                row,
                source=path,
                line_number=line_number,
            )
            if expected_split is not None and record.split != expected_split:
                raise ManifestFormatError(
                    f"{path}:{line_number} has split {record.split!r}, "
                    f"expected {expected_split!r}"
                )
            if record.sample_id in sample_ids:
                raise ManifestFormatError(
                    f"Duplicate sample ID in {path}: {record.sample_id}"
                )
            sample_ids.add(record.sample_id)
            records.append(record)
    if not records:
        raise ManifestFormatError(f"Manifest is empty: {path}")
    return records


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return TF.pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)


def _pad_to_patch(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    missing_height = max(0, patch_size - height)
    missing_width = max(0, patch_size - width)
    left = missing_width // 2
    right = missing_width - left
    top = missing_height // 2
    bottom = missing_height - top
    if not any((left, right, top, bottom)):
        return tensor
    reflect_is_valid = (
        left < width
        and right < width
        and top < height
        and bottom < height
        and width > 1
        and height > 1
    )
    return F.pad(
        tensor,
        (left, right, top, bottom),
        mode="reflect" if reflect_is_valid else "replicate",
    )


def _random_int(generator: torch.Generator, high: int) -> int:
    if high <= 0:
        raise ValueError(f"high must be positive, got {high}")
    return int(torch.randint(high, (1,), generator=generator).item())


def _synchronized_train_transform(
    degraded: torch.Tensor,
    target: torch.Tensor,
    *,
    patch_size: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors = [_pad_to_patch(degraded, patch_size), _pad_to_patch(target, patch_size)]
    height, width = tensors[0].shape[-2:]
    if tensors[1].shape[-2:] != (height, width):
        raise ValueError(
            "Input/target size mismatch before crop: "
            f"{tensors[0].shape[-2:]} != {tensors[1].shape[-2:]}"
        )
    top = _random_int(generator, height - patch_size + 1)
    left = _random_int(generator, width - patch_size + 1)
    tensors = [
        tensor[:, top : top + patch_size, left : left + patch_size]
        for tensor in tensors
    ]
    if _random_int(generator, 2):
        tensors = [torch.flip(tensor, dims=(-1,)) for tensor in tensors]
    if _random_int(generator, 2):
        tensors = [torch.flip(tensor, dims=(-2,)) for tensor in tensors]
    rotation = _random_int(generator, 4)
    if rotation:
        tensors = [torch.rot90(tensor, rotation, dims=(-2, -1)) for tensor in tensors]
    return tensors[0], tensors[1]


class CDD11ManifestDataset(Dataset):
    """Load native validation/test pairs or deterministic training patches."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        patch_size: Optional[int] = None,
        validate_paths: bool = False,
    ):
        if split == "train" and (patch_size is None or patch_size <= 0):
            raise ValueError("Training requires a positive patch_size")
        if split != "train" and patch_size is not None:
            raise ValueError("Validation/test must keep native resolution")
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.patch_size = patch_size
        self.records = load_manifest(self.manifest_path, expected_split=split)
        if validate_paths:
            missing = [
                str(path)
                for record in self.records
                for path in (record.input_path, record.target_path)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Manifest references {len(missing)} missing files; first: {missing[:10]}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, request: Union[int, SampleRequest]) -> Dict[str, object]:
        if isinstance(request, tuple):
            if len(request) != 2:
                raise ValueError(f"Sample request must be (index, seed), got {request}")
            index, sample_seed = int(request[0]), int(request[1])
        else:
            index = int(request)
            if self.split == "train":
                raise ValueError(
                    "Training samples require deterministic (index, seed) requests from "
                    "BalancedDegradationBatchSampler"
                )
            sample_seed = 0

        record = self.records[index]
        degraded = _load_rgb_tensor(record.input_path)
        target = _load_rgb_tensor(record.target_path)
        if degraded.shape != target.shape:
            raise ValueError(
                f"Pair shape mismatch for {record.sample_id}: "
                f"{tuple(degraded.shape)} != {tuple(target.shape)}"
            )
        if self.split == "train":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(sample_seed)
            degraded, target = _synchronized_train_transform(
                degraded,
                target,
                patch_size=int(self.patch_size),
                generator=generator,
            )
        return {
            "degraded": degraded,
            "target": target,
            "degradation": record.degradation,
            "arity": DEGRADATION_ARITY[record.degradation],
            "sample_id": record.sample_id,
            "scene_id": record.scene_id,
            "record_index": index,
            "sample_seed": sample_seed,
        }


class BalancedDegradationBatchSampler(Sampler[List[SampleRequest]]):
    """Yield one deterministic, scene-uniform sample from every degradation."""

    def __init__(
        self,
        dataset: CDD11ManifestDataset,
        *,
        start_step: int,
        num_batches: int,
        seed: int,
    ):
        if dataset.split != "train":
            raise ValueError(
                "BalancedDegradationBatchSampler requires a training dataset"
            )
        if start_step < 0 or num_batches < 0:
            raise ValueError("start_step and num_batches must be non-negative")
        self.dataset = dataset
        self.start_step = int(start_step)
        self.num_batches = int(num_batches)
        self.seed = int(seed)

        grouped: Dict[str, Dict[str, List[int]]] = {
            degradation: defaultdict(list) for degradation in DEGRADATIONS
        }
        for index, record in enumerate(dataset.records):
            grouped[record.degradation][record.scene_id].append(index)
        self.indices_by_degradation_scene: Dict[str, Dict[str, Tuple[int, ...]]] = {}
        self.scenes_by_degradation: Dict[str, Tuple[str, ...]] = {}
        for degradation in DEGRADATIONS:
            if not grouped[degradation]:
                raise ValueError(
                    f"Training manifest contains no {degradation!r} samples"
                )
            by_scene = {
                scene_id: tuple(sorted(indices))
                for scene_id, indices in sorted(grouped[degradation].items())
            }
            self.indices_by_degradation_scene[degradation] = by_scene
            self.scenes_by_degradation[degradation] = tuple(by_scene)

    @property
    def batch_size(self) -> int:
        return len(DEGRADATIONS)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[SampleRequest]]:
        for offset in range(self.num_batches):
            global_step = self.start_step + offset
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                deterministic_seed(
                    f"{CDD11_PROTOCOL_VERSION}:balanced-batch:{self.seed}:{global_step}"
                )
            )
            requests: List[SampleRequest] = []
            for degradation in DEGRADATIONS:
                scenes = self.scenes_by_degradation[degradation]
                scene_id = scenes[_random_int(generator, len(scenes))]
                indices = self.indices_by_degradation_scene[degradation][scene_id]
                record_index = indices[_random_int(generator, len(indices))]
                sample_seed = _random_int(generator, MAX_TORCH_SEED)
                requests.append((record_index, sample_seed))
            permutation = torch.randperm(len(requests), generator=generator).tolist()
            yield [requests[index] for index in permutation]


def build_train_dataloader(
    manifest_path: Path,
    *,
    patch_size: int,
    start_step: int,
    num_batches: int,
    seed: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    validate_paths: bool = False,
) -> Tuple[DataLoader, CDD11ManifestDataset, BalancedDegradationBatchSampler]:
    dataset = CDD11ManifestDataset(
        manifest_path,
        split="train",
        patch_size=patch_size,
        validate_paths=validate_paths,
    )
    batch_sampler = BalancedDegradationBatchSampler(
        dataset,
        start_step=start_step,
        num_batches=num_batches,
        seed=seed,
    )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(
        deterministic_seed(f"{CDD11_PROTOCOL_VERSION}:train-loader:{seed}")
    )
    loader_options = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "generator": loader_generator,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = 2
        loader_options["multiprocessing_context"] = "spawn"
    return DataLoader(**loader_options), dataset, batch_sampler


def build_eval_dataloader(
    manifest_path: Path,
    *,
    split: str,
    num_workers: int = 4,
    pin_memory: bool = True,
    validate_paths: bool = False,
) -> Tuple[DataLoader, CDD11ManifestDataset]:
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be 'val' or 'test'")
    dataset = CDD11ManifestDataset(
        manifest_path,
        split=split,
        patch_size=None,
        validate_paths=validate_paths,
    )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(
        deterministic_seed(f"{CDD11_PROTOCOL_VERSION}:{split}-loader")
    )
    loader_options = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "generator": loader_generator,
    }
    if num_workers > 0:
        loader_options["multiprocessing_context"] = "spawn"
    return DataLoader(**loader_options), dataset

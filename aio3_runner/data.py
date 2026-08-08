"""Manifest datasets and deterministic balanced sampling for AIO3-v1."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import functional as TF

from .protocol import AIO3_PROTOCOL_VERSION, NOISE_SIGMAS, deterministic_seed


TASKS: Tuple[str, ...] = ("denoise", "derain", "dehaze")
DEFAULT_TASK_SAMPLES: Mapping[str, int] = {
    "denoise": 4,
    "derain": 4,
    "dehaze": 4,
}
MAX_TORCH_SEED = (1 << 63) - 1
SampleRequest = Tuple[int, int]


class ManifestFormatError(ValueError):
    """Raised when a manifest row violates the AIO3-v1 schema."""


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    task: str
    split: str
    input_path: Optional[Path]
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
        required = {"id", "task", "split", "input", "target", "scene_id", "metadata"}
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
        task = str(row["task"])
        if task not in TASKS:
            raise ManifestFormatError(
                f"{source}:{line_number} has unsupported task {task!r}"
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
        input_value = row["input"]
        input_path = None if input_value is None else Path(str(input_value))
        target_path = Path(str(row["target"]))
        if not target_path.is_absolute() or (
            input_path is not None and not input_path.is_absolute()
        ):
            raise ManifestFormatError(
                f"{source}:{line_number} input and target paths must be absolute"
            )
        if task == "denoise" and input_path is not None:
            raise ManifestFormatError(
                f"{source}:{line_number} denoise input must be generated from target"
            )
        if task != "denoise" and input_path is None:
            raise ManifestFormatError(
                f"{source}:{line_number} {task} requires a real input path"
            )
        if task == "denoise":
            online = metadata.get("online")
            if split == "train" and online is not True:
                raise ManifestFormatError(
                    f"{source}:{line_number} training denoise row must be online"
                )
            if split != "train":
                sigma = metadata.get("sigma")
                seed = metadata.get("noise_seed")
                if (
                    online is not False
                    or sigma not in NOISE_SIGMAS
                    or not isinstance(seed, int)
                    or isinstance(seed, bool)
                ):
                    raise ManifestFormatError(
                        f"{source}:{line_number} fixed denoise row needs sigma and noise_seed"
                    )
        return cls(
            sample_id=sample_id,
            task=task,
            split=split,
            input_path=input_path,
            target_path=target_path,
            scene_id=scene_id,
            metadata=dict(metadata),
        )


def load_manifest(path: Path, expected_split: Optional[str] = None) -> List[ManifestRecord]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    records = []
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
    mode = "reflect" if reflect_is_valid else "replicate"
    return F.pad(tensor, (left, right, top, bottom), mode=mode)


def _random_int(generator: torch.Generator, high: int) -> int:
    if high <= 0:
        raise ValueError(f"high must be positive, got {high}")
    return int(torch.randint(high, (1,), generator=generator).item())


def _synchronized_train_transform(
    degraded: Optional[torch.Tensor],
    target: torch.Tensor,
    *,
    patch_size: int,
    generator: torch.Generator,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    tensors = [target] if degraded is None else [degraded, target]
    tensors = [_pad_to_patch(tensor, patch_size) for tensor in tensors]
    reference_height, reference_width = tensors[0].shape[-2:]
    for tensor in tensors[1:]:
        if tensor.shape[-2:] != (reference_height, reference_width):
            raise ValueError(
                "Input/target size mismatch before crop: "
                f"{tensor.shape[-2:]} != {(reference_height, reference_width)}"
            )
    top = _random_int(generator, reference_height - patch_size + 1)
    left = _random_int(generator, reference_width - patch_size + 1)
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
    if degraded is None:
        return None, tensors[0]
    return tensors[0], tensors[1]


def _add_gaussian_noise(
    clean: torch.Tensor,
    sigma: int,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(
        clean.shape,
        generator=generator,
        dtype=clean.dtype,
        device="cpu",
    )
    return clean + noise * (float(sigma) / 255.0)


class AIO3ManifestDataset(Dataset):
    """Load native validation/test images or deterministic train patches."""

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
            missing = []
            for record in self.records:
                for path in (record.input_path, record.target_path):
                    if path is not None and not path.is_file():
                        missing.append(str(path))
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
                    "BalancedTaskBatchSampler"
                )
            sample_seed = 0

        record = self.records[index]
        target = _load_rgb_tensor(record.target_path)
        degraded = (
            _load_rgb_tensor(record.input_path)
            if record.input_path is not None
            else None
        )
        if degraded is not None and degraded.shape != target.shape:
            raise ValueError(
                f"Pair shape mismatch for {record.sample_id}: "
                f"{tuple(degraded.shape)} != {tuple(target.shape)}"
            )

        sigma = -1
        if self.split == "train":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(sample_seed)
            degraded, target = _synchronized_train_transform(
                degraded,
                target,
                patch_size=int(self.patch_size),
                generator=generator,
            )
            if record.task == "denoise":
                sigma = NOISE_SIGMAS[_random_int(generator, len(NOISE_SIGMAS))]
                degraded = _add_gaussian_noise(target, sigma, generator)
        elif record.task == "denoise":
            sigma = int(record.metadata["sigma"])
            noise_generator = torch.Generator(device="cpu")
            noise_generator.manual_seed(int(record.metadata["noise_seed"]))
            degraded = _add_gaussian_noise(target, sigma, noise_generator)

        if degraded is None:
            raise RuntimeError(f"Failed to create degraded input for {record.sample_id}")
        return {
            "degraded": degraded,
            "target": target,
            "task": record.task,
            "sample_id": record.sample_id,
            "scene_id": record.scene_id,
            "sigma": sigma,
            "record_index": index,
            "sample_seed": sample_seed,
        }


class BalancedTaskBatchSampler(Sampler[List[SampleRequest]]):
    """Yield deterministic 4/4/4 batches with uniform scene sampling."""

    def __init__(
        self,
        dataset: AIO3ManifestDataset,
        *,
        start_step: int,
        num_batches: int,
        seed: int,
        samples_per_task: Mapping[str, int] = DEFAULT_TASK_SAMPLES,
    ):
        if dataset.split != "train":
            raise ValueError("BalancedTaskBatchSampler requires a training dataset")
        if start_step < 0 or num_batches < 0:
            raise ValueError("start_step and num_batches must be non-negative")
        if set(samples_per_task) != set(TASKS):
            raise ValueError(f"samples_per_task must define exactly {TASKS}")
        if any(int(value) <= 0 for value in samples_per_task.values()):
            raise ValueError("Every task must contribute at least one sample")

        self.dataset = dataset
        self.start_step = int(start_step)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.samples_per_task = {
            task: int(samples_per_task[task]) for task in TASKS
        }
        grouped: Dict[str, Dict[str, List[int]]] = {
            task: defaultdict(list) for task in TASKS
        }
        for index, record in enumerate(dataset.records):
            grouped[record.task][record.scene_id].append(index)
        self.indices_by_task_scene: Dict[str, Dict[str, Tuple[int, ...]]] = {}
        self.scenes_by_task: Dict[str, Tuple[str, ...]] = {}
        for task in TASKS:
            if not grouped[task]:
                raise ValueError(f"Training manifest contains no {task} samples")
            self.indices_by_task_scene[task] = {
                scene_id: tuple(sorted(indices))
                for scene_id, indices in sorted(grouped[task].items())
            }
            self.scenes_by_task[task] = tuple(self.indices_by_task_scene[task])

    @property
    def batch_size(self) -> int:
        return sum(self.samples_per_task.values())

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[SampleRequest]]:
        for offset in range(self.num_batches):
            global_step = self.start_step + offset
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                deterministic_seed(
                    f"{AIO3_PROTOCOL_VERSION}:balanced-batch:{self.seed}:{global_step}"
                )
            )
            requests: List[SampleRequest] = []
            for task in TASKS:
                scenes = self.scenes_by_task[task]
                for _ in range(self.samples_per_task[task]):
                    scene_id = scenes[_random_int(generator, len(scenes))]
                    indices = self.indices_by_task_scene[task][scene_id]
                    record_index = indices[_random_int(generator, len(indices))]
                    sample_seed = _random_int(generator, MAX_TORCH_SEED)
                    requests.append((record_index, sample_seed))
            permutation = torch.randperm(
                len(requests), generator=generator
            ).tolist()
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
) -> Tuple[DataLoader, AIO3ManifestDataset, BalancedTaskBatchSampler]:
    dataset = AIO3ManifestDataset(
        manifest_path,
        split="train",
        patch_size=patch_size,
        validate_paths=validate_paths,
    )
    batch_sampler = BalancedTaskBatchSampler(
        dataset,
        start_step=start_step,
        num_batches=num_batches,
        seed=seed,
    )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(
        deterministic_seed(f"{AIO3_PROTOCOL_VERSION}:train-loader:{seed}")
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
    loader = DataLoader(**loader_options)
    return loader, dataset, batch_sampler


def build_eval_dataloader(
    manifest_path: Path,
    *,
    split: str,
    num_workers: int = 4,
    pin_memory: bool = True,
    validate_paths: bool = False,
) -> Tuple[DataLoader, AIO3ManifestDataset]:
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be 'val' or 'test'")
    dataset = AIO3ManifestDataset(
        manifest_path,
        split=split,
        patch_size=None,
        validate_paths=validate_paths,
    )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(
        deterministic_seed(f"{AIO3_PROTOCOL_VERSION}:{split}-loader")
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
    loader = DataLoader(**loader_options)
    return loader, dataset

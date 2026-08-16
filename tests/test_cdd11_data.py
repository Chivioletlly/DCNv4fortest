import json
import shutil
import sys
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from cdd11_runner.data import (
    BalancedDegradationBatchSampler,
    CDD11ManifestDataset,
    build_eval_dataloader,
    build_train_dataloader,
)
from cdd11_runner.protocol import DEGRADATION_ARITY, DEGRADATIONS


def _save_rgb(path: Path, value: int, size=(9, 7)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(value, value, value)).save(path)


def _write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


@contextmanager
def _temporary_directory():
    parent = REPOSITORY_ROOT / ".tmp_cdd11_data_tests"
    parent.mkdir(exist_ok=True)
    path = parent / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)
        try:
            parent.rmdir()
        except OSError:
            pass


def _build_manifest(root: Path, *, split="train", scenes=3) -> Path:
    rows = []
    for scene_index in range(scenes):
        target = root / "images" / f"clear_{scene_index}.png"
        _save_rgb(target, 200 - scene_index)
        for degradation_index, degradation in enumerate(DEGRADATIONS):
            degraded = root / "images" / f"{degradation}_{scene_index}.png"
            _save_rgb(degraded, 20 + degradation_index + scene_index)
            rows.append(
                {
                    "id": f"{split}:{degradation}:{scene_index}",
                    "degradation": degradation,
                    "split": split,
                    "input": str(degraded.resolve()),
                    "target": str(target.resolve()),
                    "scene_id": f"cdd11:train:{scene_index}",
                    "metadata": {
                        "dataset": "CDD-11",
                        "arity": DEGRADATION_ARITY[degradation],
                    },
                }
            )
    manifest = root / f"{split}.jsonl"
    _write_manifest(manifest, rows)
    return manifest


def test_dataset_uses_synchronized_deterministic_patch_transform():
    with _temporary_directory() as root:
        target = root / "target.png"
        degraded = root / "degraded.png"
        _save_rgb(target, 110, size=(5, 6))
        _save_rgb(degraded, 100, size=(5, 6))
        manifest = root / "train.jsonl"
        _write_manifest(
            manifest,
            [
                {
                    "id": "train:low:0",
                    "degradation": "low",
                    "split": "train",
                    "input": str(degraded.resolve()),
                    "target": str(target.resolve()),
                    "scene_id": "cdd11:train:0",
                    "metadata": {"arity": 1},
                }
            ],
        )
        dataset = CDD11ManifestDataset(manifest, split="train", patch_size=8)

        first = dataset[(0, 12345)]
        repeated = dataset[(0, 12345)]

        assert first["degraded"].shape == (3, 8, 8)
        torch.testing.assert_close(first["degraded"], repeated["degraded"])
        torch.testing.assert_close(first["target"], repeated["target"])
        torch.testing.assert_close(
            first["target"] - first["degraded"],
            torch.full_like(first["target"], 10.0 / 255.0),
            rtol=0.0,
            atol=1e-6,
        )


def test_balanced_sampler_is_exact_scene_uniform_and_resume_reproducible():
    with _temporary_directory() as root:
        manifest = _build_manifest(root, scenes=3)
        dataset = CDD11ManifestDataset(manifest, split="train", patch_size=8)
        batches = list(
            BalancedDegradationBatchSampler(
                dataset,
                start_step=0,
                num_batches=20,
                seed=3407,
            )
        )
        for batch in batches:
            categories = Counter(
                dataset.records[index].degradation for index, _ in batch
            )
            assert categories == {degradation: 1 for degradation in DEGRADATIONS}
        repeated = list(
            BalancedDegradationBatchSampler(
                dataset,
                start_step=0,
                num_batches=20,
                seed=3407,
            )
        )
        resumed = list(
            BalancedDegradationBatchSampler(
                dataset,
                start_step=19,
                num_batches=1,
                seed=3407,
            )
        )
        assert repeated == batches
        assert resumed[0] == batches[19]


def test_dataloader_builders_keep_effective_batch_and_native_eval_shape():
    with _temporary_directory() as root:
        train_manifest = _build_manifest(root, scenes=2)
        train_loader, _, sampler = build_train_dataloader(
            train_manifest,
            patch_size=8,
            start_step=0,
            num_batches=1,
            seed=3407,
            num_workers=0,
        )
        train_batch = next(iter(train_loader))
        assert sampler.batch_size == 11
        assert train_batch["degraded"].shape == (11, 3, 8, 8)
        assert Counter(train_batch["degradation"]) == {
            degradation: 1 for degradation in DEGRADATIONS
        }

        val_manifest = _build_manifest(root, split="val", scenes=1)
        val_loader, _ = build_eval_dataloader(
            val_manifest,
            split="val",
            num_workers=0,
        )
        val_batch = next(iter(val_loader))
        assert val_batch["degraded"].shape == (1, 3, 7, 9)
        assert val_batch["target"].shape == (1, 3, 7, 9)

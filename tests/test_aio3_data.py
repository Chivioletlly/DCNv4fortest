import gc
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

from aio3_runner.data import (
    AIO3ManifestDataset,
    BalancedTaskBatchSampler,
    build_eval_dataloader,
    build_train_dataloader,
)


def _save_rgb(path: Path, value: int, size=(6, 5)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=(value, value, value))
    image.save(path)


def _row(
    sample_id,
    task,
    split,
    target,
    *,
    input_path=None,
    scene_id=None,
    metadata=None,
):
    return {
        "id": sample_id,
        "task": task,
        "split": split,
        "input": str(input_path) if input_path is not None else None,
        "target": str(target),
        "scene_id": scene_id or sample_id,
        "metadata": metadata or {},
    }


def _write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


@contextmanager
def _workspace_temporary_directory():
    parent = REPOSITORY_ROOT / ".tmp_aio3_manifest_tests"
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


def _build_train_manifest(root: Path) -> Path:
    rows = []
    for index in range(2):
        clean = root / "images" / f"noise_clean_{index}.png"
        _save_rgb(clean, 255 - index)
        rows.append(
            _row(
                f"bsd400:{index}",
                "denoise",
                "train",
                clean,
                scene_id=f"bsd400:{index}",
                metadata={
                    "dataset": "BSD400",
                    "degradation": "gaussian_noise",
                    "online": True,
                },
            )
        )
    for index in range(2):
        degraded = root / "images" / f"rain_{index}.png"
        target = root / "images" / f"norain_{index}.png"
        _save_rgb(degraded, 90 + index)
        _save_rgb(target, 100 + index)
        rows.append(
            _row(
                f"rain:{index}",
                "derain",
                "train",
                target,
                input_path=degraded,
                scene_id=f"rain:{index}",
                metadata={"dataset": "RainTrainL"},
            )
        )
    for scene, variants in (("h0", 1), ("h1", 9)):
        target = root / "images" / f"clear_{scene}.png"
        _save_rgb(target, 150)
        for variant in range(variants):
            degraded = root / "images" / f"haze_{scene}_{variant}.png"
            _save_rgb(degraded, 130 + variant)
            rows.append(
                _row(
                    f"ots:{scene}:{variant}",
                    "dehaze",
                    "train",
                    target,
                    input_path=degraded,
                    scene_id=f"ots:{scene}",
                    metadata={"dataset": "OTS"},
                )
            )
    manifest = root / "train.jsonl"
    _write_manifest(manifest, rows)
    return manifest


def test_synchronized_crop_padding_and_augmentation_preserve_pair_alignment():
    with _workspace_temporary_directory() as root:
        manifest = _build_train_manifest(root)
        dataset = AIO3ManifestDataset(manifest, split="train", patch_size=8)
        derain_index = next(
            index for index, record in enumerate(dataset.records) if record.task == "derain"
        )

        sample = dataset[(derain_index, 12345)]

        assert sample["degraded"].shape == (3, 8, 8)
        assert sample["target"].shape == (3, 8, 8)
        expected_difference = torch.full_like(sample["target"], 10.0 / 255.0)
        torch.testing.assert_close(
            sample["target"] - sample["degraded"],
            expected_difference,
            rtol=0.0,
            atol=1e-6,
        )


def test_online_noise_is_request_deterministic_and_not_clipped():
    with _workspace_temporary_directory() as root:
        manifest = _build_train_manifest(root)
        dataset = AIO3ManifestDataset(manifest, split="train", patch_size=8)
        denoise_index = next(
            index for index, record in enumerate(dataset.records) if record.task == "denoise"
        )

        first = dataset[(denoise_index, 777)]
        repeated = dataset[(denoise_index, 777)]
        different = dataset[(denoise_index, 778)]

        assert first["sigma"] in {15, 25, 50}
        torch.testing.assert_close(first["degraded"], repeated["degraded"])
        torch.testing.assert_close(first["target"], repeated["target"])
        assert not torch.equal(first["degraded"], different["degraded"])
        assert first["degraded"].max().item() > 1.0


def test_fixed_validation_noise_uses_manifest_seed_at_native_size():
    with _workspace_temporary_directory() as root:
        target = root / "clean.png"
        _save_rgb(target, 128, size=(11, 7))
        manifest = root / "val.jsonl"
        _write_manifest(
            manifest,
            [
                _row(
                    "wed:fixed:sigma25",
                    "denoise",
                    "val",
                    target,
                    scene_id="wed:fixed",
                    metadata={
                        "dataset": "WED",
                        "degradation": "gaussian_noise",
                        "online": False,
                        "sigma": 25,
                        "noise_seed": 123456,
                    },
                )
            ],
        )
        dataset = AIO3ManifestDataset(manifest, split="val")

        sample = dataset[0]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(123456)
        expected_noise = torch.randn(
            sample["target"].shape,
            generator=generator,
            dtype=torch.float32,
        )
        expected = sample["target"] + expected_noise * (25.0 / 255.0)

        assert sample["degraded"].shape == (3, 7, 11)
        torch.testing.assert_close(sample["degraded"], expected)
        torch.testing.assert_close(dataset[0]["degraded"], sample["degraded"])


def test_balanced_sampler_is_exact_scene_uniform_and_resume_reproducible():
    with _workspace_temporary_directory() as root:
        manifest = _build_train_manifest(root)
        dataset = AIO3ManifestDataset(manifest, split="train", patch_size=8)
        sampler = BalancedTaskBatchSampler(
            dataset,
            start_step=0,
            num_batches=1000,
            seed=3407,
        )

        dehaze_scene_counts = Counter()
        first_batches = []
        for batch_index, batch in enumerate(sampler):
            tasks = Counter(dataset.records[index].task for index, _ in batch)
            assert tasks == {"denoise": 4, "derain": 4, "dehaze": 4}
            for index, _ in batch:
                record = dataset.records[index]
                if record.task == "dehaze":
                    dehaze_scene_counts[record.scene_id] += 1
            if batch_index < 11:
                first_batches.append(batch)

        h0_fraction = dehaze_scene_counts["ots:h0"] / sum(dehaze_scene_counts.values())
        assert 0.45 < h0_fraction < 0.55

        repeated = list(
            BalancedTaskBatchSampler(
                dataset,
                start_step=0,
                num_batches=11,
                seed=3407,
            )
        )
        resumed = list(
            BalancedTaskBatchSampler(
                dataset,
                start_step=10,
                num_batches=1,
                seed=3407,
            )
        )
        assert repeated == first_batches
        assert resumed[0] == first_batches[10]


def test_dataloader_builders_collate_balanced_train_and_native_eval_batches():
    with _workspace_temporary_directory() as root:
        train_manifest = _build_train_manifest(root)
        train_loader, _, sampler = build_train_dataloader(
            train_manifest,
            patch_size=8,
            start_step=0,
            num_batches=1,
            seed=3407,
            num_workers=0,
        )
        train_batch = next(iter(train_loader))

        assert sampler.batch_size == 12
        assert train_batch["degraded"].shape == (12, 3, 8, 8)
        assert Counter(train_batch["task"]) == {
            "denoise": 4,
            "derain": 4,
            "dehaze": 4,
        }

        degraded = root / "eval_input.png"
        target = root / "eval_target.png"
        _save_rgb(degraded, 80, size=(13, 9))
        _save_rgb(target, 90, size=(13, 9))
        eval_manifest = root / "val.jsonl"
        _write_manifest(
            eval_manifest,
            [
                _row(
                    "rain:eval",
                    "derain",
                    "val",
                    target,
                    input_path=degraded,
                    metadata={"dataset": "RainTrainL"},
                )
            ],
        )
        eval_loader, _ = build_eval_dataloader(
            eval_manifest,
            split="val",
            num_workers=0,
        )
        eval_batch = next(iter(eval_loader))
        assert eval_batch["degraded"].shape == (1, 3, 9, 13)
        assert eval_batch["target"].shape == (1, 3, 9, 13)


def test_spawn_worker_context_loads_a_balanced_batch_without_forking():
    with _workspace_temporary_directory() as root:
        train_manifest = _build_train_manifest(root)
        train_loader, _, _ = build_train_dataloader(
            train_manifest,
            patch_size=8,
            start_step=0,
            num_batches=1,
            seed=3407,
            num_workers=1,
        )
        iterator = iter(train_loader)
        batch = next(iterator)
        assert Counter(batch["task"]) == {
            "denoise": 4,
            "derain": 4,
            "dehaze": 4,
        }
        del iterator
        del train_loader
        gc.collect()


if __name__ == "__main__":
    tests = [
        value
        for name, value in list(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")

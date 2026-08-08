import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from aio3_runner.manifests import AuditError, prepare_aio3_manifests
from aio3_runner.protocol import ProtocolExpectations, noise_seed


TEST_EXPECTATIONS = ProtocolExpectations(
    bsd400_images=2,
    wed_gt_images=3,
    wed_noisy_images=3,
    bsd68_images=2,
    raintrain_input_images=3,
    raintrain_target_images=3,
    rain100_input_images=2,
    rain100_target_images=2,
    ots_clear_images=3,
    ots_haze_images=6,
    ots_haze_excluded_files=1,
    sots_input_images=3,
    sots_target_images=2,
    sots_paired_images=3,
    denoise_validation_scenes=1,
    derain_validation_scenes=1,
    dehaze_validation_scenes=1,
)


def _save_image(path: Path, size=(18, 16), value=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(value, value, value)).save(path)


def _build_synthetic_aio3(root: Path):
    for index in range(2):
        _save_image(root / "BSD400" / f"bsd{index}.jpg", value=20 + index)
        _save_image(root / "BSD68" / f"test{index}.png", value=30 + index)

    for index in range(3):
        _save_image(root / "WED" / "gt" / f"wed{index}.bmp", value=40 + index)
        _save_image(root / "WED" / "noisy" / f"wed{index}.bmp", value=50 + index)
        _save_image(root / "RainTrainL" / f"rain-{index}.png", value=60 + index)
        _save_image(root / "RainTrainL" / f"norain-{index}.png", value=70 + index)
    _save_image(root / "RainTrainL" / "rainregion-0.png", value=80)
    _save_image(
        root / "RainTrainL" / ".ipynb_checkpoints" / "rain-999.png",
        value=81,
    )

    for index in range(2):
        _save_image(root / "Rain100L" / "rain" / f"rain-{index:03d}.png", value=90)
        _save_image(root / "Rain100L" / "gt" / f"norain-{index:03d}.png", value=91)

    for scene_index in range(3):
        scene_id = f"{scene_index + 1:04d}"
        _save_image(root / "OTS" / "clear" / f"{scene_id}.jpg", value=100)
        _save_image(root / "OTS" / "depth" / f"{scene_id}.png", value=101)
        for variant in range(2):
            _save_image(
                root
                / "OTS"
                / "haze"
                / f"part{variant + 1}"
                / f"{scene_id}_0.{8 + variant}_0.{1 + variant}.jpg",
                value=110 + variant,
            )
    (root / "OTS" / "haze" / ".DS_Store").write_text(
        "not an image", encoding="utf-8"
    )

    for scene_index in range(2):
        scene_id = f"{scene_index + 1:04d}"
        _save_image(
            root / "SOTS" / "outdoor" / "input" / f"{scene_id}_0.8_0.2.jpg",
            value=120,
        )
        _save_image(
            root / "SOTS" / "outdoor" / "target" / f"{scene_id}.jpg",
            value=121,
        )
    _save_image(
        root / "SOTS" / "outdoor" / "input" / "0001_0.9_0.1.jpg",
        value=122,
    )


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


@contextmanager
def _raises(expected_exception):
    try:
        yield
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception.__name__} to be raised")


@contextmanager
def _workspace_temporary_directory():
    """Use inherited workspace ACLs instead of tempfile's Windows 0o700 ACL."""

    parent = REPOSITORY_ROOT / ".tmp_aio3_manifest_tests"
    parent.mkdir(exist_ok=True)
    path = parent / uuid.uuid4().hex
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path)
        try:
            parent.rmdir()
        except OSError:
            pass


def test_manifest_counts_pairing_and_split_isolation():
    with _workspace_temporary_directory() as temporary:
        root = Path(temporary) / "AIO3"
        output = Path(temporary) / "manifests"
        _build_synthetic_aio3(root)

        audit = prepare_aio3_manifests(
            root,
            output,
            expectations=TEST_EXPECTATIONS,
            verify_images=True,
        )
        train = _read_jsonl(output / "train.jsonl")
        val = _read_jsonl(output / "val.jsonl")
        test = _read_jsonl(output / "test.jsonl")

        assert audit["status"] == "pass_with_warnings"
        assert audit["source_counts"]["RainTrainL/excluded"] == 1
        assert audit["pairing"]["RainTrainL_pairs"] == 3
        assert audit["pairing"]["Rain100L_pairs"] == 2
        assert audit["pairing"]["OTS_pairs"] == 6
        assert audit["pairing"]["OTS_clear_scenes"] == 3
        assert audit["source_counts"]["OTS/haze_files_excluded"] == 1
        assert len(audit["pairing"]["OTS_files_excluded"]) == 1
        assert audit["pairing"]["SOTS_pairs"] == 3
        assert not audit["pairing"]["SOTS_inputs_without_target"]

        assert audit["splits"]["train"]["rows_by_task"] == {
            "dehaze": 4,
            "denoise": 4,
            "derain": 2,
        }
        assert audit["splits"]["val"]["rows_by_task"] == {
            "dehaze": 1,
            "denoise": 3,
            "derain": 1,
        }
        assert audit["splits"]["test"]["rows_by_task"] == {
            "dehaze": 3,
            "denoise": 6,
            "derain": 2,
        }

        split_scenes = {
            name: {(row["task"], row["scene_id"]) for row in rows}
            for name, rows in (("train", train), ("val", val), ("test", test))
        }
        assert not (split_scenes["train"] & split_scenes["val"])
        assert not (split_scenes["train"] & split_scenes["test"])
        assert not (split_scenes["val"] & split_scenes["test"])
        assert all(row["input"] is None for row in train if row["task"] == "denoise")
        assert all(
            row["metadata"]["dataset"] != "WED/noisy"
            for row in train + val + test
        )


def test_manifests_and_noise_seeds_are_reproducible():
    with _workspace_temporary_directory() as temporary:
        root = Path(temporary) / "AIO3"
        first = Path(temporary) / "first"
        second = Path(temporary) / "second"
        _build_synthetic_aio3(root)

        prepare_aio3_manifests(root, first, expectations=TEST_EXPECTATIONS)
        prepare_aio3_manifests(root, second, expectations=TEST_EXPECTATIONS)

        for filename in ("train.jsonl", "val.jsonl", "test.jsonl", "visual_samples.json"):
            assert (first / filename).read_bytes() == (second / filename).read_bytes()
        assert noise_seed("test", "bsd68:test001", 25) == noise_seed(
            "test", "bsd68:test001", 25
        )
        assert noise_seed("test", "bsd68:test001", 15) != noise_seed(
            "test", "bsd68:test001", 25
        )


def test_missing_rain_pair_fails_instead_of_pairing_by_index():
    with _workspace_temporary_directory() as temporary:
        root = Path(temporary) / "AIO3"
        _build_synthetic_aio3(root)
        (root / "RainTrainL" / "norain-1.png").unlink()

        with _raises(AuditError):
            prepare_aio3_manifests(
                root,
                Path(temporary) / "output",
                expectations=TEST_EXPECTATIONS,
            )


def test_pair_size_mismatch_fails_before_writing_manifests():
    with _workspace_temporary_directory() as temporary:
        root = Path(temporary) / "AIO3"
        output = Path(temporary) / "output"
        _build_synthetic_aio3(root)
        _save_image(root / "Rain100L" / "gt" / "norain-000.png", size=(17, 16))

        with _raises(AuditError):
            prepare_aio3_manifests(root, output, expectations=TEST_EXPECTATIONS)
        assert not (output / "train.jsonl").exists()


def test_existing_outputs_are_not_overwritten_without_permission():
    with _workspace_temporary_directory() as temporary:
        root = Path(temporary) / "AIO3"
        output = Path(temporary) / "output"
        _build_synthetic_aio3(root)
        prepare_aio3_manifests(root, output, expectations=TEST_EXPECTATIONS)

        original = (output / "train.jsonl").read_bytes()
        with _raises(AuditError):
            prepare_aio3_manifests(root, output, expectations=TEST_EXPECTATIONS)
        assert (output / "train.jsonl").read_bytes() == original


if __name__ == "__main__":
    tests = [
        value
        for name, value in list(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")

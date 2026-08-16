import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from cdd11_runner.manifests import AuditError, prepare_cdd11_manifests
from cdd11_runner.protocol import DEGRADATIONS, ProtocolExpectations


@contextmanager
def _temporary_directory():
    parent = REPOSITORY_ROOT / ".tmp_cdd11_manifest_tests"
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


def _make_source(root: Path, train_scenes=4, test_scenes=2, size=(13, 9)):
    for split, scenes, offset in (("train", train_scenes, 0), ("test", test_scenes, 100)):
        for directory in ("clear", *DEGRADATIONS):
            (root / split / directory).mkdir(parents=True, exist_ok=True)
        for scene in range(scenes):
            filename = f"scene_{scene:03d}.png"
            clear_value = 20 + offset + scene
            Image.new("RGB", size, color=(clear_value,) * 3).save(
                root / split / "clear" / filename
            )
            for degradation_index, degradation in enumerate(DEGRADATIONS):
                value = (clear_value + degradation_index + 1) % 256
                Image.new("RGB", size, color=(value,) * 3).save(
                    root / split / degradation / filename
                )


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_manifests_freezes_scene_disjoint_1x11_splits():
    with _temporary_directory() as root:
        data_root = root / "CDD-11"
        output = root / "manifests"
        _make_source(data_root)
        expectations = ProtocolExpectations(
            official_train_scenes=4,
            official_test_scenes=2,
            validation_scenes=2,
            image_width=13,
            image_height=9,
        )
        audit = prepare_cdd11_manifests(
            data_root,
            output,
            expectations=expectations,
        )

        train = _read_jsonl(output / "train.jsonl")
        val = _read_jsonl(output / "val.jsonl")
        test = _read_jsonl(output / "test.jsonl")
        assert (len(train), len(val), len(test)) == (22, 22, 22)
        assert {row["scene_id"] for row in train}.isdisjoint(
            {row["scene_id"] for row in val}
        )
        assert audit["splits"]["val"]["rows_by_degradation"] == {
            degradation: 2 for degradation in DEGRADATIONS
        }
        assert audit["image_audit"]["allowed_sizes"] == [
            {"width": 13, "height": 9},
            {"width": 9, "height": 13},
        ]
        visuals = json.loads((output / "visual_samples.json").read_text("utf-8"))
        assert len(visuals["ordered_sample_ids"]) == 2 * len(DEGRADATIONS)


def test_prepare_manifests_accepts_mixed_landscape_and_portrait_pairs():
    with _temporary_directory() as root:
        data_root = root / "CDD-11"
        output = root / "manifests"
        _make_source(data_root)
        for directory in ("clear", *DEGRADATIONS):
            Image.new("RGB", (9, 13), color=(42,) * 3).save(
                data_root / "train" / directory / "scene_000.png"
            )
        expectations = ProtocolExpectations(
            official_train_scenes=4,
            official_test_scenes=2,
            validation_scenes=2,
            image_width=13,
            image_height=9,
        )

        audit = prepare_cdd11_manifests(
            data_root,
            output,
            expectations=expectations,
        )

        assert audit["status"] == "pass"
        assert audit["image_audit"]["verified_files"] == 72
        assert audit["image_audit"]["observed_sizes"] == [
            {"width": 9, "height": 13, "files": 12},
            {"width": 13, "height": 9, "files": 60},
        ]


def test_prepare_manifests_rejects_size_outside_allowed_orientations():
    with _temporary_directory() as root:
        data_root = root / "CDD-11"
        _make_source(data_root)
        Image.new("RGB", (8, 13), color=(42,) * 3).save(
            data_root / "train" / "clear" / "scene_000.png"
        )
        expectations = ProtocolExpectations(
            official_train_scenes=4,
            official_test_scenes=2,
            validation_scenes=2,
            image_width=13,
            image_height=9,
        )

        try:
            prepare_cdd11_manifests(
                data_root,
                root / "manifests",
                expectations=expectations,
            )
        except AuditError as error:
            assert "not in (13x9, 9x13)" in str(error)
        else:
            raise AssertionError("Expected CDD-11 audit to reject an unsupported size")


def test_prepare_manifests_rejects_pair_orientation_mismatch():
    with _temporary_directory() as root:
        data_root = root / "CDD-11"
        _make_source(data_root)
        Image.new("RGB", (9, 13), color=(42,) * 3).save(
            data_root / "train" / "snow" / "scene_000.png"
        )
        expectations = ProtocolExpectations(
            official_train_scenes=4,
            official_test_scenes=2,
            validation_scenes=2,
            image_width=13,
            image_height=9,
        )

        try:
            prepare_cdd11_manifests(
                data_root,
                root / "manifests",
                expectations=expectations,
            )
        except AuditError as error:
            assert "Pair metadata mismatch" in str(error)
        else:
            raise AssertionError("Expected CDD-11 audit to reject pair orientation drift")


def test_prepare_manifests_rejects_category_filename_mismatch():
    with _temporary_directory() as root:
        data_root = root / "CDD-11"
        _make_source(data_root)
        (data_root / "train" / "snow" / "scene_000.png").unlink()
        expectations = ProtocolExpectations(
            official_train_scenes=4,
            official_test_scenes=2,
            validation_scenes=2,
            image_width=13,
            image_height=9,
        )
        try:
            prepare_cdd11_manifests(
                data_root,
                root / "manifests",
                expectations=expectations,
            )
        except AuditError as error:
            assert "Filename mismatch" in str(error)
        else:
            raise AssertionError("Expected CDD-11 audit to fail")

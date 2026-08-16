"""CDD-11 source audit and deterministic scene-disjoint manifests."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from PIL import Image

from .protocol import (
    CDD11_PROTOCOL_VERSION,
    DEFAULT_EXPECTATIONS,
    DEGRADATION_ARITY,
    DEGRADATIONS,
    ProtocolExpectations,
    sample_sort_key,
    split_sort_key,
)


OUTPUT_FILES = (
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
    "data_audit.json",
    "visual_samples.json",
)


class AuditError(RuntimeError):
    """Raised when CDD-11 does not match the frozen protocol."""


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    mode: str
    image_format: str


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    _write_text_atomic(path, text)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _files_by_name(directory: Path, *, require_png: bool) -> Dict[str, Path]:
    if not directory.is_dir():
        raise AuditError(f"Missing CDD-11 directory: {directory}")
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]
    if require_png:
        unexpected = sorted(str(path) for path in paths if path.suffix.casefold() != ".png")
        if unexpected:
            raise AuditError(f"CDD-11 contains non-PNG files in {directory}: {unexpected[:20]}")
        paths = [path for path in paths if path.suffix.casefold() == ".png"]
    else:
        paths = [
            path
            for path in paths
            if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]
    result: Dict[str, Path] = {}
    for path in sorted(paths, key=lambda value: value.name.casefold()):
        key = path.name.casefold()
        if key in result:
            raise AuditError(f"Case-insensitive duplicate filename in {directory}: {path.name}")
        result[key] = _absolute(path)
    return result


def _inspect_image(
    path: Path,
    *,
    expectations: ProtocolExpectations,
    cache: MutableMapping[Path, ImageInfo],
    verify_images: bool,
) -> ImageInfo:
    cached = cache.get(path)
    if cached is not None:
        return cached
    try:
        with Image.open(path) as image:
            info = ImageInfo(
                width=int(image.width),
                height=int(image.height),
                mode=str(image.mode),
                image_format=str(image.format or "unknown"),
            )
            if verify_images:
                image.load()
    except Exception as error:
        raise AuditError(f"Failed to decode CDD-11 image {path}: {error}") from error
    if (info.width, info.height) not in expectations.allowed_image_sizes:
        allowed = ", ".join(
            f"{width}x{height}"
            for width, height in expectations.allowed_image_sizes
        )
        raise AuditError(
            f"Unexpected CDD-11 image size for {path}: "
            f"{info.width}x{info.height} not in ({allowed})"
        )
    if expectations.require_rgb and info.mode != "RGB":
        raise AuditError(f"CDD-11 image is not RGB: {path} has mode {info.mode!r}")
    cache[path] = info
    return info


def _audit_source_split(
    split_root: Path,
    *,
    expected_scenes: int,
    expectations: ProtocolExpectations,
    verify_images: bool,
    image_cache: MutableMapping[Path, ImageInfo],
) -> Tuple[Dict[str, Path], Dict[str, Dict[str, Path]]]:
    clear = _files_by_name(split_root / "clear", require_png=expectations.require_png)
    if len(clear) != expected_scenes:
        raise AuditError(
            f"{split_root.name}/clear count mismatch: {len(clear)} != {expected_scenes}"
        )
    degraded: Dict[str, Dict[str, Path]] = {}
    clear_names = set(clear)
    for degradation in DEGRADATIONS:
        values = _files_by_name(
            split_root / degradation,
            require_png=expectations.require_png,
        )
        missing = sorted(clear_names - set(values))
        extra = sorted(set(values) - clear_names)
        if missing or extra:
            raise AuditError(
                f"Filename mismatch for {split_root.name}/{degradation}; "
                f"missing={missing[:20]}, extra={extra[:20]}"
            )
        degraded[degradation] = values

    for filename, target_path in clear.items():
        target_info = _inspect_image(
            target_path,
            expectations=expectations,
            cache=image_cache,
            verify_images=verify_images,
        )
        for degradation in DEGRADATIONS:
            input_path = degraded[degradation][filename]
            input_info = _inspect_image(
                input_path,
                expectations=expectations,
                cache=image_cache,
                verify_images=verify_images,
            )
            if input_info != target_info:
                raise AuditError(
                    f"Pair metadata mismatch for {split_root.name}/{degradation}/{filename}: "
                    f"{input_info} != {target_info}"
                )
    return clear, degraded


def _record(
    *,
    source_split: str,
    split: str,
    degradation: str,
    filename: str,
    input_path: Path,
    target_path: Path,
) -> Dict[str, object]:
    scene_id = f"cdd11:{source_split}:{filename}"
    return {
        "id": f"{split}:{degradation}:{filename}",
        "degradation": degradation,
        "split": split,
        "input": str(input_path),
        "target": str(target_path),
        "scene_id": scene_id,
        "metadata": {
            "dataset": "CDD-11",
            "source_split": source_split,
            "filename": filename,
            "arity": DEGRADATION_ARITY[degradation],
        },
    }


def _rows_for_filenames(
    filenames: Iterable[str],
    *,
    clear: Mapping[str, Path],
    degraded: Mapping[str, Mapping[str, Path]],
    source_split: str,
    split: str,
) -> List[Dict[str, object]]:
    rows = []
    for degradation in DEGRADATIONS:
        for filename in sorted(filenames):
            rows.append(
                _record(
                    source_split=source_split,
                    split=split,
                    degradation=degradation,
                    filename=filename,
                    input_path=degraded[degradation][filename],
                    target_path=clear[filename],
                )
            )
    return rows


def _count_records(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    categories = Counter(str(row["degradation"]) for row in rows)
    scenes = {str(row["scene_id"]) for row in rows}
    return {
        "rows": len(rows),
        "scenes": len(scenes),
        "rows_by_degradation": {
            degradation: categories[degradation] for degradation in DEGRADATIONS
        },
    }


def _assert_split_isolation(
    train_rows: Sequence[Mapping[str, object]],
    val_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
) -> None:
    values = {
        split: {str(row["scene_id"]) for row in rows}
        for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows))
    }
    conflicts = []
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(values[first] & values[second])
        if overlap:
            conflicts.append(f"{first}/{second}: {overlap[:20]}")
    if conflicts:
        raise AuditError("CDD-11 scene leakage across splits: " + "; ".join(conflicts))


def _assert_no_exact_clear_duplicates(
    train_clear: Mapping[str, Path],
    test_clear: Mapping[str, Path],
) -> Dict[str, object]:
    train_by_hash: Dict[str, List[str]] = {}
    for filename, path in train_clear.items():
        train_by_hash.setdefault(file_sha256(path), []).append(filename)
    test_by_hash: Dict[str, List[str]] = {}
    for filename, path in test_clear.items():
        test_by_hash.setdefault(file_sha256(path), []).append(filename)
    within_split = {
        "train": [values for values in train_by_hash.values() if len(values) > 1],
        "test": [values for values in test_by_hash.values() if len(values) > 1],
    }
    if within_split["train"] or within_split["test"]:
        raise AuditError(f"Exact duplicate clear scenes within official split: {within_split}")
    duplicates = []
    for digest, test_filenames in test_by_hash.items():
        if digest in train_by_hash:
            duplicates.append(
                {
                    "test": test_filenames,
                    "train": train_by_hash[digest],
                    "sha256": digest,
                }
            )
    if duplicates:
        raise AuditError(f"Exact clear-image duplicates across official splits: {duplicates[:20]}")
    return {
        "train_unique_sha256": len(train_by_hash),
        "test_unique_sha256": len(test_by_hash),
        "within_split_exact_duplicates": 0,
        "cross_split_exact_duplicates": 0,
    }


def _select_visual_samples(val_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    scene_ids = sorted(
        {str(row["scene_id"]) for row in val_rows},
        key=lambda value: sample_sort_key("visual-scene", value),
    )[:2]
    lookup = {
        (str(row["scene_id"]), str(row["degradation"])): str(row["id"])
        for row in val_rows
    }
    ordered = [
        lookup[(scene_id, degradation)]
        for scene_id in scene_ids
        for degradation in DEGRADATIONS
    ]
    if len(ordered) != 2 * len(DEGRADATIONS):
        raise AuditError("Could not select two complete validation scenes for visualization")
    return {
        "protocol": CDD11_PROTOCOL_VERSION,
        "selection_rule": "two lowest SHA256(cdd11-v1:visual-scene:<scene_id>)",
        "scene_ids": scene_ids,
        "ordered_sample_ids": ordered,
    }


def prepare_cdd11_manifests(
    data_root: Path,
    output_dir: Path,
    *,
    expectations: ProtocolExpectations = DEFAULT_EXPECTATIONS,
    verify_images: bool = True,
    overwrite: bool = False,
    protocol_document: Optional[Path] = None,
) -> Dict[str, object]:
    """Audit the official CDD-11 layout and freeze train/val/test manifests."""

    data_root = _absolute(Path(data_root))
    output_dir = _absolute(Path(output_dir))
    if not data_root.is_dir():
        raise AuditError(f"CDD-11 data root does not exist: {data_root}")
    if expectations.validation_scenes <= 0:
        raise AuditError("validation_scenes must be positive")
    if expectations.training_scenes <= 0:
        raise AuditError("validation_scenes must be smaller than official_train_scenes")
    if protocol_document is not None:
        protocol_document = _absolute(Path(protocol_document))
        if not protocol_document.is_file():
            raise AuditError(f"Protocol document does not exist: {protocol_document}")
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise AuditError(
            "Refusing to overwrite existing CDD-11 manifest outputs: "
            + ", ".join(str(path) for path in existing)
        )

    image_cache: Dict[Path, ImageInfo] = {}
    train_clear, train_degraded = _audit_source_split(
        data_root / "train",
        expected_scenes=expectations.official_train_scenes,
        expectations=expectations,
        verify_images=verify_images,
        image_cache=image_cache,
    )
    test_clear, test_degraded = _audit_source_split(
        data_root / "test",
        expected_scenes=expectations.official_test_scenes,
        expectations=expectations,
        verify_images=verify_images,
        image_cache=image_cache,
    )
    duplicate_audit = _assert_no_exact_clear_duplicates(train_clear, test_clear)

    ordered_train_filenames = sorted(train_clear, key=split_sort_key)
    val_filenames = set(ordered_train_filenames[: expectations.validation_scenes])
    fit_filenames = set(ordered_train_filenames[expectations.validation_scenes :])
    train_rows = _rows_for_filenames(
        fit_filenames,
        clear=train_clear,
        degraded=train_degraded,
        source_split="train",
        split="train",
    )
    val_rows = _rows_for_filenames(
        val_filenames,
        clear=train_clear,
        degraded=train_degraded,
        source_split="train",
        split="val",
    )
    test_rows = _rows_for_filenames(
        test_clear,
        clear=test_clear,
        degraded=test_degraded,
        source_split="test",
        split="test",
    )
    _assert_split_isolation(train_rows, val_rows, test_rows)

    observed_sizes = Counter(
        (info.width, info.height) for info in image_cache.values()
    )

    expected_counts = {
        "train": expectations.training_scenes * len(DEGRADATIONS),
        "val": expectations.validation_scenes * len(DEGRADATIONS),
        "test": expectations.official_test_scenes * len(DEGRADATIONS),
    }
    actual_counts = {
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
    }
    if actual_counts != expected_counts:
        raise AuditError(f"CDD-11 split row counts mismatch: {actual_counts} != {expected_counts}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "val.jsonl", val_rows)
    _write_jsonl(output_dir / "test.jsonl", test_rows)
    visual_samples = _select_visual_samples(val_rows)
    _write_json(output_dir / "visual_samples.json", visual_samples)

    manifest_entries = {}
    for filename in ("train.jsonl", "val.jsonl", "test.jsonl"):
        path = output_dir / filename
        manifest_entries[filename] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    audit: Dict[str, object] = {
        "protocol": CDD11_PROTOCOL_VERSION,
        "status": "pass",
        "data_root": str(data_root),
        "expectations": expectations.to_dict(),
        "source_counts": {
            "official_train_clear_scenes": len(train_clear),
            "official_test_clear_scenes": len(test_clear),
            "degradations": len(DEGRADATIONS),
        },
        "splits": {
            "train": _count_records(train_rows),
            "val": _count_records(val_rows),
            "test": _count_records(test_rows),
        },
        "scene_split": {
            "seed": 3407,
            "rule": "lowest SHA256(cdd11-v1:split:3407:<filename>) assigned to val",
            "validation_filenames": sorted(val_filenames),
        },
        "duplicate_audit": duplicate_audit,
        "image_audit": {
            "verified_files": len(image_cache),
            "allowed_sizes": [
                {"width": width, "height": height}
                for width, height in expectations.allowed_image_sizes
            ],
            "observed_sizes": [
                {"width": width, "height": height, "files": files}
                for (width, height), files in sorted(observed_sizes.items())
            ],
            "required_mode": "RGB" if expectations.require_rgb else None,
        },
        "manifests": manifest_entries,
        "visual_samples": {
            "sha256": file_sha256(output_dir / "visual_samples.json"),
            "count": len(visual_samples["ordered_sample_ids"]),
        },
    }
    if protocol_document is not None:
        audit["protocol_document"] = {
            "path": str(protocol_document),
            "sha256": file_sha256(protocol_document),
        }
    _write_json(output_dir / "data_audit.json", audit)
    return audit

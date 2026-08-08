"""Strict AIO3-v1 data pairing, auditing, and manifest generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from PIL import Image

from .protocol import (
    AIO3_PROTOCOL_VERSION,
    DEFAULT_EXPECTATIONS,
    NOISE_SIGMAS,
    ProtocolExpectations,
    deterministic_seed,
    noise_seed,
    split_sort_key,
)


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

OUTPUT_FILES = (
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
    "visual_samples.json",
    "data_audit.json",
)


class AuditError(RuntimeError):
    """Raised when source data cannot satisfy the frozen protocol."""


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    mode: str
    image_format: str


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        relative_parts = path.parts
    return any(part.startswith(".") for part in relative_parts)


def find_images(directory: Path, recursive: bool = False) -> List[Path]:
    if not directory.is_dir():
        raise AuditError(f"Required image directory does not exist: {directory}")
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    paths = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.casefold() in IMAGE_EXTENSIONS
        and not _is_hidden(path, directory)
    ]
    return sorted(paths, key=lambda value: (value.as_posix().casefold(), value.as_posix()))


def find_excluded_files(directory: Path, recursive: bool = False) -> List[Path]:
    """List files intentionally excluded by the image scanner.

    Files inside notebook checkpoint directories are ignored entirely. Other
    hidden files (for example .DS_Store) are reported so a raw `find` count can
    be reconciled with the number of usable images.
    """

    if not directory.is_dir():
        raise AuditError(f"Required directory does not exist: {directory}")
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    paths = []
    for path in iterator:
        if not path.is_file():
            continue
        relative_parts = path.relative_to(directory).parts
        if ".ipynb_checkpoints" in relative_parts:
            continue
        if path.suffix.casefold() not in IMAGE_EXTENSIONS or _is_hidden(path, directory):
            paths.append(path)
    return sorted(paths, key=lambda value: (value.as_posix().casefold(), value.as_posix()))


def _unique_map(
    paths: Iterable[Path],
    key_fn,
    label: str,
) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    duplicates: Dict[str, List[str]] = defaultdict(list)
    for path in paths:
        raw_key = str(key_fn(path))
        key = raw_key.casefold()
        if key in mapping:
            duplicates[key].extend([str(mapping[key]), str(path)])
        else:
            mapping[key] = path
    if duplicates:
        details = "; ".join(
            f"{key}: {sorted(set(values))}" for key, values in sorted(duplicates.items())
        )
        raise AuditError(f"Duplicate {label} keys detected: {details}")
    return mapping


def _stem_map(paths: Iterable[Path], label: str) -> Dict[str, Path]:
    return _unique_map(paths, lambda path: path.stem, label)


def _expect_count(
    errors: List[str],
    label: str,
    actual: int,
    expected: int,
) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected}, found {actual}")


def _rain_pairs(
    inputs: Sequence[Path],
    targets: Sequence[Path],
    input_prefix: str,
    target_prefix: str,
    label: str,
) -> Tuple[List[Tuple[str, Path, Path]], List[str], List[str]]:
    input_pattern = re.compile(rf"^{re.escape(input_prefix)}(.+)$", re.IGNORECASE)
    target_pattern = re.compile(rf"^{re.escape(target_prefix)}(.+)$", re.IGNORECASE)

    def suffix(path: Path, pattern: re.Pattern) -> str:
        match = pattern.match(path.stem)
        if match is None:
            raise AuditError(f"Unexpected {label} filename: {path.name}")
        return match.group(1)

    input_map = _unique_map(inputs, lambda path: suffix(path, input_pattern), f"{label} input")
    target_map = _unique_map(targets, lambda path: suffix(path, target_pattern), f"{label} target")
    shared = sorted(set(input_map) & set(target_map))
    pairs = [(key, input_map[key], target_map[key]) for key in shared]
    missing_targets = [str(input_map[key]) for key in sorted(set(input_map) - set(target_map))]
    missing_inputs = [str(target_map[key]) for key in sorted(set(target_map) - set(input_map))]
    return pairs, missing_targets, missing_inputs


def _reside_scene_id(path: Path) -> str:
    stem = path.stem
    if "_" not in stem:
        raise AuditError(
            f"RESIDE degraded filename does not contain a scene separator '_': {path}"
        )
    scene_id = stem.split("_", 1)[0]
    if not scene_id:
        raise AuditError(f"RESIDE degraded filename has an empty scene ID: {path}")
    return scene_id


def _reside_pairs(
    degraded_paths: Sequence[Path],
    target_paths: Sequence[Path],
    label: str,
) -> Tuple[List[Tuple[str, Path, Path]], List[str], List[str], Dict[str, int]]:
    target_map = _stem_map(target_paths, f"{label} target")
    pairs: List[Tuple[str, Path, Path]] = []
    missing_targets: List[str] = []
    variants = Counter()
    matched_target_keys = set()
    for degraded_path in degraded_paths:
        scene_id = _reside_scene_id(degraded_path)
        key = scene_id.casefold()
        target_path = target_map.get(key)
        if target_path is None:
            missing_targets.append(str(degraded_path))
            continue
        pairs.append((scene_id, degraded_path, target_path))
        variants[scene_id] += 1
        matched_target_keys.add(key)
    missing_inputs = [
        str(target_map[key]) for key in sorted(set(target_map) - matched_target_keys)
    ]
    return pairs, missing_targets, missing_inputs, dict(sorted(variants.items()))


def _inspect_image(
    path: Path,
    cache: MutableMapping[Path, ImageInfo],
    verify_images: bool,
) -> ImageInfo:
    cached = cache.get(path)
    if cached is not None:
        return cached
    try:
        with Image.open(path) as image:
            width, height = image.size
            info = ImageInfo(
                width=width,
                height=height,
                mode=str(image.mode),
                image_format=str(image.format or "unknown"),
            )
            if verify_images:
                image.load()
    except Exception as error:
        raise AuditError(f"Failed to decode image {path}: {error}") from error
    if info.width <= 0 or info.height <= 0:
        raise AuditError(f"Image has an invalid size: {path} -> {info.width}x{info.height}")
    cache[path] = info
    return info


def _record(
    *,
    sample_id: str,
    task: str,
    split: str,
    scene_id: str,
    target: Path,
    input_path: Optional[Path] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return {
        "id": sample_id,
        "task": task,
        "split": split,
        "input": str(input_path) if input_path is not None else None,
        "target": str(target),
        "scene_id": scene_id,
        "metadata": dict(metadata or {}),
    }


def _denoise_record(
    dataset: str,
    path: Path,
    split: str,
    sigma: Optional[int] = None,
) -> Dict[str, object]:
    base_id = f"{dataset.casefold()}:{path.stem}"
    sample_id = base_id if sigma is None else f"{base_id}:sigma{sigma}"
    metadata: Dict[str, object] = {
        "dataset": dataset,
        "degradation": "gaussian_noise",
        "online": sigma is None,
    }
    if sigma is not None:
        metadata.update(
            {
                "sigma": sigma,
                "noise_seed": noise_seed(split, base_id, sigma),
            }
        )
    return _record(
        sample_id=sample_id,
        task="denoise",
        split=split,
        scene_id=base_id,
        target=path,
        metadata=metadata,
    )


def _sort_records(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        records,
        key=lambda row: (
            str(row["task"]),
            str(row["scene_id"]),
            str(row["id"]),
            str(row.get("input") or ""),
        ),
    )


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    _write_text_atomic(path, text)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_text_atomic(path, text)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _count_records(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    task_rows = Counter(str(row["task"]) for row in records)
    task_scenes: Dict[str, set] = defaultdict(set)
    for row in records:
        task_scenes[str(row["task"])].add(str(row["scene_id"]))
    return {
        "rows": len(records),
        "rows_by_task": dict(sorted(task_rows.items())),
        "scenes_by_task": {
            task: len(scenes) for task, scenes in sorted(task_scenes.items())
        },
    }


def _assert_no_split_leakage(
    train_records: Sequence[Mapping[str, object]],
    val_records: Sequence[Mapping[str, object]],
    test_records: Sequence[Mapping[str, object]],
) -> None:
    split_keys = {}
    for name, rows in (
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ):
        split_keys[name] = {
            (str(row["task"]), str(row["scene_id"])) for row in rows
        }
    conflicts = []
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(split_keys[first] & split_keys[second])
        if overlap:
            conflicts.append(f"{first}/{second}: {overlap[:20]}")
    if conflicts:
        raise AuditError("Scene leakage across splits: " + "; ".join(conflicts))


def _assert_unique_record_ids(
    split: str,
    records: Sequence[Mapping[str, object]],
) -> None:
    counts = Counter(str(row["id"]) for row in records)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        raise AuditError(f"Duplicate sample IDs in {split} manifest: {duplicates[:20]}")


def _select_visual_samples(val_records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    selected: Dict[str, List[str]] = {}
    for sigma in NOISE_SIGMAS:
        candidates = [
            row
            for row in val_records
            if row["task"] == "denoise" and row["metadata"].get("sigma") == sigma
        ]
        candidates.sort(key=lambda row: split_sort_key("visual", str(row["id"])))
        selected[f"denoise_sigma{sigma}"] = [str(row["id"]) for row in candidates[:2]]
    for task, count in (("derain", 4), ("dehaze", 4)):
        candidates = [row for row in val_records if row["task"] == task]
        candidates.sort(key=lambda row: split_sort_key("visual", str(row["id"])))
        selected[task] = [str(row["id"]) for row in candidates[:count]]
    return {
        "protocol": AIO3_PROTOCOL_VERSION,
        "selection_rule": "lowest SHA256(aio3-v1:visual:<sample_id>)",
        "samples": selected,
    }


def prepare_aio3_manifests(
    data_root: Path,
    output_dir: Path,
    *,
    expectations: ProtocolExpectations = DEFAULT_EXPECTATIONS,
    verify_images: bool = True,
    overwrite: bool = False,
    protocol_document: Optional[Path] = None,
) -> Dict[str, object]:
    """Audit AIO-3 and write deterministic train/val/test manifests."""

    data_root = _absolute_without_resolving_symlinks(Path(data_root))
    output_dir = _absolute_without_resolving_symlinks(Path(output_dir))
    if not data_root.is_dir():
        raise AuditError(f"AIO-3 data root does not exist: {data_root}")
    if protocol_document is not None:
        protocol_document = _absolute_without_resolving_symlinks(Path(protocol_document))
        if not protocol_document.is_file():
            raise AuditError(f"Protocol document does not exist: {protocol_document}")
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise AuditError(
            "Refusing to overwrite existing manifest outputs: "
            + ", ".join(str(path) for path in existing)
        )

    bsd400 = find_images(data_root / "BSD400")
    bsd68 = find_images(data_root / "BSD68")
    wed_gt = find_images(data_root / "WED" / "gt")
    wed_noisy = find_images(data_root / "WED" / "noisy")

    raintrain_root = data_root / "RainTrainL"
    raintrain_all = find_images(raintrain_root)
    raintrain_inputs = [
        path for path in raintrain_all if re.match(r"^rain-.+", path.stem, re.IGNORECASE)
    ]
    raintrain_targets = [
        path for path in raintrain_all if re.match(r"^norain-.+", path.stem, re.IGNORECASE)
    ]

    rain100_inputs = find_images(data_root / "Rain100L" / "rain")
    rain100_targets = find_images(data_root / "Rain100L" / "gt")
    ots_clear = find_images(data_root / "OTS" / "clear")
    ots_haze = find_images(data_root / "OTS" / "haze", recursive=True)
    ots_haze_excluded_files = find_excluded_files(
        data_root / "OTS" / "haze", recursive=True
    )
    ots_depth = find_images(data_root / "OTS" / "depth")
    sots_inputs = find_images(data_root / "SOTS" / "outdoor" / "input")
    sots_targets = find_images(data_root / "SOTS" / "outdoor" / "target")

    source_counts = {
        "BSD400": len(bsd400),
        "BSD68": len(bsd68),
        "WED/gt": len(wed_gt),
        "WED/noisy_ignored": len(wed_noisy),
        "RainTrainL/all": len(raintrain_all),
        "RainTrainL/rain": len(raintrain_inputs),
        "RainTrainL/norain": len(raintrain_targets),
        "RainTrainL/excluded": (
            len(raintrain_all) - len(raintrain_inputs) - len(raintrain_targets)
        ),
        "Rain100L/rain": len(rain100_inputs),
        "Rain100L/gt": len(rain100_targets),
        "OTS/clear": len(ots_clear),
        "OTS/haze": len(ots_haze),
        "OTS/haze_files_excluded": len(ots_haze_excluded_files),
        "OTS/depth_ignored": len(ots_depth),
        "SOTS/outdoor/input": len(sots_inputs),
        "SOTS/outdoor/target": len(sots_targets),
    }

    errors: List[str] = []
    _expect_count(errors, "BSD400", len(bsd400), expectations.bsd400_images)
    _expect_count(errors, "BSD68", len(bsd68), expectations.bsd68_images)
    _expect_count(errors, "WED/gt", len(wed_gt), expectations.wed_gt_images)
    _expect_count(errors, "WED/noisy", len(wed_noisy), expectations.wed_noisy_images)
    _expect_count(
        errors,
        "RainTrainL/rain",
        len(raintrain_inputs),
        expectations.raintrain_input_images,
    )
    _expect_count(
        errors,
        "RainTrainL/norain",
        len(raintrain_targets),
        expectations.raintrain_target_images,
    )
    _expect_count(
        errors,
        "Rain100L/rain",
        len(rain100_inputs),
        expectations.rain100_input_images,
    )
    _expect_count(
        errors,
        "Rain100L/gt",
        len(rain100_targets),
        expectations.rain100_target_images,
    )
    _expect_count(errors, "OTS/clear", len(ots_clear), expectations.ots_clear_images)
    _expect_count(errors, "OTS/haze", len(ots_haze), expectations.ots_haze_images)
    _expect_count(
        errors,
        "OTS/haze excluded files",
        len(ots_haze_excluded_files),
        expectations.ots_haze_excluded_files,
    )
    _expect_count(
        errors,
        "SOTS/outdoor/input",
        len(sots_inputs),
        expectations.sots_input_images,
    )
    _expect_count(
        errors,
        "SOTS/outdoor/target",
        len(sots_targets),
        expectations.sots_target_images,
    )

    raintrain_pairs, raintrain_missing_targets, raintrain_missing_inputs = _rain_pairs(
        raintrain_inputs,
        raintrain_targets,
        input_prefix="rain-",
        target_prefix="norain-",
        label="RainTrainL",
    )
    rain100_pairs, rain100_missing_targets, rain100_missing_inputs = _rain_pairs(
        rain100_inputs,
        rain100_targets,
        input_prefix="rain-",
        target_prefix="norain-",
        label="Rain100L",
    )
    ots_pairs, ots_missing_targets, ots_missing_inputs, ots_variants = _reside_pairs(
        ots_haze,
        ots_clear,
        label="OTS",
    )
    sots_pairs, sots_missing_targets, sots_missing_inputs, _ = _reside_pairs(
        sots_inputs,
        sots_targets,
        label="SOTS Outdoor",
    )

    _expect_count(
        errors,
        "RainTrainL complete pairs",
        len(raintrain_pairs),
        expectations.raintrain_input_images,
    )
    _expect_count(
        errors,
        "Rain100L complete pairs",
        len(rain100_pairs),
        expectations.rain100_input_images,
    )
    _expect_count(
        errors,
        "OTS paired clear scenes",
        len(ots_variants),
        expectations.ots_clear_images,
    )
    _expect_count(
        errors,
        "SOTS complete pairs",
        len(sots_pairs),
        expectations.sots_paired_images,
    )
    for label, values in (
        ("RainTrainL inputs without target", raintrain_missing_targets),
        ("RainTrainL targets without input", raintrain_missing_inputs),
        ("Rain100L inputs without target", rain100_missing_targets),
        ("Rain100L targets without input", rain100_missing_inputs),
        ("OTS haze without clear", ots_missing_targets),
        ("OTS clear without haze", ots_missing_inputs),
    ):
        if values:
            errors.append(f"{label}: {values[:20]}")
    if errors:
        raise AuditError("AIO3 source-count or pairing audit failed:\n- " + "\n- ".join(errors))

    wed_sorted = sorted(
        wed_gt,
        key=lambda path: split_sort_key("denoise", f"wed:{path.stem}"),
    )
    wed_val = set(wed_sorted[: expectations.denoise_validation_scenes])

    raintrain_sorted = sorted(
        raintrain_pairs,
        key=lambda item: split_sort_key("derain", f"raintrainl:{item[0]}"),
    )
    raintrain_val_suffixes = {
        suffix
        for suffix, _, _ in raintrain_sorted[: expectations.derain_validation_scenes]
    }

    ots_scene_ids = sorted(
        ots_variants,
        key=lambda scene_id: split_sort_key("dehaze", f"ots:{scene_id}"),
    )
    ots_val_scenes = set(ots_scene_ids[: expectations.dehaze_validation_scenes])

    train_records: List[Dict[str, object]] = []
    val_records: List[Dict[str, object]] = []
    test_records: List[Dict[str, object]] = []

    train_records.extend(_denoise_record("BSD400", path, "train") for path in bsd400)
    for path in wed_gt:
        if path in wed_val:
            for sigma in NOISE_SIGMAS:
                val_records.append(_denoise_record("WED", path, "val", sigma=sigma))
        else:
            train_records.append(_denoise_record("WED", path, "train"))

    for suffix, input_path, target_path in raintrain_pairs:
        split = "val" if suffix in raintrain_val_suffixes else "train"
        row = _record(
            sample_id=f"raintrainl:{suffix}",
            task="derain",
            split=split,
            scene_id=f"raintrainl:{suffix}",
            input_path=input_path,
            target=target_path,
            metadata={"dataset": "RainTrainL"},
        )
        (val_records if split == "val" else train_records).append(row)

    ots_by_scene: Dict[str, List[Tuple[str, Path, Path]]] = defaultdict(list)
    for pair in ots_pairs:
        ots_by_scene[pair[0]].append(pair)
    for scene_id, candidates in sorted(ots_by_scene.items()):
        candidates.sort(key=lambda item: item[1].as_posix())
        if scene_id in ots_val_scenes:
            selection_seed = deterministic_seed(
                f"{AIO3_PROTOCOL_VERSION}:dehaze-val:{scene_id}"
            )
            selected = candidates[selection_seed % len(candidates)]
            _, input_path, target_path = selected
            val_records.append(
                _record(
                    sample_id=f"ots:{input_path.stem}",
                    task="dehaze",
                    split="val",
                    scene_id=f"ots:{scene_id}",
                    input_path=input_path,
                    target=target_path,
                    metadata={"dataset": "OTS", "clear_scene_id": scene_id},
                )
            )
        else:
            for _, input_path, target_path in candidates:
                train_records.append(
                    _record(
                        sample_id=f"ots:{input_path.stem}",
                        task="dehaze",
                        split="train",
                        scene_id=f"ots:{scene_id}",
                        input_path=input_path,
                        target=target_path,
                        metadata={"dataset": "OTS", "clear_scene_id": scene_id},
                    )
                )

    for path in bsd68:
        for sigma in NOISE_SIGMAS:
            test_records.append(_denoise_record("BSD68", path, "test", sigma=sigma))
    for suffix, input_path, target_path in rain100_pairs:
        test_records.append(
            _record(
                sample_id=f"rain100l:{suffix}",
                task="derain",
                split="test",
                scene_id=f"rain100l:{suffix}",
                input_path=input_path,
                target=target_path,
                metadata={"dataset": "Rain100L"},
            )
        )
    for scene_id, input_path, target_path in sots_pairs:
        test_records.append(
            _record(
                sample_id=f"sots:{input_path.stem}",
                task="dehaze",
                split="test",
                scene_id=f"sots:{scene_id}",
                input_path=input_path,
                target=target_path,
                metadata={
                    "dataset": "SOTS-outdoor",
                    "clear_scene_id": scene_id,
                },
            )
        )

    train_records = _sort_records(train_records)
    val_records = _sort_records(val_records)
    test_records = _sort_records(test_records)
    _assert_unique_record_ids("train", train_records)
    _assert_unique_record_ids("val", val_records)
    _assert_unique_record_ids("test", test_records)
    _assert_no_split_leakage(train_records, val_records, test_records)

    image_cache: Dict[Path, ImageInfo] = {}
    used_paths = set()
    for row in train_records + val_records + test_records:
        target_path = Path(str(row["target"]))
        used_paths.add(target_path)
        target_info = _inspect_image(target_path, image_cache, verify_images)
        input_value = row.get("input")
        if input_value is not None:
            input_path = Path(str(input_value))
            used_paths.add(input_path)
            input_info = _inspect_image(input_path, image_cache, verify_images)
            if (input_info.width, input_info.height) != (
                target_info.width,
                target_info.height,
            ):
                raise AuditError(
                    "Paired image size mismatch for "
                    f"{row['id']}: input={input_info.width}x{input_info.height}, "
                    f"target={target_info.width}x{target_info.height}"
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = {
        "train.jsonl": train_records,
        "val.jsonl": val_records,
        "test.jsonl": test_records,
    }
    for filename, rows in manifest_rows.items():
        _write_jsonl(output_dir / filename, rows)
    visual_samples = _select_visual_samples(val_records)
    _write_json(output_dir / "visual_samples.json", visual_samples)

    mode_counts = Counter(info.mode for info in image_cache.values())
    format_counts = Counter(info.image_format for info in image_cache.values())
    widths = [info.width for info in image_cache.values()]
    heights = [info.height for info in image_cache.values()]
    warnings = [
        "WED/noisy is counted but intentionally excluded; AIO3-v1 synthesizes Gaussian noise.",
        (
            "SOTS Outdoor contains "
            f"{len(sots_inputs)} degraded inputs mapped to {len(sots_targets)} unique "
            f"target files; all {len(sots_pairs)} inputs form strict evaluation pairs."
        ),
    ]
    audit: Dict[str, object] = {
        "status": "pass_with_warnings" if warnings else "pass",
        "protocol": AIO3_PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "expectations": expectations.to_dict(),
        "source_counts": source_counts,
        "pairing": {
            "RainTrainL_pairs": len(raintrain_pairs),
            "Rain100L_pairs": len(rain100_pairs),
            "OTS_pairs": len(ots_pairs),
            "OTS_clear_scenes": len(ots_variants),
            "OTS_haze_variants_per_scene_min": min(ots_variants.values()),
            "OTS_haze_variants_per_scene_max": max(ots_variants.values()),
            "OTS_files_excluded": [
                str(path) for path in ots_haze_excluded_files
            ],
            "SOTS_pairs": len(sots_pairs),
            "SOTS_inputs_without_target": sots_missing_targets,
            "SOTS_targets_without_input": sots_missing_inputs,
        },
        "splits": {
            "train": _count_records(train_records),
            "val": _count_records(val_records),
            "test": _count_records(test_records),
        },
        "image_validation": {
            "full_decode_verify": verify_images,
            "unique_used_files": len(used_paths),
            "modes": dict(sorted(mode_counts.items())),
            "formats": dict(sorted(format_counts.items())),
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
        },
        "leakage_check": "pass",
        "warnings": warnings,
    }
    if protocol_document is not None:
        audit["protocol_document"] = {
            "path": str(protocol_document),
            "sha256": file_sha256(protocol_document),
        }
    audit["manifests"] = {
        filename: {
            "rows": len(rows),
            "sha256": file_sha256(output_dir / filename),
        }
        for filename, rows in manifest_rows.items()
    }
    audit["visual_samples"] = {
        "path": str(output_dir / "visual_samples.json"),
        "sha256": file_sha256(output_dir / "visual_samples.json"),
    }
    _write_json(output_dir / "data_audit.json", audit)
    return audit

"""Command-line entry point for AIO3-v1 manifest preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifests import AuditError, prepare_aio3_manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit AIO-3 and generate frozen AIO3-v1 manifests."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-document",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "AIO3_TRAINING_EVALUATION_PROTOCOL.md",
    )
    parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="Development only: skip Pillow decode verification. Never use for a formal run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the known manifest/audit files in output-dir.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        audit = prepare_aio3_manifests(
            data_root=args.data_root,
            output_dir=args.output_dir,
            verify_images=not args.skip_image_verification,
            overwrite=args.overwrite,
            protocol_document=args.protocol_document,
        )
    except AuditError as error:
        raise SystemExit(f"AIO3 AUDIT FAILED\n{error}") from error
    summary = {
        "status": audit["status"],
        "protocol": audit["protocol"],
        "data_root": audit["data_root"],
        "output_dir": audit["output_dir"],
        "source_counts": audit["source_counts"],
        "pairing": audit["pairing"],
        "splits": audit["splits"],
        "manifests": audit["manifests"],
        "warnings": audit["warnings"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

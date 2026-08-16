"""Command-line entry point for frozen CDD-11 manifest preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .manifests import prepare_cdd11_manifests


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit CDD-11 and create scene-disjoint cdd11-v1 manifests."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-image-decode",
        action="store_true",
        help="Inspect image headers only; formal preparation should not use this flag.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol_document = REPOSITORY_ROOT / "docs" / "CDD11_TRAINING_EVALUATION_PROTOCOL.md"
    audit = prepare_cdd11_manifests(
        args.data_root,
        args.output_dir,
        verify_images=not args.skip_image_decode,
        overwrite=args.overwrite,
        protocol_document=protocol_document,
    )
    print(
        "CDD-11 manifests prepared: "
        f"train={audit['splits']['train']['rows']} "
        f"val={audit['splits']['val']['rows']} "
        f"test={audit['splits']['test']['rows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

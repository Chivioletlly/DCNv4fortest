"""Minimal W&B connectivity check required before an AIO3 smoke run."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .runtime import WANDB_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check W&B logging from the AIO3 server.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--entity")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import wandb
    except ImportError as error:
        raise SystemExit("wandb is not installed in the active Python environment") from error
    if wandb.__version__ != WANDB_VERSION:
        raise SystemExit(
            f"AIO3-v1 requires wandb=={WANDB_VERSION}, got {wandb.__version__}"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root.expanduser().resolve() / "wandb-connectivity" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = args.output_root.expanduser().resolve() / ".wandb_cache"
    data_dir = args.output_root.expanduser().resolve() / ".wandb_staging"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(run_dir)
    os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
    os.environ["WANDB_DATA_DIR"] = str(data_dir)
    os.environ["WANDB_MODE"] = args.mode
    os.environ["WANDB_PROJECT"] = "aio3-restoration"
    if args.entity is not None:
        os.environ["WANDB_ENTITY"] = args.entity
    run_id = uuid.uuid4().hex[:16]
    run = wandb.init(
        entity=args.entity,
        project="aio3-restoration",
        group="aio3-v1-connectivity",
        job_type="connectivity-check",
        name=f"connectivity-{timestamp}",
        id=run_id,
        resume="never",
        dir=str(run_dir),
        mode=args.mode,
        force=args.mode == "online",
        tags=["aio3-v1", "connectivity"],
        config={"protocol": "aio3-v1", "purpose": "connectivity-check"},
    )
    try:
        run.define_metric("global_step")
        run.define_metric("connectivity/*", step_metric="global_step")
        run.log({"global_step": 0, "connectivity/value": 1.0})
        result = {
            "status": "pass",
            "mode": args.mode,
            "run_id": run_id,
            "entity": getattr(run, "entity", None),
            "project": getattr(run, "project", None),
            "url": getattr(run, "url", None),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (run_dir / "connectivity_result.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()

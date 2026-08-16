"""Minimal W&B connectivity check before CDD-11 runs are created."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .runtime import WANDB_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check CDD-11 W&B logging")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--entity")
    parser.add_argument("--project", default="cdd11-restoration")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import wandb
    except ImportError as error:
        raise SystemExit("wandb is not installed in the active environment") from error
    if wandb.__version__ != WANDB_VERSION:
        raise SystemExit(
            f"CDD-11-v1 requires wandb=={WANDB_VERSION}, got {wandb.__version__}"
        )

    output_root = args.output_root.expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / "wandb-connectivity" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = output_root / ".wandb_cache"
    data_dir = output_root / ".wandb_staging"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(run_dir)
    os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
    os.environ["WANDB_DATA_DIR"] = str(data_dir)
    run_id = uuid.uuid4().hex[:16]
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="cdd11-v1-connectivity",
        job_type="connectivity-check",
        name=f"connectivity-{timestamp}",
        id=run_id,
        resume="never",
        dir=str(run_dir),
        mode=args.mode,
        force=args.mode == "online",
        tags=["cdd11-v1", "connectivity"],
        config={"protocol": "cdd11-v1", "purpose": "connectivity-check"},
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
        (run_dir / "connectivity_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()

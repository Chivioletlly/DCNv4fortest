"""Command-line entry point for frozen AIO3-v1 DCNv4 U-Net training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from .checkpoint import load_checkpoint
from .runtime import MODEL_VARIANTS, load_run_config, prepare_new_run, seed_everything
from .training import run_training


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a signed-residual DCNv4 U-Net under AIO3-v1."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume exactly from an existing latest.pth checkpoint.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="Directory containing the frozen train/val/test manifests and audit.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="AIO3-v1 output root; a model-specific run directory is created.",
    )
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        help=(
            "Model for a new run. Omit for the frozen baseline; use "
            "degradation-aware for DAM + gated skips + dual-domain modulation."
        ),
    )
    parser.add_argument(
        "--run-kind",
        choices=("smoke", "pilot", "formal"),
        default="smoke",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--pause-at-step",
        type=int,
        help=(
            "Stop cleanly after this optimizer step, atomically save latest.pth, "
            "and mark the run paused. This execution-only control is intended for "
            "resume acceptance tests."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Use offline when the server cannot reach W&B; disabled is for diagnostics only.",
    )
    parser.add_argument(
        "--wandb-entity",
        help="Optional W&B account/team; when omitted the SDK default is resolved and recorded.",
    )
    return parser


def _resolve_run(args) -> tuple:
    resume_checkpoint: Optional[Path] = None
    if args.resume is not None:
        if (
            args.manifest_dir is not None
            or args.output_root is not None
            or args.model_variant is not None
        ):
            raise SystemExit(
                "--resume cannot be combined with "
                "--manifest-dir/--output-root/--model-variant"
            )
        resume_checkpoint = args.resume.expanduser().resolve()
        if resume_checkpoint.name != "latest.pth":
            raise SystemExit("Exact training resume must use checkpoints/latest.pth")
        checkpoint = load_checkpoint(resume_checkpoint)
        run_dir = Path(str(checkpoint["run_dir"])).expanduser().resolve()
        config = load_run_config(run_dir / "config.yaml")
        if Path(str(config["paths"]["run_dir"])).expanduser().resolve() != run_dir:
            raise SystemExit("Checkpoint run_dir differs from config.yaml")
        return run_dir, config, resume_checkpoint

    if args.manifest_dir is None or args.output_root is None:
        raise SystemExit("New runs require both --manifest-dir and --output-root")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    run_dir, config = prepare_new_run(
        repository_root=REPOSITORY_ROOT,
        manifest_dir=args.manifest_dir,
        output_root=args.output_root,
        run_kind=args.run_kind,
        seed=args.seed,
        num_workers=args.num_workers,
        wandb_mode=args.wandb_mode,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
        model_variant=args.model_variant or "baseline",
    )
    return run_dir, config, resume_checkpoint


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("AIO3 DCNv4 training requires an available CUDA GPU")
    run_dir, config, resume_checkpoint = _resolve_run(args)
    seed_everything(int(config["seed"]))
    print(f"AIO3 run directory: {run_dir}", flush=True)
    print(
        f"run_kind={config['run_kind']} seed={config['seed']} "
        f"max_steps={config['training']['max_steps']}",
        flush=True,
    )
    run_training(
        repository_root=REPOSITORY_ROOT,
        run_dir=run_dir,
        config=config,
        resume_checkpoint=resume_checkpoint,
        pause_at_step=args.pause_at_step,
    )


if __name__ == "__main__":
    main()

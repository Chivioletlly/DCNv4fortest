"""Command-line entry point for all three CDD-11-v1 training runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from .checkpoint import load_checkpoint
from .models import MODEL_IDS
from .runtime import load_run_config, prepare_new_run, seed_everything
from .training import run_training


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one frozen CDD-11-v1 model")
    parser.add_argument("--model", choices=MODEL_IDS)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--uformer-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-kind", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--inference-mode", choices=("native", "tiled"), default="native")
    parser.add_argument("--pause-at-step", type=int)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Use online for live dashboards or offline for later wandb sync.",
    )
    parser.add_argument(
        "--wandb-entity",
        help="Optional W&B account/team; the SDK default is used when omitted.",
    )
    parser.add_argument(
        "--wandb-project",
        default="cdd11-restoration",
        help="W&B project shared by all three comparison models.",
    )
    return parser


def _resolve_run(args):
    resume_checkpoint: Optional[Path] = None
    if args.resume is not None:
        forbidden = (
            args.model is not None
            or args.manifest_dir is not None
            or args.output_root is not None
            or args.uformer_root is not None
        )
        if forbidden:
            raise SystemExit(
                "--resume cannot be combined with --model/--manifest-dir/"
                "--output-root/--uformer-root"
            )
        resume_checkpoint = args.resume.expanduser().resolve()
        if resume_checkpoint.name != "latest.pth":
            raise SystemExit("Exact resume requires checkpoints/latest.pth")
        checkpoint = load_checkpoint(resume_checkpoint)
        run_dir = Path(str(checkpoint["run_dir"])).expanduser().resolve()
        config = load_run_config(run_dir / "config.yaml")
        return run_dir, config, resume_checkpoint
    if args.model is None or args.manifest_dir is None or args.output_root is None:
        raise SystemExit("New runs require --model, --manifest-dir, and --output-root")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    return (*prepare_new_run(
        repository_root=REPOSITORY_ROOT,
        manifest_dir=args.manifest_dir,
        output_root=args.output_root,
        model_id=args.model,
        run_kind=args.run_kind,
        seed=args.seed,
        num_workers=args.num_workers,
        microbatch_size=args.microbatch_size,
        inference_mode=args.inference_mode,
        uformer_root=args.uformer_root,
        run_name=args.run_name,
        wandb_mode=args.wandb_mode,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    ), None)


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CDD-11 training requires an available CUDA GPU")
    run_dir, config, resume_checkpoint = _resolve_run(args)
    seed_everything(int(config["seed"]))
    print(f"CDD-11 run directory: {run_dir}", flush=True)
    run_training(
        repository_root=REPOSITORY_ROOT,
        run_dir=run_dir,
        config=config,
        resume_checkpoint=resume_checkpoint,
        pause_at_step=args.pause_at_step,
    )


if __name__ == "__main__":
    main()

"""Failure-isolated Weights & Biases monitoring for AIO3-v1."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from .checkpoint import preserve_rng_state
from .runtime import MANIFEST_FILES, WANDB_VERSION, append_jsonl, atomic_write_json


class WandbMonitor:
    """Mirror already-persisted local metrics to W&B without affecting training."""

    def __init__(
        self,
        *,
        config: Mapping[str, object],
        run_dir: Path,
        resume: bool,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.resume = bool(resume)
        self.mode = str(config["monitoring"]["mode"])
        self.run = None
        self.wandb = None
        self.errors = 0
        if self.mode == "disabled":
            self._write_state(active=False, resolved_entity=None)
            return
        try:
            import wandb
        except ImportError as error:
            self._write_state(
                active=False,
                resolved_entity=None,
                initialization_error="wandb is not installed",
            )
            raise RuntimeError(
                "W&B monitoring is enabled but the wandb SDK is not installed"
            ) from error

        self.wandb = wandb
        if wandb.__version__ != WANDB_VERSION:
            raise RuntimeError(f"AIO3-v1 requires wandb=={WANDB_VERSION}, got {wandb.__version__}")
        output_root = Path(str(config["paths"]["output_root"]))
        cache_dir = output_root / ".wandb_cache"
        data_dir = output_root / ".wandb_staging"
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(self.run_dir)
        os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
        os.environ["WANDB_DATA_DIR"] = str(data_dir)
        os.environ["WANDB_MODE"] = self.mode
        os.environ["WANDB_PROJECT"] = str(config["monitoring"]["project"])
        requested_entity = config["monitoring"].get("entity")
        if requested_entity is not None:
            os.environ["WANDB_ENTITY"] = str(requested_entity)
        configured_tags = config["monitoring"].get("tags")
        tags = (
            [str(value) for value in configured_tags]
            if configured_tags is not None
            else ["aio3-v1", "dcnv4", "signed-residual", str(config["run_kind"])]
        )
        try:
            with (self.run_dir / "environment.json").open("r", encoding="utf-8") as stream:
                environment = json.load(stream)
            wandb_config = dict(config)
            wandb_config["environment"] = environment
            with preserve_rng_state():
                self.run = wandb.init(
                    entity=config["monitoring"].get("entity"),
                    project=str(config["monitoring"]["project"]),
                    group=str(config["monitoring"]["group"]),
                    job_type="train",
                    name=str(config["run_name"]),
                    id=str(config["monitoring"]["wandb_run_id"]),
                    resume="must" if resume else "never",
                    dir=str(self.run_dir),
                    config=wandb_config,
                    tags=tags,
                    mode=self.mode,
                    force=self.mode == "online",
                )
        except Exception as error:
            self._write_state(
                active=False,
                resolved_entity=None,
                initialization_error=f"{type(error).__name__}: {error}",
            )
            raise RuntimeError(
                f"Could not initialize W&B in {self.mode!r} mode; "
                "use --wandb-mode offline when network access is unavailable"
            ) from error
        resolved_entity = getattr(self.run, "entity", None)
        self._write_state(active=True, resolved_entity=resolved_entity)
        self._safe_call("define_metrics", self._define_metrics)
        if not self.resume:
            self._safe_call("manifest_artifact", self._log_manifest_artifact)

    @property
    def active(self) -> bool:
        return self.run is not None

    def _write_state(
        self,
        *,
        active: bool,
        resolved_entity: Optional[str],
        initialization_error: Optional[str] = None,
    ) -> None:
        value = {
            "provider": "wandb",
            "mode": self.mode,
            "active": bool(active),
            "run_id": self.config["monitoring"]["wandb_run_id"],
            "run_name": self.config["run_name"],
            "project": self.config["monitoring"]["project"],
            "group": self.config["monitoring"]["group"],
            "requested_entity": self.config["monitoring"].get("entity"),
            "resolved_entity": resolved_entity,
            "resume": self.resume,
            "errors": self.errors,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if initialization_error is not None:
            value["initialization_error"] = initialization_error
        atomic_write_json(self.run_dir / "wandb_state.json", value)

    def _record_error(self, operation: str, error: Exception) -> None:
        self.errors += 1
        append_jsonl(
            self.run_dir / "logs" / "wandb_errors.jsonl",
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "operation": operation,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        self._write_state(
            active=self.active,
            resolved_entity=getattr(self.run, "entity", None),
        )
        print(f"W&B warning during {operation}: {type(error).__name__}: {error}", flush=True)

    def _safe_call(self, operation: str, function: Callable[[], None]) -> None:
        if not self.active:
            return
        try:
            with preserve_rng_state():
                function()
        except Exception as error:
            self._record_error(operation, error)

    def _define_metrics(self) -> None:
        self.run.define_metric("global_step")
        for namespace in (
            "train/*",
            "diagnostics/*",
            "system/*",
            "val/*",
            "test/*",
        ):
            self.run.define_metric(namespace, step_metric="global_step")

        # W&B 0.25.1 does not reliably apply a one-star namespace definition
        # to metric names below another slash.  Define every nested scalar and
        # media key exactly so charts never fall back to W&B's internal Step.
        nested_metrics = [
            "diagnostics/denoise/residual_negative_fraction",
            "diagnostics/derain/residual_negative_fraction",
            "diagnostics/dehaze/residual_negative_fraction",
            "val/denoise/mean/psnr",
            "val/denoise/mean/ssim",
            "val/denoise/raw_l1",
            "val/denoise/residual_negative_fraction",
            "val/derain/psnr",
            "val/derain/ssim",
            "val/derain/images",
            "val/derain/raw_l1",
            "val/derain/residual_negative_fraction",
            "val/dehaze/psnr",
            "val/dehaze/ssim",
            "val/dehaze/images",
            "val/dehaze/raw_l1",
            "val/dehaze/residual_negative_fraction",
            "test/bsd68/mean/psnr",
            "test/bsd68/mean/ssim",
            "test/rain100l/psnr",
            "test/rain100l/ssim",
            "test/rain100l/images",
            "test/sots_outdoor/psnr",
            "test/sots_outdoor/ssim",
            "test/sots_outdoor/images",
            "test/macro/psnr",
            "test/macro/ssim",
            "test/per_image_metrics",
            "test/fixed_gallery",
        ]
        for sigma in (15, 25, 50):
            nested_metrics.extend(
                f"val/denoise/sigma{sigma}/{metric}" for metric in ("psnr", "ssim", "images")
            )
            nested_metrics.extend(
                f"test/bsd68/sigma{sigma}/{metric}" for metric in ("psnr", "ssim", "images")
            )
        for metric in nested_metrics:
            self.run.define_metric(metric, step_metric="global_step")

        self.run.define_metric(
            "val/macro/psnr",
            step_metric="global_step",
            summary="max",
        )
        self.run.define_metric(
            "val/macro/ssim",
            step_metric="global_step",
            summary="max",
        )

    def _log_manifest_artifact(self) -> None:
        with (self.run_dir / "manifests" / "data_audit.json").open("r", encoding="utf-8") as stream:
            audit = json.load(stream)
        artifact = self.wandb.Artifact(
            name="aio3-v1-manifests",
            type="dataset",
            metadata={
                "protocol": self.config["protocol"],
                "manifest_sha256": self.config["data"]["manifest_sha256"],
                "data_root": audit.get("data_root"),
            },
        )
        manifest_dir = self.run_dir / "manifests"
        for filename in (*MANIFEST_FILES, "data_audit.json", "visual_samples.json"):
            artifact.add_file(str(manifest_dir / filename), name=filename)
        artifact.add_file(
            str(self.run_dir / "AIO3_TRAINING_EVALUATION_PROTOCOL.md"),
            name="AIO3_TRAINING_EVALUATION_PROTOCOL.md",
        )
        ablation_plan = self.run_dir / "AIO3_DCNV4_ABLATION_PLAN.md"
        if ablation_plan.is_file():
            artifact.add_file(
                str(ablation_plan),
                name="AIO3_DCNV4_ABLATION_PLAN.md",
            )
        artifact.add_file(str(self.run_dir / "config.yaml"), name="config.yaml")
        self.run.log_artifact(artifact)

    def log_scalars(self, metrics: Mapping[str, object]) -> None:
        payload = dict(metrics)
        payload.pop("timestamp_utc", None)
        self._safe_call("log_scalars", lambda: self.run.log(payload))

    def log_validation(
        self,
        *,
        global_step: int,
        summary: Mapping[str, float],
        visuals: Sequence[Mapping[str, object]],
        residual_histogram: Sequence[float],
    ) -> None:
        def log() -> None:
            payload = {"global_step": int(global_step)}
            payload.update({f"val/{key}": value for key, value in summary.items()})
            if visuals:
                columns = [
                    "global_step",
                    "task",
                    "sigma",
                    "sample_id",
                    "input",
                    "prediction",
                    "target",
                    "absolute_error",
                    "signed_residual",
                    "psnr",
                    "ssim",
                    "residual_mean",
                    "residual_negative_fraction",
                ]
                table = self.wandb.Table(columns=columns)
                for visual in visuals:
                    table.add_data(
                        int(global_step),
                        visual["task"],
                        visual.get("sigma"),
                        visual["sample_id"],
                        self.wandb.Image(str(visual["input_path"])),
                        self.wandb.Image(str(visual["prediction_path"])),
                        self.wandb.Image(str(visual["target_path"])),
                        self.wandb.Image(str(visual["absolute_error_path"])),
                        self.wandb.Image(str(visual["signed_residual_path"])),
                        visual["psnr"],
                        visual["ssim"],
                        visual["residual_mean"],
                        visual["residual_negative_fraction"],
                    )
                payload["val/fixed_samples"] = table
            if residual_histogram:
                payload["diagnostics/validation_signed_residual"] = self.wandb.Histogram(
                    residual_histogram
                )
            self.run.log(payload)

        self._safe_call("log_validation", log)

    def update_best_summary(self, best_metrics: Mapping[str, object]) -> None:
        def update() -> None:
            self.run.summary["best/val_macro_psnr"] = best_metrics["macro_psnr"]
            self.run.summary["best/val_macro_ssim"] = best_metrics["macro_ssim"]
            self.run.summary["best/global_step"] = best_metrics["global_step"]

        self._safe_call("update_best_summary", update)

    def log_best_checkpoint(
        self,
        checkpoint_path: Path,
        best_metrics: Mapping[str, object],
    ) -> None:
        def log() -> None:
            artifact = self.wandb.Artifact(
                name=f"aio3-v1-dcnv4-unet-seed{self.config['seed']}",
                type="model",
                metadata={
                    "protocol": self.config["protocol"],
                    "run_id": self.config["monitoring"]["wandb_run_id"],
                    "repository_commit": self.config["source"]["repository_commit"],
                    "manifest_sha256": self.config["data"]["manifest_sha256"],
                    "best_metrics": dict(best_metrics),
                },
            )
            artifact.add_file(str(checkpoint_path), name="best_macro_psnr.pth")
            artifact.add_file(str(self.run_dir / "config.yaml"), name="config.yaml")
            artifact.add_file(
                str(self.run_dir / "environment.json"),
                name="environment.json",
            )
            artifact.add_file(
                str(self.run_dir / "manifests" / "data_audit.json"),
                name="data_audit.json",
            )
            final_step = int(self.config["training"]["max_steps"])
            artifact.add_file(
                str(self.run_dir / "validation" / f"metrics_step_{final_step:06d}.json"),
                name="final_validation_metrics.json",
            )
            self.run.log_artifact(
                artifact,
                aliases=["best", f"seed{self.config['seed']}"],
            )

        self._safe_call("best_checkpoint_artifact", log)

    def log_test_evaluation(
        self,
        *,
        global_step: int,
        summary: Mapping[str, float],
        per_image: Sequence[Mapping[str, object]],
        visuals: Sequence[Mapping[str, object]],
    ) -> None:
        """Append frozen formal-test metrics to the resumed training run."""

        def log() -> None:
            payload = {"global_step": int(global_step)}
            payload.update({f"test/{key}": value for key, value in summary.items()})
            metric_columns = [
                "dataset",
                "task",
                "sigma",
                "sample_id",
                "psnr",
                "ssim",
                "inference_time_seconds",
            ]
            metric_table = self.wandb.Table(columns=metric_columns)
            for row in per_image:
                metric_table.add_data(*(row[column] for column in metric_columns))
            payload["test/per_image_metrics"] = metric_table

            gallery_columns = [
                "task",
                "sigma",
                "sample_id",
                "input",
                "prediction",
                "target",
                "absolute_error",
                "signed_residual",
                "psnr",
                "ssim",
                "residual_mean",
                "residual_negative_fraction",
            ]
            gallery_table = self.wandb.Table(columns=gallery_columns)
            for visual in visuals:
                gallery_table.add_data(
                    visual["task"],
                    visual.get("sigma"),
                    visual["sample_id"],
                    self.wandb.Image(str(visual["input_path"])),
                    self.wandb.Image(str(visual["prediction_path"])),
                    self.wandb.Image(str(visual["target_path"])),
                    self.wandb.Image(str(visual["absolute_error_path"])),
                    self.wandb.Image(str(visual["signed_residual_path"])),
                    visual["psnr"],
                    visual["ssim"],
                    visual["residual_mean"],
                    visual["residual_negative_fraction"],
                )
            payload["test/fixed_gallery"] = gallery_table
            self.run.log(payload)

        self._safe_call("log_test_evaluation", log)

    def log_evaluation_artifact(
        self,
        test_dir: Path,
        *,
        metadata: Mapping[str, object],
    ) -> None:
        """Upload numerical test outputs and the fixed gallery, never all predictions."""

        def log() -> None:
            artifact = self.wandb.Artifact(
                name=f"aio3-v1-dcnv4-unet-seed{self.config['seed']}-evaluation",
                type="evaluation",
                metadata=dict(metadata),
            )
            test_dir_path = Path(test_dir)
            for filename in (
                "metrics.json",
                "metrics.csv",
                "per_image_metrics.csv",
                "gallery.json",
                "gallery_selection.json",
            ):
                artifact.add_file(str(test_dir_path / filename), name=filename)
            gallery_dir = test_dir_path / "gallery"
            for path in sorted(gallery_dir.rglob("*.png")):
                relative = path.relative_to(test_dir_path).as_posix()
                artifact.add_file(str(path), name=relative)
            self.run.log_artifact(
                artifact,
                aliases=["final", f"seed{self.config['seed']}"],
            )

        self._safe_call("evaluation_artifact", log)

    def finish(self) -> None:
        if not self.active:
            return
        resolved_entity = getattr(self.run, "entity", None)
        self._safe_call("finish", self.run.finish)
        self.run = None
        self._write_state(active=False, resolved_entity=resolved_entity)

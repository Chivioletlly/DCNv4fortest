"""Failure-isolated Weights & Biases monitoring for CDD-11-v1."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from aio3_runner.checkpoint import preserve_rng_state

from .protocol import ARITY_GROUPS, DEGRADATIONS
from .runtime import (
    AUXILIARY_DATA_FILES,
    MANIFEST_FILES,
    WANDB_VERSION,
    append_jsonl,
    atomic_write_json,
)


class CDD11WandbMonitor:
    """Mirror locally persisted CDD-11 metrics to W&B without changing training."""

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
        monitoring = config["monitoring"]
        self.mode = str(monitoring.get("mode", "disabled"))
        self.run = None
        self.wandb = None
        self.errors = 0
        if self.mode == "disabled":
            self._write_state(active=False, resolved_entity=None, url=None)
            return

        try:
            import wandb
        except ImportError as error:
            self._write_state(
                active=False,
                resolved_entity=None,
                url=None,
                initialization_error="wandb is not installed",
            )
            raise RuntimeError(
                "W&B monitoring is enabled but the wandb SDK is not installed"
            ) from error
        if wandb.__version__ != WANDB_VERSION:
            raise RuntimeError(
                f"CDD-11-v1 requires wandb=={WANDB_VERSION}, got {wandb.__version__}"
            )

        self.wandb = wandb
        output_root = Path(str(config["paths"]["output_root"]))
        cache_dir = output_root / ".wandb_cache"
        data_dir = output_root / ".wandb_staging"
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(self.run_dir)
        os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
        os.environ["WANDB_DATA_DIR"] = str(data_dir)
        os.environ["WANDB_MODE"] = self.mode
        os.environ["WANDB_PROJECT"] = str(monitoring["project"])
        requested_entity = monitoring.get("entity")
        if requested_entity is not None:
            os.environ["WANDB_ENTITY"] = str(requested_entity)

        try:
            with preserve_rng_state():
                self.run = wandb.init(
                    entity=requested_entity,
                    project=str(monitoring["project"]),
                    group=str(monitoring["group"]),
                    job_type="train",
                    name=str(config["run_name"]),
                    id=str(monitoring["wandb_run_id"]),
                    resume="must" if resume else "never",
                    dir=str(self.run_dir),
                    config=dict(config),
                    tags=[str(value) for value in monitoring["tags"]],
                    mode=self.mode,
                    force=self.mode == "online",
                )
        except Exception as error:
            self._write_state(
                active=False,
                resolved_entity=None,
                url=None,
                initialization_error=f"{type(error).__name__}: {error}",
            )
            raise RuntimeError(
                f"Could not initialize W&B in {self.mode!r} mode; verify login/network "
                "or create a new run with --wandb-mode offline"
            ) from error

        self._write_state(
            active=True,
            resolved_entity=getattr(self.run, "entity", None),
            url=getattr(self.run, "url", None),
        )
        print(
            "W&B monitoring: "
            f"mode={self.mode} id={monitoring['wandb_run_id']} "
            f"url={getattr(self.run, 'url', None)}",
            flush=True,
        )
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
        url: Optional[str],
        initialization_error: Optional[str] = None,
    ) -> None:
        monitoring = self.config["monitoring"]
        value = {
            "provider": "wandb",
            "mode": self.mode,
            "active": bool(active),
            "run_id": monitoring.get("wandb_run_id"),
            "run_name": self.config["run_name"],
            "project": monitoring.get("project"),
            "group": monitoring.get("group"),
            "requested_entity": monitoring.get("entity"),
            "resolved_entity": resolved_entity,
            "url": url,
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
            url=getattr(self.run, "url", None),
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
        for namespace in ("train/*", "system/*", "val/*"):
            self.run.define_metric(namespace, step_metric="global_step")

        nested = []
        for degradation in DEGRADATIONS:
            nested.append(f"train/{degradation}_l1")
            nested.extend(
                f"val/{degradation}/{metric}"
                for metric in ("psnr", "ssim", "images", "raw_l1")
            )
        for group_name in ARITY_GROUPS:
            nested.extend(
                f"val/{group_name}/{metric}"
                for metric in ("psnr", "ssim", "categories")
            )
        nested.extend(("val/images", "val/fixed_samples"))
        for metric in nested:
            self.run.define_metric(metric, step_metric="global_step")
        for metric in ("val/macro/psnr", "val/macro/ssim"):
            self.run.define_metric(metric, step_metric="global_step", summary="max")

    def _log_manifest_artifact(self) -> None:
        manifest_dir = self.run_dir / "manifests"
        audit = json.loads(
            (manifest_dir / "data_audit.json").read_text(encoding="utf-8")
        )
        artifact = self.wandb.Artifact(
            name="cdd11-v1-frozen-manifests",
            type="dataset",
            metadata={
                "protocol": self.config["protocol"],
                "manifest_sha256": self.config["data"]["manifest_sha256"],
                "source_counts": audit.get("source_counts"),
            },
        )
        for filename in (*MANIFEST_FILES, *AUXILIARY_DATA_FILES):
            artifact.add_file(str(manifest_dir / filename), name=filename)
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
    ) -> None:
        def log() -> None:
            payload = {"global_step": int(global_step)}
            payload.update({f"val/{key}": value for key, value in summary.items()})
            if visuals:
                columns = [
                    "global_step",
                    "degradation",
                    "arity",
                    "sample_id",
                    "input",
                    "prediction",
                    "target",
                    "absolute_error",
                    "signed_residual",
                    "psnr",
                    "ssim",
                ]
                table = self.wandb.Table(columns=columns)
                for visual in visuals:
                    table.add_data(
                        int(global_step),
                        visual["degradation"],
                        visual["arity"],
                        visual["sample_id"],
                        self.wandb.Image(str(visual["input_path"])),
                        self.wandb.Image(str(visual["prediction_path"])),
                        self.wandb.Image(str(visual["target_path"])),
                        self.wandb.Image(str(visual["absolute_error_path"])),
                        self.wandb.Image(str(visual["signed_residual_path"])),
                        visual["psnr"],
                        visual["ssim"],
                    )
                payload["val/fixed_samples"] = table
            self.run.log(payload)

        self._safe_call("log_validation", log)

    def update_best_summary(self, best_metrics: Mapping[str, object]) -> None:
        def update() -> None:
            self.run.summary["best/val_macro_psnr"] = best_metrics["macro_psnr"]
            self.run.summary["best/val_macro_ssim"] = best_metrics["macro_ssim"]
            self.run.summary["best/global_step"] = best_metrics["global_step"]

        self._safe_call("update_best_summary", update)

    def finish(self) -> None:
        if not self.active:
            return
        resolved_entity = getattr(self.run, "entity", None)
        url = getattr(self.run, "url", None)
        self._safe_call("finish", self.run.finish)
        self.run = None
        self._write_state(active=False, resolved_entity=resolved_entity, url=url)


__all__ = ["CDD11WandbMonitor"]

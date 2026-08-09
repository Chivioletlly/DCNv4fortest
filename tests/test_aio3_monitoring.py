import json
import random
import shutil
import sys
import types
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from aio3_runner.monitoring import WandbMonitor


@contextmanager
def _workspace_temporary_directory():
    parent = REPOSITORY_ROOT / ".tmp_aio3_monitoring_tests"
    parent.mkdir(exist_ok=True)
    path = parent / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)
        try:
            parent.rmdir()
        except OSError:
            pass


class _FakeArtifact:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.files = []

    def add_file(self, path, name=None):
        assert Path(path).is_file()
        self.files.append((path, name))


class _FakeTable:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def add_data(self, *values):
        self.rows.append(values)


class _FakeRun:
    def __init__(self):
        self.entity = "unit-entity"
        self.project = "aio3-restoration"
        self.url = "https://example.invalid/unit-run"
        self.summary = {}
        self.logs = []
        self.artifacts = []
        self.metric_definitions = []
        self.finished = False

    def define_metric(self, *args, **kwargs):
        random.random()
        torch.rand(1)
        self.metric_definitions.append((args, kwargs))

    def log(self, value):
        random.random()
        torch.rand(1)
        self.logs.append(value)

    def log_artifact(self, artifact, aliases=None):
        random.random()
        torch.rand(1)
        self.artifacts.append((artifact, aliases))

    def finish(self):
        random.random()
        torch.rand(1)
        self.finished = True


def _install_fake_wandb():
    module = types.ModuleType("wandb")
    module.__version__ = "0.25.1"
    module.run = _FakeRun()

    def init(**kwargs):
        random.random()
        torch.rand(1)
        module.init_kwargs = kwargs
        return module.run

    module.init = init
    module.Artifact = lambda **kwargs: _FakeArtifact(**kwargs)
    module.Table = lambda columns: _FakeTable(columns)
    module.Image = lambda path: ("image", path)
    module.Histogram = lambda values: ("histogram", tuple(values))
    sys.modules["wandb"] = module
    return module


def _config(run_dir: Path, mode="online"):
    return {
        "protocol": "aio3-v1",
        "run_kind": "smoke",
        "run_name": "monitoring-unit-test",
        "seed": 3407,
        "training": {"max_steps": 10},
        "data": {"manifest_sha256": {"train.jsonl": "abc"}},
        "source": {"repository_commit": "0123456789"},
        "monitoring": {
            "mode": mode,
            "version": "0.25.1",
            "entity": None,
            "project": "aio3-restoration",
            "group": "aio3-v1",
            "wandb_run_id": "monitor-unit-id",
        },
        "paths": {"run_dir": str(run_dir), "output_root": str(run_dir)},
    }


def _prepare_run_files(root: Path):
    (root / "logs").mkdir()
    manifest_dir = root / "manifests"
    manifest_dir.mkdir()
    for filename in (
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
        "data_audit.json",
        "visual_samples.json",
    ):
        (manifest_dir / filename).write_text("{}\n", encoding="utf-8")
    (root / "environment.json").write_text(
        json.dumps({"torch": torch.__version__}) + "\n",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "AIO3_TRAINING_EVALUATION_PROTOCOL.md").write_text(
        "# unit protocol\n", encoding="utf-8"
    )


def test_wandb_monitor_preserves_rng_and_logs_frozen_axes_and_artifacts():
    fake_wandb = _install_fake_wandb()
    with _workspace_temporary_directory() as root:
        _prepare_run_files(root)
        torch.manual_seed(789)
        random.seed(789)
        expected_torch = torch.rand(4)
        expected_python = random.random()
        torch.manual_seed(789)
        random.seed(789)

        monitor = WandbMonitor(config=_config(root), run_dir=root, resume=False)
        monitor.log_scalars({"global_step": 10, "train/loss": 0.5})
        monitor.log_validation(
            global_step=10,
            summary={"macro/psnr": 20.0, "macro/ssim": 0.8},
            visuals=[
                {
                    "task": "denoise",
                    "sigma": 25,
                    "sample_id": "unit-sample",
                    "input_path": root / "input.png",
                    "prediction_path": root / "prediction.png",
                    "target_path": root / "target.png",
                    "absolute_error_path": root / "error.png",
                    "signed_residual_path": root / "residual.png",
                    "psnr": 20.0,
                    "ssim": 0.8,
                    "residual_mean": -0.01,
                    "residual_negative_fraction": 0.6,
                }
            ],
            residual_histogram=[-0.1, 0.0, 0.1],
        )
        monitor.update_best_summary({"macro_psnr": 20.0, "macro_ssim": 0.8, "global_step": 10})
        monitor.finish()

        torch.testing.assert_close(torch.rand(4), expected_torch)
        assert random.random() == expected_python
        assert fake_wandb.init_kwargs["resume"] == "never"
        assert fake_wandb.init_kwargs["id"] == "monitor-unit-id"
        definitions = {
            args[0]: kwargs for args, kwargs in fake_wandb.run.metric_definitions
        }
        for nested_metric in (
            "diagnostics/denoise/residual_negative_fraction",
            "val/denoise/sigma25/psnr",
            "val/derain/ssim",
            "val/dehaze/raw_l1",
            "test/bsd68/sigma50/psnr",
            "test/sots_outdoor/ssim",
            "test/fixed_gallery",
        ):
            assert definitions[nested_metric]["step_metric"] == "global_step"
        assert definitions["val/macro/psnr"] == {
            "step_metric": "global_step",
            "summary": "max",
        }
        assert definitions["val/macro/ssim"] == {
            "step_metric": "global_step",
            "summary": "max",
        }
        assert all(log["global_step"] == 10 for log in fake_wandb.run.logs)
        assert "val/fixed_samples" in fake_wandb.run.logs[-1]
        assert fake_wandb.run.summary["best/global_step"] == 10
        assert fake_wandb.run.artifacts
        assert fake_wandb.run.finished
        with (root / "wandb_state.json").open("r", encoding="utf-8") as stream:
            state = json.load(stream)
        assert state["active"] is False
        assert state["errors"] == 0


def test_disabled_monitor_does_not_require_wandb_sdk():
    existing_wandb = sys.modules.get("wandb")
    sys.modules["wandb"] = None
    try:
        with _workspace_temporary_directory() as root:
            _prepare_run_files(root)
            monitor = WandbMonitor(
                config=_config(root, mode="disabled"),
                run_dir=root,
                resume=False,
            )
            assert not monitor.active
            monitor.log_scalars({"global_step": 1, "train/loss": 1.0})
            monitor.finish()
    finally:
        if existing_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = existing_wandb


def test_resumed_monitor_logs_formal_test_tables_and_small_artifact():
    fake_wandb = _install_fake_wandb()
    with _workspace_temporary_directory() as root:
        _prepare_run_files(root)
        test_dir = root / "test"
        gallery_dir = test_dir / "gallery" / "sample"
        gallery_dir.mkdir(parents=True)
        for filename in (
            "metrics.json",
            "metrics.csv",
            "per_image_metrics.csv",
            "gallery.json",
            "gallery_selection.json",
        ):
            (test_dir / filename).write_text("{}\n", encoding="utf-8")
        gallery_image = gallery_dir / "prediction.png"
        gallery_image.write_bytes(b"unit")
        visual_paths = {}
        for name in (
            "input",
            "prediction",
            "target",
            "absolute_error",
            "signed_residual",
        ):
            path = root / f"{name}.png"
            path.write_bytes(b"unit")
            visual_paths[f"{name}_path"] = path

        monitor = WandbMonitor(config=_config(root), run_dir=root, resume=True)
        monitor.log_test_evaluation(
            global_step=10,
            summary={"macro/psnr": 20.0, "macro/ssim": 0.8},
            per_image=[
                {
                    "dataset": "BSD68_sigma25",
                    "task": "denoise",
                    "sigma": 25,
                    "sample_id": "unit-sample",
                    "psnr": 20.0,
                    "ssim": 0.8,
                    "inference_time_seconds": 0.01,
                }
            ],
            visuals=[
                {
                    "task": "denoise",
                    "sigma": 25,
                    "sample_id": "unit-sample",
                    "psnr": 20.0,
                    "ssim": 0.8,
                    "residual_mean": -0.01,
                    "residual_negative_fraction": 0.6,
                    **visual_paths,
                }
            ],
        )
        monitor.log_evaluation_artifact(
            test_dir,
            metadata={"checkpoint_sha256": "unit"},
        )
        monitor.finish()

        assert fake_wandb.init_kwargs["resume"] == "must"
        test_log = fake_wandb.run.logs[-1]
        assert test_log["global_step"] == 10
        assert test_log["test/macro/psnr"] == 20.0
        assert len(test_log["test/per_image_metrics"].rows) == 1
        assert len(test_log["test/fixed_gallery"].rows) == 1
        artifact, aliases = fake_wandb.run.artifacts[-1]
        assert artifact.kwargs["type"] == "evaluation"
        assert aliases == ["final", "seed3407"]
        uploaded_names = {name for _, name in artifact.files}
        assert "gallery/sample/prediction.png" in uploaded_names
        assert all("predictions/" not in name for name in uploaded_names)


if __name__ == "__main__":
    tests = [
        test_wandb_monitor_preserves_rng_and_logs_frozen_axes_and_artifacts,
        test_disabled_monitor_does_not_require_wandb_sdk,
        test_resumed_monitor_logs_formal_test_tables_and_small_artifact,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")

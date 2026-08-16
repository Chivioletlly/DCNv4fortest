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

from cdd11_runner.monitoring import CDD11WandbMonitor
from cdd11_runner.protocol import CDD11_PROTOCOL_VERSION
from cdd11_runner.runtime import AUXILIARY_DATA_FILES, MANIFEST_FILES


@contextmanager
def _temporary_directory():
    parent = REPOSITORY_ROOT / ".tmp_cdd11_monitoring_tests"
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
        self.project = "cdd11-restoration"
        self.url = "https://example.invalid/cdd11-run"
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
    sys.modules["wandb"] = module
    return module


def _config(root: Path, mode="online"):
    return {
        "protocol": CDD11_PROTOCOL_VERSION,
        "run_kind": "pilot",
        "run_name": "unet-pilot-wandb-test",
        "seed": 3407,
        "model": {"id": "unet"},
        "training": {"max_steps": 5000},
        "data": {"manifest_sha256": {"train.jsonl": "abc"}},
        "source": {"repository_commit": "main-commit", "uformer_commit": None},
        "monitoring": {
            "provider": "wandb",
            "version": "0.25.1",
            "mode": mode,
            "entity": None,
            "project": "cdd11-restoration",
            "group": "cdd11-v1-pilot-native-seed3407",
            "wandb_run_id": "cdd11-monitor-id",
            "tags": ["cdd11-v1", "pilot", "unet"],
        },
        "paths": {
            "run_dir": str(root),
            "output_root": str(root.parent),
        },
    }


def _prepare_run_files(root: Path):
    (root / "logs").mkdir()
    (root / "config.yaml").write_text("protocol: cdd11-v1\n", encoding="utf-8")
    manifest_dir = root / "manifests"
    manifest_dir.mkdir()
    for filename in (*MANIFEST_FILES, *AUXILIARY_DATA_FILES):
        value = {"source_counts": {}} if filename == "data_audit.json" else {}
        (manifest_dir / filename).write_text(
            json.dumps(value) + "\n", encoding="utf-8"
        )


def test_cdd11_monitor_preserves_rng_logs_scalars_visuals_and_artifacts():
    fake_wandb = _install_fake_wandb()
    with _temporary_directory() as root:
        _prepare_run_files(root)
        visual_paths = {}
        for name in ("input", "prediction", "target", "absolute_error", "signed_residual"):
            path = root / f"{name}.png"
            path.write_bytes(b"unit")
            visual_paths[f"{name}_path"] = path

        torch.manual_seed(3407)
        random.seed(3407)
        expected_torch = torch.rand(4)
        expected_python = random.random()
        torch.manual_seed(3407)
        random.seed(3407)

        monitor = CDD11WandbMonitor(
            config=_config(root), run_dir=root, resume=False
        )
        monitor.log_scalars({"global_step": 50, "train/loss": 0.1})
        monitor.log_validation(
            global_step=5000,
            summary={"macro/psnr": 20.0, "macro/ssim": 0.8},
            visuals=[
                {
                    "degradation": "haze",
                    "arity": 1,
                    "sample_id": "unit-sample",
                    "psnr": 20.0,
                    "ssim": 0.8,
                    **visual_paths,
                }
            ],
        )
        monitor.update_best_summary(
            {"macro_psnr": 20.0, "macro_ssim": 0.8, "global_step": 5000}
        )
        monitor.finish()

        torch.testing.assert_close(torch.rand(4), expected_torch)
        assert random.random() == expected_python
        assert fake_wandb.init_kwargs["resume"] == "never"
        assert fake_wandb.init_kwargs["id"] == "cdd11-monitor-id"
        definitions = {
            args[0]: kwargs for args, kwargs in fake_wandb.run.metric_definitions
        }
        assert definitions["val/haze/psnr"]["step_metric"] == "global_step"
        assert definitions["val/macro/psnr"]["summary"] == "max"
        assert fake_wandb.run.logs[0]["global_step"] == 50
        assert "val/fixed_samples" in fake_wandb.run.logs[-1]
        assert len(fake_wandb.run.logs[-1]["val/fixed_samples"].rows) == 1
        assert fake_wandb.run.summary["best/global_step"] == 5000
        assert fake_wandb.run.artifacts
        assert fake_wandb.run.finished
        state = json.loads((root / "wandb_state.json").read_text(encoding="utf-8"))
        assert state["active"] is False
        assert state["errors"] == 0
        assert state["url"] == "https://example.invalid/cdd11-run"


def test_cdd11_monitor_resume_requires_same_run_id():
    fake_wandb = _install_fake_wandb()
    with _temporary_directory() as root:
        _prepare_run_files(root)
        monitor = CDD11WandbMonitor(
            config=_config(root), run_dir=root, resume=True
        )
        monitor.finish()
        assert fake_wandb.init_kwargs["resume"] == "must"
        assert fake_wandb.init_kwargs["id"] == "cdd11-monitor-id"


def test_disabled_cdd11_monitor_does_not_import_sdk():
    existing = sys.modules.get("wandb")
    sys.modules["wandb"] = None
    try:
        with _temporary_directory() as root:
            _prepare_run_files(root)
            monitor = CDD11WandbMonitor(
                config=_config(root, mode="disabled"),
                run_dir=root,
                resume=False,
            )
            assert not monitor.active
            monitor.log_scalars({"global_step": 1, "train/loss": 1.0})
            monitor.finish()
    finally:
        if existing is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = existing


if __name__ == "__main__":
    tests = [
        test_cdd11_monitor_preserves_rng_logs_scalars_visuals_and_artifacts,
        test_cdd11_monitor_resume_requires_same_run_id,
        test_disabled_cdd11_monitor_does_not_import_sdk,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")

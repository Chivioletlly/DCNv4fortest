import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from aio3_runner.schedule import WarmupCosineScheduler
from cdd11_runner.checkpoint import (
    atomic_torch_save,
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    validate_checkpoint_identity,
)
from cdd11_runner.runtime import build_run_config


@contextmanager
def _temporary_directory():
    parent = REPOSITORY_ROOT / ".tmp_cdd11_checkpoint_tests"
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


def _config(run_dir: Path):
    return build_run_config(
        model_id="unet",
        run_kind="smoke",
        seed=3407,
        run_name="checkpoint-test",
        run_dir=run_dir,
        manifest_hashes={"train.jsonl": "a", "val.jsonl": "b"},
        repository_state={"commit": "main-commit", "dirty": False},
        uformer_state=None,
        uformer_root=None,
        num_workers=0,
        microbatch_size=1,
        inference_mode="native",
    )


def test_checkpoint_round_trip_restores_model_optimizer_scheduler_and_identity():
    with _temporary_directory() as root:
        torch.manual_seed(3407)
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = WarmupCosineScheduler(
            optimizer,
            base_lr=2e-4,
            min_lr=1e-6,
            warmup_steps=10,
            max_steps=100,
        )
        loss = model(torch.rand(2, 4)).abs().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        saved_parameters = [value.detach().clone() for value in model.parameters()]
        config = _config(root)
        checkpoint = build_checkpoint(
            model=model,
            architecture={"name": "test"},
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=1,
            best_metrics={"macro_psnr": None},
            config=config,
        )
        path = root / "latest.pth"
        atomic_torch_save(checkpoint, path)
        loaded = load_checkpoint(path)
        validate_checkpoint_identity(
            loaded,
            config=config,
            repository_commit="main-commit",
            uformer_commit=None,
        )

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        restore_training_state(
            loaded,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        assert scheduler.completed_steps == 1
        for actual, expected in zip(model.parameters(), saved_parameters):
            torch.testing.assert_close(actual, expected)


def test_checkpoint_identity_rejects_manifest_drift():
    with _temporary_directory() as root:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = WarmupCosineScheduler(
            optimizer,
            base_lr=2e-4,
            min_lr=1e-6,
            warmup_steps=1,
            max_steps=100,
        )
        config = _config(root)
        checkpoint = build_checkpoint(
            model=model,
            architecture={"name": "test"},
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=0,
            best_metrics={},
            config=config,
        )
        changed = _config(root)
        changed["data"]["manifest_sha256"]["train.jsonl"] = "different"
        try:
            validate_checkpoint_identity(
                checkpoint,
                config=changed,
                repository_commit="main-commit",
                uformer_commit=None,
            )
        except RuntimeError as error:
            assert "manifest_sha256" in str(error)
        else:
            raise AssertionError("Expected manifest drift to block resume")

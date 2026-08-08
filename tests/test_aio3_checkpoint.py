import random
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from aio3_runner.checkpoint import (
    atomic_torch_save,
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    validate_checkpoint_identity,
)
from aio3_runner.schedule import WarmupCosineScheduler


@contextmanager
def _workspace_temporary_directory():
    parent = REPOSITORY_ROOT / ".tmp_aio3_checkpoint_tests"
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
    return {
        "protocol": "aio3-v1",
        "run_name": "unit-test",
        "data": {"manifest_sha256": {"train.jsonl": "abc"}},
        "source": {"repository_commit": "0123456789"},
        "monitoring": {"wandb_run_id": "test-run-id"},
        "paths": {"run_dir": str(run_dir)},
    }


def _model_optimizer_scheduler():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=2e-4,
        min_lr=1e-6,
        warmup_steps=2,
        max_steps=8,
    )
    return model, optimizer, scheduler


def test_checkpoint_round_trip_restores_model_optimizer_scheduler_and_rng():
    torch.manual_seed(123)
    random.seed(123)
    model, optimizer, scheduler = _model_optimizer_scheduler()
    fixed_input = torch.randn(3, 4)
    loss = model(fixed_input).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_output = model(fixed_input).detach().clone()

    with _workspace_temporary_directory() as root:
        config = _config(root)
        checkpoint = build_checkpoint(
            model=model,
            architecture={"model_name": "unit-test"},
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=1,
            best_metrics={"macro_psnr": 10.0, "global_step": 1},
            config=config,
        )
        expected_torch_random = torch.rand(5)
        expected_python_random = random.random()
        path = root / "checkpoint.pth"
        atomic_torch_save(checkpoint, path)

        restored_model, restored_optimizer, restored_scheduler = (
            _model_optimizer_scheduler()
        )
        loaded = load_checkpoint(path)
        validate_checkpoint_identity(
            loaded,
            config=config,
            current_repository_commit="0123456789",
        )
        restore_training_state(
            loaded,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
        )

        torch.testing.assert_close(restored_model(fixed_input), expected_output)
        assert restored_scheduler.completed_steps == 1
        assert restored_optimizer.param_groups[0]["lr"] == scheduler.get_last_lr()[0]
        torch.testing.assert_close(torch.rand(5), expected_torch_random)
        assert random.random() == expected_python_random


def test_checkpoint_identity_rejects_manifest_or_commit_changes():
    model, optimizer, scheduler = _model_optimizer_scheduler()
    with _workspace_temporary_directory() as root:
        config = _config(root)
        checkpoint = build_checkpoint(
            model=model,
            architecture={"model_name": "unit-test"},
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=0,
            best_metrics={},
            config=config,
        )
        changed_config = _config(root)
        changed_config["data"]["manifest_sha256"]["train.jsonl"] = "different"
        try:
            validate_checkpoint_identity(
                checkpoint,
                config=changed_config,
                current_repository_commit="different-commit",
            )
        except RuntimeError as error:
            message = str(error)
            assert "config" in message
            assert "manifest_sha256" in message
            assert "repository_commit" in message
        else:
            raise AssertionError("Incompatible checkpoint identity was accepted")


if __name__ == "__main__":
    tests = [
        test_checkpoint_round_trip_restores_model_optimizer_scheduler_and_rng,
        test_checkpoint_identity_rejects_manifest_or_commit_changes,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")

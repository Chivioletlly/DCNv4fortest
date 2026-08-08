"""Optimizer-step learning-rate schedule frozen by AIO3-v1."""

from __future__ import annotations

import math
from typing import Dict, Mapping

from torch.optim import Optimizer


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay with unambiguous resume state.

    ``completed_steps`` is the number of successful ``optimizer.step()`` calls.
    The scheduler sets the optimizer LR for the *next* update. Call ``step()``
    exactly once immediately after every optimizer update.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        base_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
        completed_steps: int = 0,
    ) -> None:
        if base_lr <= 0.0:
            raise ValueError("base_lr must be positive")
        if min_lr < 0.0 or min_lr > base_lr:
            raise ValueError("min_lr must satisfy 0 <= min_lr <= base_lr")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if warmup_steps < 0 or warmup_steps > max_steps:
            raise ValueError("warmup_steps must satisfy 0 <= warmup_steps <= max_steps")
        if completed_steps < 0 or completed_steps > max_steps:
            raise ValueError("completed_steps must satisfy 0 <= completed_steps <= max_steps")
        if not optimizer.param_groups:
            raise ValueError("optimizer must contain at least one parameter group")

        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.completed_steps = int(completed_steps)
        self._set_optimizer_lr(self.lr_for_update(self.completed_steps))

    def lr_for_update(self, update_index: int) -> float:
        """Return LR for zero-based optimizer update ``update_index``."""

        if update_index < 0 or update_index >= self.max_steps:
            if update_index == self.max_steps:
                return self.min_lr
            raise ValueError(
                f"update_index must be in [0, {self.max_steps}], got {update_index}"
            )
        if self.warmup_steps > 0 and update_index < self.warmup_steps:
            return self.base_lr * float(update_index + 1) / float(self.warmup_steps)
        if self.warmup_steps == self.max_steps:
            return self.base_lr

        if self.warmup_steps == 0:
            if self.max_steps == 1:
                progress = 1.0
            else:
                progress = float(update_index) / float(self.max_steps - 1)
        else:
            progress = float(update_index - self.warmup_steps + 1) / float(
                self.max_steps - self.warmup_steps
            )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def _set_optimizer_lr(self, learning_rate: float) -> None:
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def get_last_lr(self) -> list:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def step(self) -> None:
        if self.completed_steps >= self.max_steps:
            raise RuntimeError("Scheduler was stepped beyond max_steps")
        self.completed_steps += 1
        self._set_optimizer_lr(self.lr_for_update(self.completed_steps))

    def state_dict(self) -> Dict[str, object]:
        return {
            "scheduler_type": "warmup_cosine",
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        expected = {
            "scheduler_type": "warmup_cosine",
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
        }
        mismatches = {
            key: (state_dict.get(key), expected_value)
            for key, expected_value in expected.items()
            if state_dict.get(key) != expected_value
        }
        if mismatches:
            details = ", ".join(
                f"{key}={actual!r} (expected {wanted!r})"
                for key, (actual, wanted) in mismatches.items()
            )
            raise RuntimeError(f"Incompatible scheduler state: {details}")
        completed_steps = int(state_dict["completed_steps"])
        if completed_steps < 0 or completed_steps > self.max_steps:
            raise RuntimeError(f"Invalid completed_steps in scheduler state: {completed_steps}")
        self.completed_steps = completed_steps
        self._set_optimizer_lr(self.lr_for_update(self.completed_steps))

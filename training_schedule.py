"""Utilities for Restormer-style progressive patch-size training."""

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True)
class PatchStage:
    start_epoch: int
    patch_size: int


def parse_patch_schedule(
    entries: Iterable[str],
    size_multiple: int = 8,
) -> Tuple[PatchStage, ...]:
    """Parse one-indexed ``START_EPOCH:PATCH_SIZE`` entries."""
    stages = []
    for entry in entries or ():
        try:
            epoch_text, size_text = str(entry).split(":", 1)
            stage = PatchStage(int(epoch_text), int(size_text))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid patch stage {entry!r}; expected START_EPOCH:PATCH_SIZE"
            ) from exc
        stages.append(stage)

    if not stages:
        return ()
    if stages[0].start_epoch != 1:
        raise ValueError("The first patch stage must start at epoch 1")

    previous_epoch = 0
    previous_size = 0
    for stage in stages:
        if stage.start_epoch <= previous_epoch:
            raise ValueError("Patch stage start epochs must be strictly increasing")
        if stage.patch_size <= 0 or stage.patch_size % size_multiple != 0:
            raise ValueError(
                f"Patch size {stage.patch_size} must be a positive multiple of {size_multiple}"
            )
        if stage.patch_size < previous_size:
            raise ValueError("Patch sizes must be non-decreasing")
        previous_epoch = stage.start_epoch
        previous_size = stage.patch_size
    return tuple(stages)


def patch_size_for_epoch(stages: Sequence[PatchStage], epoch: int) -> int:
    """Return the patch size for a zero-indexed training epoch."""
    if not stages:
        raise ValueError("A non-empty patch schedule is required")
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")

    one_indexed_epoch = epoch + 1
    selected = stages[0]
    for stage in stages[1:]:
        if stage.start_epoch > one_indexed_epoch:
            break
        selected = stage
    return selected.patch_size


def format_patch_schedule(stages: Sequence[PatchStage]) -> Tuple[str, ...]:
    return tuple(f"{stage.start_epoch}:{stage.patch_size}" for stage in stages)

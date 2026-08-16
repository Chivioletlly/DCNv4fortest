"""Cross-run fairness audit for the three CDD-11 comparison configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .models import MODEL_IDS
from .protocol import CDD11_PROTOCOL_VERSION
from .runtime import atomic_write_json, load_run_config


def _common_config(config: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: config[key]
        for key in (
            "protocol",
            "run_kind",
            "seed",
            "data",
            "training",
            "optimizer",
            "scheduler",
            "validation",
            "checkpoint",
            "monitoring",
        )
    }


def audit_comparison_configs(
    configs: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if len(configs) != len(MODEL_IDS):
        raise ValueError("CDD-11 comparison audit requires exactly three configs")
    by_model = {str(config["model"]["id"]): config for config in configs}
    if set(by_model) != set(MODEL_IDS):
        raise ValueError(f"Comparison must contain exactly one run for each {MODEL_IDS}")
    ordered = [by_model[model_id] for model_id in MODEL_IDS]
    reference = _common_config(ordered[0])
    mismatches = [
        model_id
        for model_id, config in zip(MODEL_IDS[1:], ordered[1:])
        if _common_config(config) != reference
    ]
    if mismatches:
        raise RuntimeError(
            "Cross-model CDD-11 protocol mismatch for: " + ", ".join(mismatches)
        )
    main_commits = {
        str(config["source"]["repository_commit"]) for config in ordered
    }
    if len(main_commits) != 1:
        raise RuntimeError("The three runs must use the same main repository commit")
    if reference["protocol"] != CDD11_PROTOCOL_VERSION:
        raise RuntimeError("Comparison config protocol mismatch")
    serialized = json.dumps(reference, sort_keys=True, separators=(",", ":"))
    return {
        "protocol": CDD11_PROTOCOL_VERSION,
        "status": "pass",
        "run_kind": reference["run_kind"],
        "models": list(MODEL_IDS),
        "run_names": {
            model_id: str(by_model[model_id]["run_name"]) for model_id in MODEL_IDS
        },
        "main_repository_commit": next(iter(main_commits)),
        "uformer_commit": by_model["uformer"]["source"]["uformer_commit"],
        "common_config_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "common_config": reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit three CDD-11 run configs")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_dir) != 3:
        raise SystemExit("Pass --run-dir exactly three times")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite suite audit: {args.output}")
    configs = [load_run_config(path / "config.yaml") for path in args.run_dir]
    result = audit_comparison_configs(configs)
    atomic_write_json(args.output, result)
    print(
        f"CDD-11 comparison audit passed: {result['common_config_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

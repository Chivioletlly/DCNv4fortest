import copy
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from cdd11_runner.audit_suite import audit_comparison_configs
from cdd11_runner.models import MODEL_IDS
from cdd11_runner.runtime import build_run_config


def _configs():
    values = []
    for model_id in MODEL_IDS:
        values.append(
            build_run_config(
                model_id=model_id,
                run_kind="formal",
                seed=3407,
                run_name=f"{model_id}-formal",
                run_dir=Path("C:/runs") / model_id,
                manifest_hashes={"train.jsonl": "a", "val.jsonl": "b"},
                repository_state={"commit": "main-commit", "dirty": False},
                uformer_state=(
                    {"commit": "uformer-commit", "dirty": False}
                    if model_id == "uformer"
                    else None
                ),
                uformer_root=(Path("C:/uformer") if model_id == "uformer" else None),
                num_workers=8,
                microbatch_size=1,
                inference_mode="tiled",
            )
        )
    return values


def test_suite_audit_accepts_only_one_of_each_model_with_identical_protocol():
    result = audit_comparison_configs(_configs())
    assert result["status"] == "pass"
    assert result["models"] == list(MODEL_IDS)
    assert len(result["common_config_sha256"]) == 64


def test_suite_audit_rejects_cross_model_microbatch_drift():
    configs = _configs()
    configs[1] = copy.deepcopy(configs[1])
    configs[1]["data"]["microbatch_size"] = 11
    try:
        audit_comparison_configs(configs)
    except RuntimeError as error:
        assert "protocol mismatch" in str(error)
    else:
        raise AssertionError("Expected cross-model config drift to fail")

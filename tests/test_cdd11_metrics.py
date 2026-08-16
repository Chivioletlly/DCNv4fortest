import math
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from cdd11_runner.metrics import CDD11MetricAccumulator
from cdd11_runner.protocol import ARITY_GROUPS, DEGRADATION_ARITY, DEGRADATIONS


def test_metric_summary_is_category_macro_and_reports_arity_groups():
    accumulator = CDD11MetricAccumulator()
    for index, degradation in enumerate(DEGRADATIONS):
        accumulator.add_values(
            sample_id=f"sample-{index}",
            degradation=degradation,
            arity=DEGRADATION_ARITY[degradation],
            psnr=float(index + 1),
            ssim=float(index + 1) / 100.0,
        )
    summary = accumulator.summarize()

    assert math.isclose(summary["macro/psnr"], 6.0)
    assert math.isclose(summary["macro/ssim"], 0.06)
    for group_name, categories in ARITY_GROUPS.items():
        expected = sum(DEGRADATIONS.index(value) + 1 for value in categories) / len(categories)
        assert math.isclose(summary[f"{group_name}/psnr"], expected)
        assert summary[f"{group_name}/categories"] == float(len(categories))
    assert summary["images"] == 11.0


def test_metric_accumulator_rejects_wrong_arity():
    accumulator = CDD11MetricAccumulator()
    try:
        accumulator.add_values(
            sample_id="bad",
            degradation="low_haze",
            arity=1,
            psnr=20.0,
            ssim=0.8,
        )
    except ValueError as error:
        assert "Arity" in str(error)
    else:
        raise AssertionError("Expected wrong arity to be rejected")

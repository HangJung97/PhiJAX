import json
from pathlib import Path

import numpy as np
import pytest

from phijax.data import HostPool, save_prediction_artifact
from phijax.evaluation import RegressionEvaluator, evaluate_prediction_artifact, regression_metrics


def test_regression_metrics_match_direct_computation() -> None:
    """Verify all aggregate errors use the documented prediction-target difference."""
    target = np.asarray([[1.0], [2.0]])
    prediction = np.asarray([[2.0], [0.0]])

    metrics = regression_metrics(prediction, target)

    assert metrics["relative_l2_error"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["mean_absolute_error"] == pytest.approx(1.5)
    assert metrics["max_absolute_error"] == pytest.approx(2.0)


def test_evaluate_predictions_loads_artifact_and_writes_json(tmp_path: Path) -> None:
    """Verify the evaluator loads flat arrays and persists reproducible JSON metrics.

    Args:
        tmp_path: Temporary prediction and metric artifact directory.
    """
    predictions_path = tmp_path / "run" / "predictions" / "dataset.npz"
    output_dir = tmp_path / "run" / "predictions" / "results"
    predictions_path.parent.mkdir(parents=True)
    target = np.asarray([[1.0], [2.0]], dtype=np.float32)
    pool = HostPool(
        inputs=np.asarray([[0.0], [1.0]], dtype=np.float32),
        targets=target,
        aux={},
        metadata={"coordinate_names": ("x",), "output_names": ("u",)},
        reference_shape=(2,),
        flat_index=np.arange(2),
    )
    save_prediction_artifact(predictions_path, target, pool)
    result = RegressionEvaluator(predictions_path).evaluate()

    assert result.metrics["relative_l2_error"] == 0.0
    assert evaluate_prediction_artifact(predictions_path) == result.metrics
    assert json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")) == result.metrics


@pytest.mark.parametrize(
    ("prediction", "target", "match"),
    [
        (np.ones((2, 1)), np.ones((2, 2)), "same non-empty shape"),
        (np.ones((1, 1)), np.zeros((1, 1)), "nonzero norm"),
        (np.asarray([[np.nan]]), np.ones((1, 1)), "finite"),
    ],
)
def test_regression_metrics_reject_invalid_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    match: str,
) -> None:
    """Verify invalid reference comparisons fail with actionable errors.

    Args:
        prediction: Invalid predicted values.
        target: Invalid or incompatible reference values.
        match: Expected validation-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        regression_metrics(prediction, target)

from pathlib import Path

import numpy as np
import pytest

from phijax.evaluation import (
    EvaluationResult,
    finite_mask,
    normalized_rmse,
    robust_summary,
    squared_correlation,
    subtract_weighted_frame_means,
    vector_max_magnitude,
    vector_normalized_rmse,
    write_evaluation_outputs,
)


def test_metric_helpers_support_masks_and_weights() -> None:
    """Verify scalar and vector metrics consistently apply masks and positive weights."""
    reference = np.asarray([1.0, 2.0, np.nan])
    prediction = np.asarray([2.0, 2.0, 10.0])
    mask = np.asarray([True, True, False])
    weights = np.asarray([1.0, 3.0, 1.0])

    assert finite_mask(reference, prediction).tolist() == [True, True, False]
    assert normalized_rmse(prediction, reference, denominator=2.0, mask=mask, weights=weights) == pytest.approx(25.0)
    assert vector_normalized_rmse(
        (prediction, prediction),
        (reference, reference),
        denominator=np.sqrt(8.0),
        mask=mask,
        weights=weights,
    ) == pytest.approx(25.0)
    assert vector_max_magnitude((reference, reference), mask=mask) == pytest.approx(np.sqrt(8.0))
    assert np.isnan(squared_correlation(prediction[:2], reference[:2]))


def test_pressure_gauge_alignment_removes_each_weighted_frame_mean() -> None:
    """Verify pressure gauge alignment operates independently along the final frame axis."""
    values = np.asarray([[[1.0, 10.0], [3.0, 14.0]]])
    weights = np.asarray([[[1.0, 1.0], [3.0, 3.0]]])
    mask = np.ones_like(values, dtype=bool)

    aligned, means = subtract_weighted_frame_means(values, weights=weights, mask=mask)

    np.testing.assert_allclose(means, [2.5, 13.0])
    np.testing.assert_allclose(np.sum(aligned * weights, axis=(0, 1)), 0.0, atol=1e-12)


def test_evaluation_outputs_convert_non_finite_metrics_and_write_frame_csv(tmp_path: Path) -> None:
    """Verify evaluation output writers create strict JSON and stable frame tables.

    Args:
        tmp_path: Temporary output directory.
    """
    result = EvaluationResult(
        prediction_path=tmp_path / "prediction.npz",
        output_dir=tmp_path / "evaluation",
        metrics={"summary": robust_summary([1.0, 2.0]), "undefined": float("nan")},
        per_frame=({"frame": 0, "metric": 1.0},),
    )

    metrics_path, frame_path = write_evaluation_outputs(result)

    assert metrics_path.is_file()
    assert '"undefined": null' in metrics_path.read_text(encoding="utf-8")
    assert frame_path is not None
    assert frame_path.is_file()

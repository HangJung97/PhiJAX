from typing import TYPE_CHECKING

from phijax._lazy import lazy_exports

if TYPE_CHECKING:
    from phijax.evaluation.base import EvaluationResult, PredictionEvaluator
    from phijax.evaluation.metrics import (
        finite_mask,
        max_abs,
        normalized_rmse,
        regression_metrics,
        robust_summary,
        squared_correlation,
        subtract_weighted_frame_means,
        vector_max_magnitude,
        vector_normalized_rmse,
    )
    from phijax.evaluation.outputs import json_ready, resolve_evaluation_output_dir, write_evaluation_outputs
    from phijax.evaluation.regression import RegressionEvaluator, evaluate_prediction_artifact

__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "EvaluationResult": "phijax.evaluation.base",
        "PredictionEvaluator": "phijax.evaluation.base",
        "RegressionEvaluator": "phijax.evaluation.regression",
        "evaluate_prediction_artifact": "phijax.evaluation.regression",
        "finite_mask": "phijax.evaluation.metrics",
        "json_ready": "phijax.evaluation.outputs",
        "max_abs": "phijax.evaluation.metrics",
        "normalized_rmse": "phijax.evaluation.metrics",
        "regression_metrics": "phijax.evaluation.metrics",
        "resolve_evaluation_output_dir": "phijax.evaluation.outputs",
        "robust_summary": "phijax.evaluation.metrics",
        "squared_correlation": "phijax.evaluation.metrics",
        "subtract_weighted_frame_means": "phijax.evaluation.metrics",
        "vector_max_magnitude": "phijax.evaluation.metrics",
        "vector_normalized_rmse": "phijax.evaluation.metrics",
        "write_evaluation_outputs": "phijax.evaluation.outputs",
    },
)

__all__ = [
    "EvaluationResult",
    "PredictionEvaluator",
    "RegressionEvaluator",
    "evaluate_prediction_artifact",
    "finite_mask",
    "json_ready",
    "max_abs",
    "normalized_rmse",
    "regression_metrics",
    "resolve_evaluation_output_dir",
    "robust_summary",
    "squared_correlation",
    "subtract_weighted_frame_means",
    "vector_max_magnitude",
    "vector_normalized_rmse",
    "write_evaluation_outputs",
]

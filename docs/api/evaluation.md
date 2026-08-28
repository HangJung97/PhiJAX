# Evaluation

Evaluation runs on the host and reads self-contained prediction artifacts. Framework evaluators remain independent of
application packages. Application evaluators handle physical units, coordinate weights, gauge freedoms, domain
metrics, and plots.

## Contracts and outputs

::: phijax.evaluation.EvaluationResult

::: phijax.evaluation.PredictionEvaluator

::: phijax.evaluation.write_evaluation_outputs

::: phijax.evaluation.resolve_evaluation_output_dir

::: phijax.evaluation.json_ready

## Generic regression

::: phijax.evaluation.RegressionEvaluator

::: phijax.evaluation.evaluate_prediction_artifact

::: phijax.evaluation.regression_metrics

## Metric primitives

::: phijax.evaluation.finite_mask

::: phijax.evaluation.normalized_rmse

::: phijax.evaluation.vector_normalized_rmse

::: phijax.evaluation.squared_correlation

::: phijax.evaluation.max_abs

::: phijax.evaluation.vector_max_magnitude

::: phijax.evaluation.robust_summary

::: phijax.evaluation.subtract_weighted_frame_means

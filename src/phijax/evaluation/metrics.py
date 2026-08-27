from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def regression_metrics(prediction: ArrayLike, target: ArrayLike) -> dict[str, float]:
    """Compute aggregate regression errors for prediction and reference arrays.

    Args:
        prediction: Finite predicted values with any non-empty shape.
        target: Finite reference values with the same shape as `prediction`.

    Returns:
        Relative L2 error, root mean squared error, mean absolute error, and maximum absolute error.

    Raises:
        ValueError: If arrays are empty, mismatched, non-finite, or the reference has zero L2 norm.
    """
    predicted = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    if predicted.shape != reference.shape or predicted.size == 0:
        raise ValueError("Prediction and target arrays must have the same non-empty shape.")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("Prediction and target arrays must contain only finite values.")
    reference_norm = np.linalg.norm(reference.reshape(-1))
    if reference_norm == 0.0:
        raise ValueError("Relative L2 error requires a target with nonzero norm.")
    difference = predicted - reference
    return {
        "relative_l2_error": float(np.linalg.norm(difference.reshape(-1)) / reference_norm),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "max_absolute_error": float(np.max(np.abs(difference))),
    }


def finite_mask(*arrays: ArrayLike) -> NDArray[np.bool_]:
    """Select entries that are finite in every broadcast-compatible array.

    Args:
        arrays: One or more arrays with broadcast-compatible shapes.

    Returns:
        Boolean array selecting jointly finite entries.

    Raises:
        ValueError: If no arrays are supplied.
    """
    if not arrays:
        raise ValueError("At least one array is required to build a finite mask.")
    converted = tuple(np.asarray(array) for array in arrays)
    mask = np.ones(np.broadcast_shapes(*(array.shape for array in converted)), dtype=bool)
    for array in converted:
        mask &= np.isfinite(array)
    return mask


def squared_correlation(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> float:
    """Compute squared Pearson correlation over finite masked entries.

    Args:
        prediction: Predicted scalar field.
        reference: Reference scalar field.
        mask: Optional Boolean validity mask.

    Returns:
        Squared correlation, or `nan` when fewer than two varying values exist.
    """
    predicted = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    valid = finite_mask(predicted, target)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    predicted_values = predicted[valid].reshape(-1)
    target_values = target[valid].reshape(-1)
    if predicted_values.size < 2 or np.std(predicted_values) == 0.0 or np.std(target_values) == 0.0:
        return float("nan")
    correlation = np.corrcoef(predicted_values, target_values)[0, 1]
    return float(correlation * correlation)


def normalized_rmse(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    denominator: float,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> float:
    """Compute percent normalized RMSE over finite masked entries.

    Args:
        prediction: Predicted scalar field.
        reference: Reference scalar field.
        denominator: Positive global normalization value.
        mask: Optional Boolean validity mask.
        weights: Optional non-negative mean-square weights.

    Returns:
        Normalized RMSE in percent, or `nan` when it is undefined.
    """
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    predicted = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    valid = finite_mask(predicted, target)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    resolved_weights = None if weights is None else np.asarray(weights, dtype=np.float64)
    if resolved_weights is not None:
        valid &= np.isfinite(resolved_weights) & (resolved_weights > 0.0)
    if not np.any(valid):
        return float("nan")
    squared_error = np.square(predicted[valid] - target[valid])
    if resolved_weights is None:
        mean_squared_error = np.mean(squared_error)
    else:
        valid_weights = resolved_weights[valid]
        mean_squared_error = np.sum(squared_error * valid_weights) / np.sum(valid_weights)
    return float(np.sqrt(mean_squared_error) / denominator * 100.0)


def vector_normalized_rmse(
    prediction_components: Sequence[ArrayLike],
    reference_components: Sequence[ArrayLike],
    *,
    denominator: float,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> float:
    """Compute percent normalized RMSE for vector-valued fields.

    Args:
        prediction_components: Predicted scalar components in a common basis.
        reference_components: Reference scalar components in matching order.
        denominator: Positive global vector-magnitude normalization.
        mask: Optional Boolean validity mask.
        weights: Optional non-negative mean-square weights.

    Returns:
        Vector normalized RMSE in percent, or `nan` when it is undefined.

    Raises:
        ValueError: If component counts differ or no components are supplied.
    """
    if not prediction_components or len(prediction_components) != len(reference_components):
        raise ValueError("Prediction and reference must contain the same nonzero component count.")
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    predicted = tuple(np.asarray(value, dtype=np.float64) for value in prediction_components)
    reference = tuple(np.asarray(value, dtype=np.float64) for value in reference_components)
    valid = finite_mask(*predicted, *reference)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    resolved_weights = None if weights is None else np.asarray(weights, dtype=np.float64)
    if resolved_weights is not None:
        valid &= np.isfinite(resolved_weights) & (resolved_weights > 0.0)
    if not np.any(valid):
        return float("nan")
    squared_error = np.zeros_like(predicted[0], dtype=np.float64)
    for predicted_component, reference_component in zip(predicted, reference, strict=True):
        squared_error += np.square(predicted_component - reference_component)
    if resolved_weights is None:
        mean_squared_error = np.mean(squared_error[valid])
    else:
        valid_weights = resolved_weights[valid]
        mean_squared_error = np.sum(squared_error[valid] * valid_weights) / np.sum(valid_weights)
    return float(np.sqrt(mean_squared_error) / denominator * 100.0)


def max_abs(values: ArrayLike, *, mask: ArrayLike | None = None) -> float:
    """Return the finite masked maximum absolute value.

    Args:
        values: Scalar values to reduce.
        mask: Optional Boolean validity mask.

    Returns:
        Maximum absolute value, or `nan` when no valid value exists.
    """
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return float(np.max(np.abs(array[valid]))) if np.any(valid) else float("nan")


def vector_max_magnitude(components: Sequence[ArrayLike], *, mask: ArrayLike | None = None) -> float:
    """Return the finite masked maximum vector magnitude.

    Args:
        components: Non-empty scalar components in a shared basis.
        mask: Optional Boolean validity mask.

    Returns:
        Maximum magnitude, or `nan` when no valid value exists.

    Raises:
        ValueError: If no components are supplied.
    """
    if not components:
        raise ValueError("At least one component is required.")
    arrays = tuple(np.asarray(component, dtype=np.float64) for component in components)
    valid = finite_mask(*arrays)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return float("nan")
    squared_magnitude = np.zeros_like(arrays[0])
    for array in arrays:
        squared_magnitude += np.square(array)
    return float(np.sqrt(squared_magnitude[valid]).max())


def robust_summary(values: Sequence[float]) -> dict[str, float]:
    """Summarize a metric series using its range, median, and MAD scale.

    Args:
        values: Numeric metric values, possibly including non-finite entries.

    Returns:
        Mapping containing `min`, `max`, `median`, and `robust_sigma`.
    """
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {name: float("nan") for name in ("min", "max", "median", "robust_sigma")}
    median = float(np.median(array))
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "median": median,
        "robust_sigma": float(1.4826 * np.median(np.abs(array - median))),
    }


def subtract_weighted_frame_means(
    values: ArrayLike,
    *,
    weights: ArrayLike,
    mask: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Subtract a weighted spatial mean independently from every final-axis frame.

    Args:
        values: Scalar field with at least one spatial axis followed by a frame axis.
        weights: Non-negative weights broadcast-compatible with `values`.
        mask: Boolean validity mask shaped like `values`.

    Returns:
        Gauge-aligned values and the subtracted mean for every frame.

    Raises:
        ValueError: If `values` has fewer than two dimensions or other arrays are not broadcast-compatible.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError("Frame-wise mean subtraction requires at least one spatial axis and one frame axis.")
    resolved_weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), array.shape)
    resolved_mask = np.broadcast_to(np.asarray(mask, dtype=bool), array.shape)
    aligned = array.copy()
    means = np.full(array.shape[-1], np.nan, dtype=np.float64)
    for frame in range(array.shape[-1]):
        frame_values = array[..., frame]
        frame_weights = resolved_weights[..., frame]
        valid = resolved_mask[..., frame] & np.isfinite(frame_values) & np.isfinite(frame_weights) & (frame_weights > 0)
        if np.any(valid):
            means[frame] = np.sum(frame_values[valid] * frame_weights[valid]) / np.sum(frame_weights[valid])
            aligned[..., frame] -= means[frame]
    return aligned, means


__all__ = [
    "finite_mask",
    "max_abs",
    "normalized_rmse",
    "regression_metrics",
    "robust_summary",
    "squared_correlation",
    "subtract_weighted_frame_means",
    "vector_max_magnitude",
    "vector_normalized_rmse",
]

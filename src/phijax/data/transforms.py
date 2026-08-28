from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def log_compress(
    values: ArrayLike,
    *,
    percentile: float = 99.0,
    dynamic_range: float = 40.0,
) -> NDArray[np.floating[Any]]:
    """Compress power-like values into a normalized relative-decibel range.

    Finite values above the percentile-derived reference are clipped, non-positive finite values are mapped to the
    floating-point epsilon before applying the logarithm, and the retained decibel interval is mapped to `[0, 1]`.
    Non-finite entries are preserved.

    Args:
        values: Power-like values to compress.
        percentile: Finite upper-reference percentile in `(0, 100]`.
        dynamic_range: Positive relative-decibel range retained below the reference.

    Returns:
        Floating array with finite values in `[0, 1]` and the same shape as `values`.

    Raises:
        ValueError: If `percentile` or `dynamic_range` is invalid.
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError("`percentile` must lie in `(0, 100]`.")
    if not np.isfinite(dynamic_range) or dynamic_range <= 0.0:
        raise ValueError("`dynamic_range` must be positive.")

    working = _as_floating_array(values)
    finite_mask = np.isfinite(working)
    finite = working[finite_mask]
    if finite.size == 0:
        return working.copy()

    epsilon = np.finfo(working.dtype).eps
    maximum = max(float(np.percentile(finite, percentile)), epsilon)
    positive = np.clip(finite, epsilon, maximum)
    relative_db = 10.0 * np.log10(positive / maximum)
    compressed = working.copy()
    compressed[finite_mask] = (np.clip(relative_db, -dynamic_range, 0.0) + dynamic_range) / dynamic_range
    return compressed


def minmax_scale(
    values: ArrayLike,
    *,
    minimum: ArrayLike | None = None,
    maximum: ArrayLike | None = None,
    feature_range: Sequence[float] = (0.0, 1.0),
) -> NDArray[np.floating[Any]]:
    """Scale finite values from a supplied or data-derived interval.

    Explicit `minimum` and `maximum` arrays may be reused across datasets and follow NumPy broadcasting. When either
    statistic is omitted, it is derived globally from finite entries in `values`. A constant source interval maps to
    the lower endpoint of `feature_range`, and non-finite entries are preserved.

    Args:
        values: Values to scale.
        minimum: Optional finite lower statistic. Defaults to the finite global minimum of `values`.
        maximum: Optional finite upper statistic. Defaults to the finite global maximum of `values`.
        feature_range: Two finite values defining the strictly increasing output interval.

    Returns:
        Floating array with the same shape as `values`.

    Raises:
        ValueError: If the output interval or explicit source statistics are invalid.
    """
    if len(feature_range) != 2:
        raise ValueError("`feature_range` must contain exactly two values.")
    lower, upper = (float(value) for value in feature_range)
    if not np.isfinite((lower, upper)).all() or upper <= lower:
        raise ValueError("`feature_range` must contain finite, strictly increasing values.")

    working = _as_floating_array(values)
    finite = working[np.isfinite(working)]
    if minimum is not None and not np.isfinite(np.asarray(minimum)).all():
        raise ValueError("`minimum` and `maximum` must contain only finite values.")
    if maximum is not None and not np.isfinite(np.asarray(maximum)).all():
        raise ValueError("`minimum` and `maximum` must contain only finite values.")
    if finite.size == 0 and (minimum is None or maximum is None):
        return working.copy()
    source_minimum = np.asarray(np.min(finite) if minimum is None else minimum, dtype=working.dtype)
    source_maximum = np.asarray(np.max(finite) if maximum is None else maximum, dtype=working.dtype)
    if not np.isfinite(source_minimum).all() or not np.isfinite(source_maximum).all():
        raise ValueError("`minimum` and `maximum` must contain only finite values.")
    if np.any(source_maximum < source_minimum):
        raise ValueError("`maximum` must be greater than or equal to `minimum`.")

    denominator = source_maximum - source_minimum
    normalized = np.zeros_like(working)
    np.divide(working - source_minimum, denominator, out=normalized, where=denominator > 0.0)
    scaled = normalized * (upper - lower) + lower
    return np.where(np.isfinite(working), scaled, working)


def standardize(
    values: ArrayLike,
    *,
    mean: ArrayLike | None = None,
    std: ArrayLike | None = None,
    eps: float = 1e-12,
) -> NDArray[np.floating[Any]]:
    """Center and scale finite values with reusable statistics.

    Explicit `mean` and `std` arrays follow NumPy broadcasting. Missing statistics are derived globally from finite
    entries in `values`. A standard deviation no greater than `eps` produces zeros for finite entries, and non-finite
    entries are preserved.

    Args:
        values: Values to standardize.
        mean: Optional finite centering statistic. Defaults to the finite global mean of `values`.
        std: Optional finite non-negative scale statistic. Defaults to the finite global standard deviation.
        eps: Non-negative threshold below which the scale is treated as zero.

    Returns:
        Floating array with the same shape as `values`.

    Raises:
        ValueError: If `eps`, `mean`, or `std` is invalid.
    """
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("`eps` must be finite and non-negative.")

    working = _as_floating_array(values)
    finite = working[np.isfinite(working)]
    if mean is not None and not np.isfinite(np.asarray(mean)).all():
        raise ValueError("`mean` and `std` must contain only finite values.")
    if std is not None and not np.isfinite(np.asarray(std)).all():
        raise ValueError("`mean` and `std` must contain only finite values.")
    if finite.size == 0 and (mean is None or std is None):
        return working.copy()
    center = np.asarray(np.mean(finite) if mean is None else mean, dtype=working.dtype)
    scale = np.asarray(np.std(finite) if std is None else std, dtype=working.dtype)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("`mean` and `std` must contain only finite values.")
    if np.any(scale < 0.0):
        raise ValueError("`std` must be non-negative.")

    standardized = np.zeros_like(working)
    np.divide(working - center, scale, out=standardized, where=scale > eps)
    return np.where(np.isfinite(working), standardized, working)


def scale_by_max(
    values: ArrayLike,
    *,
    percentile: float | None = None,
    clip: bool | None = None,
) -> NDArray[np.floating[Any]]:
    """Scale finite values by their maximum or an upper percentile.

    Percentile scaling clips by default so outliers do not produce values above one. Exact-maximum scaling does not
    clip by default. Non-finite entries are excluded from statistic estimation and preserved in the returned array.

    Args:
        values: Values to scale.
        percentile: Optional finite upper percentile in `(0, 100]` used instead of the maximum.
        clip: Whether to clip finite values to the divisor before scaling. Defaults to whether `percentile` is set.

    Returns:
        Floating array with the same shape as `values`. A zero divisor maps finite entries to zero.

    Raises:
        ValueError: If `percentile` is outside `(0, 100]`.
    """
    if percentile is not None and not 0.0 < percentile <= 100.0:
        raise ValueError("`percentile` must lie in `(0, 100]`.")

    working = _as_floating_array(values)
    finite_mask = np.isfinite(working)
    finite = working[finite_mask]
    if finite.size == 0:
        return working.copy()
    divisor = float(np.max(finite) if percentile is None else np.percentile(finite, percentile))
    scaled = working.copy()
    if divisor == 0.0:
        scaled[finite_mask] = 0.0
        return scaled
    should_clip = percentile is not None if clip is None else clip
    finite_values = np.minimum(finite, divisor) if should_clip else finite
    scaled[finite_mask] = finite_values / divisor
    return scaled


def _as_floating_array(values: ArrayLike) -> NDArray[np.floating[Any]]:
    """Convert values to an array with at least `float32` precision.

    Args:
        values: Array-like values to convert.

    Returns:
        NumPy array whose dtype is the result of combining the source dtype with `float32`.
    """
    array = np.asarray(values)
    return np.asarray(array, dtype=np.result_type(array.dtype, np.float32))

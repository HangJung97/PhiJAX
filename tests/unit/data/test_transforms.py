import numpy as np
import pytest

from phijax.data import log_compress, minmax_scale, scale_by_max, standardize


def test_log_compress_maps_decibels_and_preserves_nonfinite_values() -> None:
    """Verify relative-decibel compression, clipping, dtype promotion, and missing-value preservation."""
    values = np.asarray([0.0, 1.0, 10.0, 100.0, 1000.0, np.nan], dtype=np.float32)

    compressed = log_compress(values, percentile=75.0, dynamic_range=20.0)

    assert compressed.dtype == np.float32
    np.testing.assert_allclose(compressed[:4], [0.0, 0.0, 0.5, 1.0], atol=1e-6)
    assert compressed[4] == pytest.approx(1.0)
    assert np.isnan(compressed[5])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"percentile": 0.0}, "percentile"),
        ({"percentile": 101.0}, "percentile"),
        ({"dynamic_range": 0.0}, "dynamic_range"),
        ({"dynamic_range": np.nan}, "dynamic_range"),
    ],
)
def test_log_compress_rejects_invalid_configuration(kwargs: dict[str, float], message: str) -> None:
    """Verify invalid compression settings fail before transforming values.

    Args:
        kwargs: Invalid keyword argument passed to :func:`log_compress`.
        message: Expected validation-message fragment.
    """
    with pytest.raises(ValueError, match=message):
        log_compress([1.0], **kwargs)


def test_minmax_scale_supports_reused_broadcast_statistics_and_constant_values() -> None:
    """Verify explicit channel statistics, custom ranges, and stable constant scaling."""
    values = np.asarray([[0.0, 10.0], [5.0, 20.0]], dtype=np.float32)

    scaled = minmax_scale(values, minimum=[0.0, 0.0], maximum=[10.0, 20.0], feature_range=(-1.0, 1.0))
    constant = minmax_scale(np.full(3, 4.0, dtype=np.float32))

    np.testing.assert_allclose(scaled, [[-1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(constant, np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"feature_range": (1.0,)}, "feature_range"),
        ({"feature_range": (1.0, 0.0)}, "feature_range"),
        ({"minimum": 2.0, "maximum": 1.0}, "maximum"),
    ],
)
def test_minmax_scale_rejects_invalid_statistics(kwargs: dict[str, object], message: str) -> None:
    """Verify invalid source and destination intervals raise clear errors.

    Args:
        kwargs: Invalid keyword arguments passed to :func:`minmax_scale`.
        message: Expected validation-message fragment.
    """
    with pytest.raises(ValueError, match=message):
        minmax_scale([1.0, 2.0], **kwargs)


def test_standardize_uses_finite_statistics_and_supports_reuse() -> None:
    """Verify finite-only fitting, non-finite preservation, and broadcast explicit statistics."""
    values = np.asarray([[1.0, 10.0], [3.0, np.nan]], dtype=np.float32)

    derived = standardize(values)
    reused = standardize(values, mean=[1.0, 10.0], std=[2.0, 5.0])
    constant = standardize(np.full(2, 3.0, dtype=np.float32))

    finite_derived = derived[np.isfinite(derived)]
    assert np.mean(finite_derived) == pytest.approx(0.0, abs=1e-6)
    assert np.std(finite_derived) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(reused[0], [0.0, 0.0])
    np.testing.assert_allclose(reused[1, 0], 1.0)
    assert np.isnan(reused[1, 1])
    np.testing.assert_array_equal(constant, np.zeros(2, dtype=np.float32))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"eps": -1.0}, "eps"), ({"mean": np.nan, "std": 1.0}, "mean"), ({"mean": 0.0, "std": -1.0}, "std")],
)
def test_standardize_rejects_invalid_statistics(kwargs: dict[str, object], message: str) -> None:
    """Verify invalid standardization statistics fail explicitly.

    Args:
        kwargs: Invalid keyword arguments passed to :func:`standardize`.
        message: Expected validation-message fragment.
    """
    with pytest.raises(ValueError, match=message):
        standardize([1.0, 2.0], **kwargs)


def test_scale_by_max_supports_exact_percentile_and_zero_scaling() -> None:
    """Verify maximum scaling, percentile clipping, zero handling, and non-finite preservation."""
    values = np.asarray([0.0, 2.0, 4.0, 6.0, 100.0, np.inf], dtype=np.float32)

    exact = scale_by_max(values)
    percentile = scale_by_max(values, percentile=75.0)
    zero = scale_by_max(np.zeros(3, dtype=np.float32))

    np.testing.assert_allclose(exact[:5], [0.0, 0.02, 0.04, 0.06, 1.0])
    np.testing.assert_allclose(percentile[:5], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0])
    assert np.isinf(exact[-1])
    assert np.isinf(percentile[-1])
    np.testing.assert_array_equal(zero, np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize("percentile", [0.0, 101.0])
def test_scale_by_max_rejects_invalid_percentiles(percentile: float) -> None:
    """Verify max scaling rejects percentiles outside the supported interval.

    Args:
        percentile: Invalid percentile passed to :func:`scale_by_max`.
    """
    with pytest.raises(ValueError, match="percentile"):
        scale_by_max([1.0], percentile=percentile)


def test_transforms_preserve_arrays_without_finite_values() -> None:
    """Verify data-derived transforms return all-non-finite inputs without warnings or fabricated statistics."""
    values = np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32)

    for transformed in (log_compress(values), minmax_scale(values), standardize(values), scale_by_max(values)):
        np.testing.assert_array_equal(transformed, values)

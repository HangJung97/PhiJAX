from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _immutable_array(value: Any) -> NDArray[Any]:
    """Copy an array-like value and disable writes on the result.

    Args:
        value: Value accepted by :func:`numpy.array`.

    Returns:
        Independent read-only NumPy array.
    """
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _immutable_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Recursively freeze a field mapping and copy array-like values.

    Args:
        values: Optional field mapping. Nested mappings are frozen, NumPy arrays and lists are copied into immutable
            arrays, and structural values such as tuples and strings are retained.

    Returns:
        Read-only mapping proxy containing isolated array storage.
    """
    frozen: dict[str, Any] = {}
    for name, value in (values or {}).items():
        if isinstance(value, Mapping):
            frozen[name] = _immutable_mapping(value)
        elif isinstance(value, (np.ndarray, list)):
            frozen[name] = _immutable_array(value)
        else:
            frozen[name] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True, init=False)
class HostPool:
    """Represent one immutable host-side subset with stable row ordering.

    Attributes:
        inputs: Rank-two coordinate array with shape `[samples, input_features]`.
        targets: Rank-two target array with shape `[samples, target_features]`; unsupervised pools may use zero width.
        aux: Read-only mapping of sample-wise auxiliary arrays such as periods, weights, and normals.
        metadata: Read-only mapping of structural or reconstruction metadata.
        reference_shape: Original dense grid shape used to reconstruct flat predictions.
        flat_index: Indices mapping pool rows back to the flattened reference grid.
    """

    inputs: NDArray[Any]
    targets: NDArray[Any]
    aux: Mapping[str, Any]
    metadata: Mapping[str, Any]
    reference_shape: tuple[int, ...]
    flat_index: NDArray[Any]

    def __init__(
        self,
        inputs: NDArray[Any],
        targets: NDArray[Any] | None = None,
        aux: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        reference_shape: tuple[int, ...] | None = None,
        flat_index: NDArray[Any] | None = None,
    ) -> None:
        """Copy arrays, fill safe defaults, validate dimensions, and freeze storage.

        Args:
            inputs: Rank-two coordinate array with shape `[samples, input_features]`.
            targets: Optional rank-two target array. Omission creates a zero-width array.
            aux: Optional sample-wise auxiliary arrays.
            metadata: Optional structural or reconstruction metadata.
            reference_shape: Optional dense reconstruction shape. Omission uses `(sample_count,)`.
            flat_index: Optional flattened reconstruction indices. Omission uses sequential row indices.

        Raises:
            ValueError: If `inputs` or `targets` is not rank two, or if any sample-wise array has a mismatched leading
                row count.
        """
        frozen_inputs = _immutable_array(inputs)
        if frozen_inputs.ndim != 2:
            raise ValueError("Pool `inputs` must be a rank-two array.")
        frozen_targets = _immutable_array(
            np.empty((frozen_inputs.shape[0], 0), dtype=frozen_inputs.dtype) if targets is None else targets
        )
        frozen_flat_index = _immutable_array(
            np.arange(frozen_inputs.shape[0], dtype=np.int64) if flat_index is None else flat_index
        )
        if frozen_targets.ndim != 2:
            raise ValueError("Pool `inputs` and `targets` must be rank-two arrays.")
        if frozen_targets.shape[0] != frozen_inputs.shape[0] or frozen_flat_index.shape[0] != frozen_inputs.shape[0]:
            raise ValueError("Pool arrays must share the same leading row count.")
        frozen_aux = _immutable_mapping(aux)
        for name, value in frozen_aux.items():
            if isinstance(value, np.ndarray) and value.shape[0] != frozen_inputs.shape[0]:
                raise ValueError(f"Auxiliary field `{name}` does not match the pool row count.")
        resolved_shape = (frozen_inputs.shape[0],) if reference_shape is None else reference_shape
        object.__setattr__(self, "inputs", frozen_inputs)
        object.__setattr__(self, "targets", frozen_targets)
        object.__setattr__(self, "flat_index", frozen_flat_index)
        object.__setattr__(self, "aux", frozen_aux)
        object.__setattr__(self, "metadata", _immutable_mapping(metadata))
        object.__setattr__(self, "reference_shape", tuple(int(value) for value in resolved_shape))

    def fields(self) -> Mapping[str, NDArray[Any]]:
        """Collect dense sample-wise fields for one device transfer.

        Returns:
            Read-only mapping containing `inputs`, `targets`, and NumPy-array entries from `aux`.
        """
        fields = {"inputs": self.inputs, "targets": self.targets}
        fields.update({name: value for name, value in self.aux.items() if isinstance(value, np.ndarray)})
        return MappingProxyType(fields)


def input_statistics(pools: Mapping[str, HostPool]) -> tuple[NDArray[Any], NDArray[Any]]:
    """Compute shared coordinate normalization statistics across named pools.

    Args:
        pools: Non-empty named host pools with a common input width.

    Returns:
        Per-coordinate mean and safely floored standard deviation as `float32` arrays.

    Raises:
        ValueError: If no pools are supplied or their input widths differ.
    """
    if not pools:
        raise ValueError("At least one host pool is required to derive input statistics.")
    widths = {pool.inputs.shape[1] for pool in pools.values()}
    if len(widths) != 1:
        raise ValueError("Every pool must use the same input width.")
    inputs = np.concatenate([pool.inputs for pool in pools.values()], axis=0).astype(np.float32)
    mean = inputs.mean(axis=0, dtype=np.float32)
    std = inputs.std(axis=0, dtype=np.float32)
    return mean, np.maximum(std, np.finfo(np.float32).eps)


def reconstruct_predictions(predictions: NDArray[Any], pool: HostPool) -> NDArray[Any]:
    """Scatter flat predictions back to a pool's dense reference grid.

    Args:
        predictions: Flat prediction array with shape `[samples, output_features]`.
        pool: Host pool carrying reference shape and flattened row indices.

    Returns:
        Dense prediction array with shape `[*reference_shape, output_features]`.

    Raises:
        ValueError: If prediction rows, indices, or reference shape are inconsistent.
    """
    values = np.asarray(predictions)
    if values.ndim != 2 or values.shape[0] != pool.inputs.shape[0]:
        raise ValueError("Predictions must be rank two and match the pool row count.")
    dense_count = int(np.prod(pool.reference_shape))
    if np.any(pool.flat_index < 0) or np.any(pool.flat_index >= dense_count):
        raise ValueError("Pool flat indices fall outside the reference grid.")
    dense = np.full((dense_count, values.shape[1]), np.nan, dtype=values.dtype)
    dense[pool.flat_index] = values
    return dense.reshape(*pool.reference_shape, values.shape[1])

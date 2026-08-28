import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.io import savemat

from phijax.data.pools import HostPool, reconstruct_predictions

_RESERVED_ARTIFACT_NAMES = frozenset(
    {
        "artifact_schema_version",
        "coordinate_names",
        "dense_inputs",
        "flat_index",
        "flat_prediction",
        "flat_target",
        "inputs",
        "mask",
        "output_names",
        "prediction",
        "reference_shape",
        "target",
        "value_space",
    }
)
_SUPPORTED_METADATA_KINDS = frozenset("biufUS")
_MATLAB_FIELD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_ARTIFACT_SCHEMA_VERSION = 2
_PHYSICAL_VALUE_SPACE = "physical"
_STORED_DATA_NAMES = frozenset(
    {
        "dense_inputs",
        "flat_index",
        "flat_prediction",
        "flat_target",
        "inputs",
        "mask",
        "prediction",
        "reference_shape",
        "target",
    }
)


@dataclass(frozen=True, slots=True)
class PredictionArtifact:
    """Expose one validated physical prediction artifact and named channel views.

    Attributes:
        prediction: Dense physical predictions with shape `[*reference_shape, outputs]`.
        target: Dense physical targets with zero or all output channels.
        flat_prediction: Physical predictions at the rows represented by `flat_index`.
        flat_target: Physical targets at the rows represented by `flat_index`.
        inputs: Flat model coordinates in the application's configured coordinate space.
        dense_inputs: Coordinates reconstructed on the reference grid.
        mask: Boolean reference-grid mask selecting represented rows.
        flat_index: Flat reference-grid index for every represented row.
        reference_shape: Dense reference-grid shape.
        coordinate_names: Semantic name of every input coordinate.
        output_names: Semantic name of every predicted output.
        output_scales: Physical multiplier applied to each normalized model output during serialization.
        outputs: Read-only mapping from each output name to its dense physical prediction view.
        targets: Read-only mapping from each output name to its dense physical target view, or an empty mapping when
            the prediction pool has no targets.
        metadata: Read-only mapping containing schema, structural, and application-provided metadata arrays.
    """

    prediction: NDArray[Any]
    target: NDArray[Any]
    flat_prediction: NDArray[Any]
    flat_target: NDArray[Any]
    inputs: NDArray[Any]
    dense_inputs: NDArray[Any]
    mask: NDArray[np.bool_]
    flat_index: NDArray[np.int64]
    reference_shape: tuple[int, ...]
    coordinate_names: tuple[str, ...]
    output_names: tuple[str, ...]
    output_scales: NDArray[Any]
    outputs: Mapping[str, NDArray[Any]]
    targets: Mapping[str, NDArray[Any]]
    metadata: Mapping[str, NDArray[Any]]


def save_prediction_artifact(
    output_path: str | Path,
    predictions: object,
    pool: HostPool,
    *,
    save_mat: bool = False,
    mat_field_names: Mapping[str, str] | None = None,
) -> Path:
    """Save a canonical NumPy prediction artifact and an optional MATLAB sidecar.

    Model predictions and compatible pool targets are multiplied by application-provided `output_scales` before
    serialization. Missing scales resolve to ones. Both formats store physical values, the applied scales, schema
    version `2`, and `value_space="physical"`.

    Args:
        output_path: Destination compressed NumPy artifact path ending in `.npz`.
        predictions: Flat normalized model predictions ordered like `pool.inputs`.
        pool: Host prediction pool supplying normalized targets, reconstruction metadata, and optional
            `artifact_metadata`.
        save_mat: Whether to also save a `.mat` file beside the NumPy artifact.
        mat_field_names: Optional mapping from generic prediction-array names to MATLAB variable names.

    Returns:
        Absolute path to the saved `.npz` artifact.

    Raises:
        KeyError: If `mat_field_names` refers to an array that is not exported.
        ValueError: If the destination suffix, predictions, targets, output scales, or MATLAB names are invalid.
    """
    path = Path(output_path).expanduser().resolve()
    if path.suffix.lower() != ".npz":
        raise ValueError("Prediction artifact `output_path` must end in `.npz`.")
    normalized_predictions = np.asarray(predictions)
    metadata = _prediction_metadata(pool, output_width=normalized_predictions.shape[-1])
    output_scales = _resolve_output_scales(
        metadata.get("output_scales"),
        output_width=normalized_predictions.shape[-1],
        output_dtype=normalized_predictions.dtype,
    )
    metadata.update(
        {
            "artifact_schema_version": np.asarray(_ARTIFACT_SCHEMA_VERSION, dtype=np.int64),
            "output_scales": output_scales,
            "value_space": np.asarray(_PHYSICAL_VALUE_SPACE, dtype=np.str_),
        }
    )
    flat_predictions = _scale_outputs(normalized_predictions, output_scales, name="predictions")
    flat_targets = _scale_outputs(np.asarray(pool.targets), output_scales, name="targets")
    dense_predictions = reconstruct_predictions(flat_predictions, pool)
    dense_targets = reconstruct_predictions(flat_targets, pool)
    dense_inputs = reconstruct_predictions(np.asarray(pool.inputs), pool)
    validity_mask = _dense_validity_mask(pool)
    matlab_arrays = None
    if save_mat:
        matlab_arrays = to_matlab_prediction_arrays(
            _matlab_prediction_arrays(
                dense_predictions,
                dense_targets,
                output_names=tuple(str(name) for name in metadata["output_names"].tolist()),
                flat_index=np.asarray(pool.flat_index),
                reference_shape=pool.reference_shape,
                metadata=metadata,
            ),
            field_names=mat_field_names,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz = cast(Any, np.savez_compressed)
    save_npz(
        path,
        prediction=dense_predictions,
        flat_prediction=flat_predictions,
        target=dense_targets,
        flat_target=flat_targets,
        inputs=pool.inputs,
        dense_inputs=dense_inputs,
        mask=validity_mask,
        flat_index=pool.flat_index,
        reference_shape=np.asarray(pool.reference_shape, dtype=np.int64),
        **metadata,
    )
    if matlab_arrays is not None:
        savemat(path.with_suffix(".mat"), matlab_arrays, do_compression=True, oned_as="row")
    return path


def load_prediction_artifact(path: str | Path) -> PredictionArtifact:
    """Load and validate a physical PhiJAX prediction artifact without pickle support.

    Missing `output_scales` default to one for every output channel, leaving stored physical values unchanged.

    Args:
        path: Schema-version-2 `.npz` prediction artifact.

    Returns:
        Immutable artifact with combined arrays, named output views, and separated metadata.

    Raises:
        FileNotFoundError: If `path` does not identify a file.
        KeyError: If a required schema array is absent.
        ValueError: If schema metadata, shapes, names, indices, scales, or values are inconsistent.
    """
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Prediction artifact does not exist at `{artifact_path}`.")
    required_names = _STORED_DATA_NAMES | {
        "artifact_schema_version",
        "coordinate_names",
        "output_names",
        "value_space",
    }
    with np.load(artifact_path, allow_pickle=False) as source:
        missing = tuple(sorted(required_names - set(source.files)))
        if missing:
            raise KeyError(f"Prediction artifact is missing required arrays: {missing}.")
        arrays = {name: _readonly_array(source[name]) for name in source.files}

    schema_version = arrays["artifact_schema_version"]
    if schema_version.shape != () or schema_version.item() != _ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Prediction artifacts require schema version `{_ARTIFACT_SCHEMA_VERSION}`.")
    value_space = arrays["value_space"]
    if value_space.shape != () or value_space.item() != _PHYSICAL_VALUE_SPACE:
        raise ValueError(f"Prediction artifacts require `{_PHYSICAL_VALUE_SPACE}` value space.")

    prediction = arrays["prediction"]
    target = arrays["target"]
    flat_prediction = arrays["flat_prediction"]
    flat_target = arrays["flat_target"]
    inputs = arrays["inputs"]
    dense_inputs = arrays["dense_inputs"]
    mask = np.asarray(arrays["mask"], dtype=bool)
    flat_index = np.asarray(arrays["flat_index"], dtype=np.int64)
    reference_shape_array = np.asarray(arrays["reference_shape"], dtype=np.int64)
    if reference_shape_array.ndim != 1 or reference_shape_array.size == 0 or np.any(reference_shape_array <= 0):
        raise ValueError("Prediction artifact `reference_shape` must contain positive dimensions.")
    reference_shape = tuple(int(value) for value in reference_shape_array)
    if prediction.ndim != len(reference_shape) + 1 or prediction.shape[:-1] != reference_shape:
        raise ValueError("Dense predictions must match `reference_shape` and expose one final output dimension.")
    output_width = prediction.shape[-1]
    if (
        target.ndim != prediction.ndim
        or target.shape[:-1] != reference_shape
        or target.shape[-1] not in {0, output_width}
    ):
        raise ValueError("Dense targets must match `reference_shape` and contain zero or all outputs.")
    if (
        dense_inputs.ndim != prediction.ndim
        or mask.shape != reference_shape
        or dense_inputs.shape[:-1] != reference_shape
    ):
        raise ValueError("Prediction mask and dense inputs must match `reference_shape`.")
    if flat_prediction.ndim != 2 or flat_prediction.shape[1] != output_width:
        raise ValueError("Flat predictions must be rank two and match the dense output width.")
    row_count = flat_prediction.shape[0]
    if inputs.ndim != 2 or inputs.shape[0] != row_count or flat_index.shape != (row_count,):
        raise ValueError("Flat inputs and indices must align with flat prediction rows.")
    if flat_target.shape != (row_count, target.shape[-1]):
        raise ValueError("Flat targets must align with flat prediction rows and dense target width.")
    dense_count = int(np.prod(reference_shape))
    if np.any(flat_index < 0) or np.any(flat_index >= dense_count) or len(np.unique(flat_index)) != row_count:
        raise ValueError("Prediction `flat_index` must contain unique in-range reference-grid indices.")
    expected_mask = np.zeros(dense_count, dtype=bool)
    expected_mask[flat_index] = True
    if not np.array_equal(mask.reshape(-1), expected_mask):
        raise ValueError("Prediction mask must select exactly the rows represented by `flat_index`.")

    coordinate_name_array = arrays["coordinate_names"]
    output_name_array = arrays["output_names"]
    if coordinate_name_array.ndim != 1 or output_name_array.ndim != 1:
        raise ValueError("Prediction coordinate and output names must be one-dimensional arrays.")
    coordinate_names = tuple(str(name) for name in coordinate_name_array.tolist())
    output_names = tuple(str(name) for name in output_name_array.tolist())
    if len(coordinate_names) != inputs.shape[-1] or dense_inputs.shape[-1] != inputs.shape[-1]:
        raise ValueError("Prediction coordinate names must match flat and dense input widths.")
    if len(output_names) != output_width or len(set(output_names)) != output_width:
        raise ValueError("Prediction output names must be unique and match the output width.")
    output_scales = _readonly_array(
        _resolve_output_scales(
            arrays.get("output_scales"),
            output_width=output_width,
            output_dtype=prediction.dtype,
        )
    )
    arrays["output_scales"] = output_scales

    readonly_mask = _readonly_array(mask)
    readonly_index = _readonly_array(flat_index)
    output_views = MappingProxyType({name: prediction[..., index] for index, name in enumerate(output_names)})
    target_views = (
        MappingProxyType({})
        if target.shape[-1] == 0
        else MappingProxyType({name: target[..., index] for index, name in enumerate(output_names)})
    )
    metadata = MappingProxyType({name: value for name, value in arrays.items() if name not in _STORED_DATA_NAMES})
    return PredictionArtifact(
        prediction=prediction,
        target=target,
        flat_prediction=flat_prediction,
        flat_target=flat_target,
        inputs=inputs,
        dense_inputs=dense_inputs,
        mask=readonly_mask,
        flat_index=readonly_index,
        reference_shape=reference_shape,
        coordinate_names=coordinate_names,
        output_names=output_names,
        output_scales=output_scales,
        outputs=output_views,
        targets=target_views,
        metadata=metadata,
    )


def _readonly_array(values: Any) -> NDArray[Any]:
    """Copy an array-like value into immutable artifact-owned storage.

    Args:
        values: Array-like value loaded from an artifact.

    Returns:
        Independent non-writeable NumPy array.
    """
    array = np.array(values, copy=True)
    array.setflags(write=False)
    return array


def to_matlab_prediction_arrays(
    arrays: Mapping[str, Any],
    *,
    field_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Rename reconstructed prediction arrays for MATLAB export.

    Args:
        arrays: Reconstructed prediction arrays and optional nested metadata keyed by generic PhiJAX names.
        field_names: Optional mapping from generic top-level or metadata names to MATLAB variable names.

    Returns:
        Independent mapping suitable for :func:`scipy.io.savemat`.

    Raises:
        KeyError: If `field_names` contains a source name absent from `arrays`.
        TypeError: If a configured source or destination name is not a string.
        ValueError: If a destination is not a MATLAB identifier or multiple arrays use the same destination.
    """
    resolved_names: dict[str, str] = dict(field_names or {})
    if any(not isinstance(name, str) for name in resolved_names):
        raise TypeError("`mat_field_names` source names must be strings.")
    if any(not isinstance(name, str) for name in resolved_names.values()):
        raise TypeError("`mat_field_names` destination names must be strings.")
    metadata = arrays.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("MATLAB prediction `metadata` must be a mapping when present.")
    available_names = set(arrays) | set(metadata)
    unknown_names = set(resolved_names) - available_names
    if unknown_names:
        raise KeyError(f"`mat_field_names` contains unknown prediction arrays: {sorted(unknown_names)}")

    matlab_arrays: dict[str, Any] = {}
    for name, value in arrays.items():
        if name == "metadata":
            continue
        matlab_name = resolved_names.get(name, name)
        _validate_matlab_field_name(matlab_name)
        if matlab_name in matlab_arrays:
            raise ValueError(f"Multiple prediction arrays map to MATLAB field `{matlab_name}`.")
        matlab_arrays[matlab_name] = np.asarray(value)
    matlab_metadata: dict[str, np.ndarray] = {}
    for name, value in metadata.items():
        if not isinstance(name, str):
            raise TypeError("MATLAB prediction metadata keys must be strings.")
        matlab_name = resolved_names.get(name, name)
        _validate_matlab_field_name(matlab_name)
        if matlab_name in matlab_metadata:
            raise ValueError(f"Multiple prediction metadata arrays map to MATLAB field `{matlab_name}`.")
        matlab_metadata[matlab_name] = np.asarray(value)
    if "metadata" in arrays:
        metadata_name = resolved_names.get("metadata", "metadata")
        _validate_matlab_field_name(metadata_name)
        if metadata_name in matlab_arrays:
            raise ValueError(f"MATLAB metadata field `{metadata_name}` collides with a prediction array.")
        matlab_arrays[metadata_name] = matlab_metadata
    return matlab_arrays


def _matlab_prediction_arrays(
    dense_predictions: np.ndarray,
    dense_targets: np.ndarray,
    *,
    output_names: tuple[str, ...],
    flat_index: np.ndarray,
    reference_shape: tuple[int, ...],
    metadata: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Build the compact reconstructed-array schema used by MATLAB prediction files.

    Args:
        dense_predictions: Physical channel-last predictions with shape `reference_shape + (C,)`.
        dense_targets: Physical channel-last targets with either `C` or zero output components.
        output_names: Stable name for each of the `C` output channels.
        flat_index: Flattened grid indices represented by the prediction pool.
        reference_shape: Dense spatial or spatiotemporal grid shape.
        metadata: Validated artifact metadata, including output names, scales, schema version, and value space.

    Returns:
        Dense physical predictions and targets, one array per predicted output, and one nested metadata mapping.

    Raises:
        ValueError: If output names collide, target width is invalid, or `output_scales` is inconsistent.
    """
    output_width = dense_predictions.shape[-1]
    if len(output_names) != output_width:
        raise ValueError("MATLAB prediction output names must match the output width.")
    if len(set(output_names)) != len(output_names):
        raise ValueError("MATLAB prediction output names must be unique.")
    if dense_targets.shape[:-1] != dense_predictions.shape[:-1] or dense_targets.shape[-1] not in {0, output_width}:
        raise ValueError("MATLAB prediction targets must share the prediction grid and contain zero or all outputs.")
    structural_names = {
        "metadata",
        "prediction",
        "target",
    }
    collisions = structural_names.intersection(output_names)
    if collisions:
        raise ValueError(f"MATLAB prediction output names collide with structural arrays: {sorted(collisions)}")
    scales = _resolve_output_scales(
        metadata.get("output_scales"),
        output_width=output_width,
        output_dtype=dense_predictions.dtype,
    )
    resolved_metadata = {**metadata, "output_scales": scales}
    arrays = {
        "metadata": {
            **resolved_metadata,
            "flat_index": np.asarray(flat_index, dtype=np.int64),
            "reference_shape": np.asarray(reference_shape, dtype=np.int64),
        },
        "prediction": dense_predictions,
        "target": dense_targets,
    }
    arrays.update({name: dense_predictions[..., index] for index, name in enumerate(output_names)})
    return arrays


def _resolve_output_scales(
    values: Any,
    *,
    output_width: int,
    output_dtype: np.dtype[Any],
) -> np.ndarray:
    """Resolve a finite positive physical scale for every model output.

    Args:
        values: Optional application-provided scale vector.
        output_width: Number of predicted output components.
        output_dtype: Normalized prediction dtype used to select a safe floating-point scale dtype.

    Returns:
        Scale vector with shape `(output_width,)`; absent scales resolve to ones.

    Raises:
        ValueError: If the output width or configured scale vector is invalid.
    """
    if output_width < 1:
        raise ValueError("Prediction artifacts require at least one output component.")
    scale_dtype = np.result_type(output_dtype, np.float32)
    scales = np.ones(output_width, dtype=scale_dtype) if values is None else np.asarray(values, dtype=scale_dtype)
    if scales.shape != (output_width,) or not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("Prediction `output_scales` must be a finite positive vector matching the output width.")
    return scales


def _scale_outputs(values: np.ndarray, scales: np.ndarray, *, name: str) -> np.ndarray:
    """Convert normalized channel-last arrays to physical values.

    Args:
        values: Normalized output array.
        scales: One physical multiplier per predicted output.
        name: Array label used in validation errors.

    Returns:
        Physical values, or an unchanged copy for an empty target array.

    Raises:
        ValueError: If `values` has no output axis or a nonempty output width inconsistent with `scales`.
    """
    if values.ndim < 1:
        raise ValueError(f"Prediction artifact `{name}` must expose a final output dimension.")
    if values.shape[-1] == 0:
        return np.array(values, copy=True)
    if values.shape[-1] != scales.shape[0]:
        raise ValueError(f"Prediction artifact `{name}` output width must match `output_scales`.")
    return values * scales


def _validate_matlab_field_name(name: str) -> None:
    """Validate one configured MATLAB variable name.

    Args:
        name: Candidate MATLAB variable name.

    Raises:
        ValueError: If `name` does not begin with a letter or contains unsupported characters.
    """
    if _MATLAB_FIELD_PATTERN.fullmatch(name) is None:
        raise ValueError(
            f"`{name}` is not a valid MATLAB variable name. Names must start with a letter and contain only letters, "
            "digits, and underscores."
        )


def _dense_validity_mask(pool: HostPool) -> np.ndarray:
    """Reconstruct the Boolean mask selecting valid rows in a dense prediction grid.

    Args:
        pool: Host prediction pool with reference shape and flat indices.

    Returns:
        Boolean validity mask with shape `pool.reference_shape`.
    """
    mask = np.zeros(int(np.prod(pool.reference_shape)), dtype=bool)
    mask[np.asarray(pool.flat_index, dtype=np.int64)] = True
    return mask.reshape(pool.reference_shape)


def _prediction_metadata(pool: HostPool, *, output_width: int) -> dict[str, np.ndarray]:
    """Convert structural names and application-owned artifact metadata into safe arrays.

    Args:
        pool: Host prediction pool containing optional structural names and an `artifact_metadata` mapping.
        output_width: Number of model output components in the prediction artifact.

    Returns:
        Artifact metadata containing standardized names and validated application-defined arrays.

    Raises:
        TypeError: If `artifact_metadata` is not a mapping or contains a non-string key.
        ValueError: If structural names are inconsistent or application metadata is reserved or unsafe.
    """
    input_width = int(pool.inputs.shape[-1])
    coordinate_names = tuple(pool.metadata.get("coordinate_names", (f"x{index}" for index in range(input_width))))
    configured_output_names = pool.metadata.get("output_names", pool.metadata.get("target_names"))
    output_names = (
        tuple(configured_output_names)
        if configured_output_names is not None
        else tuple(f"output_{index}" for index in range(output_width))
    )
    if len(coordinate_names) != input_width:
        raise ValueError("Prediction coordinate names must match the input feature count.")
    if len(output_names) != output_width:
        raise ValueError("Prediction output names must match the model output feature count.")
    metadata = {
        "coordinate_names": np.asarray(coordinate_names, dtype=np.str_),
        "output_names": np.asarray(output_names, dtype=np.str_),
    }
    application_metadata = pool.metadata.get("artifact_metadata", {})
    if not isinstance(application_metadata, Mapping):
        raise TypeError("Pool `artifact_metadata` must be a mapping.")
    for name, value in application_metadata.items():
        if not isinstance(name, str):
            raise TypeError("Artifact metadata keys must be strings.")
        if not name.strip():
            raise ValueError("Artifact metadata keys must not be empty.")
        if name in _RESERVED_ARTIFACT_NAMES:
            raise ValueError(f"Artifact metadata key `{name}` is reserved by the prediction schema.")
        if name == "output_scales" and value is None:
            continue
        metadata[name] = _safe_metadata_array(name, value)
    return metadata


def _safe_metadata_array(name: str, value: Any) -> np.ndarray:
    """Convert one application metadata value without enabling NumPy pickle loading.

    Args:
        name: Artifact metadata field name used in validation errors.
        value: Scalar or array-like Boolean, numeric, byte-string, or Unicode value.

    Returns:
        Independent NumPy array safe to load with `allow_pickle=False`.

    Raises:
        ValueError: If the value requires an object array or has an unsupported dtype.
    """
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in _SUPPORTED_METADATA_KINDS:
        raise ValueError(
            f"Artifact metadata `{name}` must contain only Boolean, real numeric, byte-string, or Unicode values."
        )
    return np.array(array, copy=True)


__all__ = ["PredictionArtifact", "load_prediction_artifact", "save_prediction_artifact", "to_matlab_prediction_arrays"]

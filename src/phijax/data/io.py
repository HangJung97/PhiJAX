from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import h5py
import mat73
import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

ArrayFormat = Literal["auto", "npz", "mat", "hdf5"]

_SUFFIX_FORMATS: Mapping[str, ArrayFormat] = {
    ".h5": "hdf5",
    ".hdf": "hdf5",
    ".hdf5": "hdf5",
    ".mat": "mat",
    ".npz": "npz",
}
_SUPPORTED_FORMATS = frozenset({"auto", "npz", "mat", "hdf5"})


def _materialize_hdf5(node: h5py.Dataset | h5py.Group) -> Any:
    """Materialize one HDF5 node as NumPy arrays and nested dictionaries.

    Args:
        node: Open HDF5 dataset or group.

    Returns:
        A copied NumPy array for a dataset or a recursively materialized dictionary for a group.
    """
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[()])
    return {name: _materialize_hdf5(child) for name, child in node.items()}


def _load_hdf5(path: Path) -> dict[str, Any]:
    """Load an HDF5 file and close it before returning.

    Args:
        path: Existing HDF5 file path.

    Returns:
        Nested dictionary containing eagerly materialized arrays.
    """
    with h5py.File(path, "r") as handle:
        return {name: _materialize_hdf5(node) for name, node in handle.items()}


def _load_mat(path: Path) -> dict[str, Any]:
    """Load a classic or HDF5-backed MATLAB file.

    SciPy handles classic MAT files. MATLAB v7.3 artifacts are HDF5 containers decoded by :mod:`mat73`, which restores
    MATLAB's logical dimension order and data conventions instead of exposing raw HDF5 storage axes.

    Args:
        path: Existing MATLAB file path.

    Returns:
        Dictionary containing user-defined MATLAB variables.
    """
    try:
        contents = loadmat(path, squeeze_me=True, struct_as_record=False)
    except (NotImplementedError, ValueError):
        if h5py.is_hdf5(path):
            decoded = mat73.loadmat(path)
            if decoded is None:
                raise ValueError(f"MATLAB v7.3 decoder returned no fields for `{path}`.") from None
            return decoded
        raise
    return {name: value for name, value in contents.items() if not name.startswith("__")}


def _load_npz(path: Path) -> dict[str, Any]:
    """Load a NumPy archive without permitting pickled Python objects.

    Args:
        path: Existing NumPy archive path.

    Returns:
        Dictionary containing independent in-memory arrays.
    """
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _resolve_format(path: Path, file_format: ArrayFormat) -> ArrayFormat:
    """Resolve an explicit format or infer one from a file path and header.

    Args:
        path: Existing data file path.
        file_format: Explicit format or `"auto"`.

    Returns:
        Resolved `"npz"`, `"mat"`, or `"hdf5"` format.

    Raises:
        ValueError: If `file_format` is unsupported or automatic detection fails.
    """
    if file_format not in _SUPPORTED_FORMATS:
        choices = ", ".join(sorted(_SUPPORTED_FORMATS))
        raise ValueError(f"Unsupported array file format `{file_format}`. Available formats: {choices}.")
    if file_format != "auto":
        return file_format
    suffix_format = _SUFFIX_FORMATS.get(path.suffix.lower())
    if suffix_format is not None:
        return suffix_format
    if h5py.is_hdf5(path):
        return "hdf5"
    supported_suffixes = ", ".join(sorted(_SUFFIX_FORMATS))
    raise ValueError(f"Cannot infer the format of `{path}`. Supported file suffixes: {supported_suffixes}.")


def load_arrays(path: str | Path, *, file_format: ArrayFormat = "auto") -> dict[str, Any]:
    """Load an NPZ, MATLAB, or HDF5 artifact into host memory.

    File access and decoding remain entirely NumPy-based so this function can run without initializing a JAX backend.
    HDF5 groups are represented by nested dictionaries, while all HDF5 datasets are materialized before the file is
    closed. Automatic mode uses the file suffix and recognizes HDF5 content when the suffix is unknown.

    Args:
        path: Source `.npz`, `.mat`, `.h5`, `.hdf`, or `.hdf5` path.
        file_format: Explicit file format or `"auto"` for detection.

    Returns:
        Mapping of source field names to NumPy arrays or nested field mappings.

    Raises:
        FileNotFoundError: If `path` is not an existing file.
        ValueError: If the requested format is invalid or automatic detection fails.
    """
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Array data file not found: {resolved_path}")
    resolved_format = _resolve_format(resolved_path, file_format)
    loaders = {"npz": _load_npz, "mat": _load_mat, "hdf5": _load_hdf5}
    return loaders[resolved_format](resolved_path)


def get_array(arrays: Mapping[str, Any], key: str) -> NDArray[Any]:
    """Read one array using a top-level key or slash-delimited nested path.

    A direct mapping key takes precedence over path traversal so NPZ fields containing `/` remain addressable.

    Args:
        arrays: Mapping returned by :func:`load_arrays` or an equivalent array tree.
        key: Top-level field name or nested path such as `"results/velocity"`.

    Returns:
        Requested value represented as a NumPy array.

    Raises:
        KeyError: If the key or one of its path components is absent.
        TypeError: If the key resolves to a group rather than an array value.
        ValueError: If `key` is empty.
    """
    normalized_key = key.strip("/")
    if not normalized_key:
        raise ValueError("An array key must not be empty.")
    if key in arrays:
        value = arrays[key]
    elif normalized_key in arrays:
        value = arrays[normalized_key]
    else:
        value: Any = arrays
        traversed: list[str] = []
        for component in normalized_key.split("/"):
            traversed.append(component)
            if not isinstance(value, Mapping) or component not in value:
                location = "/".join(traversed)
                raise KeyError(f"Array field `{normalized_key}` is missing path component `{location}`.")
            value = value[component]
    if isinstance(value, Mapping):
        raise TypeError(f"Array field `{normalized_key}` resolves to a group rather than an array.")
    return np.asarray(value)

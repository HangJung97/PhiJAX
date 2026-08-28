from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from phijax.data import get_array, load_arrays


def test_load_arrays_reads_npz_without_pickled_objects(tmp_path: Path) -> None:
    """Verify NPZ fields are eagerly materialized and direct keys containing slashes remain readable.

    Args:
        tmp_path: Temporary directory used for the source archive.
    """
    path = tmp_path / "fields.npz"
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.savez(path, inputs=expected, **{"group/direct": expected + 1.0})

    arrays = load_arrays(path)

    np.testing.assert_array_equal(get_array(arrays, "inputs"), expected)
    np.testing.assert_array_equal(get_array(arrays, "group/direct"), expected + 1.0)


def test_load_arrays_reads_classic_mat_and_filters_metadata(tmp_path: Path) -> None:
    """Verify classic MATLAB variables load through SciPy without bookkeeping entries.

    Args:
        tmp_path: Temporary directory used for the MATLAB artifact.
    """
    path = tmp_path / "fields.mat"
    expected = np.arange(4, dtype=np.float64).reshape(2, 2)
    savemat(path, {"solution": expected})

    arrays = load_arrays(path)

    assert not any(name.startswith("__") for name in arrays)
    np.testing.assert_array_equal(get_array(arrays, "solution"), expected)


def test_load_arrays_reads_nested_hdf5_and_mat_v73_containers(tmp_path: Path) -> None:
    """Verify generic HDF5 paths and MATLAB-aware v7.3 dimension decoding.

    Args:
        tmp_path: Temporary directory used for HDF5 artifacts.
    """
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    hdf5_path = tmp_path / "fields.h5"
    mat_path = tmp_path / "fields.mat"
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset("results/velocity", data=expected)
    with h5py.File(mat_path, "w") as handle:
        dataset = handle.create_dataset("velocity", data=expected.T)
        dataset.attrs["MATLAB_class"] = np.bytes_("single")

    hdf5_arrays = load_arrays(hdf5_path)
    mat_arrays = load_arrays(mat_path)

    np.testing.assert_array_equal(get_array(hdf5_arrays, "/results/velocity/"), expected)
    np.testing.assert_array_equal(get_array(mat_arrays, "velocity"), expected)


def test_load_arrays_detects_hdf5_content_with_an_unknown_suffix(tmp_path: Path) -> None:
    """Verify automatic detection examines HDF5 content when no known suffix is present.

    Args:
        tmp_path: Temporary directory used for an extensionless HDF5 artifact.
    """
    path = tmp_path / "fields.data"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("value", data=np.asarray([1.0, 2.0]))

    arrays = load_arrays(path)

    np.testing.assert_array_equal(get_array(arrays, "value"), np.asarray([1.0, 2.0]))


def test_array_io_reports_invalid_files_formats_and_fields(tmp_path: Path) -> None:
    """Verify invalid paths, formats, keys, and group selections fail with actionable errors.

    Args:
        tmp_path: Temporary directory used for invalid and grouped artifacts.
    """
    with pytest.raises(FileNotFoundError, match="Array data file not found"):
        load_arrays(tmp_path / "missing.npz")

    unknown_path = tmp_path / "unknown.data"
    unknown_path.write_bytes(b"not an array file")
    with pytest.raises(ValueError, match="Cannot infer"):
        load_arrays(unknown_path)
    with pytest.raises(ValueError, match="Unsupported array file format"):
        load_arrays(unknown_path, file_format="csv")  # type: ignore[arg-type]

    arrays = {"results": {"velocity": np.ones((2, 1))}}
    with pytest.raises(KeyError, match="missing path component"):
        get_array(arrays, "results/pressure")
    with pytest.raises(TypeError, match="group rather than an array"):
        get_array(arrays, "results")
    with pytest.raises(ValueError, match="must not be empty"):
        get_array(arrays, "/")

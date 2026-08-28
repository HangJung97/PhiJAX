from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from phijax.data import build_array_pools, reconstruct_predictions


def _schema(path: Path, *, initial_index: int | list[int] = 0) -> dict[str, object]:
    """Create a compact declarative pool schema for tests.

    Args:
        path: NPZ source path.
        initial_index: Scalar time index or half-open time-index range.

    Returns:
        Keyword arguments accepted by :func:`build_array_pools`.
    """
    return {
        "source": {"path": path},
        "coordinates": {
            "t": {"key": "t"},
            "x": {"key": "x", "bounds": [-2.0, 2.0]},
        },
        "fields": {
            "u": {"key": "usol", "axes": ["t", "x"]},
            "weight": {"key": "weights", "axes": ["t"]},
        },
        "pools": {
            "initial": {
                "inputs": ["t", "x"],
                "targets": ["u"],
                "aux": {"sample_weight": "weight"},
                "slice": {"t": {"index": initial_index}},
            },
            "pde": {
                "inputs": ["t", "x"],
                "sampling": {"method": "uniform", "size": 7, "seed": 13},
            },
            "predict": {"inputs": ["t", "x"], "targets": ["u"], "grid": "full"},
        },
    }


def _write_source(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write one structured space-time source artifact.

    Args:
        path: NPZ destination.

    Returns:
        Time coordinates, spatial coordinates, and solution field.
    """
    times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    positions = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    solution = np.arange(12, dtype=np.float64).reshape(3, 4)
    np.savez(path, t=times, x=positions, usol=solution, weights=np.asarray([1.0, 2.0, 3.0]))
    return times, positions, solution


def test_build_array_pools_assembles_grid_slice_fields_and_uniform_samples(tmp_path: Path) -> None:
    """Verify one schema builds sliced, sampled, and prediction pools from shared coordinates.

    Args:
        tmp_path: Temporary directory used for the source archive.
    """
    path = tmp_path / "structured.npz"
    times, positions, solution = _write_source(path)

    pools = build_array_pools(**_schema(path))

    assert set(pools) == {"initial", "pde", "predict"}
    np.testing.assert_array_equal(pools["initial"].inputs[:, 0], np.full(positions.size, times[0]))
    np.testing.assert_array_equal(pools["initial"].inputs[:, 1], positions)
    np.testing.assert_array_equal(pools["initial"].targets[:, 0], solution[0].astype(np.float32))
    np.testing.assert_array_equal(pools["initial"].aux["sample_weight"], np.ones((positions.size, 1)))
    np.testing.assert_array_equal(pools["initial"].flat_index, np.arange(positions.size))
    assert pools["initial"].reference_shape == solution.shape

    assert pools["pde"].inputs.shape == (7, 2)
    assert np.all((pools["pde"].inputs[:, 0] >= times[0]) & (pools["pde"].inputs[:, 0] <= times[-1]))
    assert np.all((pools["pde"].inputs[:, 1] >= -2.0) & (pools["pde"].inputs[:, 1] <= 2.0))
    assert pools["pde"].metadata["sampling_bounds"] == (("t", 0.0, 1.0), ("x", -2.0, 2.0))

    np.testing.assert_array_equal(pools["predict"].targets[:, 0], solution.astype(np.float32).reshape(-1))
    assert pools["predict"].metadata["sampling_bounds"] == (("t", 0.0, 1.0), ("x", -2.0, 2.0))
    dense = reconstruct_predictions(pools["predict"].targets, pools["predict"])
    np.testing.assert_array_equal(dense[..., 0], solution.astype(np.float32))


def test_build_array_pools_treats_list_index_as_half_open_range(tmp_path: Path) -> None:
    """Verify `[start, stop]` selects a range while retaining the selected coordinate axis.

    Args:
        tmp_path: Temporary directory used for the source archive.
    """
    path = tmp_path / "structured.npz"
    times, positions, solution = _write_source(path)

    pools = build_array_pools(**_schema(path, initial_index=[1, 3]))
    selected = pools["initial"]

    np.testing.assert_array_equal(selected.inputs[:, 0], np.repeat(times[1:3], positions.size))
    np.testing.assert_array_equal(selected.targets[:, 0], solution[1:3].astype(np.float32).reshape(-1))
    np.testing.assert_array_equal(selected.flat_index, np.arange(positions.size, solution.size))
    assert selected.inputs.shape == (2 * positions.size, 2)


def test_build_array_pools_aligns_fields_declared_in_a_different_axis_order(tmp_path: Path) -> None:
    """Verify named field axes transpose source storage into pool-input order.

    Args:
        tmp_path: Temporary directory used for the source archive.
    """
    path = tmp_path / "transposed.npz"
    times = np.asarray([0.0, 1.0])
    positions = np.asarray([-1.0, 0.0, 1.0])
    solution = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.savez(path, t=times, x=positions, transposed=solution.T)

    pools = build_array_pools(
        source={"path": path},
        coordinates={"t": "t", "x": "x"},
        fields={"u": {"key": "transposed", "axes": ["x", "t"]}},
        pools={"predict": {"inputs": ["t", "x"], "targets": ["u"]}},
    )

    np.testing.assert_array_equal(pools["predict"].targets[:, 0], solution.reshape(-1))


def test_build_array_pools_runs_an_optional_source_preparation_hook(tmp_path: Path) -> None:
    """Verify an application generation hook can prepare a missing artifact without defining a data builder.

    Args:
        tmp_path: Temporary directory used for the generated source archive.
    """
    path = tmp_path / "prepared.npz"

    def prepare(destination: str | Path) -> Path:
        """Write a minimal source artifact.

        Args:
            destination: Requested source destination.

        Returns:
            Generated source path.
        """
        resolved = Path(destination)
        np.savez(resolved, t=np.asarray([0.0, 1.0]), u=np.asarray([2.0, 3.0]))
        return resolved

    pools = build_array_pools(
        source={"path": path, "prepare": prepare},
        coordinates={"t": "t"},
        fields={"u": {"key": "u", "axes": ["t"]}},
        pools={"predict": {"inputs": ["t"], "targets": ["u"]}},
    )

    assert path.is_file()
    np.testing.assert_array_equal(pools["predict"].targets[:, 0], np.asarray([2.0, 3.0]))


def test_build_array_pools_broadcasts_keyed_and_inline_constant_fields(tmp_path: Path) -> None:
    """Verify scalar and feature-vector constants can be loaded or defined inline and broadcast as auxiliary fields.

    Args:
        tmp_path: Temporary directory used for the source archive.
    """
    path = tmp_path / "constants.npz"
    times = np.asarray([0.0, 0.5, 1.0])
    np.savez(path, t=times, **{"parameters/viscosity": np.asarray([[0.25]])})

    pools = build_array_pools(
        source={"path": path},
        coordinates={"t": "t"},
        fields={
            "viscosity": {"kind": "constant", "key": "parameters/viscosity"},
            "direction": {"kind": "constant", "value": [1.0, -1.0]},
        },
        pools={
            "predict": {
                "inputs": ["t"],
                "aux": {"nu": "viscosity", "direction": "direction"},
            },
            "pde": {
                "inputs": ["t"],
                "aux": {"nu": "viscosity"},
                "sampling": {"method": "uniform", "size": 4, "seed": 2},
            },
        },
    )

    np.testing.assert_array_equal(pools["predict"].aux["nu"], np.full((times.size, 1), 0.25))
    np.testing.assert_array_equal(
        pools["predict"].aux["direction"],
        np.tile(np.asarray([[1.0, -1.0]]), (times.size, 1)),
    )
    np.testing.assert_array_equal(pools["pde"].aux["nu"], np.full((4, 1), 0.25))


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ({"kind": "constant"}, "exactly one"),
        ({"kind": "constant", "key": "constant", "value": 1.0}, "exactly one"),
        ({"kind": "constant", "value": 1.0, "axes": ["t"]}, "cannot declare sample axes"),
        ({"kind": "constant", "value": [[1.0, 2.0], [3.0, 4.0]]}, "scalar or non-empty feature vector"),
        ({"kind": "derived", "value": 1.0}, "unsupported kind"),
        ({"kind": "source", "value": 1.0, "axes": []}, "cannot define an inline"),
    ],
)
def test_build_array_pools_rejects_invalid_constant_field_definitions(
    tmp_path: Path,
    field: dict[str, object],
    match: str,
) -> None:
    """Verify constant kinds enforce exclusive sources, axis independence, and fixed feature shapes.

    Args:
        tmp_path: Temporary directory used for the source archive.
        field: Invalid field specification.
        match: Expected validation-message fragment.
    """
    path = tmp_path / "constants.npz"
    np.savez(path, t=np.asarray([0.0, 1.0]), constant=np.asarray(2.0))

    with pytest.raises(ValueError, match=match):
        build_array_pools(
            source={"path": path},
            coordinates={"t": "t"},
            fields={"invalid": field},
            pools={"predict": {"inputs": ["t"], "aux": ["invalid"]}},
        )


def test_uniform_pool_rejects_grid_aligned_auxiliary_fields(tmp_path: Path) -> None:
    """Verify generated coordinates cannot be paired implicitly with source-grid auxiliary rows.

    Args:
        tmp_path: Temporary directory used for the source archive.
    """
    path = tmp_path / "varying.npz"
    np.savez(path, t=np.asarray([0.0, 1.0]), weight=np.asarray([1.0, 2.0]))

    with pytest.raises(ValueError, match="only accepts constant auxiliary fields"):
        build_array_pools(
            source={"path": path},
            coordinates={"t": "t"},
            fields={"weight": {"key": "weight", "axes": ["t"]}},
            pools={
                "pde": {
                    "inputs": ["t"],
                    "aux": ["weight"],
                    "sampling": {"method": "uniform", "size": 2},
                }
            },
        )


@pytest.mark.parametrize(
    ("index", "match"),
    [
        ([0], "must contain"),
        ([1, 1], "selects no values"),
        (3, "falls outside"),
        ("first", "must be an integer"),
    ],
)
def test_build_array_pools_rejects_invalid_scalar_and_range_indices(
    tmp_path: Path,
    index: object,
    match: str,
) -> None:
    """Verify malformed, empty, and out-of-range slice indices fail eagerly.

    Args:
        tmp_path: Temporary directory used for the source archive.
        index: Invalid scalar or range index.
        match: Expected validation-message fragment.
    """
    path = tmp_path / "structured.npz"
    _write_source(path)
    schema = _schema(path)
    initial = schema["pools"]["initial"]  # type: ignore[index]
    initial["slice"]["t"]["index"] = index  # type: ignore[index]

    with pytest.raises(ValueError, match=match):
        build_array_pools(**schema)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda schema: schema.update(dtype="int32"), "floating"),
        (lambda schema: schema["coordinates"].update(missing="absent"), "missing"),
        (lambda schema: schema["pools"]["pde"]["sampling"].update(size=0), "positive integer"),
        (lambda schema: schema["pools"]["pde"]["sampling"].update(method="normal"), "only supports"),
    ],
)
def test_build_array_pools_rejects_invalid_schema(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    """Verify invalid dtype, fields, and sampling policies fail eagerly.

    Args:
        tmp_path: Temporary directory used for the source archive.
        mutate: Schema mutation introducing one invalid setting.
        match: Expected validation-message fragment.
    """
    path = tmp_path / "structured.npz"
    _write_source(path)
    schema = _schema(path)
    mutate(schema)

    with pytest.raises((KeyError, ValueError), match=match):
        build_array_pools(**schema)

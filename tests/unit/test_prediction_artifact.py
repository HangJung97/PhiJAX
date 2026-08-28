from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import loadmat

from phijax.data import HostPool, load_prediction_artifact, save_prediction_artifact, to_matlab_prediction_arrays


def _prediction_pool(artifact_metadata: Any) -> HostPool:
    """Build a compact prediction pool with configurable application metadata.

    Args:
        artifact_metadata: Value stored under the application-owned `artifact_metadata` namespace.

    Returns:
        Two-row prediction pool.
    """
    return HostPool(
        inputs=np.asarray([[0.0], [1.0]], dtype=np.float32),
        targets=np.asarray([[1.0], [2.0]], dtype=np.float32),
        aux={},
        metadata={
            "coordinate_names": ("x",),
            "output_names": ("u",),
            "artifact_metadata": artifact_metadata,
        },
        reference_shape=(2,),
        flat_index=np.arange(2),
    )


def test_prediction_artifact_serializes_application_owned_metadata(tmp_path: Path) -> None:
    """Verify safe namespaced metadata is preserved without pickle-backed arrays.

    Args:
        tmp_path: Temporary prediction artifact directory.
    """
    pool = _prediction_pool(
        {
            "output_scales": np.asarray([2.5]),
            "density": 1060.0,
            "description": "physical scaling",
            "enabled": True,
        }
    )

    path = save_prediction_artifact(tmp_path / "prediction.npz", pool.targets, pool)

    with np.load(path, allow_pickle=False) as artifact:
        assert artifact["coordinate_names"].tolist() == ["x"]
        assert artifact["output_names"].tolist() == ["u"]
        np.testing.assert_allclose(artifact["output_scales"], [2.5])
        np.testing.assert_allclose(artifact["prediction"], [[2.5], [5.0]])
        np.testing.assert_allclose(artifact["target"], [[2.5], [5.0]])
        assert artifact["artifact_schema_version"].item() == 2
        assert artifact["value_space"].item() == "physical"
        assert artifact["density"].item() == 1060.0
        assert artifact["description"].item() == "physical scaling"
        assert artifact["enabled"].item() is True
    assert not path.with_suffix(".mat").exists()

    loaded = load_prediction_artifact(path)
    np.testing.assert_allclose(loaded.outputs["u"], [2.5, 5.0])
    np.testing.assert_allclose(loaded.targets["u"], [2.5, 5.0])
    assert loaded.reference_shape == (2,)
    assert loaded.output_names == ("u",)
    assert loaded.metadata["description"].item() == "physical scaling"
    with pytest.raises(ValueError, match="read-only"):
        loaded.outputs["u"][0] = 0.0


def test_prediction_artifact_writes_physical_numpy_and_matlab_arrays(tmp_path: Path) -> None:
    """Verify NPZ and MATLAB artifacts share physical outputs, targets, scales, and schema metadata.

    Args:
        tmp_path: Temporary prediction artifact directory.
    """
    pool = _prediction_pool({"output_scales": np.asarray([2.5])})

    path = save_prediction_artifact(
        tmp_path / "prediction.npz",
        pool.targets,
        pool,
        save_mat=True,
        mat_field_names={"prediction": "pred", "u": "u_pred", "flat_index": "indices"},
    )

    with np.load(path, allow_pickle=False) as artifact:
        np.testing.assert_allclose(artifact["prediction"], [[2.5], [5.0]])
        np.testing.assert_allclose(artifact["target"], [[2.5], [5.0]])
        np.testing.assert_allclose(artifact["output_scales"], [2.5])
    matlab = loadmat(path.with_suffix(".mat"), simplify_cells=True)
    np.testing.assert_allclose(matlab["pred"], [2.5, 5.0])
    np.testing.assert_allclose(matlab["target"], [2.5, 5.0])
    np.testing.assert_allclose(matlab["u_pred"], [2.5, 5.0])
    metadata = matlab["metadata"]
    assert float(metadata["output_scales"]) == pytest.approx(2.5)
    assert int(metadata["artifact_schema_version"]) == 2
    assert metadata["value_space"] == "physical"
    np.testing.assert_array_equal(metadata["indices"], [0, 1])
    assert int(metadata["reference_shape"]) == 2


@pytest.mark.parametrize("artifact_metadata", [{"description": "unscaled"}, {"output_scales": None}])
def test_prediction_artifact_defaults_missing_output_scales_to_one(
    tmp_path: Path,
    artifact_metadata: dict[str, Any],
) -> None:
    """Verify absent or null output scales preserve values and are stored as identity scales.

    Args:
        tmp_path: Temporary prediction artifact directory.
        artifact_metadata: Application metadata omitting scales or setting them to `None`.
    """
    pool = _prediction_pool(artifact_metadata)

    path = save_prediction_artifact(tmp_path / "prediction.npz", pool.targets, pool, save_mat=True)

    artifact = load_prediction_artifact(path)
    np.testing.assert_allclose(artifact.output_scales, [1.0])
    np.testing.assert_allclose(artifact.prediction, pool.targets)
    np.testing.assert_allclose(artifact.target, pool.targets)
    matlab = loadmat(path.with_suffix(".mat"), simplify_cells=True)
    assert float(matlab["metadata"]["output_scales"]) == pytest.approx(1.0)


def test_prediction_artifact_loader_defaults_absent_stored_scales_to_one(tmp_path: Path) -> None:
    """Verify schema-version-2 artifacts remain loadable when the scale array is absent.

    Args:
        tmp_path: Temporary prediction artifact directory.
    """
    source_path = save_prediction_artifact(tmp_path / "source.npz", np.ones((2, 1)), _prediction_pool({}))
    without_scales_path = tmp_path / "without_scales.npz"
    with np.load(source_path, allow_pickle=False) as source:
        np.savez_compressed(
            without_scales_path, **{name: source[name] for name in source.files if name != "output_scales"}
        )

    artifact = load_prediction_artifact(without_scales_path)

    np.testing.assert_allclose(artifact.output_scales, [1.0])
    np.testing.assert_allclose(artifact.metadata["output_scales"], [1.0])


@pytest.mark.parametrize(
    ("field_names", "exception", "match"),
    [
        ({"missing": "value"}, KeyError, "unknown prediction arrays"),
        ({"prediction": "u", "u": "u"}, ValueError, "Multiple prediction arrays"),
        ({"u": "1_invalid"}, ValueError, "valid MATLAB variable name"),
        ({"u": 1}, TypeError, "destination names must be strings"),
    ],
)
def test_matlab_prediction_field_names_are_validated(
    field_names: dict[str, Any],
    exception: type[Exception],
    match: str,
) -> None:
    """Verify MATLAB mappings reject unknown, duplicated, or invalid destinations.

    Args:
        field_names: Invalid source-to-destination mapping.
        exception: Expected validation exception.
        match: Expected validation-message fragment.
    """
    arrays = {
        "prediction": np.ones((2, 1)),
        "u": np.ones(2),
        "flat_index": np.arange(2),
        "reference_shape": np.asarray([2]),
    }
    with pytest.raises(exception, match=match):
        to_matlab_prediction_arrays(arrays, field_names=field_names)


@pytest.mark.parametrize(
    ("artifact_metadata", "exception", "match"),
    [
        ({"prediction": 1.0}, ValueError, "reserved"),
        ({"settings": {"nested": True}}, ValueError, "Boolean, real numeric"),
        (["not", "a", "mapping"], TypeError, "must be a mapping"),
        ({1: 2.0}, TypeError, "keys must be strings"),
        ({"output_scales": [1.0, 2.0]}, ValueError, "output_scales"),
        ({"output_scales": [0.0]}, ValueError, "positive"),
    ],
)
def test_prediction_artifact_rejects_unsafe_or_ambiguous_metadata(
    tmp_path: Path,
    artifact_metadata: Any,
    exception: type[Exception],
    match: str,
) -> None:
    """Verify application metadata cannot overwrite the schema or require pickle loading.

    Args:
        tmp_path: Temporary prediction artifact directory.
        artifact_metadata: Invalid application metadata payload.
        exception: Expected validation exception.
        match: Expected validation-message fragment.
    """
    with pytest.raises(exception, match=match):
        save_prediction_artifact(tmp_path / "prediction.npz", np.ones((2, 1)), _prediction_pool(artifact_metadata))

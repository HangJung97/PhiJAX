from pathlib import Path

import numpy as np
import pytest

from phijax.callbacks import PredictionContext, PredictionWriter
from phijax.data import HostPool


def _prediction_pool() -> HostPool:
    """Build a compact prediction pool for writer lifecycle tests.

    Returns:
        Two-row scalar prediction pool with dense reconstruction metadata.
    """
    return HostPool(
        inputs=np.asarray([[0.0], [1.0]], dtype=np.float32),
        targets=np.asarray([[1.0], [2.0]], dtype=np.float32),
        aux={},
        metadata={"coordinate_names": ("x",), "output_names": ("u",)},
        reference_shape=(2,),
        flat_index=np.arange(2),
    )


def test_prediction_writer_saves_final_rank_zero_outputs(tmp_path: Path) -> None:
    """Verify prediction-end dispatch writes the canonical artifact and retains its path.

    Args:
        tmp_path: Temporary prediction output directory.
    """
    output_path = tmp_path / "prediction.npz"
    callback = PredictionWriter(tmp_path, save_file_name="prediction")
    callback.setup()

    callback.on_predict_end(
        PredictionContext(
            outputs=np.asarray([[3.0], [4.0]], dtype=np.float32),
            batch_index=None,
            metadata={},
            pool=_prediction_pool(),
            is_global_zero=True,
        )
    )

    assert callback.artifact_path == output_path.resolve()
    with np.load(output_path, allow_pickle=False) as artifact:
        np.testing.assert_allclose(artifact["prediction"], [[3.0], [4.0]])


def test_prediction_writer_respects_rank_zero_policy(tmp_path: Path) -> None:
    """Verify a nonzero distributed process does not write a shared artifact.

    Args:
        tmp_path: Temporary prediction output directory.
    """
    output_path = tmp_path / "prediction.npz"
    callback = PredictionWriter(tmp_path, save_file_name="prediction", rank_zero_only=True)

    callback.on_predict_end(
        PredictionContext(
            outputs=np.asarray([[3.0], [4.0]], dtype=np.float32),
            batch_index=None,
            metadata={},
            pool=_prediction_pool(),
            is_global_zero=False,
        )
    )

    assert callback.artifact_path is None
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("context", "match"),
    [
        (PredictionContext(None, None, {}, pool=_prediction_pool()), "collected outputs"),
        (PredictionContext(np.ones((2, 1)), None, {}), "source exposing"),
    ],
)
def test_prediction_writer_requires_collected_outputs_and_pool(context: PredictionContext, match: str) -> None:
    """Verify incomplete final prediction contexts fail with actionable messages.

    Args:
        context: Invalid final prediction context.
        match: Expected validation-message fragment.
    """
    callback = PredictionWriter(".")

    with pytest.raises(RuntimeError, match=match):
        callback.on_predict_end(context)


def test_prediction_writer_validates_directory_and_suffix_free_filename() -> None:
    """Verify writer destinations use a directory and one safe suffix-free filename stem."""
    with pytest.raises(ValueError, match="non-null"):
        PredictionWriter(None)
    with pytest.raises(ValueError, match="suffix-free"):
        PredictionWriter(".", save_file_name="prediction.npz")
    with pytest.raises(ValueError, match="suffix-free"):
        PredictionWriter(".", save_file_name="nested/prediction")

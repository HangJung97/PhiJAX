from collections.abc import Mapping
from pathlib import Path

from phijax.callbacks.base import Callback, PredictionContext
from phijax.data.artifacts import save_prediction_artifact


class PredictionWriter(Callback):
    """Write reconstructed prediction artifacts after a prediction lifecycle completes.

    The writer keeps persistence outside the model and prediction entrypoint. It receives the concatenated host
    predictions and immutable reconstruction pool through :class:`PredictionContext`. The artifact serializer applies
    configured output scales so NPZ and optional MATLAB values share physical units.

    Attributes:
        output_path: Resolved canonical NPZ destination.
        artifact_path: Saved canonical path after successful prediction, otherwise `None`.
    """

    def __init__(
        self,
        output_dir: str | Path | None,
        *,
        save_file_name: str = "predictions",
        save_mat: bool = False,
        mat_field_names: Mapping[str, str] | None = None,
        rank_zero_only: bool = True,
    ) -> None:
        """Initialize prediction artifact policy.

        Args:
            output_dir: Directory receiving the canonical NPZ and optional MATLAB sidecar.
            save_file_name: Suffix-free filename stem without parent-directory components.
            save_mat: Whether to write a MATLAB sidecar with the same stem.
            mat_field_names: Optional generic-to-MATLAB variable-name mapping.
            rank_zero_only: Whether only global rank zero may write artifacts.

        Raises:
            ValueError: If `output_dir` is absent or `save_file_name` is not a suffix-free local filename stem.
        """
        if output_dir is None or not str(output_dir).strip():
            raise ValueError("Prediction writer requires a non-null `output_dir`.")
        if not isinstance(save_file_name, str) or not save_file_name.strip():
            raise ValueError("Prediction writer `save_file_name` must be a non-empty string.")
        filename = Path(save_file_name)
        if filename.name != save_file_name or filename.suffix:
            raise ValueError("Prediction writer `save_file_name` must be a suffix-free local filename stem.")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.save_file_name = save_file_name
        self.output_path = self.output_dir / f"{self.save_file_name}.npz"
        self.save_mat = bool(save_mat)
        self.mat_field_names = dict(mat_field_names or {})
        self.rank_zero_only = bool(rank_zero_only)
        self.artifact_path: Path | None = None

    def setup(self) -> None:
        """Reset the artifact result before a new prediction lifecycle."""
        self.artifact_path = None

    def on_predict_end(self, context: PredictionContext) -> None:
        """Reconstruct and save the completed prediction on the permitted process.

        Args:
            context: Final prediction context containing concatenated host outputs and the reconstruction pool.

        Raises:
            RuntimeError: If prediction collection was disabled or the prediction source exposed no host pool.
        """
        if self.rank_zero_only and not context.is_global_zero:
            return
        if context.outputs is None:
            raise RuntimeError("PredictionWriter requires collected outputs; use `return_predictions=True`.")
        if context.pool is None:
            raise RuntimeError("PredictionWriter requires a prediction source exposing an immutable `pool`.")
        self.artifact_path = save_prediction_artifact(
            self.output_path,
            context.outputs,
            context.pool,
            save_mat=self.save_mat,
            mat_field_names=self.mat_field_names,
        )


__all__ = ["PredictionWriter"]

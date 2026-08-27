from pathlib import Path

from phijax.data.artifacts import load_prediction_artifact
from phijax.evaluation.base import EvaluationResult
from phijax.evaluation.metrics import regression_metrics
from phijax.evaluation.outputs import resolve_evaluation_output_dir, write_evaluation_outputs


def evaluate_prediction_artifact(prediction_path: str | Path) -> dict[str, float]:
    """Evaluate flat prediction and target arrays from a PhiJAX artifact.

    Args:
        prediction_path: `.npz` artifact containing `flat_prediction` and `flat_target`.

    Returns:
        Aggregate regression metrics.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        KeyError: If required arrays are absent.
        ValueError: If artifact arrays cannot be compared.
    """
    path = Path(prediction_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prediction artifact does not exist at `{path}`.")
    artifact = load_prediction_artifact(path)
    return regression_metrics(artifact.flat_prediction, artifact.flat_target)


class RegressionEvaluator:
    """Evaluate aggregate errors in a self-contained PhiJAX prediction artifact.

    Args:
        prediction_path: Prediction `.npz` artifact path.
        output_dir: Optional directory receiving `metrics.json`. Defaults to `results` beside the artifact.
    """

    def __init__(self, prediction_path: str | Path, output_dir: str | Path | None = None) -> None:
        """Store resolved input and output paths.

        Args:
            prediction_path: Prediction `.npz` artifact path.
            output_dir: Optional directory receiving `metrics.json`.
        """
        self.prediction_path = Path(prediction_path).expanduser().resolve()
        self.output_dir = resolve_evaluation_output_dir(self.prediction_path, output_dir)

    def evaluate(self) -> EvaluationResult:
        """Compute aggregate metrics and persist them as JSON.

        Returns:
            Completed aggregate evaluation result.
        """
        result = EvaluationResult(
            prediction_path=self.prediction_path,
            output_dir=self.output_dir,
            metrics=evaluate_prediction_artifact(self.prediction_path),
        )
        write_evaluation_outputs(result)
        return result


__all__ = ["RegressionEvaluator", "evaluate_prediction_artifact"]

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Collect metrics and files produced by one prediction evaluator.

    Attributes:
        prediction_path: Prediction artifact used as evaluator input.
        output_dir: Directory containing generated evaluation outputs.
        metrics: JSON-serializable global and summary metrics.
        per_frame: Optional frame-wise metric rows written to CSV.
    """

    prediction_path: Path
    output_dir: Path
    metrics: dict[str, Any]
    per_frame: tuple[dict[str, float | int], ...] = ()


@runtime_checkable
class PredictionEvaluator(Protocol):
    """Define a host-only evaluator selected through Hydra configuration."""

    def evaluate(self) -> EvaluationResult:
        """Evaluate one saved prediction artifact and write configured outputs.

        Returns:
            Completed evaluation result.
        """
        ...


__all__ = ["EvaluationResult", "PredictionEvaluator"]

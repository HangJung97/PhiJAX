import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from phijax.evaluation.base import EvaluationResult


def resolve_evaluation_output_dir(
    prediction_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Resolve an explicit destination or a `results` directory beside the prediction artifact.

    Args:
        prediction_path: Prediction artifact path.
        output_dir: Optional explicit output directory.

    Returns:
        Resolved explicit directory or `<prediction_directory>/results`.
    """
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    artifact_path = Path(prediction_path).expanduser().resolve()
    return artifact_path.parent / "results"


def json_ready(value: Any) -> Any:
    """Convert NumPy and non-finite values into strict JSON-compatible objects.

    Args:
        value: Arbitrary value from an evaluation metrics tree.

    Returns:
        Recursively converted JSON-compatible value.
    """
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_evaluation_outputs(result: EvaluationResult) -> tuple[Path, Path | None]:
    """Write metrics JSON and optional per-frame CSV files.

    Args:
        result: Completed evaluation result.

    Returns:
        Metrics JSON path and optional per-frame CSV path.
    """
    result.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result.output_dir / "metrics.json"
    metrics_path.write_text(f"{json.dumps(json_ready(result.metrics), indent=2, sort_keys=True)}\n", encoding="utf-8")
    if not result.per_frame:
        return metrics_path, None
    csv_path = result.output_dir / "per_frame_metrics.csv"
    fieldnames = sorted({key for row in result.per_frame for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.per_frame)
    return metrics_path, csv_path


__all__ = ["json_ready", "resolve_evaluation_output_dir", "write_evaluation_outputs"]

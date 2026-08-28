import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from phijax.training import ConsoleLogger, CSVLogger, ExperimentLogger, LoggerCollection, TensorBoardLogger, WandbLogger
from phijax.training.loggers import scalar_metrics


class _MemoryLogger(ExperimentLogger):
    """Capture logger calls for fan-out tests."""

    def __init__(self) -> None:
        """Initialize empty call collections."""
        self.parameters: list[Mapping[str, Any]] = []
        self.metrics: list[tuple[Mapping[str, float], int]] = []
        self.artifacts: list[Path] = []
        self.statuses: list[str] = []

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        """Capture parameters.

        Args:
            parameters: Resolved parameters.
        """
        self.parameters.append(parameters)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Capture metrics.

        Args:
            metrics: Scalar metrics.
            step: Optimizer step.
        """
        self.metrics.append((metrics, step))

    def log_artifact(self, path: str | Path) -> None:
        """Capture an artifact path.

        Args:
            path: Artifact path.
        """
        self.artifacts.append(Path(path))

    def finalize(self, status: str) -> None:
        """Capture terminal status.

        Args:
            status: Terminal run status.
        """
        self.statuses.append(status)


def test_logger_collection_fans_out_every_operation(tmp_path: Path) -> None:
    """Verify several logger choices receive identical lifecycle operations."""
    first = _MemoryLogger()
    second = _MemoryLogger()
    collection = LoggerCollection((first, second))
    artifact = tmp_path / "model.json"
    collection.log_hyperparameters({"seed": 3})
    collection.log_metrics({"loss": 1.0}, 4)
    collection.log_artifact(artifact)
    collection.finalize("success")
    for logger in (first, second):
        assert logger.parameters == [{"seed": 3}]
        assert logger.metrics == [({"loss": 1.0}, 4)]
        assert logger.artifacts == [artifact]
        assert logger.statuses == ["success"]


def test_csv_logger_writes_long_form_dynamic_metrics(tmp_path: Path) -> None:
    """Verify CSV logging supports metric names that vary between steps."""
    logger = CSVLogger(tmp_path)
    logger.log_metrics({"loss/a": 1.0, "loss/b": 2.0}, 1)
    logger.log_metrics({"loss/c": 3.0}, 2)
    logger.finalize("success")
    rows = (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert rows == ["step,metric,value", "1,loss/a,1.0", "1,loss/b,2.0", "2,loss/c,3.0"]


def test_scalar_metrics_rejects_vector_values() -> None:
    """Verify logger conversion does not silently reduce vector diagnostics."""
    with pytest.raises(ValueError, match="shape"):
        scalar_metrics({"vector": [1.0, 2.0]})


def test_console_logger_accepts_standard_lifecycle_calls() -> None:
    """Verify the built-in console backend implements the shared logger contract."""
    logger = ConsoleLogger("phijax.tests.trainer")
    logger.log_hyperparameters({"seed": 1})
    logger.log_metrics({"loss": 2.0}, 3)
    logger.log_artifact("checkpoint")
    logger.finalize("success")


def test_tensorboard_logger_reports_missing_optional_dependency(tmp_path: Path) -> None:
    """Verify TensorBoard logging works when installed and otherwise reports the required extra."""
    if importlib.util.find_spec("tensorboard") is None:
        with pytest.raises(ModuleNotFoundError, match="extra tensorboard"):
            TensorBoardLogger(tmp_path)
        return
    logger = TensorBoardLogger(tmp_path)
    logger.log_metrics({"loss": 1.0}, 1)
    logger.finalize("success")
    assert tuple(tmp_path.glob("events.out.tfevents.*"))


def test_wandb_logger_owns_and_idempotently_finishes_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify W&B initialization, logging, artifacts, and terminal exit status without network access."""
    run = SimpleNamespace(
        config=SimpleNamespace(update=Mock()),
        log=Mock(),
        log_artifact=Mock(),
        finish=Mock(),
    )
    init = Mock(return_value=run)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    logger = WandbLogger(
        "phijax-tests",
        save_dir=tmp_path,
        entity="team",
        name="run",
        run_id="stable-id",
        tags=("pinn", "smoke"),
        mode="disabled",
    )
    logger.log_hyperparameters({"seed": 3})
    logger.log_metrics({"loss": 1.0}, 2)
    logger.log_artifact(tmp_path / "checkpoint")
    logger.finalize("failed")
    logger.finalize("success")

    init.assert_called_once()
    assert init.call_args.kwargs["project"] == "phijax-tests"
    assert init.call_args.kwargs["id"] == "stable-id"
    assert init.call_args.kwargs["tags"] == ("pinn", "smoke")
    run.config.update.assert_called_once_with({"seed": 3}, allow_val_change=True)
    run.log.assert_called_once_with({"loss": 1.0}, step=2)
    run.log_artifact.assert_called_once_with(str(tmp_path / "checkpoint"), type="model")
    run.finish.assert_called_once_with(exit_code=1)

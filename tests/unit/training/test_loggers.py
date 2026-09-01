import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from phijax.training import ConsoleLogger, CSVLogger, ExperimentLogger, LoggerCollection, TensorBoardLogger, WandbLogger
from phijax.training.loggers import create_default_logger, scalar_metrics


class _MemoryLogger(ExperimentLogger):
    """Capture logger calls for fan-out tests."""

    def __init__(self) -> None:
        """Initialize empty call collections."""
        self.parameters: list[Mapping[str, Any]] = []
        self.metrics: list[tuple[Mapping[str, float], int]] = []
        self.artifacts: list[Path] = []
        self.statuses: list[str] = []

    def log_hyperparams(self, parameters: Mapping[str, Any]) -> None:
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
    collection.log_hyperparams({"seed": 3})
    collection.log_metrics({"loss": 1.0}, 4)
    collection.log_artifact(artifact)
    collection.finalize("success")
    for logger in (first, second):
        assert logger.parameters == [{"seed": 3}]
        assert logger.metrics == [({"loss": 1.0}, 4)]
        assert logger.artifacts == [artifact]
        assert logger.statuses == ["success"]


def test_logger_collection_can_finalize_after_partial_setup_failure() -> None:
    """Verify every backend receives cleanup when a later setup fails."""

    class SetupLogger(_MemoryLogger):
        """Record setup and optionally fail."""

        def __init__(self, *, fail: bool = False) -> None:
            """Initialize setup behavior.

            Args:
                fail: Whether setup raises after recording the call.
            """
            super().__init__()
            self.fail = fail
            self.setup_calls = 0

        def setup(self) -> None:
            """Record setup and raise when configured."""
            self.setup_calls += 1
            if self.fail:
                raise RuntimeError("setup failed")

    first = SetupLogger()
    second = SetupLogger(fail=True)
    collection = LoggerCollection((first, second))

    with pytest.raises(RuntimeError, match="setup failed"):
        collection.setup()
    collection.finalize("failed")

    assert first.setup_calls == 1
    assert second.setup_calls == 1
    assert first.statuses == ["failed"]
    assert second.statuses == ["failed"]


def test_csv_logger_writes_long_form_dynamic_metrics(tmp_path: Path) -> None:
    """Verify CSV logging supports metric names that vary between steps."""
    save_dir = tmp_path / "csv"
    logger = CSVLogger(save_dir)
    assert not save_dir.exists()

    logger.finalize("failed")
    logger.log_metrics({"loss/a": 1.0, "loss/b": 2.0}, 1)
    logger.log_metrics({"loss/c": 3.0}, 2)
    logger.finalize("success")
    logger.finalize("success")
    logger.setup()
    logger.finalize("success")

    rows = (save_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert rows == ["step,metric,value", "1,loss/a,1.0", "1,loss/b,2.0", "2,loss/c,3.0"]


def test_default_logger_uses_versioned_csv_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify dependency-safe default logging allocates consecutive local versions.

    Args:
        monkeypatch: Pytest fixture used to hide TensorBoard.
        tmp_path: Temporary logging root.
    """
    monkeypatch.setattr("phijax.training.loggers.find_spec", lambda name: None)
    first = create_default_logger(tmp_path)
    second = create_default_logger(tmp_path)
    first.log_hyperparams({"seed": 3})
    first.finalize("success")
    second.finalize("success")

    assert isinstance(first.loggers[0], CSVLogger)
    assert first.name == "phijax_logs"
    assert first.version == 0
    assert first.log_dir == tmp_path / "phijax_logs" / "version_0"
    assert second.version == 1
    assert (first.log_dir / "hparams.yaml").read_text(encoding="utf-8") == "seed: 3\n"


def test_default_logger_does_not_create_files_outside_global_rank(tmp_path: Path) -> None:
    """Verify only global rank zero owns local logger resources."""
    logger = create_default_logger(tmp_path, is_global_zero=False)

    assert not logger
    assert not (tmp_path / "phijax_logs").exists()


def test_default_logger_prefers_tensorboard_when_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify TensorBoard takes precedence over the CSV fallback when importable.

    Args:
        monkeypatch: Pytest fixture used to isolate dependency discovery.
        tmp_path: Temporary logging root.
    """
    created: list[tuple[Path, str, int | str | None]] = []

    class FakeTensorBoardLogger(ExperimentLogger):
        """Record default TensorBoard construction without importing its optional dependency."""

        def __init__(self, save_dir: str | Path, *, name: str, version: int | str | None) -> None:
            """Capture resolved default logger fields.

            Args:
                save_dir: Versioned run directory.
                name: Default experiment name.
                version: Allocated run version.
            """
            self._log_dir = Path(save_dir)
            self._name = name
            self._version = version
            created.append((self._log_dir, name, version))

        @property
        def name(self) -> str:
            """Return the captured logger name.

            Returns:
                Configured logger name.
            """
            return self._name

        @property
        def version(self) -> int | str | None:
            """Return the captured logger version.

            Returns:
                Configured logger version.
            """
            return self._version

        @property
        def log_dir(self) -> Path:
            """Return the captured run directory.

            Returns:
                Versioned run directory.
            """
            return self._log_dir

    monkeypatch.setattr("phijax.training.loggers.find_spec", lambda name: object())
    monkeypatch.setattr("phijax.training.loggers.TensorBoardLogger", FakeTensorBoardLogger)
    logger = create_default_logger(tmp_path)

    expected = tmp_path / "phijax_logs" / "version_0"
    assert created == [(expected, "phijax_logs", 0)]
    assert logger.log_dir == expected


def test_scalar_metrics_rejects_vector_values() -> None:
    """Verify logger conversion does not silently reduce vector diagnostics."""
    with pytest.raises(ValueError, match="shape"):
        scalar_metrics({"vector": [1.0, 2.0]})


def test_console_logger_accepts_standard_lifecycle_calls() -> None:
    """Verify the built-in console backend implements the shared logger contract."""
    logger = ConsoleLogger("phijax.tests.trainer")
    logger.log_hyperparams({"seed": 1})
    logger.log_metrics({"loss": 2.0}, 3)
    logger.log_artifact("checkpoint")
    logger.finalize("success")


def test_tensorboard_logger_reports_missing_optional_dependency(tmp_path: Path) -> None:
    """Verify TensorBoard logging works when installed and otherwise reports the required extra."""
    save_dir = tmp_path / "tensorboard"
    logger = TensorBoardLogger(save_dir)
    assert not save_dir.exists()

    if importlib.util.find_spec("tensorboard") is None:
        with pytest.raises(ModuleNotFoundError, match="extra tensorboard"):
            logger.setup()
        logger.finalize("failed")
        return
    logger.log_metrics({"loss": 1.0}, 1)
    logger.setup()
    logger.finalize("success")
    logger.finalize("success")
    logger.setup()
    logger.finalize("success")
    assert tuple(save_dir.glob("events.out.tfevents.*"))


def test_wandb_logger_owns_and_idempotently_finishes_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify W&B initialization, logging, artifacts, and terminal exit status without network access."""
    run = SimpleNamespace(
        config=SimpleNamespace(update=Mock()),
        log=Mock(),
        log_artifact=Mock(),
        finish=Mock(),
    )
    init = Mock(return_value=run)
    register_finalizer = Mock()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    monkeypatch.setattr("phijax.training.loggers.register_task_finalizer", register_finalizer)
    save_dir = tmp_path / "wandb"
    logger = WandbLogger(
        "phijax-tests",
        save_dir=save_dir,
        entity="team",
        name="run",
        run_id="stable-id",
        tags=("pinn", "smoke"),
        mode="disabled",
    )
    assert not save_dir.exists()
    init.assert_not_called()
    register_finalizer.assert_not_called()

    logger.log_hyperparams({"seed": 3})
    logger.log_metrics({"loss": 1.0}, 2)
    logger.log_artifact(tmp_path / "checkpoint")
    logger.finalize("failed")
    logger.finalize("success")

    init.assert_called_once()
    register_finalizer.assert_called_once_with(logger.finalize)
    assert init.call_args.kwargs["project"] == "phijax-tests"
    assert init.call_args.kwargs["id"] == "stable-id"
    assert init.call_args.kwargs["tags"] == ("pinn", "smoke")
    run.config.update.assert_called_once_with({"seed": 3}, allow_val_change=True)
    run.log.assert_called_once_with({"loss": 1.0}, step=2)
    run.log_artifact.assert_called_once_with(str(tmp_path / "checkpoint"), type="model")
    run.finish.assert_called_once_with(exit_code=1)


def test_wandb_logger_can_start_a_new_run_after_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify a finalized logger can lazily own a later task run.

    Args:
        monkeypatch: Pytest fixture used to replace the optional W&B SDK.
        tmp_path: Temporary local W&B directory.
    """
    first_run = SimpleNamespace(finish=Mock())
    second_run = SimpleNamespace(finish=Mock())
    init = Mock(side_effect=(first_run, second_run))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    logger = WandbLogger("phijax-tests", save_dir=tmp_path / "wandb", mode="disabled")

    assert logger.run is first_run
    logger.setup()
    logger.finalize("success")
    logger.finalize("failed")
    assert logger.run is second_run
    logger.finalize("failed")

    assert init.call_count == 2
    first_run.finish.assert_called_once_with(exit_code=0)
    second_run.finish.assert_called_once_with(exit_code=1)


def test_wandb_logger_reports_missing_optional_dependency_on_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify W&B dependency discovery is delayed until setup.

    Args:
        monkeypatch: Pytest fixture used to replace dynamic imports.
    """
    logger = WandbLogger("phijax-tests")

    def missing_wandb(name: str) -> Any:
        """Reject the optional W&B import.

        Args:
            name: Requested module name.

        Raises:
            ModuleNotFoundError: For the simulated missing W&B package.
        """
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("phijax.training.loggers.import_module", missing_wandb)

    with pytest.raises(ModuleNotFoundError, match="extra wandb"):
        logger.setup()
    logger.finalize("failed")

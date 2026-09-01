import csv
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phijax.utils.pylogger import get_colorlogger
from phijax.utils.utils import register_task_finalizer


class ExperimentLogger:
    """Define the minimal interface shared by experiment-logging backends.

    Logger constructors must not create files, start remote runs, or acquire other external resources. Resource
    acquisition belongs in the idempotent :meth:`setup` hook, and :meth:`finalize` must tolerate calls before setup and
    repeated calls after cleanup.
    """

    def setup(self) -> None:
        """Prepare task-local logger resources idempotently."""

    @property
    def name(self) -> str:
        """Return the logger or experiment name.

        Returns:
            Stable logger name.
        """
        return type(self).__name__

    @property
    def version(self) -> int | str | None:
        """Return the experiment version when available.

        Returns:
            Integer or string version, or `None` for unversioned loggers.
        """
        return None

    @property
    def log_dir(self) -> Path | None:
        """Return the local run directory when available.

        Returns:
            Local run directory, or `None` for remote or stream loggers.
        """
        return None

    def log_hyperparams(self, parameters: Mapping[str, Any]) -> None:
        """Record resolved experiment parameters.

        Args:
            parameters: Resolved scalar or nested experiment configuration.
        """

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Record scalar metrics for one optimizer step.

        Args:
            metrics: Host scalar metric mapping.
            step: Optimizer step associated with `metrics`.

        Raises:
            RuntimeError: If TensorBoard does not create an event writer.
        """

    def log_artifact(self, path: str | Path) -> None:
        """Record or register an experiment artifact.

        Args:
            path: Existing local artifact path.
        """

    def finalize(self, status: str) -> None:
        """Flush resources and record the terminal run status.

        Args:
            status: Terminal status such as `success` or `failed`.
        """


class LoggerCollection(ExperimentLogger):
    """Fan each logging operation out to multiple independent backends."""

    def __init__(self, loggers: Iterable[ExperimentLogger] = ()) -> None:
        """Initialize a logger fan-out collection.

        Args:
            loggers: Logging backends receiving every operation.
        """
        self.loggers = tuple(loggers)

    def __bool__(self) -> bool:
        """Return whether the collection contains a configured backend.

        Returns:
            Whether at least one logger is configured.
        """
        return bool(self.loggers)

    def setup(self) -> None:
        """Prepare every configured backend for a new task."""
        for logger in self.loggers:
            logger.setup()

    @property
    def name(self) -> str:
        """Return the primary logger name.

        Returns:
            First backend name, or this collection's class name when empty.
        """
        return self.loggers[0].name if self.loggers else type(self).__name__

    @property
    def version(self) -> int | str | None:
        """Return the primary logger version.

        Returns:
            First backend version, or `None` when empty.
        """
        return self.loggers[0].version if self.loggers else None

    @property
    def log_dir(self) -> Path | None:
        """Return the primary local log directory.

        Returns:
            First backend log directory, or `None` when empty.
        """
        return self.loggers[0].log_dir if self.loggers else None

    def log_hyperparams(self, parameters: Mapping[str, Any]) -> None:
        """Forward hyperparameters to all backends.

        Args:
            parameters: Resolved experiment configuration.
        """
        for logger in self.loggers:
            logger.log_hyperparams(parameters)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Forward scalar metrics to all backends.

        Args:
            metrics: Host scalar metric mapping.
            step: Optimizer step associated with `metrics`.
        """
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def log_artifact(self, path: str | Path) -> None:
        """Forward one artifact to all backends.

        Args:
            path: Existing local artifact path.
        """
        for logger in self.loggers:
            logger.log_artifact(path)

    def finalize(self, status: str) -> None:
        """Finalize every backend.

        Args:
            status: Terminal run status.
        """
        first_error: Exception | None = None
        for logger in self.loggers:
            try:
                logger.finalize(status)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


class ConsoleLogger(ExperimentLogger):
    """Write compact scalar training progress through PhiJAX's color logger."""

    def __init__(self, name: str = "phijax.training", level: int = logging.INFO) -> None:
        """Initialize console logging.

        Args:
            name: Python logger name.
            level: Standard-library logging level.
        """
        self.logger = get_colorlogger(name, level)

    def log_hyperparams(self, parameters: Mapping[str, Any]) -> None:
        """Log the number of resolved top-level parameters.

        Args:
            parameters: Resolved experiment configuration.
        """
        self.logger.debug("Resolved %d top-level hyperparameters.", len(parameters))

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log one compact, deterministically ordered metric line.

        Args:
            metrics: Host scalar metric mapping.
            step: Optimizer step associated with `metrics`.
        """
        rendered = ", ".join(f"{name}={value:.6g}" for name, value in sorted(metrics.items()))
        self.logger.info("step=%d | %s", step, rendered)

    def log_artifact(self, path: str | Path) -> None:
        """Log the local artifact path.

        Args:
            path: Existing local artifact path.
        """
        self.logger.info("artifact=%s", path)

    def finalize(self, status: str) -> None:
        """Log the terminal run status.

        Args:
            status: Terminal run status.
        """
        self.logger.info("training status=%s", status)


class CSVLogger(ExperimentLogger):
    """Append dynamically named scalar metrics to a long-form CSV file.

    Attributes:
        path: Destination CSV path.
    """

    def __init__(
        self,
        save_dir: str | Path,
        filename: str = "metrics.csv",
        *,
        name: str = "csv",
        version: int | str | None = None,
    ) -> None:
        """Initialize a long-form CSV logger.

        Args:
            save_dir: Directory containing the metrics file.
            filename: CSV filename within `save_dir`.
            name: Logger or experiment name.
            version: Optional run version displayed by progress callbacks.
        """
        self._log_dir = Path(save_dir)
        self._name = name
        self._version = version
        self.path = self._log_dir / filename
        self._file: Any = None
        self._writer: Any = None

    def setup(self) -> None:
        """Open the CSV stream for a new task when it is currently closed."""
        if self._file is not None and not self._file.closed:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        if self.path.stat().st_size == 0:
            self._writer.writerow(("step", "metric", "value"))

    @property
    def name(self) -> str:
        """Return the experiment name.

        Returns:
            Configured experiment name.
        """
        return self._name

    @property
    def version(self) -> int | str | None:
        """Return the experiment version.

        Returns:
            Configured version.
        """
        return self._version

    @property
    def log_dir(self) -> Path:
        """Return the local run directory.

        Returns:
            Directory containing metrics and hyperparameters.
        """
        return self._log_dir

    def log_hyperparams(self, parameters: Mapping[str, Any]) -> None:
        """Write resolved parameters to `hparams.yaml`.

        Args:
            parameters: Resolved experiment configuration.
        """
        self.setup()
        _write_hyperparameters(self.log_dir, parameters)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Append one row per scalar metric.

        Args:
            metrics: Host scalar metric mapping.
            step: Optimizer step associated with `metrics`.
        """
        self.setup()
        self._writer.writerows((step, name, value) for name, value in sorted(metrics.items()))
        self._file.flush()

    def log_artifact(self, path: str | Path) -> None:
        """Ignore artifacts because CSV has no artifact store.

        Args:
            path: Existing local artifact path.
        """
        del path

    def finalize(self, status: str) -> None:
        """Flush and close the metrics file.

        Args:
            status: Terminal run status, unused by this backend.
        """
        del status
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
        self._file = None
        self._writer = None


class TensorBoardLogger(ExperimentLogger):
    """Write scalar summaries using TensorBoard's dependency-light event writer."""

    def __init__(
        self,
        save_dir: str | Path,
        *,
        name: str = "tensorboard",
        version: int | str | None = None,
    ) -> None:
        """Initialize TensorBoard event logging.

        Args:
            save_dir: Directory receiving TensorBoard event files.
            name: Logger or experiment name.
            version: Optional run version displayed by progress callbacks.
        """
        self.save_dir = Path(save_dir)
        self._name = name
        self._version = version
        self._writer_class: Any | None = None
        self._writer: Any | None = None

    def setup(self) -> None:
        """Create a TensorBoard writer for a new task when needed.

        Raises:
            ModuleNotFoundError: If the optional `tensorboard` extra is not installed.
        """
        if self._writer is not None:
            return
        writer_class = self._writer_class
        if writer_class is None:
            try:
                writer_class = import_module("tensorboard.summary.writer.event_file_writer").EventFileWriter
            except ModuleNotFoundError as error:
                raise ModuleNotFoundError(
                    "TensorBoard logging requires `uv sync --extra tensorboard` in addition to a JAX backend extra."
                ) from error
            self._writer_class = writer_class
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._writer = writer_class(str(self.save_dir))

    @property
    def name(self) -> str:
        """Return the experiment name.

        Returns:
            Configured experiment name.
        """
        return self._name

    @property
    def version(self) -> int | str | None:
        """Return the experiment version.

        Returns:
            Configured version.
        """
        return self._version

    @property
    def log_dir(self) -> Path:
        """Return the TensorBoard run directory.

        Returns:
            Directory containing event files and hyperparameters.
        """
        return self.save_dir

    def log_hyperparams(self, parameters: Mapping[str, Any]) -> None:
        """Write resolved parameters to `hparams.yaml`.

        Args:
            parameters: Resolved experiment configuration.
        """
        self.setup()
        _write_hyperparameters(self.log_dir, parameters)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Write scalar TensorBoard summaries.

        Args:
            metrics: Host scalar metric mapping.
            step: Optimizer step associated with `metrics`.
        """
        self.setup()
        event_class = import_module("tensorboard.compat.proto.event_pb2").Event
        summary_class = import_module("tensorboard.compat.proto.summary_pb2").Summary

        values = [summary_class.Value(tag=name, simple_value=float(value)) for name, value in sorted(metrics.items())]
        writer = self._writer
        if writer is None:
            raise RuntimeError("TensorBoard did not create an event writer.")
        writer.add_event(event_class(wall_time=time.time(), step=step, summary=summary_class(value=values)))

    def log_artifact(self, path: str | Path) -> None:
        """Ignore artifacts because the scalar writer has no artifact store.

        Args:
            path: Existing local artifact path.
        """
        del path

    def finalize(self, status: str) -> None:
        """Flush and close the TensorBoard event writer.

        Args:
            status: Terminal run status, unused by this backend.
        """
        del status
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


class WandbLogger(ExperimentLogger):
    """Log experiments through the optional Weights & Biases SDK.

    The owned W&B run is finalized idempotently by :meth:`finalize`. When constructed inside :func:`task_wrapper`, the
    logger also registers the same finalizer as a failure-safe for errors outside :meth:`Trainer.fit`.

    Attributes:
        run: Lazily initialized W&B run returned by `wandb.init()`.
    """

    def __init__(
        self,
        project: str,
        *,
        save_dir: str | Path | None = None,
        entity: str | None = None,
        name: str | None = None,
        run_id: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        tags: Sequence[str] = (),
        mode: str | None = None,
        resume: str | bool | None = None,
        artifact_type: str = "model",
        init_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize one W&B run without making the SDK a core dependency.

        Args:
            project: W&B project name.
            save_dir: Local directory for W&B run files.
            entity: Optional W&B team or account name.
            name: Optional human-readable run name.
            run_id: Optional stable run identifier used for resumption.
            group: Optional run group.
            job_type: Optional job-type label.
            tags: Optional run tags.
            mode: W&B mode such as `online`, `offline`, or `disabled`.
            resume: W&B resume policy. When omitted with `run_id`, defaults to `allow`.
            artifact_type: Default type assigned to logged artifact paths.
            init_kwargs: Additional keyword arguments forwarded to `wandb.init()`.
        """
        self._log_dir = None if save_dir is None else Path(save_dir)
        self._name = project
        self._version = run_id
        self.artifact_type = artifact_type
        self._run: Any | None = None
        self._options = dict(init_kwargs or {})
        self._options.update(
            {
                "project": project,
                "entity": entity,
                "dir": None if self._log_dir is None else str(self._log_dir),
                "name": name,
                "id": run_id,
                "group": group,
                "job_type": job_type,
                "tags": tuple(tags),
                "mode": mode,
                "resume": "allow" if run_id is not None and resume is None else resume,
            }
        )

    def setup(self) -> None:
        """Start the configured W&B run when it is not already active.

        Raises:
            ModuleNotFoundError: If the optional `wandb` extra is not installed.
            RuntimeError: If `wandb.init()` does not return a run.
        """
        if self._run is not None:
            return
        try:
            wandb = import_module("wandb")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Weights & Biases logging requires `uv sync --extra wandb` in addition to a JAX backend extra."
            ) from error
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(**self._options)
        if run is None:
            raise RuntimeError("`wandb.init()` did not return an active run.")
        self._run = run
        register_task_finalizer(self.finalize)

    @property
    def run(self) -> Any:
        """Return the active W&B run, starting it on first access.

        Returns:
            Active W&B run returned by `wandb.init()`.
        """
        self.setup()
        if self._run is None:
            raise RuntimeError("W&B logger setup completed without an active run.")
        return self._run

    @property
    def name(self) -> str:
        """Return the W&B project name.

        Returns:
            Configured project name.
        """
        return self._name

    @property
    def version(self) -> str | None:
        """Return the configured W&B run identifier.

        Returns:
            Stable run identifier, or `None` when W&B generated it.
        """
        return self._version

    @property
    def log_dir(self) -> Path | None:
        """Return the local W&B storage directory.

        Returns:
            Configured local directory, or `None`.
        """
        return self._log_dir

    def log_hyperparams(self, parameters: Mapping[str, Any]) -> None:
        """Update the W&B run configuration.

        Args:
            parameters: Resolved experiment configuration.
        """
        self.run.config.update(dict(parameters), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log one scalar metric mapping at an explicit monotonically increasing step.

        Args:
            metrics: Host scalar metric mapping.
            step: Training step associated with `metrics`.
        """
        self.run.log(dict(metrics), step=step)

    def log_artifact(self, path: str | Path) -> None:
        """Upload a local file or directory as a versioned W&B artifact.

        Args:
            path: Existing local artifact path.
        """
        self.run.log_artifact(str(path), type=self.artifact_type)

    def finalize(self, status: str) -> None:
        """Flush pending W&B data and finish the run exactly once.

        Args:
            status: Terminal run status; only `success` maps to exit code `0`.
        """
        if self._run is None:
            return
        run = self._run
        self._run = None
        run.finish(exit_code=0 if status == "success" else 1)


def scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Transfer a metric mapping to validated Python scalar values.

    Args:
        metrics: Device or host metric mapping.

    Returns:
        Scalar metric mapping suitable for every logger backend.

    Raises:
        ValueError: If any metric contains more than one value.
    """
    result: dict[str, float] = {}
    for name, value in metrics.items():
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(f"Metric `{name}` must be scalar, got shape {array.shape}.")
        result[name] = float(array.reshape(()))
    return result


def create_default_logger(default_root_dir: str | Path, *, is_global_zero: bool = True) -> LoggerCollection:
    """Create a versioned local TensorBoard logger with a CSV fallback.

    Args:
        default_root_dir: Parent directory containing the `phijax_logs` experiment directory.
        is_global_zero: Whether this process owns filesystem logging.

    Returns:
        One configured local logger on global rank zero, otherwise an empty collection.
    """
    if not is_global_zero:
        return LoggerCollection()
    root = Path(default_root_dir).expanduser().resolve() / "phijax_logs"
    version_number, run_dir = _reserve_version(root)
    try:
        tensorboard_available = find_spec("tensorboard") is not None
    except (ImportError, ValueError):
        tensorboard_available = False
    logger: ExperimentLogger
    if tensorboard_available:
        logger = TensorBoardLogger(run_dir, name="phijax_logs", version=version_number)
    else:
        logger = CSVLogger(run_dir, name="phijax_logs", version=version_number)
    return LoggerCollection((logger,))


def _reserve_version(root: Path) -> tuple[int, Path]:
    """Atomically reserve the next unused integer run directory.

    Args:
        root: Experiment directory containing `version_N` children.

    Returns:
        Smallest available version and its newly created run directory.
    """
    root.mkdir(parents=True, exist_ok=True)
    version_number = 0
    while True:
        run_dir = root / f"version_{version_number}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            version_number += 1
            continue
        return version_number, run_dir


def _write_hyperparameters(log_dir: Path, parameters: Mapping[str, Any]) -> None:
    """Write one human-readable hyperparameter mapping.

    Args:
        log_dir: Existing local run directory.
        parameters: Resolved experiment parameters.
    """
    path = log_dir / "hparams.yaml"
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(parameters), file, sort_keys=True, default_flow_style=False)


__all__ = [
    "CSVLogger",
    "ConsoleLogger",
    "ExperimentLogger",
    "LoggerCollection",
    "TensorBoardLogger",
    "WandbLogger",
    "create_default_logger",
    "scalar_metrics",
]

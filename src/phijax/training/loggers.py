import csv
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from phijax.utils.pylogger import get_colorlogger
from phijax.utils.utils import register_task_finalizer


class ExperimentLogger:
    """Define the minimal interface shared by experiment-logging backends."""

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        """Record resolved experiment parameters.

        Args:
            parameters: Resolved scalar or nested experiment configuration.
        """

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Record scalar metrics for one optimizer step.

        Args:
            metrics: Host scalar metric mapping.
            step: Optimizer step associated with `metrics`.
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
    """Fan each logging operation out to multiple independent backends.

    Attributes:
        loggers: Ordered tuple of configured logging backends.
    """

    def __init__(self, loggers: Iterable[ExperimentLogger] = ()) -> None:
        """Initialize a logger fan-out collection.

        Args:
            loggers: Logging backends receiving every operation.
        """
        self.loggers = tuple(loggers)

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        """Forward hyperparameters to all backends.

        Args:
            parameters: Resolved experiment configuration.
        """
        for logger in self.loggers:
            logger.log_hyperparameters(parameters)

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

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
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

    def __init__(self, save_dir: str | Path, filename: str = "metrics.csv") -> None:
        """Initialize a long-form CSV logger.

        Args:
            save_dir: Directory containing the metrics file.
            filename: CSV filename within `save_dir`.
        """
        self.path = Path(save_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        if self.path.stat().st_size == 0:
            self._writer.writerow(("step", "metric", "value"))

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        """Leave configuration persistence to Hydra's resolved config artifact.

        Args:
            parameters: Resolved experiment configuration.
        """
        del parameters

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Append one row per scalar metric.

        Args:
            metrics: Host scalar metric mapping.
            step: Optimizer step associated with `metrics`.
        """
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
        if not self._file.closed:
            self._file.flush()
            self._file.close()


class TensorBoardLogger(ExperimentLogger):
    """Write scalar summaries using TensorBoard's dependency-light event writer."""

    def __init__(self, save_dir: str | Path) -> None:
        """Initialize TensorBoard event logging.

        Args:
            save_dir: Directory receiving TensorBoard event files.

        Raises:
            ModuleNotFoundError: If the optional `tensorboard` extra is not installed.
        """
        try:
            writer_class = import_module("tensorboard.summary.writer.event_file_writer").EventFileWriter
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "TensorBoard logging requires `uv sync --extra tensorboard` in addition to a JAX backend extra."
            ) from error
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._writer = writer_class(str(self.save_dir))

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        """Leave configuration persistence to Hydra's resolved config artifact.

        Args:
            parameters: Resolved experiment configuration.
        """
        del parameters

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Write scalar TensorBoard summaries.

        Args:
            metrics: Host scalar metric mapping.
            step: Optimizer step associated with `metrics`.
        """
        event_class = import_module("tensorboard.compat.proto.event_pb2").Event
        summary_class = import_module("tensorboard.compat.proto.summary_pb2").Summary

        values = [summary_class.Value(tag=name, simple_value=float(value)) for name, value in sorted(metrics.items())]
        self._writer.add_event(event_class(wall_time=time.time(), step=step, summary=summary_class(value=values)))

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
        self._writer.flush()
        self._writer.close()


class WandbLogger(ExperimentLogger):
    """Log experiments through the optional Weights & Biases SDK.

    The owned W&B run is finalized idempotently by :meth:`finalize`. When constructed inside :func:`task_wrapper`, the
    logger also registers the same finalizer as a failure-safe for errors outside :meth:`Trainer.fit`.

    Attributes:
        run: Active W&B run returned by `wandb.init()`.
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

        Raises:
            ModuleNotFoundError: If the optional `wandb` extra is not installed.
            RuntimeError: If `wandb.init()` does not return a run.
        """
        try:
            wandb = import_module("wandb")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Weights & Biases logging requires `uv sync --extra wandb` in addition to a JAX backend extra."
            ) from error
        resolved_save_dir = None if save_dir is None else Path(save_dir)
        if resolved_save_dir is not None:
            resolved_save_dir.mkdir(parents=True, exist_ok=True)
        resolved_resume = "allow" if run_id is not None and resume is None else resume
        options = dict(init_kwargs or {})
        options.update(
            {
                "project": project,
                "entity": entity,
                "dir": None if resolved_save_dir is None else str(resolved_save_dir),
                "name": name,
                "id": run_id,
                "group": group,
                "job_type": job_type,
                "tags": tuple(tags),
                "mode": mode,
                "resume": resolved_resume,
            }
        )
        self.run = wandb.init(**options)
        if self.run is None:
            raise RuntimeError("`wandb.init()` did not return an active run.")
        self.artifact_type = artifact_type
        self._finished = False
        register_task_finalizer(self.finalize)

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
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
        if self._finished:
            return
        self._finished = True
        self.run.finish(exit_code=0 if status == "success" else 1)


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


__all__ = [
    "CSVLogger",
    "ConsoleLogger",
    "ExperimentLogger",
    "LoggerCollection",
    "TensorBoardLogger",
    "WandbLogger",
    "scalar_metrics",
]

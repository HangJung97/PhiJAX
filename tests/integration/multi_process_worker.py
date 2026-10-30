import json
import os
import signal
import sys
from functools import partial
from pathlib import Path
from typing import Any

import jax

from phijax.training import initialize_distributed

# Distributed initialization must precede every other JAX backend operation.
if __name__ == "__main__" and sys.argv[1] != "single-process":
    initialize_distributed(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import multihost_utils

from phijax import PhiModule, Trainer
from phijax.callbacks import Callback, ModelCheckpoint
from phijax.data import ChunkedPredictionSource, HostPool, NamedBatchSource, PhiDataModule
from phijax.data.datamodule import DataStage
from phijax.equations import base_data_fidelity
from phijax.models import build_mlp
from phijax.objectives import CompositeObjective
from phijax.training import OrbaxCheckpointIO, create_strategy


class _LineDataModule(PhiDataModule):
    """Provide one supervised pool sampled randomly at every step."""

    def setup(self, stage: DataStage) -> None:
        """Build the training pool on every process.

        Args:
            stage: Requested stage; only `fit` is used.
        """
        del stage
        inputs = np.linspace(-1.0, 1.0, 64, dtype=np.float32)[:, None]
        # Ten prediction rows in chunks of six: the last chunk is padded, and six rows do not split over four devices.
        predict_inputs = np.linspace(-0.9, 0.9, 10, dtype=np.float32)[:, None]
        self.pools = {
            "data": HostPool(inputs=inputs, targets=np.sin(3.0 * inputs)),
            "predict": HostPool(inputs=predict_inputs),
        }

    def train_batch_source(self, batch_keys: tuple[str, ...], key: jax.Array) -> NamedBatchSource:
        """Sample four rows per process and step.

        Args:
            batch_keys: Objective batch keys.
            key: Process-local sampling key supplied by the Trainer.

        Returns:
            Step-indexed batch source.
        """
        return NamedBatchSource.from_pools({"data": self.pools["data"]}, {"data": 4}, key, names=batch_keys)

    def predict_batch_source(self) -> ChunkedPredictionSource:
        """Yield ordered prediction chunks of six rows.

        Returns:
            Chunked prediction source.
        """
        return ChunkedPredictionSource(self.pools["predict"], 6)


class _BatchRecorder(Callback):
    """Record every global training batch, gathered from all processes."""

    def __init__(self) -> None:
        """Start with no recorded batches."""
        self.batches: dict[int, list[float]] = {}

    def on_train_batch_start(self, context: Any) -> None:
        """Gather the sharded batch inputs of one step.

        Args:
            context: Fit context carrying the placed global batch.
        """
        inputs = multihost_utils.process_allgather(context.batch["data"]["inputs"], tiled=True)
        self.batches[context.step] = np.asarray(inputs).reshape(-1).tolist()


class _StopOnProcessOne(Callback):
    """Request a stop on process 1 only, after step 2, to check that every process stops together."""

    def __init__(self, request: str) -> None:
        """Choose how process 1 requests the stop.

        Args:
            request: `should_stop`, `SIGINT`, or `SIGTERM`.
        """
        self.request = request
        self._trainer: Any = None

    def connect(self, trainer: Any) -> None:
        """Keep the Trainer whose `should_stop` flag this callback sets.

        Args:
            trainer: Host Trainer coordinating callback hooks.
        """
        self._trainer = trainer

    def on_train_metrics(self, context: Any) -> None:
        """Request the stop on process 1 after step 2.

        Args:
            context: Post-update fit context.
        """
        if jax.process_index() != 1 or context.step != 2:
            return
        if self.request == "should_stop":
            self._trainer.should_stop = True
        else:
            os.kill(os.getpid(), getattr(signal, self.request))


def _fit(
    directory: Path,
    max_steps: int,
    *,
    resume: bool = False,
    seed: int = 0,
    stop_request: str | None = None,
    predict: bool = False,
) -> dict[str, Any]:
    """Fit across every process and return host-readable results.

    Args:
        directory: Checkpoint directory for this run.
        max_steps: Global step at which training stops.
        resume: Whether to resume from the latest checkpoint in `directory`.
        seed: Initialization seed for this process.
        stop_request: Optional stop that process 1 alone requests after step 2.
        predict: Whether to predict after fitting and compare with a direct single-device evaluation.

    Returns:
        Recorded batches, final step, flattened parameters, and metrics.
    """
    recorder = _BatchRecorder()
    trainer = Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        devices="auto",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=directory.parent,
        callbacks=(
            recorder,
            *(() if stop_request is None else (_StopOnProcessOne(stop_request),)),
            ModelCheckpoint(OrbaxCheckpointIO(directory, enable_async_checkpointing=False), every_n_steps=2),
        ),
    )
    module = PhiModule(
        partial(build_mlp, input_dim=1, output_dim=1, hidden=(8,)),
        CompositeObjective.from_equations({"data": base_data_fidelity}),
    )
    result = trainer.fit(
        module,
        datamodule=_LineDataModule(),
        optimizer=optax.adam(1e-2),
        seed=seed,
        ckpt_path=directory if resume else None,
    )
    parameters = [np.asarray(leaf).reshape(-1).tolist() for leaf in jax.tree.leaves(result.state.model_state)]
    prediction: dict[str, Any] = {}
    if predict:
        data_module = _LineDataModule()
        predictions = trainer.predict(result, datamodule=data_module)
        # Evaluate this process's copy of the replicated state on one local device as the reference.
        local_state = jax.tree.map(lambda leaf: leaf.addressable_data(0), result.state.model_state)
        expected = result.module.predict_step(local_state, {"inputs": jnp.asarray(data_module.pools["predict"].inputs)})
        prediction = {
            "values": np.asarray(predictions).reshape(-1).tolist(),
            "max_error": float(np.max(np.abs(np.asarray(predictions) - np.asarray(expected)))),
        }
    return {
        "batches": recorder.batches,
        "step": int(np.asarray(result.state.step)),
        "parameters": parameters,
        "metrics": {name: float(np.asarray(value)) for name, value in result.metrics.items()},
        "mesh_size": len(trainer.strategy.devices),
        "stopped_early": result.stopped_early,
        "interrupted": result.interrupted,
        "prediction": prediction,
    }


def main(output: Path) -> None:
    """Run uninterrupted and resumed fits and write this process's results as JSON.

    Args:
        output: Directory shared by every process for checkpoints and results.
    """
    results: dict[str, Any] = {
        "full": _fit(output / "full", 4, predict=True),
        "first_half": _fit(output / "split", 2),
        "resumed": _fit(output / "split", 4, resume=True),
        "single_device_mesh": [
            (device.process_index, device.id) for device in create_strategy("cpu", devices=1).devices
        ],
    }
    try:
        _fit(output / "mismatch", 1, seed=jax.process_index())
    except ValueError as error:
        results["mismatch_error"] = str(error)
    results["should_stop"] = _fit(output / "should_stop", 10, stop_request="should_stop")
    results["sigint"] = _fit(output / "sigint", 10, stop_request="SIGINT")
    # JAX's preemption sync step stays fixed for the process, so the `SIGTERM` case runs last.
    try:
        _fit(output / "sigterm", 10, stop_request="SIGTERM")
    except SystemExit as error:
        saved = sorted(int(path.name) for path in (output / "sigterm").iterdir() if path.name.isdigit())
        results["sigterm"] = {"exit_code": error.code, "saved_steps": saved}
    (output / f"process_{jax.process_index()}.json").write_text(json.dumps(results))


def main_single_process(output: Path) -> None:
    """Fit and predict in one process over all local devices and write the results as JSON.

    Args:
        output: Directory for checkpoints and results.
    """
    (output / "single_process.json").write_text(json.dumps(_fit(output / "single", 2, predict=True)))


if __name__ == "__main__":
    if sys.argv[1] == "single-process":
        main_single_process(Path(sys.argv[2]))
    else:
        main(Path(sys.argv[4]))

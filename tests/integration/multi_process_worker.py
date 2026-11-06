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
from phijax.balancers import ExactNTKBalancer
from phijax.callbacks import Callback, ModelCheckpoint
from phijax.data import ChunkedPredictionSource, HostPool, NamedBatchSource, PhiDataModule
from phijax.data.datamodule import DataStage
from phijax.equations import base_data_fidelity
from phijax.models import build_mlp
from phijax.objectives import CompositeObjective
from phijax.training import OrbaxCheckpointIO, create_strategy
from phijax.training.precision import PrecisionPolicy


class _LineDataModule(PhiDataModule):
    """Provide one supervised pool sampled randomly at every step, or a small pool used whole."""

    def __init__(self, *, full_pool: bool = False) -> None:
        """Choose the training pool.

        Args:
            full_pool: Whether to train on all six rows of a small pool, which do not divide over four devices.
        """
        super().__init__()
        self.full_pool = full_pool

    def setup(self, stage: DataStage) -> None:
        """Build the training pool on every process.

        Args:
            stage: Requested stage; only `fit` is used.
        """
        del stage
        inputs = np.linspace(-1.0, 1.0, 6 if self.full_pool else 64, dtype=np.float32)[:, None]
        # Ten prediction rows with one row per device: four-row chunks on four devices, the last one padded.
        predict_inputs = np.linspace(-0.9, 0.9, 10, dtype=np.float32)[:, None]
        self.pools = {
            "data": HostPool(inputs=inputs, targets=np.sin(3.0 * inputs)),
            "predict": HostPool(inputs=predict_inputs),
        }

    def train_batch_source(self, batch_keys: tuple[str, ...], key: jax.Array) -> NamedBatchSource:
        """Sample four rows per device and step, or the whole pool.

        Args:
            batch_keys: Objective batch keys.
            key: Process-local sampling key supplied by the Trainer.

        Returns:
            Step-indexed batch source.
        """
        size = "all" if self.full_pool else 4
        return NamedBatchSource.from_pools({"data": self.pools["data"]}, {"data": size}, key, names=batch_keys)

    def predict_batch_source(self) -> ChunkedPredictionSource:
        """Yield ordered prediction chunks of one row per device.

        Returns:
            Chunked prediction source.
        """
        return ChunkedPredictionSource(self.pools["predict"], 1)


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
    ntk: bool = False,
    full_pool: bool = False,
    devices: int | str = "auto",
) -> dict[str, Any]:
    """Fit across every process and return host-readable results.

    Args:
        directory: Checkpoint directory for this run.
        max_steps: Global step at which training stops.
        resume: Whether to resume from the latest checkpoint in `directory`.
        seed: Initialization seed for this process.
        stop_request: Optional stop that process 1 alone requests after step 2.
        predict: Whether to predict after fitting and compare with a direct single-device evaluation.
        ntk: Whether to balance with exact NTK on four diagnostic rows per process, updated at every step.
        full_pool: Whether to train on a whole six-row pool with the `all` policy.
        devices: Devices per process.

    Returns:
        Recorded batches, final step, flattened parameters, and metrics.
    """
    recorder = _BatchRecorder()
    trainer = Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        devices=devices,
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
        datamodule=_LineDataModule(full_pool=full_pool),
        optimizer=optax.adam(1e-2),
        balancer=(
            ExactNTKBalancer(module.loss_names, update_every_n_steps=1, update_start_step=0, kernel_size=4)
            if ntk
            else None
        ),
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
        # The blueprint is restored from the terminal checkpoint, which stores the final model state.
        from_checkpoint = trainer.predict(module, ckpt_path=directory, datamodule=_LineDataModule())
        prediction = {
            "values": np.asarray(predictions).reshape(-1).tolist(),
            "max_error": float(np.max(np.abs(np.asarray(predictions) - np.asarray(expected)))),
            "checkpoint_max_error": float(np.max(np.abs(np.asarray(from_checkpoint) - np.asarray(predictions)))),
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
    results["full_pool"] = _fit(output / "full_pool", 2, full_pool=True)
    results["should_stop"] = _fit(output / "should_stop", 10, stop_request="should_stop")
    results["sigint"] = _fit(output / "sigint", 10, stop_request="SIGINT")
    # JAX's preemption sync step stays fixed for the process, so the `SIGTERM` case runs last.
    try:
        _fit(output / "sigterm", 10, stop_request="SIGTERM")
    except SystemExit as error:
        saved = sorted(int(path.name) for path in (output / "sigterm").iterdir() if path.name.isdigit())
        results["sigterm"] = {"exit_code": error.code, "saved_steps": saved}
    (output / f"process_{jax.process_index()}.json").write_text(json.dumps(results))


def _ntk_layout_check() -> dict[str, Any]:
    """Compile an exact-NTK update on rows split over every device and compare it with one device.

    Returns:
        All-gather count of the compiled update and the largest NTK difference from a one-device evaluation.
    """
    strategy = create_strategy("cpu", "auto")
    blueprint = PhiModule(
        partial(build_mlp, input_dim=1, output_dim=1, hidden=(16,)),
        CompositeObjective.from_equations({"data": base_data_fidelity}),
    )
    module, model_state = blueprint.prepare_model(
        key=jax.random.key(0), input_mean=None, input_std=None, precision=PrecisionPolicy.from_name("32-true")
    )
    balancer = ExactNTKBalancer(module.loss_names, update_every_n_steps=1, kernel_size=8)
    inputs = np.linspace(-1.0, 1.0, 8 * len(strategy.devices), dtype=np.float32)[:, None]
    host_batches = {"data": {"inputs": inputs, "targets": np.sin(3.0 * inputs)}}
    update = balancer.make_update(module)
    sharded = (strategy.place_state(model_state), strategy.place_batch(host_batches))
    state = strategy.place_state(balancer.initialize())
    compiled = update.lower(*sharded, state).compile()
    one_device = jax.devices()[0]
    reference = update(
        jax.device_put(model_state, one_device), jax.device_put(host_batches, one_device), balancer.initialize()
    )
    return {
        "all_gathers": compiled.as_text().count("all-gather("),
        "max_difference": float(np.max(np.abs(np.asarray(compiled(*sharded, state).traces) - reference.traces))),
    }


def main_single_process(output: Path) -> None:
    """Fit and predict in one process over all local devices and write the results as JSON.

    Args:
        output: Directory for checkpoints and results.
    """
    results = {
        "fit": _fit(output / "single", 2, predict=True, ntk=True),
        # `all` batches are deterministic, so four devices must reproduce one device.
        "full_pool": _fit(output / "full_pool", 3, full_pool=True),
        "full_pool_one_device": _fit(output / "full_pool_one_device", 3, full_pool=True, devices=1),
        "ntk_layout": _ntk_layout_check(),
    }
    (output / "single_process.json").write_text(json.dumps(results))


if __name__ == "__main__":
    if sys.argv[1] == "single-process":
        main_single_process(Path(sys.argv[2]))
    else:
        main(Path(sys.argv[4]))

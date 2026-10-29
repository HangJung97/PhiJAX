from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from phijax import PhiModule, Trainer
from phijax.balancers import ExactNTKBalancer
from phijax.callbacks import EarlyStopping, ModelCheckpoint
from phijax.data import HostPool, NamedBatchSource, PhiDataModule
from phijax.data.datamodule import DataStage
from phijax.equations import base_data_fidelity
from phijax.models import build_mlp
from phijax.objectives import CompositeObjective
from phijax.training import FitResult, OrbaxCheckpointIO


class _TwoPoolDataModule(PhiDataModule):
    """Provide two supervised pools sampled randomly at every step."""

    def setup(self, stage: DataStage) -> None:
        """Build the training pools.

        Args:
            stage: Requested stage; only `fit` is used.
        """
        del stage
        data_inputs = np.linspace(-1.0, 1.0, 32, dtype=np.float32)[:, None]
        anchor_inputs = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float32)
        self.pools = {
            "data": HostPool(inputs=data_inputs, targets=np.sin(3.0 * data_inputs)),
            "anchor": HostPool(inputs=anchor_inputs, targets=np.sin(3.0 * anchor_inputs)),
        }

    def train_batch_source(self, batch_keys: tuple[str, ...], key: jax.Array) -> NamedBatchSource:
        """Sample random rows from both pools at each step.

        Args:
            batch_keys: Objective batch keys.
            key: Explicit root sampling key folded with each global step.

        Returns:
            Step-indexed batch source.
        """
        return NamedBatchSource.from_pools(self._require_setup("fit"), {"data": 8, "anchor": 2}, key, names=batch_keys)


def _fit(checkpoints: Path, max_steps: int, *, resume: bool = False) -> tuple[FitResult, Trainer]:
    """Fit with exact-NTK balancing, early stopping, and periodic checkpoints.

    Args:
        checkpoints: Checkpoint root for this run.
        max_steps: Global step at which training stops.
        resume: Whether to resume from the latest checkpoint in `checkpoints`.

    Returns:
        Fit result and the Trainer that produced it.
    """
    model = partial(build_mlp, input_dim=1, output_dim=1, hidden=(8,))
    objective = CompositeObjective.from_equations({"data": base_data_fidelity, "anchor": base_data_fidelity})
    module = PhiModule(model, objective)
    trainer = Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=checkpoints.parent,
        callbacks=(
            EarlyStopping(monitor="train/loss", patience=100),
            ModelCheckpoint(
                OrbaxCheckpointIO(checkpoints, enable_async_checkpointing=False), every_n_steps=5, save_top_k=-1
            ),
        ),
    )
    balancer = ExactNTKBalancer(
        module.loss_names,
        update_every_n_steps=2,
        update_start_step=0,
        kernel_size={"data": 4, "anchor": "all"},
        resample=True,
    )
    result = trainer.fit(
        module,
        datamodule=_TwoPoolDataModule(),
        optimizer=optax.adam(1e-2),
        seed=0,
        balancer=balancer,
        ckpt_path=checkpoints if resume else None,
    )
    return result, trainer


def _early_stopping_state(trainer: Trainer) -> dict[str, Any]:
    """Return the state of the Trainer's early-stopping callback.

    Args:
        trainer: Trainer configured by :func:`_fit`.

    Returns:
        Early-stopping `state_dict()`.
    """
    return next(dict(callback.state_dict()) for callback in trainer.callbacks if isinstance(callback, EarlyStopping))


def test_resumed_fit_matches_an_uninterrupted_fit(tmp_path: Path) -> None:
    """Verify 5 steps plus a resume to step 10 reproduce 10 uninterrupted steps exactly.

    The run uses random batch sampling, exact-NTK updates with resampled and full-pool diagnostic batches, early
    stopping, and periodic checkpoints, so every resumable stream must continue from the checkpoint.

    Args:
        tmp_path: Temporary directory for both runs.
    """
    uninterrupted, uninterrupted_trainer = _fit(tmp_path / "uninterrupted", 10)
    _fit(tmp_path / "split", 5)
    resumed, resumed_trainer = _fit(tmp_path / "split", 10, resume=True)

    assert int(uninterrupted.state.step) == int(resumed.state.step) == 10
    assert resumed.iterations == 5
    for expected, actual in zip(jax.tree.leaves(uninterrupted.state), jax.tree.leaves(resumed.state), strict=True):
        if jnp.issubdtype(expected.dtype, jax.dtypes.prng_key):
            expected, actual = jax.random.key_data(expected), jax.random.key_data(actual)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert resumed.metrics == uninterrupted.metrics
    assert _early_stopping_state(resumed_trainer) == _early_stopping_state(uninterrupted_trainer)

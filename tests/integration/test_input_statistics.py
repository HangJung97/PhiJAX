from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import jax
import numpy as np
import optax
import pytest

from phijax import InputStatistics, PhiModule, Trainer
from phijax.balancers import StaticLossBalancer
from phijax.callbacks import ModelCheckpoint
from phijax.data import ChunkedPredictionSource, HostPool, NamedBatchSource, PhiDataModule, input_statistics
from phijax.data.datamodule import DataStage
from phijax.equations import base_data_fidelity
from phijax.models import build_mlp
from phijax.objectives import CompositeObjective
from phijax.training import OrbaxCheckpointIO, build_training_plan, load_input_statistics
from phijax.training import trainer as trainer_module


class _ShiftedDataModule(PhiDataModule):
    """Provide one supervised pool whose coordinates, and therefore statistics, depend on an offset."""

    def __init__(self, offset: float, statistics: InputStatistics | None = None) -> None:
        """Store the coordinate offset and optional statistics override.

        Args:
            offset: Shift applied to every coordinate.
            statistics: Statistics returned instead of the empirical pool statistics.
        """
        super().__init__()
        self.offset = offset
        self.statistics_override = statistics

    def setup(self, stage: DataStage) -> None:
        """Build the training or prediction pool.

        Args:
            stage: Requested `fit` or `predict` stage.
        """
        inputs = (np.linspace(0.0, 1.0, 16, dtype=np.float32) + self.offset)[:, None]
        pool = HostPool(inputs=inputs, targets=np.sin(inputs))
        self.pools = {"data": pool} if stage == "fit" else {"predict": pool}

    def train_batch_source(self, batch_keys: tuple[str, ...], key: jax.Array) -> NamedBatchSource:
        """Sample every row of the supervised pool at each step.

        Args:
            batch_keys: Objective batch keys.
            key: Explicit root sampling key.

        Returns:
            Step-indexed batch source.
        """
        return NamedBatchSource.from_pools(self._require_setup("fit"), {"data": "all"}, key, names=batch_keys)

    def input_statistics(self) -> InputStatistics | tuple[np.ndarray, np.ndarray]:
        """Return the override, or empirical statistics of the training pool.

        Returns:
            Per-coordinate mean and standard deviation.
        """
        if self.statistics_override is not None:
            return self.statistics_override
        return input_statistics(self.normalization_pools())

    def predict_batch_source(self) -> ChunkedPredictionSource:
        """Build ordered prediction chunks.

        Returns:
            Prediction source over the prediction pool.
        """
        return ChunkedPredictionSource(self._require_setup("predict")["predict"], 8)

    def prediction_pool(self) -> HostPool | None:
        """Return the prediction pool when it exists.

        Returns:
            Prediction pool, or `None` during fitting.
        """
        return self.pools.get("predict")


def _module() -> PhiModule:
    """Create a normalized MLP module with one data-fidelity loss.

    Returns:
        Unbound module blueprint.
    """
    model = partial(build_mlp, input_dim=1, output_dim=1, hidden=(8,), input_norm=True)
    return PhiModule(model, CompositeObjective.from_equations({"data": base_data_fidelity}))


def _trainer(root: Path, checkpoints: Path | None = None) -> Trainer:
    """Create a quiet CPU trainer, optionally saving a terminal checkpoint.

    Args:
        root: Default root directory for trainer outputs.
        checkpoints: Optional checkpoint root.

    Returns:
        Configured trainer.
    """
    callbacks = ()
    if checkpoints is not None:
        callbacks = (ModelCheckpoint(OrbaxCheckpointIO(checkpoints, enable_async_checkpointing=False)),)
    return Trainer(
        max_steps=2,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=root,
        callbacks=callbacks,
    )


def _statistics_of(data_module: _ShiftedDataModule) -> Mapping[str, Any]:
    """Compute a DataModule's empirical fit statistics.

    Args:
        data_module: DataModule whose fit pools define the statistics.

    Returns:
        Mapping with `mean` and `std` host arrays.
    """
    data_module.setup("fit")
    mean, std = data_module.input_statistics()
    return {"mean": np.asarray(mean), "std": np.asarray(std)}


def _assert_statistics(actual: InputStatistics | None, expected: Mapping[str, Any]) -> None:
    """Assert that statistics equal the expected host arrays.

    Args:
        actual: Statistics to check.
        expected: Mapping with expected `mean` and `std`.
    """
    assert actual is not None
    np.testing.assert_allclose(np.asarray(actual.mean), expected["mean"], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(actual.std), expected["std"], rtol=1e-6)


@pytest.fixture
def source_run(tmp_path: Path) -> tuple[Path, Trainer, Any, Mapping[str, Any]]:
    """Train on the source DataModule and save its terminal checkpoint.

    Args:
        tmp_path: Temporary directory for outputs and checkpoints.

    Returns:
        Checkpoint root, source trainer, fit result, and source statistics.
    """
    checkpoints = tmp_path / "source"
    trainer = _trainer(tmp_path, checkpoints)
    result = trainer.fit(_module(), datamodule=_ShiftedDataModule(0.0), optimizer=optax.adam(1e-2), seed=0)
    return checkpoints, trainer, result, _statistics_of(_ShiftedDataModule(0.0))


def test_fit_stores_statistics_and_resume_keeps_them(
    source_run: tuple[Path, Trainer, Any, Mapping[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fit stores the DataModule statistics and a full resume rebinds to the checkpoint statistics.

    Args:
        source_run: Source checkpoint, trainer, result, and statistics.
        tmp_path: Temporary directory for trainer outputs.
        monkeypatch: Pytest fixture used to observe the module given to the training plan.
    """
    checkpoints, _, source_result, source_statistics = source_run
    target_statistics = _statistics_of(_ShiftedDataModule(5.0))
    planned_statistics: list[InputStatistics | None] = []
    original_build = trainer_module.build_training_plan

    def recording_build(trainer: Trainer, module: PhiModule, *args: Any) -> Any:
        """Record the statistics of the module captured by the training plan.

        Args:
            trainer: Owning trainer.
            module: Module the plan compiles.
            *args: Balancer and optimizer.

        Returns:
            Training plan from the original builder.
        """
        planned_statistics.append(module.input_statistics)
        return original_build(trainer, module, *args)

    monkeypatch.setattr(trainer_module, "build_training_plan", recording_build)
    # The resuming trainer has no checkpoint callback, so the stored callback state is skipped with a warning.
    with pytest.warns(UserWarning, match="callbacks this Trainer does not have"):
        resumed = _trainer(tmp_path).fit(
            _module(),
            datamodule=_ShiftedDataModule(5.0),
            optimizer=optax.adam(1e-2),
            seed=1,
            ckpt_path=checkpoints,
        )

    _assert_statistics(source_result.state.input_statistics, source_statistics)
    _assert_statistics(source_result.module.input_statistics, source_statistics)
    _assert_statistics(resumed.state.input_statistics, source_statistics)
    _assert_statistics(resumed.module.input_statistics, source_statistics)
    _assert_statistics(planned_statistics[0], source_statistics)
    assert not np.allclose(target_statistics["mean"], source_statistics["mean"])


def test_weights_only_transfer_uses_target_statistics_by_default(
    source_run: tuple[Path, Trainer, Any, Mapping[str, Any]],
    tmp_path: Path,
) -> None:
    """Verify transfer learning uses target statistics, and `load_input_statistics` opts into the source statistics.

    Args:
        source_run: Source checkpoint, trainer, result, and statistics.
        tmp_path: Temporary directory for trainer outputs.
    """
    checkpoints, _, _, source_statistics = source_run
    target_statistics = _statistics_of(_ShiftedDataModule(5.0))
    fit = partial(_trainer(tmp_path).fit, _module(), optimizer=optax.adam(1e-2), seed=1, ckpt_path=checkpoints)

    transferred = fit(datamodule=_ShiftedDataModule(5.0), weights_only=True)
    with_source = fit(
        datamodule=_ShiftedDataModule(5.0, statistics=load_input_statistics(checkpoints)),
        weights_only=True,
    )

    _assert_statistics(transferred.state.input_statistics, target_statistics)
    _assert_statistics(transferred.module.input_statistics, target_statistics)
    _assert_statistics(with_source.state.input_statistics, source_statistics)
    _assert_statistics(with_source.module.input_statistics, source_statistics)


def test_prediction_from_checkpoint_uses_checkpoint_statistics(
    source_run: tuple[Path, Trainer, Any, Mapping[str, Any]],
) -> None:
    """Verify downstream-style prediction reproduces in-memory predictions despite different template statistics.

    Args:
        source_run: Source checkpoint, trainer, result, and statistics.
    """
    checkpoints, trainer, source_result, _ = source_run
    in_memory = trainer.predict(source_result, datamodule=_ShiftedDataModule(0.0))
    # Mirror downstream predict.py: rebuild the module from current data and a template without statistics.
    target_statistics = _statistics_of(_ShiftedDataModule(5.0))
    module, model_state = _module().prepare_model(
        key=jax.random.key(3),
        input_mean=target_statistics["mean"],
        input_std=target_statistics["std"],
        precision=trainer.precision,
    )
    template = trainer.initialize_state(
        model_state,
        optax.adam(1e-2),
        StaticLossBalancer(module.loss_names).initialize(),
        jax.random.key(4),
    )

    from_checkpoint = trainer.predict_state(module, template, ckpt_path=checkpoints, datamodule=_ShiftedDataModule(0.0))

    assert in_memory is not None
    assert from_checkpoint is not None
    np.testing.assert_allclose(from_checkpoint, in_memory, rtol=1e-6)


def test_fit_state_rejects_statistics_that_differ_from_the_compiled_module(
    source_run: tuple[Path, Trainer, Any, Mapping[str, Any]],
    tmp_path: Path,
) -> None:
    """Verify the explicit API refuses to train a plan compiled for statistics other than the restored ones.

    Args:
        source_run: Source checkpoint, trainer, result, and statistics.
        tmp_path: Temporary directory for trainer outputs.
    """
    checkpoints, _, _, _ = source_run
    trainer = _trainer(tmp_path)
    target_statistics = _statistics_of(_ShiftedDataModule(5.0))
    module, model_state = _module().prepare_model(
        key=jax.random.key(3),
        input_mean=target_statistics["mean"],
        input_std=target_statistics["std"],
        precision=trainer.precision,
    )
    balancer = StaticLossBalancer(module.loss_names)
    optimizer = optax.adam(1e-2)
    plan = build_training_plan(trainer, module, balancer, optimizer)
    template = trainer.initialize_state(model_state, optimizer, balancer.initialize(), jax.random.key(4))
    data_module = MagicMock()

    with pytest.raises(ValueError, match="restored input statistics or periods differ"):
        trainer.fit_state(module, plan, template, ckpt_path=checkpoints, datamodule=data_module)

    data_module.teardown_stage.assert_called_once_with("fit")


def test_fit_tears_down_when_checkpoint_restore_fails(tmp_path: Path) -> None:
    """Verify `fit` releases the DataModule when restoring a missing checkpoint fails.

    Args:
        tmp_path: Temporary directory for trainer outputs.
    """
    data_module = _ShiftedDataModule(0.0)
    teardown_stages: list[str] = []
    data_module.teardown_stage = teardown_stages.append  # type: ignore[method-assign]

    with pytest.raises(FileNotFoundError):
        _trainer(tmp_path).fit(
            _module(),
            datamodule=data_module,
            optimizer=optax.adam(1e-2),
            seed=0,
            ckpt_path=tmp_path / "missing",
        )

    assert teardown_stages == ["fit"]


class _PeriodicDataModule(_ShiftedDataModule):
    """Declare the only input axis periodic with a configurable physical period."""

    def __init__(self, offset: float, period: float) -> None:
        """Store the coordinate offset and the declared period.

        Args:
            offset: Shift applied to every coordinate.
            period: Physical period declared for axis 0.
        """
        super().__init__(offset)
        self.period = period

    def input_periods(self) -> dict[int, float]:
        """Declare the physical period of axis 0.

        Returns:
            Period keyed by input axis.
        """
        return {0: self.period}


def _periodic_module(override: float | None = None) -> PhiModule:
    """Create a normalized MLP module with periodic features on its only axis.

    Args:
        override: Optional configured period that overrides the declared one.

    Returns:
        Unbound module blueprint.
    """
    options = None if override is None else {"periods": {0: override}}
    model = partial(
        build_mlp,
        input_dim=1,
        output_dim=1,
        hidden=(8,),
        input_norm=True,
        periodic_features=True,
        periodic_features_kwargs=options,
    )
    return PhiModule(model, CompositeObjective.from_equations({"data": base_data_fidelity}))


def test_resume_and_prediction_use_checkpoint_periods(tmp_path: Path) -> None:
    """Verify resumed fits and checkpoint predictions apply the trained periods, not the current DataModule's.

    Args:
        tmp_path: Temporary directory for outputs and checkpoints.
    """
    checkpoints = tmp_path / "source"
    trainer = _trainer(tmp_path, checkpoints)
    fit = partial(trainer.fit, optimizer=optax.adam(1e-2), seed=0)
    source = fit(_periodic_module(), datamodule=_PeriodicDataModule(0.0, 0.6))
    in_memory = trainer.predict(source, datamodule=_PeriodicDataModule(0.0, 0.6))

    with pytest.warns(UserWarning, match="callbacks this Trainer does not have"):
        resumed = _trainer(tmp_path).fit(
            _periodic_module(),
            datamodule=_PeriodicDataModule(0.0, 0.9),
            optimizer=optax.adam(1e-2),
            seed=1,
            ckpt_path=checkpoints,
        )
    module, model_state = _periodic_module().prepare_model(
        key=jax.random.key(3),
        input_mean=None,
        input_std=None,
        precision=trainer.precision,
        input_periods={0: 0.9},
    )
    template = trainer.initialize_state(
        model_state, optax.adam(1e-2), StaticLossBalancer(module.loss_names).initialize(), jax.random.key(4)
    )
    from_checkpoint = trainer.predict_state(
        module, template, ckpt_path=checkpoints, datamodule=_PeriodicDataModule(0.0, 0.6)
    )

    assert source.state.input_periods == ((0, 0.6),)
    assert source.module.input_periods == ((0, 0.6),)
    assert resumed.state.input_periods == ((0, 0.6),)
    assert resumed.module.input_periods == ((0, 0.6),)
    assert from_checkpoint is not None
    assert in_memory is not None
    np.testing.assert_allclose(from_checkpoint, in_memory, rtol=1e-6)


def test_transfer_uses_target_periods_and_configured_periods_win(tmp_path: Path) -> None:
    """Verify transfer learning uses the target DataModule's period and a configured override beats every source.

    Args:
        tmp_path: Temporary directory for outputs and checkpoints.
    """
    checkpoints = tmp_path / "source"
    _trainer(tmp_path, checkpoints).fit(
        _periodic_module(), datamodule=_PeriodicDataModule(0.0, 0.6), optimizer=optax.adam(1e-2), seed=0
    )
    fit = partial(_trainer(tmp_path).fit, optimizer=optax.adam(1e-2), seed=1, ckpt_path=checkpoints)

    transferred = fit(_periodic_module(), datamodule=_PeriodicDataModule(5.0, 0.9), weights_only=True)
    with pytest.warns(UserWarning, match="callbacks this Trainer does not have"):
        overridden = fit(_periodic_module(override=0.4), datamodule=_PeriodicDataModule(0.0, 0.9))

    assert transferred.state.input_periods == ((0, 0.9),)
    assert transferred.module.input_periods == ((0, 0.9),)
    assert overridden.state.input_periods == ((0, 0.4),)
    assert overridden.module.input_periods == ((0, 0.4),)

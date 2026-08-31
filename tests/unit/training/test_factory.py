from typing import Any
from unittest.mock import MagicMock

import jax
import numpy as np
import pytest
from omegaconf import OmegaConf

from phijax.balancers import BalancerUpdatePlan
from phijax.callbacks import EarlyStopping, ModelSummary, TQDMProgressBar
from phijax.core import BasePhiModule, PhiModule
from phijax.data import PhiDataModule
from phijax.integrations.hydra import (
    build_trainer,
    instantiate_balancer,
    instantiate_callbacks,
    instantiate_data_module,
    instantiate_loggers,
    instantiate_model_factory,
    instantiate_module,
    instantiate_trainer,
)
from phijax.integrations.hydra import factory as factory_module
from phijax.models import InitializedModel
from phijax.training import ConsoleLogger, Trainer, TrainingPlan, build_training_plan
from phijax.types import NamedBatches


class _TestObjective:
    """Provide the smallest objective accepted by the generic module."""

    @property
    def loss_names(self) -> tuple[str, ...]:
        """Return one stable loss name.

        Returns:
            Single test loss name.
        """
        return ("loss",)

    @property
    def batch_keys(self) -> tuple[str, ...]:
        """Return the synthetic training route.

        Returns:
            One stable batch key.
        """
        return ("data",)

    def losses(
        self,
        model_apply: object,
        model_state: Any,
        batches: NamedBatches,
    ) -> dict[str, jax.Array]:
        """Return one synthetic loss.

        Args:
            model_apply: Model callable unused by the fixture.
            model_state: Explicit model state unused by the fixture.
            batches: Named batches unused by the fixture.

        Returns:
            One scalar loss mapping.
        """
        del model_apply, model_state, batches
        return {"loss": jax.numpy.asarray(0.0)}


class _CustomPhiModule(PhiModule):
    """Represent an application-defined module selected through Hydra."""


class _AdaptiveBalancer:
    """Record adaptive update-plan construction for assembly tests."""

    def __init__(self) -> None:
        """Initialize an empty build-call record."""
        self.build_call: tuple[object, tuple[str, ...]] | None = None
        self.loss_names = ("loss",)

    def initialize(self) -> jax.Array:
        """Create one scalar synthetic balancer state.

        Returns:
            Unit scalar state.
        """
        return jax.numpy.ones(1)

    def combine(self, losses: dict[str, jax.Array], state: jax.Array) -> jax.Array:
        """Return the fixture's single scalar loss.

        Args:
            losses: Scalar loss mapping.
            state: Unused synthetic balancer state.

        Returns:
            Configured scalar loss.
        """
        del state
        return losses["loss"]

    def diagnostics(self, state: jax.Array) -> dict[str, jax.Array]:
        """Expose no additional diagnostics.

        Args:
            state: Unused synthetic balancer state.

        Returns:
            Empty diagnostic mapping.
        """
        del state
        return {}

    def build_update_plan(
        self,
        module: object,
        batch_keys: tuple[str, ...],
    ) -> BalancerUpdatePlan:
        """Return one fixed-diagnostic update plan.

        Args:
            module: Configured application module.
            batch_keys: Required named training batches.

        Returns:
            Synthetic update plan using two diagnostic rows per batch.
        """
        self.build_call = (module, batch_keys)

        def update(model_state: Any, batches: NamedBatches, balancer_state: Any) -> Any:
            """Preserve synthetic balancer state.

            Args:
                model_state: Explicit model state unused by the fixture.
                batches: Diagnostic batches unused by the fixture.
                balancer_state: Synthetic balancer state.

            Returns:
                Unchanged balancer state.
            """
            del model_state, batches
            return balancer_state

        return BalancerUpdatePlan(
            update,
            every_n_steps=25,
            update_start_step=5,
            batch_sizes=dict.fromkeys(batch_keys, 2),
        )


def _model_apply(model_state: Any, inputs: jax.Array) -> jax.Array:
    """Return inputs unchanged for module-construction tests.

    Args:
        model_state: Explicit model state unused by the fixture.
        inputs: Model inputs.

    Returns:
        Unchanged model inputs.
    """
    del model_state
    return inputs


def _custom_model_factory(
    key: jax.Array,
    precision: str,
    input_mean: np.ndarray,
    input_std: np.ndarray,
) -> InitializedModel:
    """Build a non-NNX model through the generic factory contract.

    Args:
        key: Explicit initialization key.
        precision: Configured model precision name.
        input_mean: Host input-coordinate mean.
        input_std: Host input-coordinate standard deviation.

    Returns:
        Initialized affine test model.
    """
    del key, precision
    mean = jax.numpy.asarray(input_mean)
    std = jax.numpy.asarray(input_std)

    def apply(state: dict[str, jax.Array], inputs: jax.Array) -> jax.Array:
        """Apply the affine test model.

        Args:
            state: Scalar trainable weight mapping.
            inputs: Input-coordinate batch.

        Returns:
            Normalized inputs multiplied by the weight.
        """
        return state["weight"] * (inputs - mean) / std

    return InitializedModel(apply, {"weight": jax.numpy.asarray(2.0)})


def test_hydra_integration_exports_factory_helpers() -> None:
    """Verify optional integration factories remain available without eager model imports."""
    assert callable(instantiate_data_module)
    assert callable(instantiate_model_factory)
    assert callable(instantiate_balancer)


def test_instantiate_model_factory_remains_lazy_until_runtime_values_are_available() -> None:
    """Verify Hydra binds architecture options without initializing model state."""
    config = OmegaConf.create({"_target_": f"{__name__}._custom_model_factory"})

    factory = instantiate_model_factory(config)
    initialized = factory(
        key=jax.random.key(0),
        precision="32-true",
        input_mean=np.asarray([1.0], dtype=np.float32),
        input_std=np.asarray([2.0], dtype=np.float32),
    )

    assert isinstance(initialized, InitializedModel)
    assert initialized.summary is None
    np.testing.assert_allclose(initialized.apply(initialized.state, jax.numpy.asarray([[3.0]])), [[2.0]])


def test_core_assembly_returns_data_independent_adaptive_plan() -> None:
    """Verify core assembly declares source requirements without constructing or sampling a DataModule."""
    train_step = MagicMock()
    trainer = MagicMock(spec=Trainer)
    trainer.compile_train_step.return_value = train_step
    module = MagicMock(spec=BasePhiModule)
    module.loss_names = ("loss",)
    module.batch_keys = ("initial", "pde")
    balancer = _AdaptiveBalancer()
    optimizer = MagicMock()
    plan = build_training_plan(trainer, module, balancer, optimizer)

    assert isinstance(plan, TrainingPlan)
    assert plan.train_step is train_step
    assert plan.batch_keys == ("initial", "pde")
    assert plan.balancer_update is not None
    assert plan.balancer_update.every_n_steps == 25
    assert plan.balancer_update.update_start_step == 5
    assert plan.balancer_update.batch_sizes == {"initial": 2, "pde": 2}
    assert balancer.build_call == (module, ("initial", "pde"))


def test_instantiate_balancer_accepts_a_flat_hydra_target() -> None:
    """Verify balancer settings live directly on the Hydra-instantiable node."""
    config = OmegaConf.create(
        {
            "_target_": "phijax.balancers.GradNormBalancer",
            "update_every_n_steps": 25,
            "update_start_step": 5,
        }
    )

    balancer = instantiate_balancer(config, ("loss",))
    plan = balancer.build_update_plan(MagicMock(spec=BasePhiModule), ("data",))

    assert plan.every_n_steps == 25
    assert plan.update_start_step == 5


def test_instantiate_module_supports_a_custom_hydra_target() -> None:
    """Verify applications can replace the default module without changing entrypoint code."""
    config = OmegaConf.create({"_target_": f"{__name__}._CustomPhiModule"})

    module = instantiate_module(config, _custom_model_factory, _TestObjective(), name="custom")

    assert isinstance(module, BasePhiModule)
    assert isinstance(module, _CustomPhiModule)
    assert module.name == "custom"
    assert module.loss_names == ("loss",)


def test_instantiate_data_module_defers_stage_preparation_to_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify DataModule construction completes its host lifecycle before assembly.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    data_module = MagicMock(spec=PhiDataModule)
    monkeypatch.setattr(factory_module, "instantiate", lambda config: data_module)

    result = instantiate_data_module(OmegaConf.create({"_target_": "unused"}))

    assert result is data_module
    data_module.prepare_stage.assert_not_called()


def test_instantiate_data_module_rejects_an_invalid_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify configured data targets must implement the DataModule contract.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    monkeypatch.setattr(factory_module, "instantiate", lambda config: object())

    with pytest.raises(TypeError, match="PhiDataModule"):
        instantiate_data_module(OmegaConf.create({"_target_": "unused"}))


def test_build_trainer_instantiates_hydra_services() -> None:
    """Verify config-first construction wires callbacks and logger adapters into the trainer."""
    config = OmegaConf.create(
        {
            "trainer": {
                "_target_": "phijax.training.Trainer",
                "max_steps": 3,
                "accelerator": "cpu",
                "devices": 1,
            },
            "callbacks": {
                "early_stopping": {
                    "_target_": "phijax.callbacks.EarlyStopping",
                    "patience": 2,
                },
            },
            "logger": {
                "_target_": "phijax.training.ConsoleLogger",
                "name": "phijax.tests.factory",
            },
        }
    )
    trainer = build_trainer(config)
    assert isinstance(trainer, Trainer)
    assert len(trainer.callbacks) == 3
    assert isinstance(trainer.callbacks[0], EarlyStopping)
    assert type(trainer.callbacks[1]) is TQDMProgressBar
    assert type(trainer.callbacks[2]) is ModelSummary
    assert len(trainer.logger.loggers) == 1
    assert isinstance(trainer.logger.loggers[0], ConsoleLogger)


def test_instantiate_trainer_accepts_preconstructed_loggers() -> None:
    """Verify configured loggers are injected while the Trainer is constructed."""
    config = OmegaConf.create(
        {
            "_target_": "phijax.training.Trainer",
            "max_steps": 1,
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
        }
    )
    loggers = instantiate_loggers(
        OmegaConf.create(
            {
                "console": {
                    "_target_": "phijax.training.ConsoleLogger",
                    "name": "phijax.tests.factory",
                }
            }
        )
    )

    trainer = instantiate_trainer(config, logger=loggers)

    assert trainer.logger is loggers
    assert not hasattr(trainer, "set_logger")


@pytest.mark.parametrize("enabled", [True, False])
def test_instantiate_callbacks_rejects_enabled_option(enabled: bool) -> None:
    """Verify callback presence, rather than an `enabled` field, controls instantiation.

    Args:
        enabled: Legacy callback flag value under test.
    """
    config = OmegaConf.create(
        {
            "early_stopping": {
                "_target_": "phijax.callbacks.EarlyStopping",
                "enabled": enabled,
                "patience": 2,
            }
        }
    )

    with pytest.raises(ValueError, match="removed `enabled` option"):
        instantiate_callbacks(config)


def test_instantiate_callbacks_rejects_null_entries() -> None:
    """Verify disabled callbacks must be removed from the composed config."""
    config = OmegaConf.create({"early_stopping": None})

    with pytest.raises(ValueError, match="omit the entry"):
        instantiate_callbacks(config)


def test_instantiate_callbacks_requires_target() -> None:
    """Verify every configured callback names a Hydra target."""
    config = OmegaConf.create({"early_stopping": {"patience": 2}})

    with pytest.raises(ValueError, match="must define `_target_`"):
        instantiate_callbacks(config)


@pytest.mark.parametrize("enabled", [True, False])
def test_instantiate_loggers_rejects_enabled_option(enabled: bool) -> None:
    """Verify logger presence, rather than an `enabled` field, controls instantiation.

    Args:
        enabled: Legacy logger flag value under test.
    """
    config = OmegaConf.create(
        {
            "console": {
                "_target_": "phijax.training.ConsoleLogger",
                "enabled": enabled,
            }
        }
    )
    with pytest.raises(ValueError, match="removed `enabled` option"):
        instantiate_loggers(config)

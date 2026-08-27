from typing import Any
from unittest.mock import MagicMock

import jax
import numpy as np
import pytest
from omegaconf import OmegaConf

from phijax.balancers import BalancerUpdatePlan
from phijax.callbacks import EarlyStopping
from phijax.data import PhiDataModule
from phijax.integrations.hydra import (
    build_trainer,
    configure_training,
    instantiate_data_module,
    instantiate_enabled,
    instantiate_model,
    instantiate_module,
)
from phijax.integrations.hydra import factory as factory_module
from phijax.models import InitializedModel
from phijax.module import BasePhiModule, PhiModule
from phijax.training import ConsoleLogger, Trainer, TrainingPlan
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
        self.build_call: tuple[object, tuple[str, ...], dict[str, Any]] | None = None

    def build_update_plan(
        self,
        module: object,
        batch_keys: tuple[str, ...],
        options: dict[str, Any],
    ) -> BalancerUpdatePlan:
        """Return one fixed-diagnostic update plan.

        Args:
            module: Configured application module.
            batch_keys: Required named training batches.
            options: Resolved balancer-specific update options.

        Returns:
            Synthetic update plan using two diagnostic rows per batch.
        """
        self.build_call = (module, batch_keys, options)

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

        return BalancerUpdatePlan(update, dict.fromkeys(batch_keys, 2))


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


def test_hydra_integration_exports_factory_and_assembly_helpers() -> None:
    """Verify optional integration helpers remain available without eager model imports."""
    assert callable(instantiate_data_module)
    assert callable(instantiate_model)
    assert callable(configure_training)


def test_instantiate_model_accepts_a_generic_initialized_model() -> None:
    """Verify model assembly is independent of MLP and NNX graph details."""
    data_module = MagicMock(spec=PhiDataModule)
    data_module.input_statistics.return_value = (
        np.asarray([1.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
    )
    config = OmegaConf.create({"_target_": f"{__name__}._custom_model_factory"})

    initialized = instantiate_model(config, MagicMock(mode="32-true"), data_module, jax.random.key(0))

    assert isinstance(initialized, InitializedModel)
    assert initialized.summary is None
    np.testing.assert_allclose(initialized.apply(initialized.state, jax.numpy.asarray([[3.0]])), [[2.0]])


def test_configure_training_returns_data_independent_adaptive_plan() -> None:
    """Verify assembly declares source requirements without constructing or sampling a DataModule."""
    train_step = MagicMock()
    trainer = MagicMock(spec=Trainer)
    trainer.compile_train_step.return_value = train_step
    module = MagicMock(spec=BasePhiModule)
    balancer = _AdaptiveBalancer()
    optimizer = MagicMock()
    config = OmegaConf.create(
        {
            "objective": {
                "terms": {
                    "first": {"batch_key": "initial"},
                    "second": {"batch_key": "pde"},
                    "third": {"batch_key": "pde"},
                }
            },
            "balancer": {
                "update": {
                    "every_n_steps": 25,
                    "skip_first_step": False,
                    "kernel_chunk_size": 8,
                }
            },
        }
    )

    plan = configure_training(config, trainer, module, balancer, optimizer)

    assert isinstance(plan, TrainingPlan)
    assert plan.train_step is train_step
    assert plan.batch_keys == ("initial", "pde")
    assert plan.balancer_update is not None
    assert plan.balancer_update.every_n_steps == 25
    assert plan.balancer_update.skip_first_step is False
    assert plan.balancer_update.plan.batch_sizes == {"initial": 2, "pde": 2}
    assert balancer.build_call == (module, ("initial", "pde"), {"kernel_chunk_size": 8})


def test_instantiate_module_supports_a_custom_hydra_target() -> None:
    """Verify applications can replace the default module without changing entrypoint code."""
    config = OmegaConf.create({"_target_": f"{__name__}._CustomPhiModule"})

    module = instantiate_module(config, _model_apply, _TestObjective(), name="custom")

    assert isinstance(module, BasePhiModule)
    assert isinstance(module, _CustomPhiModule)
    assert module.name == "custom"
    assert module.loss_names == ("loss",)


def test_instantiate_data_module_prepares_the_requested_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify DataModule construction completes its host lifecycle before assembly.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    data_module = MagicMock(spec=PhiDataModule)
    monkeypatch.setattr(factory_module, "instantiate", lambda config: data_module)

    result = instantiate_data_module(OmegaConf.create({"_target_": "unused"}), "fit")

    assert result is data_module
    data_module.prepare_stage.assert_called_once_with("fit")


def test_instantiate_data_module_propagates_stage_preparation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DataModule stage-preparation failures propagate from the lifecycle owner.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    data_module = MagicMock(spec=PhiDataModule)
    data_module.prepare_stage.side_effect = RuntimeError("synthetic setup failure")
    monkeypatch.setattr(factory_module, "instantiate", lambda config: data_module)

    with pytest.raises(RuntimeError, match="synthetic setup failure"):
        instantiate_data_module(OmegaConf.create({"_target_": "unused"}), "predict")

    data_module.prepare_stage.assert_called_once_with("predict")


def test_build_trainer_instantiates_enabled_hydra_services() -> None:
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
                    "enabled": True,
                    "patience": 2,
                },
                "disabled": {
                    "_target_": "phijax.callbacks.EarlyStopping",
                    "enabled": False,
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
    assert len(trainer.callbacks) == 1
    assert isinstance(trainer.callbacks[0], EarlyStopping)
    assert len(trainer.logger.loggers) == 1
    assert isinstance(trainer.logger.loggers[0], ConsoleLogger)


def test_instantiate_enabled_ignores_placeholders_without_targets() -> None:
    """Verify non-service callback scheduling placeholders are not passed to Hydra."""
    config = OmegaConf.create({"prediction": {"enabled": True, "every_n_steps": 5}})
    assert instantiate_enabled(config) == ()

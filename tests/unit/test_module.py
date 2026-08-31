from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from phijax import PhiModule, PhiModuleContext
from phijax.balancers import ExactNTKBalancer
from phijax.callbacks import PredictionContext
from phijax.metrics import _collect_module_metrics
from phijax.models import InitializedModel
from phijax.training import PrecisionPolicy
from phijax.types import ModelApply, NamedBatches


class _ConfiguredObjective:
    """Provide configurable loss names and an optional residual implementation."""

    def __init__(self, loss_names: Sequence[str] = ("data",)) -> None:
        """Initialize the objective names.

        Args:
            loss_names: Names exposed to the module.
        """
        self._loss_names = tuple(loss_names)

    @property
    def loss_names(self) -> tuple[str, ...]:
        """Return configured loss names.

        Returns:
            Stable objective loss names.
        """
        return self._loss_names

    @property
    def batch_keys(self) -> tuple[str, ...]:
        """Return the configured data route.

        Returns:
            One stable batch key.
        """
        return ("data",)

    def losses(
        self,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> dict[str, jax.Array]:
        """Evaluate a mean-squared model output loss.

        Args:
            model_apply: Explicit-state model application callable.
            model_state: Scalar model state.
            batches: Nested data batch.

        Returns:
            One scalar data loss.
        """
        predictions = jax.vmap(model_apply, in_axes=(None, 0))(model_state, batches["data"]["inputs"])
        return {"data": jnp.mean(predictions**2)}


class _ResidualObjective(_ConfiguredObjective):
    """Extend the configured objective with a raw residual stream."""

    def residual_stream(
        self,
        name: str,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> jax.Array:
        """Evaluate model outputs as a synthetic residual stream.

        Args:
            name: Requested loss name.
            model_apply: Explicit-state model application callable.
            model_state: Scalar model state.
            batches: Nested data batch.

        Returns:
            Batched synthetic residuals.

        Raises:
            KeyError: If `name` is not the configured data loss.
        """
        if name != "data":
            raise KeyError(name)
        return jax.vmap(model_apply, in_axes=(None, 0))(model_state, batches["data"]["inputs"])


class _OffsetPhiModule(PhiModule):
    """Demonstrate application-specific forward customization through inheritance."""

    def forward(self, model_state: Any, inputs: jax.Array) -> jax.Array:
        """Add an offset to the configured model output.

        Args:
            model_state: Explicit scalar model state.
            inputs: One input point.

        Returns:
            Configured model output shifted by one.
        """
        return super().forward(model_state, inputs) + 1.0


def _model_apply(model_state: dict[str, jax.Array], inputs: jax.Array) -> jax.Array:
    """Apply a scalar weight to one input.

    Args:
        model_state: Mapping containing scalar `weight`.
        inputs: One input point.

    Returns:
        Weighted input point.
    """
    return model_state["weight"] * inputs


def _bind(module: PhiModule) -> PhiModule:
    """Bind one test blueprint to its configured initialized model.

    Args:
        module: Uninitialized module blueprint.

    Returns:
        Bound shallow copy ready for numerical evaluation.
    """
    bound, _ = module.prepare_model(
        key=jax.random.key(0),
        input_mean=jnp.zeros(1),
        input_std=jnp.ones(1),
        precision=PrecisionPolicy.from_name("32-true"),
    )
    return bound


def test_phi_module_delegates_forward_losses_prediction_and_residuals() -> None:
    """Verify the concrete module adapts a configured model and objective to every computation API."""
    blueprint = PhiModule(InitializedModel(_model_apply, {}), _ResidualObjective(), name="scalar PINN")
    module = _bind(blueprint)
    model_state = {"weight": jnp.asarray(2.0)}
    batches = {"data": {"inputs": jnp.asarray([[1.0], [3.0]])}}

    assert module.name == "scalar PINN"
    assert module.loss_names == ("data",)
    assert module.batch_keys == ("data",)
    with pytest.raises(RuntimeError, match="uninitialized"):
        blueprint(model_state, jnp.asarray([2.0]))
    assert jnp.array_equal(module(model_state, jnp.asarray([2.0])), jnp.asarray([4.0]))
    assert float(module.training_step(model_state, batches)["data"]) == 20.0
    metrics = module.format_training_metrics(
        jnp.asarray(4.0),
        {"data": jnp.asarray(3.0)},
        {"weight/data": jnp.asarray(2.0), "precision/loss_scale": jnp.asarray(1.0)},
    )
    assert tuple(metrics) == (
        "train/loss",
        "train/loss/data",
        "train/weight/data",
        "train/precision/loss_scale",
    )
    assert jnp.array_equal(module.predict_step(model_state, batches["data"]), jnp.asarray([[2.0], [6.0]]))
    assert jnp.array_equal(module.residual_stream("data", model_state, batches), jnp.asarray([[2.0], [6.0]]))


def test_phi_module_default_lifecycle_hooks_preserve_values_and_declare_metrics() -> None:
    """Verify default host hooks preserve values and explicitly declare scalar metric destinations."""
    module = _bind(PhiModule(InitializedModel(_model_apply, {}), _ConfiguredObjective()))
    model_state = {"weight": jnp.asarray(2.0)}
    batch = {"data": {"inputs": jnp.ones((1, 1))}}
    context = PhiModuleContext(
        step=3,
        metrics={
            "train/loss": jnp.asarray(1.0),
            "train/loss/data": jnp.asarray(0.5),
            "train/weight/data": jnp.asarray(1.0),
            "train/residual/by_region": jnp.ones(2),
        },
    )
    prediction_context = PredictionContext(outputs=None, batch_index=None, metadata={})

    assert module.setup() is None
    assert module.on_fit_start(model_state, context) is model_state
    returned_state, returned_batch = module.on_train_batch_start(model_state, batch, context)
    assert returned_state is model_state
    assert returned_batch is batch
    with _collect_module_metrics() as collector:
        returned_state, returned_metrics = module.on_train_batch_end(model_state, context)
    assert returned_state is model_state
    assert returned_metrics is context.metrics
    assert set(collector.records) == {"train/loss", "train/loss/data", "train/weight/data"}
    assert collector.records["train/loss"].logger is True
    assert collector.records["train/loss"].prog_bar is True
    assert collector.records["train/loss/data"].logger is True
    assert collector.records["train/loss/data"].prog_bar is True
    assert collector.records["train/weight/data"].logger is True
    assert collector.records["train/weight/data"].prog_bar is True
    assert module.on_fit_end(model_state, context) is model_state
    assert module.on_predict_start(model_state, prediction_context) is None
    assert module.on_predict_epoch_start(model_state, prediction_context) is None
    assert module.on_predict_batch_start(model_state, prediction_context) is None
    assert module.on_predict_batch_end(model_state, prediction_context) is None
    assert module.on_predict_epoch_end(model_state, prediction_context) is None
    assert module.on_predict_end(model_state, prediction_context) is None
    assert module.on_exception(RuntimeError("test"), context) is None
    assert module.teardown() is None
    assert module.summarize_model(model_state) is None


def test_phi_module_log_requires_the_host_batch_end_hook() -> None:
    """Verify module logging cannot introduce Python effects into compiled training code."""
    module = _bind(PhiModule(InitializedModel(_model_apply, {}), _ConfiguredObjective()))

    with pytest.raises(RuntimeError, match="only during `on_train_batch_end`"):
        module.log("train/loss", jnp.asarray(1.0), prog_bar=True)


def test_phi_module_delegates_optional_model_summary() -> None:
    """Verify the generic module forwards summary policy without depending on a model implementation."""
    calls: list[tuple[Any, int, int, bool, bool]] = []

    def summarize(
        model_state: Any,
        *,
        max_depth: int = -1,
        console_width: int = 120,
        compute_flops: bool = False,
        compute_vjp_flops: bool = False,
    ) -> str:
        """Record summary arguments and return a synthetic table.

        Args:
            model_state: Explicit model state.
            max_depth: Maximum displayed module depth.
            console_width: Rich console width.
            compute_flops: Whether forward FLOPs were requested.
            compute_vjp_flops: Whether reverse FLOPs were requested.

        Returns:
            Synthetic summary text.
        """
        calls.append((model_state, max_depth, console_width, compute_flops, compute_vjp_flops))
        return "summary"

    model_state = {"weight": jnp.asarray(2.0)}
    module = _bind(PhiModule(InitializedModel(_model_apply, {}, summarize), _ConfiguredObjective()))

    summary = module.summarize_model(model_state, max_depth=2, console_width=80, compute_flops=True)

    assert summary == "summary"
    assert calls == [(model_state, 2, 80, True, False)]


def test_phi_module_subclass_forward_is_used_by_objective_and_external_balancer() -> None:
    """Verify subclass computation feeds losses and the separately owned pointwise-mean NTK balancer."""
    module = _bind(_OffsetPhiModule(InitializedModel(_model_apply, {}), _ResidualObjective()))
    model_state = {"weight": jnp.asarray(2.0)}
    batches = {"data": {"inputs": jnp.asarray([[1.0], [3.0]])}}
    balancer = ExactNTKBalancer(
        module.loss_names, update_every_n_steps=1, kernel_size=1, moving_average_coefficient=0.0
    )

    losses = module.training_step(model_state, batches)
    balancer_state = balancer.make_update(module)(model_state, batches, balancer.initialize())

    assert float(losses["data"]) == 29.0
    assert float(balancer_state.traces[0]) == 5.0
    assert float(balancer_state.weights[0]) == 1.0


def test_phi_module_rejects_unsupported_residuals_and_invalid_configuration() -> None:
    """Verify concrete modules fail early for invalid identity, callable, names, and residual contracts."""
    module = _bind(PhiModule(InitializedModel(_model_apply, {}), _ConfiguredObjective()))
    with pytest.raises(TypeError, match="ResidualObjective"):
        module.residual_stream("data", {"weight": jnp.asarray(1.0)}, {"data": {"inputs": jnp.ones((1, 1))}})
    with pytest.raises(TypeError, match="model"):
        PhiModule(None, _ConfiguredObjective())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name"):
        PhiModule(InitializedModel(_model_apply, {}), _ConfiguredObjective(), name=" ")
    for names in ((), ("data", "data"), ("",)):
        with pytest.raises(ValueError, match="loss names"):
            PhiModule(InitializedModel(_model_apply, {}), _ConfiguredObjective(names))

    def invalid_factory(**kwargs: Any) -> object:
        """Return an invalid model value for contract validation.

        Args:
            **kwargs: Model-preparation values ignored by this invalid factory.

        Returns:
            Plain object outside the initialized-model contract.
        """
        del kwargs
        return object()

    invalid_module = PhiModule(invalid_factory, _ConfiguredObjective())
    with pytest.raises(TypeError, match="InitializedModel"):
        _bind(invalid_module)

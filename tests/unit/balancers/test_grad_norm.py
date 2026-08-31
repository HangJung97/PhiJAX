import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.balancers import GradNormBalancer


class _QuadraticLossModule:
    """Expose two scalar losses with analytically known gradient norms."""

    @property
    def loss_names(self) -> tuple[str, str]:
        """Return the stable synthetic loss names.

        Returns:
            Names of the unit-scale and double-scale quadratic losses.
        """
        return ("a", "b")

    def training_step(
        self,
        model_state: dict[str, jax.Array],
        batches: dict[str, dict[str, jax.Array]],
    ) -> dict[str, jax.Array]:
        """Evaluate two quadratic losses scaled by the current batch.

        Args:
            model_state: Mapping containing one scalar parameter under `weight`.
            batches: Mapping containing one scalar multiplier under `data.scale`.

        Returns:
            Quadratic losses whose gradient norms have ratio two.
        """
        scaled_weight = batches["data"]["scale"] * model_state["weight"]
        base_loss = jnp.sum(scaled_weight**2)
        return {"a": base_loss, "b": 2.0 * base_loss}


def test_grad_norm_update_matches_named_loss_gradient_norms_and_smoothing() -> None:
    """Verify sequential loss gradients, mean normalization, and moving-average smoothing."""
    module = _QuadraticLossModule()
    balancer = GradNormBalancer(module.loss_names, update_every_n_steps=1, eps=0.0, moving_average_coefficient=0.5)
    update = balancer.make_update(module)
    model_state = {"weight": jnp.asarray(2.0, dtype=jnp.float32)}
    batches = {"data": {"scale": jnp.asarray(1.0, dtype=jnp.float32)}}

    state = update(model_state, batches, balancer.initialize())

    np.testing.assert_allclose(state.traces, [4.0, 8.0])
    np.testing.assert_allclose(state.weights, [1.25, 0.875])


def test_grad_norm_default_regularizer_matches_relative_formula() -> None:
    """Verify the default relative denominator regularizer follows the documented formula."""
    balancer = GradNormBalancer(("a", "b"), update_every_n_steps=1, moving_average_coefficient=0.0)
    state = balancer.update_from_grad_norms(jnp.asarray([2.0, 6.0]), balancer.initialize())
    expected = 4.0 / (np.asarray([2.0, 6.0]) + 1.0e-5 * 4.0)
    np.testing.assert_allclose(state.weights, expected, rtol=1.0e-6)
    np.testing.assert_allclose(state.traces, [2.0, 6.0])


def test_grad_norm_all_zero_diagnostics_preserve_previous_weights() -> None:
    """Verify an all-zero gradient signal cannot replace finite configured weights."""
    balancer = GradNormBalancer(("a", "b"), update_every_n_steps=1, initial_weights={"a": 2.0, "b": 3.0})
    state = balancer.update_from_grad_norms(jnp.zeros(2), balancer.initialize())
    np.testing.assert_allclose(state.weights, [2.0, 3.0])
    np.testing.assert_allclose(state.traces, [0.0, 0.0])


def test_grad_norm_update_plan_reuses_current_training_batches() -> None:
    """Verify generic adaptive assembly receives a current-batch gradient-norm plan."""
    balancer = GradNormBalancer(("a", "b"), update_every_n_steps=10, update_start_step=5)

    plan = balancer.build_update_plan(_QuadraticLossModule(), ("data", "pde"))

    assert plan.batch_sizes is None
    assert plan.every_n_steps == 10
    assert plan.update_start_step == 5
    assert callable(plan.update)


def test_grad_norm_defaults_update_start_to_one_interval() -> None:
    """Verify an omitted start step preserves the previous delayed-first-update behavior."""
    balancer = GradNormBalancer(("a", "b"), update_every_n_steps=10)

    plan = balancer.build_update_plan(_QuadraticLossModule(), ("data", "pde"))

    assert balancer.update_start_step == 10
    assert plan.update_start_step == 10


@pytest.mark.parametrize("interval", [True, 0, -1, 1.5])
def test_grad_norm_rejects_invalid_update_intervals(interval: object) -> None:
    """Verify adaptive update cadence fails during balancer construction.

    Args:
        interval: Invalid update interval.
    """
    error = TypeError if isinstance(interval, bool | float) else ValueError
    with pytest.raises(error, match="update_every_n_steps"):
        GradNormBalancer(("a", "b"), update_every_n_steps=interval)  # type: ignore[arg-type]


@pytest.mark.parametrize("start_step", [True, 1.5, -1])
def test_grad_norm_rejects_invalid_update_start_steps(start_step: object) -> None:
    """Verify the first adaptive update uses a nonnegative integer step.

    Args:
        start_step: Invalid absolute update start step.
    """
    error = TypeError if isinstance(start_step, bool | float) else ValueError
    with pytest.raises(error, match="update_start_step"):
        GradNormBalancer(("a", "b"), update_every_n_steps=1, update_start_step=start_step)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"loss_names": ()}, "loss_names"),
        ({"loss_names": ("a", "a")}, "loss_names"),
        ({"loss_names": ("a",), "eps": -1.0}, "eps"),
        ({"loss_names": ("a",), "moving_average_coefficient": 1.0}, "moving_average_coefficient"),
    ],
)
def test_grad_norm_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    """Verify invalid names and numeric policies fail before compilation.

    Args:
        kwargs: Invalid constructor arguments supplied by the parameterized test case.
        message: Expected configuration field in the exception message.
    """
    with pytest.raises(ValueError, match=message):
        GradNormBalancer(update_every_n_steps=1, **kwargs)  # type: ignore[arg-type]

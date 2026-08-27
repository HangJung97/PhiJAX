import jax
import jax.numpy as jnp

from phijax.balancers import BalancerState
from phijax.training import TrainState, with_balancer_updates


def _state() -> TrainState:
    """Build a minimal state for scheduled balancer tests.

    Returns:
        Functional state with one scalar balancer weight.
    """
    return TrainState(
        model_state={"weight": jnp.asarray(0.0)},
        optimizer_state=(),
        balancer_state=BalancerState(weights=jnp.ones(1), traces=jnp.zeros(1)),
        rng_key=jax.random.key(0),
        step=jnp.asarray(0, jnp.int32),
        loss_scale=jnp.asarray(1.0),
        finite_steps=jnp.asarray(0, jnp.int32),
    )


def _train_step(state: TrainState, batches: dict[str, dict[str, jax.Array]]) -> tuple[TrainState, dict[str, jax.Array]]:
    """Increment the state step and expose the active balancer weight.

    Args:
        state: Current functional state.
        batches: Unused synthetic batches.

    Returns:
        Incremented state and current weight metric.
    """
    del batches
    return state.replace(step=state.step + 1), {"weight": state.balancer_state.weights[0]}


def _update_balancer(
    model_state: dict[str, jax.Array],
    batches: dict[str, dict[str, jax.Array]],
    state: BalancerState,
) -> BalancerState:
    """Increment a synthetic balancer weight.

    Args:
        model_state: Unused explicit model state.
        batches: Unused fixed diagnostic batches.
        state: Current balancer state.

    Returns:
        Balancer state with weight and trace incremented by one.
    """
    del model_state, batches
    return BalancerState(weights=state.weights + 1.0, traces=state.traces + 1.0)


def test_with_balancer_updates_refreshes_on_configured_steps() -> None:
    """Verify scheduled balancer state is refreshed before due optimizer updates."""
    batches = {"data": {"inputs": jnp.ones((1, 1))}}
    step = with_balancer_updates(_train_step, _update_balancer, batches, every_n_steps=2, skip_first_step=False)

    state, metrics = step(_state(), batches)
    assert float(metrics["weight"]) == 2.0
    state, metrics = step(state, batches)
    assert float(metrics["weight"]) == 2.0
    state, metrics = step(state, batches)
    assert float(metrics["weight"]) == 3.0


def test_with_balancer_updates_does_not_trace_an_update_before_it_is_due() -> None:
    """Verify the host scheduler keeps the diagnostic executable out of ordinary steps."""
    batches = {"data": {"inputs": jnp.ones((1, 1))}}
    update_steps: list[int] = []

    def update(
        model_state: dict[str, jax.Array],
        diagnostic_batches: dict[str, dict[str, jax.Array]],
        state: BalancerState,
    ) -> BalancerState:
        """Record an actually scheduled diagnostic update.

        Args:
            model_state: Unused explicit model state.
            diagnostic_batches: Unused fixed diagnostic batches.
            state: Current balancer state.

        Returns:
            Balancer state with its weight incremented.
        """
        del model_state, diagnostic_batches
        update_steps.append(len(update_steps))
        return BalancerState(weights=state.weights + 1.0, traces=state.traces)

    step = with_balancer_updates(_train_step, update, batches, every_n_steps=2, skip_first_step=True)
    state, _ = step(_state(), batches)
    state, _ = step(state, batches)
    assert update_steps == []
    step(state, batches)
    assert update_steps == [0]


def test_with_balancer_updates_can_refresh_from_current_training_batches() -> None:
    """Verify adaptive balancers can use the current batch instead of a fixed diagnostic sample."""
    initial_batches = {"data": {"inputs": jnp.asarray([[2.0]])}}
    later_batches = {"data": {"inputs": jnp.asarray([[5.0]])}}

    def update(
        model_state: dict[str, jax.Array],
        batches: dict[str, dict[str, jax.Array]],
        state: BalancerState,
    ) -> BalancerState:
        """Use the current batch value as the synthetic adaptive weight.

        Args:
            model_state: Unused explicit model state.
            batches: Current training batches selected by the scheduler.
            state: Current balancer state.

        Returns:
            State whose weight records the current batch value.
        """
        del model_state
        return BalancerState(weights=batches["data"]["inputs"].reshape(-1), traces=state.traces)

    step = with_balancer_updates(_train_step, update, None, every_n_steps=1, skip_first_step=False)
    state, metrics = step(_state(), initial_batches)
    assert float(metrics["weight"]) == 2.0
    _, metrics = step(state, later_batches)
    assert float(metrics["weight"]) == 5.0

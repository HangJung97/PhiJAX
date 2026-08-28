from typing import Any

import pytest

from phijax.training import BalancerUpdateSchedule, TrainingPlan


def _update_plan(state: Any, batch: Any) -> tuple[Any, dict[str, float]]:
    """Return unchanged state and empty diagnostics for schedule validation tests."""
    return state, {}


def _train_step(state: Any, batch: Any) -> tuple[Any, dict[str, float]]:
    """Return unchanged state and empty metrics for plan validation tests."""
    return state, {}


@pytest.mark.parametrize("interval", [True, 1.5, "1"])
def test_balancer_update_schedule_requires_an_integer_interval(interval: object) -> None:
    """Verify Boolean and non-integer adaptive update intervals are rejected."""
    with pytest.raises(TypeError, match="integer"):
        BalancerUpdateSchedule(_update_plan, interval)  # type: ignore[arg-type]


def test_balancer_update_schedule_validates_positive_interval_and_boolean_skip() -> None:
    """Verify schedule values fail early when their host-side semantics are invalid."""
    with pytest.raises(ValueError, match="positive"):
        BalancerUpdateSchedule(_update_plan, 0)
    with pytest.raises(TypeError, match="Boolean"):
        BalancerUpdateSchedule(_update_plan, 1, skip_first_step=1)  # type: ignore[arg-type]
    schedule = BalancerUpdateSchedule(_update_plan, 3, skip_first_step=False)
    assert schedule.every_n_steps == 3


def test_training_plan_validates_step_and_batch_keys() -> None:
    """Verify training plans require callable steps and unique non-empty batch names."""
    with pytest.raises(TypeError, match="callable"):
        TrainingPlan(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        TrainingPlan(_train_step, ("pde", " "))
    with pytest.raises(ValueError, match="unique"):
        TrainingPlan(_train_step, ("pde", "pde"))
    plan = TrainingPlan(_train_step, ("initial", "pde"))
    assert plan.batch_keys == ("initial", "pde")

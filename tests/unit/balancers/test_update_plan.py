from typing import Any

import pytest

from phijax.balancers import BalancerUpdatePlan


def _balancer_update(model_state: Any, batches: Any, state: Any) -> Any:
    """Return unchanged balancer state for update-plan validation tests."""
    del model_state, batches
    return state


@pytest.mark.parametrize("interval", [True, 1.5, "1"])
def test_balancer_update_plan_requires_an_integer_interval(interval: object) -> None:
    """Verify Boolean and non-integer adaptive update intervals are rejected."""
    with pytest.raises(TypeError, match="integer"):
        BalancerUpdatePlan(_balancer_update, interval, 0)  # type: ignore[arg-type]


def test_balancer_update_plan_validates_and_freezes_batch_sizes() -> None:
    """Verify one update plan owns an immutable diagnostic sampling policy."""
    batch_sizes = {"pde": 8}
    plan = BalancerUpdatePlan(_balancer_update, 10, update_start_step=5, batch_sizes=batch_sizes)
    batch_sizes["pde"] = 16

    assert plan.every_n_steps == 10
    assert plan.update_start_step == 5
    assert plan.batch_sizes == {"pde": 8}
    with pytest.raises(TypeError):
        plan.batch_sizes["pde"] = 4  # type: ignore[index,union-attr]
    with pytest.raises(ValueError, match="positive"):
        BalancerUpdatePlan(_balancer_update, 0, 0)
    with pytest.raises(TypeError, match="update_start_step"):
        BalancerUpdatePlan(_balancer_update, 1, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="update_start_step"):
        BalancerUpdatePlan(_balancer_update, 1, -1)
    with pytest.raises(TypeError, match="batch names"):
        BalancerUpdatePlan(_balancer_update, 1, 0, batch_sizes={"": 1})
    with pytest.raises(ValueError, match="batch sizes"):
        BalancerUpdatePlan(_balancer_update, 1, 0, batch_sizes={"pde": 0})


def test_balancer_update_plan_accepts_all_rows_and_validates_resample() -> None:
    """Verify `"all"` diagnostic sizes are accepted and `resample` requires sampled diagnostic batches."""
    plan = BalancerUpdatePlan(_balancer_update, 1, 0, batch_sizes={"pde": 8, "boundary": "all"}, resample=True)

    assert plan.batch_sizes == {"pde": 8, "boundary": "all"}
    assert plan.resample is True
    assert BalancerUpdatePlan(_balancer_update, 1, 0, batch_sizes={"pde": 8}).resample is False
    with pytest.raises(ValueError, match="batch sizes"):
        BalancerUpdatePlan(_balancer_update, 1, 0, batch_sizes={"pde": "half"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="`resample` requires `batch_sizes`"):
        BalancerUpdatePlan(_balancer_update, 1, 0, resample=True)
    with pytest.raises(TypeError, match="resample"):
        BalancerUpdatePlan(_balancer_update, 1, 0, batch_sizes={"pde": 8}, resample=1)  # type: ignore[arg-type]

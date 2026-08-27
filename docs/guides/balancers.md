# Creating a custom loss balancer

Loss balancers convert the named scalar losses produced by `PhiModule` into the single scalar differentiated by the
optimizer. They are numerical training components rather than callbacks: their weights participate in the compiled
loss, their state is checkpointed with `TrainState`, and callbacks only observe the resulting diagnostics.

This guide implements a fixed balancer that normalizes configured nonnegative weights to have mean one. The same
contract can express other fixed weighting rules without changing the trainer or module.

## Balancer contract

The training factory injects `module.loss_names` into the configured balancer constructor. A balancer must expose:

| Member                                  | Responsibility                                                           |
| --------------------------------------- | ------------------------------------------------------------------------ |
| `loss_names`                            | Immutable names defining component and weight-vector order               |
| `initialize() -> BalancerState`         | Create JAX-compatible weights and diagnostic arrays                      |
| `combine(losses, state) -> (total, xs)` | Return the weighted scalar total and ordered unweighted component vector |

Do not configure `loss_names` in YAML. They come from the objective after equation-local names and any explicit term
aliases have been resolved. This prevents a balancer from silently assigning a weight to the wrong residual.

`combine` runs inside the JIT-compiled training step and must therefore be pure. It must not mutate Python state,
convert traced arrays to NumPy, perform file IO, log metrics, or change its output PyTree structure between calls.

## Implement a fixed normalized balancer

Create `src/my_project/balancers/normalized_static.py`:

```python
import math
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from phijax.balancers.base import BalancerState


class NormalizedStaticBalancer:
    """Apply fixed nonnegative weights normalized to have mean one.

    Attributes:
        loss_names: Stable names defining component and weight ordering.
        initial_weights: Normalized `float32` weight vector.
    """

    def __init__(
        self,
        loss_names: Sequence[str],
        *,
        weights: Mapping[str, float] | None = None,
        default_weight: float = 1.0,
    ) -> None:
        """Initialize the configured loss weights.

        Args:
            loss_names: Unique non-empty loss names injected by training assembly.
            weights: Optional weights keyed by exact objective loss name.
            default_weight: Weight used for names absent from `weights`.

        Raises:
            ValueError: If names are empty or duplicated, a weight is non-finite or negative, or every weight is zero.
        """
        if not loss_names or len(set(loss_names)) != len(loss_names):
            raise ValueError("`loss_names` must be non-empty and unique.")
        configured = weights or {}
        values = tuple(float(configured.get(name, default_weight)) for name in loss_names)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Loss weights must be finite and nonnegative.")
        weight_sum = sum(values)
        if weight_sum == 0.0:
            raise ValueError("At least one loss weight must be positive.")

        self.loss_names = tuple(loss_names)
        normalization = len(values) / weight_sum
        self.initial_weights = jnp.asarray(values, dtype=jnp.float32) * normalization

    def initialize(self) -> BalancerState:
        """Create fixed weights and zero diagnostic placeholders.

        Returns:
            JIT-compatible balancer state aligned with :attr:`loss_names`.
        """
        return BalancerState(
            weights=self.initial_weights,
            traces=jnp.zeros_like(self.initial_weights),
        )

    def combine(
        self,
        losses: Mapping[str, jax.Array],
        state: BalancerState,
    ) -> jax.Array:
        """Combine named scalar losses in stable order.

        Args:
            losses: Scalar arrays keyed by every entry in :attr:`loss_names`.
            state: Current fixed balancer state.

        Returns:
            Weighted scalar total.

        Raises:
            KeyError: If a configured loss is absent from `losses`.
        """
        components = jnp.stack([losses[name] for name in self.loss_names]).astype(jnp.float32)
        return jnp.vdot(state.weights, components)

    def diagnostics(self, state: BalancerState) -> Mapping[str, jax.Array]:
        """Expose scalar weights for logging.

        Args:
            state: Current fixed balancer state.

        Returns:
            Weights keyed by their associated loss name.
        """
        return {f"weight/{name}": state.weights[index] for index, name in enumerate(self.loss_names)}
```

Using `BalancerState` keeps the custom balancer compatible with progress metrics, precision handling, Orbax
checkpointing, and weight restoration. The `traces` field may remain zero for a fixed policy; adaptive built-ins use it
for their most recent gradient-norm or NTK diagnostic.

Re-export the class from `phijax.balancers` when it is part of PhiJAX itself. An external application package can
instead use its complete import target directly in Hydra.

## Add a Hydra config

Create `src/my_project/configs/model/balancer/normalized_static.yaml`:

```yaml
# Fixed normalized loss weights with no periodic diagnostic update.
name: normalized_static
factory:
  _target_: my_project.balancers.NormalizedStaticBalancer
  # Keys must match the final names exposed by the composed objective.
  weights:
    initial/u: 1.0
    boundary/u: 2.0
    pde/heat: 4.0
  # Names omitted above receive this value before all weights are normalized.
  default_weight: 1.0
# Fixed policies do not compile or schedule a separate balancer refresh.
update: null
```

Select it from an experiment defaults list:

```yaml
defaults:
  - override /model/balancer: normalized_static
  - _self_
```

For a one-off run, use a Hydra override:

```bash
python -m my_project.train experiment=heat_static_1d model/balancer=normalized_static
```

The training factory constructs the balancer as if it were:

```python
balancer = NormalizedStaticBalancer(loss_names=module.loss_names, **configured_options)
```

It then initializes `TrainState`, compiles `combine` into the optimizer step, and logs every current weight as
`train/weight/<loss_name>`.

## Test the balancer independently

Add a focused unit test under `tests/unit/balancers/`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from phijax.balancers import NormalizedStaticBalancer


def test_normalized_static_balancer_preserves_name_order_and_gradients() -> None:
    """Verify normalized weights, mapping order independence, and finite loss gradients."""
    balancer = NormalizedStaticBalancer(("data", "pde"), weights={"data": 1.0, "pde": 3.0})
    state = balancer.initialize()

    def combined_loss(components: jax.Array) -> jax.Array:
        """Combine a synthetic component vector.

        Args:
            components: Ordered synthetic scalar losses.

        Returns:
            Weighted total produced from a deliberately reversed mapping.
        """
        losses = {"pde": components[1], "data": components[0]}
        return balancer.combine(losses, state)

    components = jnp.asarray([2.0, 4.0], dtype=jnp.float32)
    total = jax.jit(combined_loss)(components)
    gradients = jax.grad(combined_loss)(components)

    np.testing.assert_allclose(state.weights, [0.5, 1.5])
    np.testing.assert_allclose(total, 7.0)
    np.testing.assert_allclose(gradients, state.weights)
```

Also add a Hydra composition test that instantiates the complete experiment and checks:

```python
assert objective.loss_names == balancer.loss_names
assert balancer.initialize().weights.shape == (len(objective.loss_names),)
```

Test invalid names and numeric options before testing the compiled path. For adaptive numerical methods, additionally
test zero diagnostics, finite gradients, deterministic updates, singleton batches, and exact behavior on an analytic
toy problem.

## Adaptive custom balancers

An adaptive balancer additionally needs a pure compiled update with the shape:

```python
update(model_state, named_batches, balancer_state) -> balancer_state
```

PhiJAX schedules this executable outside the ordinary optimizer executable with `with_balancer_updates`, preventing an
expensive diagnostic branch from being embedded in every step. Adaptive implementations also satisfy
`AdaptiveBalancer` by returning a `BalancerUpdatePlan`:

```python
from collections.abc import Mapping, Sequence
from typing import Any

from phijax.balancers import BalancerUpdatePlan
from phijax.module import BasePhiModule


def build_update_plan(
    self,
    module: BasePhiModule,
    batch_keys: Sequence[str],
    options: Mapping[str, Any],
) -> BalancerUpdatePlan:
    """Build the compiled update and declare its sampling policy."""
    del batch_keys
    if options:
        raise ValueError(f"Unsupported update options: {tuple(options)}")
    return BalancerUpdatePlan(update=self.make_update(module), batch_sizes=None)
```

`batch_sizes=None` reuses the current training batches, as gradient-norm balancing does. A mapping such as
`dict.fromkeys(batch_keys, kernel_size)` asks assembly to draw one fixed diagnostic batch per key, as exact NTK does.
The common `every_n_steps` and `skip_first_step` keys remain host scheduling policy; all remaining `update` keys are
passed to `build_update_plan()` for method-specific validation. Consequently, a new adaptive balancer no longer
requires edits to training assembly and its mathematical update remains outside callbacks.

## Common mistakes

| Symptom                           | Likely cause                                                               |
| --------------------------------- | -------------------------------------------------------------------------- |
| Missing loss `KeyError`           | Configured weight key does not match the objective's final loss name       |
| Weight attached to the wrong loss | Components were built from mapping iteration order instead of `loss_names` |
| JAX concretization error          | `combine` converted a traced value to Python or NumPy                      |
| Retracing on every update         | State, batch shapes, dtypes, or PyTree structure change between calls      |
| Checkpoint structure mismatch     | Balancer state structure changed between save and restore                  |
| Adaptive weights lag unexpectedly | Update scheduling or `skip_first_step` does not match the intended cadence |

Keep weights, components, diagnostics, and totals in `float32` unless a configured precision policy explicitly
requires otherwise. Stop gradients through derived adaptive weights so optimization does not differentiate the
balancing rule itself.

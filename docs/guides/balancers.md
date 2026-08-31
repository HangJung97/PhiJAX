# Creating a custom loss balancer

This guide assumes an objective already exposes stable named losses. Read
[Building equations and objectives](objectives.md) first if `module.loss_names` or residual groups are unfamiliar.

Loss balancers combine the named losses from `PhiModule` into one scalar for differentiation. They are numerical
training components, not callbacks. Their weights are part of the compiled loss, and their state is stored in
`TrainState`. Callbacks only observe their diagnostics.

This guide implements a fixed balancer that normalizes configured nonnegative weights to have mean one. The same
contract can express other fixed weighting rules without changing the trainer or module.

## Balancer contract

Pass `module.loss_names` to the balancer constructor. A balancer must expose:

| Member                            | Responsibility                                                       |
| --------------------------------- | -------------------------------------------------------------------- |
| `loss_names`                      | Immutable names defining component and weight-vector order           |
| `initialize() -> BalancerState`   | Create JAX-compatible weights and diagnostic arrays                  |
| `combine(losses, state) -> total` | Return the weighted scalar total                                     |
| `diagnostics(state) -> mapping`   | Expose stable scalar values for progress bars and experiment loggers |

Do not duplicate `loss_names` in project settings. They come from the objective after equation-local names and any
explicit term aliases have been resolved. This prevents a balancer from silently assigning a weight to the wrong
residual.

`combine` runs inside the JIT-compiled training step and must therefore be pure. It must not mutate Python state,
convert traced arrays to NumPy, perform file IO, log metrics, or change its output PyTree structure between calls.

## Implement a fixed normalized balancer

Create `src/my_project/balancers/normalized_static.py`:

```python
import math
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from phijax.balancers import BalancerState


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

Using `BalancerState` makes the custom balancer work with progress metrics, precision handling, checkpoints, and
restoration. A fixed policy can leave `traces` at zero. Adaptive built-ins use it for the latest gradient-norm or NTK
diagnostic.

Re-export the class from `phijax.balancers` when it is part of PhiJAX itself. An application can otherwise import the
class from its own package.

## Construct the balancer

Use the objective's resolved names directly:

```python
balancer = NormalizedStaticBalancer(
    loss_names=module.loss_names,
    weights={
        "initial/u": 1.0,
        "boundary/u": 2.0,
        "pde/heat": 4.0,
    },
    default_weight=1.0,
)
```

Pass the balancer to the concise Trainer API. The Trainer initializes its state and compiles `combine` into the update:

```python
result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=0,
    balancer=balancer,
)
```

The balancer diagnostics expose every current weight as `weight/<loss_name>`. Advanced applications can initialize the
balancer state directly and pass an explicit plan to `Trainer.fit_state()`.

## Adaptive custom balancers

An adaptive balancer additionally needs a pure compiled update with the shape:

```python
update(model_state, named_batches, balancer_state) -> balancer_state
```

PhiJAX schedules this function from the Trainer's host loop, outside the ordinary optimizer update. This avoids putting
an expensive diagnostic branch in every step. Adaptive balancers return a `BalancerUpdatePlan`:

Store `update_every_n_steps` and `update_start_step` as validated constructor attributes on the custom balancer. Put
any method-specific settings, such as a diagnostic sample count, on the same constructor. The plan then resolves
those typed settings for the Trainer:

```python
from collections.abc import Sequence

from phijax.balancers import BalancerUpdatePlan
from phijax.core import BasePhiModule


def build_update_plan(
    self,
    module: BasePhiModule,
    batch_keys: Sequence[str],
) -> BalancerUpdatePlan:
    """Build the compiled update and declare its scheduling and sampling policy."""
    del batch_keys
    return BalancerUpdatePlan(
        update=self.make_update(module),
        every_n_steps=self.update_every_n_steps,
        update_start_step=self.update_start_step,
        batch_sizes=None,
    )
```

`batch_sizes=None` reuses the current training batches, as gradient-norm balancing does. A mapping such as
`dict.fromkeys(batch_keys, kernel_size)` requests one fixed diagnostic batch per key, as exact NTK does.
Put cadence and method-specific diagnostic settings on the balancer itself:

```python
from phijax.balancers import ExactNTKBalancer

balancer = ExactNTKBalancer(
    module.loss_names,
    update_every_n_steps=100,
    kernel_size=256,
)

result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=0,
    balancer=balancer,
)
```

The constructor validates numerical and scheduling options before fitting. When `update_start_step` is omitted, the
first update occurs after one complete interval. The cadence is anchored to an explicit start: a start of `50` and an
interval of `100` updates at steps `50`, `150`, and `250`. `build_update_plan()` converts that typed configuration into
the single runtime contract consumed by the Trainer. A new adaptive balancer therefore needs no Trainer changes, and
its numerical update stays outside callbacks. Advanced users place a complete `BalancerUpdatePlan` in
`TrainingPlan.balancer_update`; no separate schedule wrapper is required.

## Testing a custom balancer

Start with a focused unit test under `tests/unit/balancers/`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from my_project.balancers import NormalizedStaticBalancer


def test_normalized_static_balancer_preserves_name_order_and_gradients() -> None:
    """Verify normalized weights, name order, and finite loss gradients."""
    balancer = NormalizedStaticBalancer(("data", "pde"), weights={"data": 1.0, "pde": 3.0})
    state = balancer.initialize()

    def combined_loss(components: jax.Array) -> jax.Array:
        """Combine two synthetic loss values.

        Args:
            components: Loss values in balancer order.

        Returns:
            Weighted total loss.
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

Add a small integration test to confirm that the objective and balancer use the same loss order:

```python
balancer = NormalizedStaticBalancer(objective.loss_names)
assert balancer.loss_names == objective.loss_names
assert balancer.initialize().weights.shape == (len(objective.loss_names),)
```

Also test invalid names and numerical options. For an adaptive balancer, test zero diagnostics, deterministic updates,
singleton batches, finite gradients, and one case with a known analytic result.

## Common mistakes

| Symptom                           | Likely cause                                                                   |
| --------------------------------- | ------------------------------------------------------------------------------ |
| Missing loss `KeyError`           | Configured weight key does not match the objective's final loss name           |
| Weight attached to the wrong loss | Components were built from mapping iteration order instead of `loss_names`     |
| JAX concretization error          | `combine` converted a traced value to Python or NumPy                          |
| Retracing on every update         | State, batch shapes, dtypes, or PyTree structure change between calls          |
| Checkpoint structure mismatch     | Balancer state structure changed between save and restore                      |
| Adaptive weights lag unexpectedly | `update_start_step` or the update interval does not match the intended cadence |

Keep weights, components, diagnostics, and totals in `float32` unless a configured precision policy explicitly
requires otherwise. Stop gradients through derived adaptive weights so optimization does not differentiate the
balancing rule itself.

See the [balancer API](../api/balancers.md) for built-in static, gradient-norm, and exact-NTK policies.

For Hydra-based application assembly, see the
[PhiJAX Hydra template](https://github.com/HangJung97/phijax-hydra-template) and the
[configuration integration API](../api/configuration.md).

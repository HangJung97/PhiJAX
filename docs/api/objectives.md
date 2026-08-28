# Objectives

Objectives convert raw equation streams into stable named scalar losses. They do not combine weighted losses; that
policy belongs to a separately selected balancer.

## Contracts

The protocols are structural and runtime-checkable. Custom implementations do not need to inherit them.

::: phijax.objectives.ObjectiveTerm

::: phijax.objectives.Objective

::: phijax.objectives.ResidualObjective

## Residual terms

Each outer residual group becomes one scalar loss. `ResidualTerm` sums the mean square of every array in the group.
For example, one phase-wrapped loss can combine cosine and sine residuals without a custom objective class.

If `names` is absent, metadata attached by `residual_equation` produces names such as `pde/continuity`.

```python
from functools import partial

import math

from phijax.equations import burgers_1d
from phijax.objectives import ResidualTerm

pde_term = ResidualTerm(
    batch_key="pde",
    residual_fn=partial(burgers_1d, viscosity_coefficient=0.01 / math.pi),
)
```

::: phijax.objectives.ResidualTerm

## Composite objectives

The term mapping establishes stable composition names and preserves declaration order. Loss names must be globally
unique across terms.

```python
from phijax.objectives import CompositeObjective

objective = CompositeObjective(
    terms={
        "initial": initial_term,
        "boundary": boundary_term,
        "pde": pde_term,
    }
)
```

::: phijax.objectives.CompositeObjective

See [Building equations and objectives](../guides/objectives.md) for a complete custom equation composition.

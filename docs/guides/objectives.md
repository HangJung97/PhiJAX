# Building equations and objectives

This guide assumes you have run the [heat-equation quickstart](../getting-started/quickstart.md) and read
[Core concepts](../getting-started/concepts.md). It shows how to test an equation callable and combine it with reusable
initial and boundary terms.

PhiJAX treats equations and objective reduction as separate layers:

- an **equation callable** evaluates one or more raw residual arrays and declares their local names;
- a **`ResidualTerm`** prefixes those names with its batch key or applies an explicit application-level override; and
- a **`CompositeObjective`** merges independently configured terms.

Most applications only need to implement equation callables and compose existing objective classes in Python.

## Equation callable contract

An equation has this conceptual signature:

```python
def equation(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    stream: ResidualStream = "residual",
    **equation_options: Any,
) -> ResidualGroups: ...
```

The arguments have deliberately generic meanings:

| Argument      | Meaning                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------ |
| `model_apply` | Pure callable mapping `(model_state, inputs)` to model outputs                             |
| `model_state` | Explicit differentiable parameter PyTree                                                   |
| `batch`       | Fields from the configured `HostPool`, including `inputs`, `targets`, and auxiliary arrays |
| `stream`      | Either physical residuals or an explicitly supported raw-output stream                     |

An equation must not import or inspect a particular model class. This lets the same physics work with NNX or another
model implementation exposing the same pure application contract.

## Residual groups

`ResidualGroups` is a nested tuple:

```python
tuple[tuple[jax.Array, ...], ...]
```

The outer tuple matches the names declared by `@residual_equation`. Arrays in one inner tuple contribute to the same
scalar loss:

```text
(
    (residual_a,),                  -> local_names[0]
    (residual_b1, residual_b2),     -> local_names[1]
)
```

`ResidualTerm` reduces each group as:

```math
L_i = \sum_j \operatorname{mean}(r_{ij}^2).
```

Every residual array must retain a leading sample axis. Arrays in the same group must have equal sample counts so they
can also be flattened into one derivative-balancing stream.

## Application equation pattern

Use `@residual_equation` to attach stable local names to a pure residual function:

```python
@residual_equation(names=("heat",))
def heat_1d(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    diffusivity: float = 0.1,
    stream: ResidualStream = "residual",
) -> ResidualGroups:
    if stream != "residual":
        raise ValueError("The heat equation supports only the `residual` stream.")
    ...
    return ((residuals,),)
```

Keep coordinate order, output meaning, coefficients, supported streams, and returned shapes explicit. Use selective
helpers such as `value_and_jacobian` and `hessian_diagonal` to avoid tracing derivatives the equation does not need,
then apply `jax.vmap` over collocation points.

The [executable heat-equation quickstart](https://github.com/HangJung97/PhiJAX/blob/main/examples/quickstart.py)
contains the complete `du/dt - diffusivity * d2u/dx2` implementation and its DataModule. Re-export application
equations from a stable package path so objectives and tests do not depend on internal file layout.

## Built-in Navier--Stokes equations

PhiJAX provides incompressible Navier--Stokes equations for the following coordinate systems:

| Equation target                               | Input order       | Output order            | Residual names                                            |
| --------------------------------------------- | ----------------- | ----------------------- | --------------------------------------------------------- |
| `phijax.equations.cartesian_2d_navier_stokes` | `[x, y, t]`       | `[u_x, u_y, p]`         | `continuity`, `momentum_x`, `momentum_y`                  |
| `phijax.equations.cartesian_3d_navier_stokes` | `[x, y, z, t]`    | `[u_x, u_y, u_z, p]`    | `continuity`, `momentum_x`, `momentum_y`, `momentum_z`    |
| `phijax.equations.polar_navier_stokes`        | `[r, th, t]`      | `[u_r, u_th, p]`        | `continuity`, `momentum_r`, `momentum_th`                 |
| `phijax.equations.spherical_navier_stokes`    | `[r, th, phi, t]` | `[u_r, u_th, u_phi, p]` | `continuity`, `momentum_r`, `momentum_th`, `momentum_phi` |

All four accept `pressure_coefficient` and `viscosity_coefficient`. Use `pressure_coefficient: 1.0` and
`viscosity_coefficient: 1 / Re` for a consistently nondimensional equation. Setting `viscosity_coefficient: 0.0`
selects an inviscid path that does not trace velocity Hessians.

The spherical function weights residuals by `r sin(th)`. Away from coordinate singularities, this does not change the
zero-residual solutions. It reduces terms that divide by `sin(th)` near the polar axis. `radius_epsilon` and
`sine_epsilon` protect the remaining denominators.

A term can bind equation options with `functools.partial`:

```python
from functools import partial

from phijax.equations import cartesian_3d_navier_stokes
from phijax.objectives import ResidualTerm

pde_term = ResidualTerm(
    batch_key="pde",
    ntk_stream="residual",
    residual_fn=partial(
        cartesian_3d_navier_stokes,
        pressure_coefficient=1.0,
        viscosity_coefficient=0.01,
    ),
)
```

## Compose the objective

Initial and boundary conditions are direct supervised comparisons, so they can reuse
`phijax.equations.base_data_fidelity`. Only the PDE needs the new equation callable.

```python
from functools import partial

from phijax.equations import base_data_fidelity
from phijax.objectives import CompositeObjective, ResidualTerm

fidelity = partial(
    base_data_fidelity,
    output_indices=(0,),
    target_indices=(0,),
)
objective = CompositeObjective(
    terms={
        "initial": ResidualTerm(
            names=("initial/u",),
            batch_key="initial",
            ntk_stream="output",
            residual_fn=fidelity,
        ),
        "boundary": ResidualTerm(
            names=("boundary/u",),
            batch_key="boundary",
            ntk_stream="output",
            residual_fn=fidelity,
        ),
        "pde": ResidualTerm(
            batch_key="pde",
            ntk_stream="residual",
            residual_fn=partial(heat_1d, diffusivity=0.1, output_index=0),
        ),
    }
)
```

`partial` binds equation options such as `diffusivity`. The compiled training step still supplies `model_apply`,
`model_state`, `batch`, and `stream`.

The heat equation declares the local name `heat`. Its term uses `batch_key="pde"`, so the full loss name is `pde/heat`.
Explicit names remain useful for supervised terms. For example, an application can rename the generic fidelity name
`data` to `initial/u`.

## Choosing `ntk_stream`

`ntk_stream` controls what an exact-NTK balancer differentiates:

- `residual` differentiates the physical equation residual;
- `output` differentiates selected raw model outputs and must be explicitly supported by the equation.

Use `output` for fidelity or boundary functions that expose raw outputs. Use `residual` for PDEs and equations without
an output stream. Static and gradient-norm balancers do not use `Objective.residual_stream`. A correct declaration
still lets the same objective work with exact NTK balancing.

## Multiple losses from one equation

One equation may return several outer groups. For example, a coupled PDE could return:

```python
@residual_equation(names=("continuity", "momentum_x", "momentum_y"))
def fluid_equation(...) -> ResidualGroups:
    ...
    return (
        (continuity_residual,),
        (momentum_x_residual,),
        (momentum_y_residual,),
    )
```

Its term only provides the grouping prefix:

```python
term = ResidualTerm(batch_key="pde", residual_fn=fluid_equation)
```

This produces `pde/continuity`, `pde/momentum_x`, and `pde/momentum_y`. PhiJAX checks the returned group count at
runtime. Stale metadata therefore raises an error instead of assigning a residual to the wrong loss.

Use multiple arrays inside one group only when they jointly define one scalar objective, such as sine and cosine errors
for a wrapped phase.

## Explicit name overrides

Set `ResidualTerm.names` only when the application needs names different from `batch_key/local_name`:

```python
measurement_term = ResidualTerm(
    names=("measurement/axial_velocity",),
    batch_key="measurement",
    residual_fn=base_data_fidelity,
)
```

Explicit names must remain unique and match the equation's outer group count. They do not modify the reusable
equation's metadata.

## Objective tests

Test equations independently before composing a training experiment:

- residual shapes and dtypes;
- analytic or manufactured-solution values;
- finite parameter and input gradients;
- singleton batches;
- every supported stream;
- invalid coordinate widths and coefficients; and
- residual-group count and order.

Then compose the objective and verify:

```python
assert objective.loss_names == ("initial/u", "boundary/u", "pde/heat")
```

The [generic objective-term tests](https://github.com/HangJung97/PhiJAX/blob/main/tests/unit/objectives/test_terms.py)
show grouped reduction and stream validation in isolation. Continue with the [loss-balancer guide](balancers.md) after
the objective exposes stable loss names.

For Hydra-based application assembly, see the
[PhiJAX Hydra template](https://github.com/HangJung97/phijax-hydra-template) and the
[configuration integration API](../api/configuration.md).

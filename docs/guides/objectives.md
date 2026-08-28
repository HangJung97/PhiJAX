# Building equations and objectives

This guide assumes you have run the [heat-equation quickstart](../getting-started/quickstart.md) and read
[Core concepts](../getting-started/concepts.md). It shows how to test an equation callable and combine it with reusable
initial and boundary terms.

PhiJAX treats equations and objective reduction as separate layers:

- an **equation callable** evaluates one or more raw residual arrays and declares their local names;
- a **`ResidualTerm`** prefixes those names with its batch key or applies an explicit application-level override; and
- a **`CompositeObjective`** merges independently configured terms.

Most applications only need to implement equation callables and compose existing objective classes in Python or in
their project-owned Hydra configuration.

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

## Example heat equation

Create `src/my_project/equations/heat.py` in the application repository:

```python
from typing import Any

import jax
import jax.numpy as jnp

from phijax.derivatives import hessian_diagonal, value_and_jacobian
from phijax.equations import residual_equation
from phijax.types import ArrayMapping, ModelApply, ResidualGroups, ResidualStream


@residual_equation(names=("heat",))
def heat_1d(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    diffusivity: float = 0.1,
    output_index: int = 0,
    stream: ResidualStream = "residual",
) -> ResidualGroups:
    """Evaluate the one-dimensional heat-equation residual.

    Coordinates are ordered as `[t, x]`. The returned residual is
    `du/dt - diffusivity * d2u/dx2`.

    Args:
        model_apply: Pure explicit-state model application callable.
        model_state: Differentiable model parameter PyTree.
        batch: Arrays containing rank-two coordinates under `inputs`.
        diffusivity: Positive coefficient multiplying `d2u/dx2`.
        output_index: Scalar model-output component representing `u`.
        stream: Requested residual representation; only `residual` is supported.

    Returns:
        One single-array residual group with shape `[samples, 1]`.

    Raises:
        ValueError: If the stream, coordinates, diffusivity, or output index is invalid.
    """
    if stream != "residual":
        raise ValueError("The heat equation supports only the `residual` stream.")
    inputs = batch["inputs"]
    if inputs.ndim != 2 or inputs.shape[-1] != 2:
        raise ValueError("Heat-equation inputs must have columns `[t, x]`.")
    if diffusivity <= 0.0:
        raise ValueError("`diffusivity` must be positive.")
    if output_index < 0:
        raise ValueError("`output_index` must be nonnegative.")

    def scalar_prediction(
        state: Any,
        time_coordinate: jax.Array,
        position: jax.Array,
    ) -> jax.Array:
        """Select the modeled scalar field from split coordinates.

        Args:
            state: Explicit differentiable model state.
            time_coordinate: Scalar time coordinate.
            position: Scalar spatial coordinate.

        Returns:
            Selected scalar model output.
        """
        point = jnp.stack((time_coordinate, position))
        return model_apply(state, point)[output_index]

    value_and_first_derivatives = value_and_jacobian(scalar_prediction, (1, 2))
    second_spatial_derivative = hessian_diagonal(scalar_prediction, 2)

    def point_residual(point: jax.Array) -> jax.Array:
        """Evaluate one scalar heat-equation residual.

        Args:
            point: Coordinate vector ordered as `[t, x]`.

        Returns:
            Scalar value `du/dt - diffusivity * d2u/dx2`.
        """
        _, (du_dt, _) = value_and_first_derivatives(model_state, point[0], point[1])
        (d2u_dx2,) = second_spatial_derivative(model_state, point[0], point[1])
        coefficient = jnp.asarray(diffusivity, dtype=du_dt.dtype)
        return du_dt - coefficient * d2u_dx2

    residuals = jax.vmap(point_residual)(inputs)
    return ((residuals[:, None],),)
```

`value_and_jacobian` reuses the model evaluation for the selected first derivatives. `hessian_diagonal` computes only
`d2u/dx2`, not the unused `d2u/dt2` or mixed derivatives. Both helpers work on scalar positional arguments. Apply
`jax.vmap` outside them to batch collocation points.

Re-export the function from the application's `my_project.equations` package so Python and Hydra can use the stable
target:

```text
my_project.equations.heat_1d
```

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

A configured objective term needs only the selected public target:

```yaml
pde:
  _target_: phijax.objectives.ResidualTerm
  batch_key: pde
  ntk_stream: residual
  residual_fn:
    _target_: phijax.equations.cartesian_3d_navier_stokes
    _partial_: true
    pressure_coefficient: 1.0
    viscosity_coefficient: 0.01
```

## Objective configuration

Initial and boundary conditions are direct supervised comparisons, so they can reuse
`phijax.equations.base_data_fidelity`. Only the PDE needs the new equation callable.

For a Hydra-based application, create `src/my_project/configs/model/objective/heat_1d.yaml`:

```yaml
_target_: phijax.objectives.CompositeObjective
terms:
  initial:
    _target_: phijax.objectives.ResidualTerm
    names: [initial/u]
    batch_key: initial
    ntk_stream: output
    residual_fn:
      _target_: phijax.equations.base_data_fidelity
      _partial_: true
      output_indices: [0]
      target_indices: [0]
  boundary:
    _target_: phijax.objectives.ResidualTerm
    names: [boundary/u]
    batch_key: boundary
    ntk_stream: output
    residual_fn:
      _target_: phijax.equations.base_data_fidelity
      _partial_: true
      output_indices: [0]
      target_indices: [0]
  pde:
    _target_: phijax.objectives.ResidualTerm
    batch_key: pde
    ntk_stream: residual
    residual_fn:
      _target_: my_project.equations.heat_1d
      _partial_: true
      diffusivity: 0.1
      output_index: 0
```

Hydra's `_partial_: true` binds options such as `diffusivity`. The compiled training step still supplies `model_apply`,
`model_state`, `batch`, and `stream`.

The heat equation declares the local name `heat`. Its term uses `batch_key: pde`, so the full loss name is `pde/heat`.
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

```yaml
_target_: phijax.objectives.ResidualTerm
batch_key: pde
residual_fn:
  _target_: package.equations.fluid_equation
  _partial_: true
```

This produces `pde/continuity`, `pde/momentum_x`, and `pde/momentum_y`. PhiJAX checks the returned group count at
runtime. Stale metadata therefore raises an error instead of assigning a residual to the wrong loss.

Use multiple arrays inside one group only when they jointly define one scalar objective, such as sine and cosine errors
for a wrapped phase.

## Explicit name overrides

Set `ResidualTerm.names` only when the application needs names different from `batch_key/local_name`:

```yaml
names: [measurement/axial_velocity]
batch_key: measurement
residual_fn:
  _target_: phijax.equations.base_data_fidelity
  _partial_: true
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

Then instantiate the YAML objective and verify:

```python
objective = hydra.utils.instantiate(config.model.objective)
assert objective.loss_names == ("initial/u", "boundary/u", "pde/heat")
```

The [generic objective-term tests](https://github.com/HangJung97/PhiJAX/blob/main/tests/unit/objectives/test_terms.py)
show grouped reduction and stream validation in isolation. Continue with the [loss-balancer guide](balancers.md) after
the objective exposes stable loss names.

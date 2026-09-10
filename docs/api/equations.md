# Equations

Equation functions use JAX arrays and a generic model callable, not a concrete model class. Each public PDE accepts an
`ArrayMapping` batch and returns named residual groups for `ResidualTerm`. One equation callable can compute all of its
coupled components together.

## Residual metadata

Attach static local names to every reusable equation. When a `ResidualTerm` omits explicit names, it prefixes these
with the term's `batch_key`.

```python
@residual_equation(names=("heat",))
def heat_equation(model_apply, model_state, batch, *, stream="residual"):
    return ((residual,),)
```

::: phijax.equations.residual_equation

::: phijax.equations.get_residual_names

::: phijax.equations.get_default_ntk_stream

## Data fidelity

[`base_data_fidelity`](#phijax.equations.base_data_fidelity) supports direct component supervision, optional sample
weights, vector projection, and target negation.
[`phase_wrapped_fidelity`](#phijax.equations.phase_wrapped_fidelity) supports the same options and represents periodic
observations with cosine and sine residuals in one loss group.

Set `weight_key="confidence"` on either equation to read residual weights from `batch["confidence"]`. The default
is `"weight"`; a missing field or `weight_key=None` disables weighting. Weights multiply residuals before loss
reduction, including both cosine and sine residuals for phase-wrapped fidelity. Use shape `(N, 1)` for sample-wise
weights that broadcast across residual components. The output stream remains unweighted.

Set `projection_key="direction"` on either equation to read projection directions from `batch["direction"]`. The
default is `"projection"`; a missing field or `projection_key=None` disables projection. Directions must match the
shape and component order selected by `output_indices`. Projection takes the dot product along the last axis and
keeps a trailing singleton dimension, so `target_indices` must select one component. Phase-wrapped fidelity projects
predictions before converting them to phases.

Phase-wrapped fidelity also accepts `period_key="wrapping_period"` to read sample-wise periods from
`batch["wrapping_period"]`. It defaults to `"period"`. The configured field is required for residuals; the output
stream does not read it.

For example, configure phase-wrapped fidelity with custom weight, projection, and period fields:

```python
from functools import partial

from phijax.equations import phase_wrapped_fidelity

fidelity = partial(
    phase_wrapped_fidelity,
    output_indices=(0, 1),
    target_indices=(0,),
    weight_key="confidence",
    projection_key="direction",
    period_key="wrapping_period",
)
```

Here, `batch["direction"]` has shape `(N, 2)`, and both the selected targets and `batch["wrapping_period"]` have shape
`(N, 1)`. Both equations support
`stream="output"` for output-based diagnostics; this stream returns selected model outputs before projection.
The array-level residual functions accept the direction array directly through `projection`.

Phase-wrapped fidelity now uses a `projection` field when present. Set `projection_key=None` to retain direct
supervision for batches that contain an unrelated field with that name.

::: phijax.equations.base_data_fidelity_residual

::: phijax.equations.base_data_fidelity

::: phijax.equations.phase_wrapped_residuals

::: phijax.equations.phase_wrapped_fidelity

## Boundary conditions

[`no_slip_boundary`](#phijax.equations.no_slip_boundary) constrains every selected velocity component to the wall
velocity in `batch["targets"]`. Use zero targets for a stationary wall or prescribed velocities for a moving wall.
`output_indices` and `target_indices` default to `(0, 1)` and must select the same number of components in matching
order. For three velocity components, set both to `(0, 1, 2)`. The equation returns one `no_slip` group containing
the component-wise residual array. Its default NTK stream is `"output"`, which returns selected predictions without
reading targets. No wall normals are required.

[`free_slip_boundary`](#phijax.equations.free_slip_boundary) accepts `normals_key="wall_normals"` to read wall-normal
directions from `batch["wall_normals"]`. The default remains `"normals"`. Normal components must follow the order
selected by `output_indices` and have the same final width. The configured field is required for residuals;
`stream="output"` returns the selected model outputs without reading normals.

::: phijax.equations.base_boundary_residual

::: phijax.equations.no_slip_residual

::: phijax.equations.no_slip_boundary

::: phijax.equations.free_slip_residual

::: phijax.equations.free_slip_boundary

## One-dimensional Burgers equation

Inputs use `[t, x]`; `output_index` selects the scalar solution. The equation is

$$
\frac{\partial u}{\partial t}
+ u\frac{\partial u}{\partial x}
- \nu\frac{\partial^2 u}{\partial x^2}
= 0,
$$

where $\nu$ is `viscosity_coefficient`. The function returns the left-hand side as the residual optimized during
training.

::: phijax.equations.burgers_1d

## Cartesian Navier--Stokes

| Dimension | Input order    | Output order         | Residual order                     |
| --------- | -------------- | -------------------- | ---------------------------------- |
| 2D        | `[x, y, t]`    | `[u_x, u_y, p]`      | continuity, x momentum, y momentum |
| 3D        | `[x, y, z, t]` | `[u_x, u_y, u_z, p]` | continuity, x/y/z momentum         |

A zero `viscosity_coefficient` selects an inviscid path that does not trace second derivatives.

::: phijax.equations.cartesian_2d_navier_stokes

::: phijax.equations.cartesian_3d_navier_stokes

## Polar Navier--Stokes

Polar inputs use `[r, th, t]` and outputs use `[u_r, u_th, p]`. `radius_epsilon` protects terms that divide by the
radius. Equation weighting keeps the same zero-residual solutions while reducing singular behavior near the origin.

::: phijax.equations.polar_navier_stokes

## Spherical Navier--Stokes

Spherical inputs use `[r, th, phi, t]` and outputs use `[u_r, u_th, u_phi, p]`. `radius_epsilon` and `sine_epsilon`
guard geometric singularities. Residuals are weighted by `r * sin(th)` as described in the function documentation.

::: phijax.equations.spherical_navier_stokes

See [Building equations and objectives](../guides/objectives.md) for residual-group design and analytic tests.

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

`base_data_fidelity` supports direct component supervision, optional sample weights, vector projection, and target
negation. `phase_wrapped_fidelity` represents periodic observations with cosine and sine residuals in one loss group.
Both support the model-output stream used by output-based diagnostics.

::: phijax.equations.base_data_fidelity_residual

::: phijax.equations.base_data_fidelity

::: phijax.equations.phase_wrapped_residuals

::: phijax.equations.phase_wrapped_fidelity

## Boundary conditions

::: phijax.equations.base_boundary_residual

::: phijax.equations.no_slip_residual

::: phijax.equations.free_slip_residual

::: phijax.equations.free_slip_boundary

## One-dimensional Burgers equation

Inputs use `[t, x]`; `output_index` selects the scalar solution. The returned residual is

```text
du/dt + u * du/dx - viscosity_coefficient * d2u/dx2.
```

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

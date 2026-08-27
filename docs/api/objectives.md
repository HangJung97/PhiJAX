# Objectives

Objectives convert raw equation streams into stable named scalar losses. They do not combine weighted losses; that
policy belongs to a separately selected balancer.

## Contracts

The protocols are structural and runtime-checkable. Custom implementations do not need to inherit them.

::: phijax.objectives.ObjectiveTerm

::: phijax.objectives.Objective

::: phijax.objectives.ResidualObjective

## Residual terms

One outer equation residual group becomes one scalar loss. Every array in that group contributes its mean square and
the means are summed. This allows a phase-wrapped term, for example, to combine cosine and sine residuals without an
equation-specific objective class.

If `names` is absent, metadata attached by `residual_equation` produces names such as `pde/continuity`.

```yaml
pde:
  _target_: phijax.objectives.ResidualTerm
  batch_key: pde
  residual_fn:
    _target_: phijax.equations.burgers_1d
    _partial_: true
    viscosity_coefficient: ${op:truediv,0.01,${math:pi}}
```

::: phijax.objectives.ResidualTerm

## Composite objectives

The term mapping establishes stable Hydra override paths and preserves declaration order. Loss names must be globally
unique across terms.

```yaml
_target_: phijax.objectives.CompositeObjective
terms:
  initial: ${initial_term}
  boundary: ${boundary_term}
  pde: ${pde_term}
```

::: phijax.objectives.CompositeObjective

See [Building equations and objectives](../guides/objectives.md) for a complete custom equation composition.

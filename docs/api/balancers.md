# Loss balancers

A balancer combines ordered, unweighted objective losses. It is independent of `PhiModule`, so the same module and
objective can use static, gradient-norm, exact-NTK, or custom weighting.

## Shared state and update contract

`BalancerState.weights` is aligned with objective `loss_names`. The `traces` field stores the latest adaptive
diagnostic: parameter-gradient norms for `GradNormBalancer`, or mean pointwise diagonal NTKs for `ExactNTKBalancer`.

`BalancerUpdatePlan` contains a compiled functional update, its host-side cadence, and optional diagnostic batch
sizes. A `None` batch-size mapping reuses the current training batches.

Adaptive balancers accept `update_every_n_steps` and an optional `update_start_step`. By default, the first update
occurs after one complete interval. An explicit start anchors the cadence, so a start of `50` with an interval of `100`
updates at steps `50`, `150`, and `250`.

## Logged metrics

A balancer's `diagnostics()` method returns names such as `weight/pde/heat`. `PhiModule` adds the `train/` prefix before
the values reach callbacks, progress bars, or experiment loggers. For a loss named `pde/heat`, the built-in balancers
produce these metrics:

| Balancer           | Metric                     | Meaning                                                                            |
| ------------------ | -------------------------- | ---------------------------------------------------------------------------------- |
| All built-ins      | `train/weight/pde/heat`    | Current multiplier applied to the unweighted `pde/heat` loss                       |
| `GradNormBalancer` | `train/grad_norm/pde/heat` | L2 norm of the unweighted loss gradient over all model parameters                  |
| `ExactNTKBalancer` | `train/ntk/pde/heat`       | Mean pointwise diagonal empirical NTK for the configured residual or output stream |

Every scalar is sent to configured experiment loggers at the Trainer's logging cadence. Loss weights also appear in
the progress metrics by default. Gradient-norm and NTK diagnostics are logger-only unless a module explicitly routes
them to the progress bar with `self.log(..., prog_bar=True)`.

`StaticLossBalancer` reports the configured weights throughout fitting. Adaptive diagnostics start at zero and change
only when their scheduled update runs. Their most recent values are reported between updates; they are not recomputed
on every optimizer step.

::: phijax.balancers.BalancerState

::: phijax.balancers.BalancerUpdatePlan

::: phijax.balancers.LossBalancer

::: phijax.balancers.AdaptiveBalancer

## Static weighting

::: phijax.balancers.StaticLossBalancer

```python
from phijax.balancers import StaticLossBalancer

balancer = StaticLossBalancer(
    module.loss_names,
    weights={"initial/u": 1.0, "pde/burgers": 1.0},
)
```

Pass the module's resolved `loss_names`; do not duplicate them in project settings.

## Gradient-norm weighting

This policy balances inverse parameter-gradient magnitudes. It is not the training-rate-aware GradNorm multitask
algorithm. Losses are differentiated individually to limit retained intermediate state. Each
`train/grad_norm/<loss_name>` value is the L2 norm of one unweighted scalar loss gradient with respect to every model
parameter. The update uses the current training batches.

::: phijax.balancers.GradNormBalancer

```python
balancer = GradNormBalancer(
    module.loss_names,
    update_every_n_steps=100,
)
```

## Exact NTK weighting

Exact NTK balancing computes the squared parameter-Jacobian norm at each point. It averages these values for each
residual stream and normalizes the loss weights from those means. Named streams are evaluated one at a time.
`kernel_chunk_size=1` minimizes peak memory; `None` vectorizes over all samples.

Each `train/ntk/<loss_name>` value is the resulting mean for one named residual or output stream. Each scheduled update
evaluates the current model on diagnostic batches drawn from the training source.

### Choose the diagnostic samples

`kernel_size` sets how many rows each objective batch contributes to the NTK estimate:

| `kernel_size`                      | Diagnostic rows                                              |
| ---------------------------------- | ------------------------------------------------------------ |
| `256`                              | 256 random rows from every objective batch                   |
| `"all"`                            | Every row of every finite pool                               |
| `{"pde": 1024, "boundary": "all"}` | Set per objective batch key; the mapping must name every key |

A random sample estimates the mean NTK of a term. `"all"` computes it exactly, which suits small boundary and initial
condition sets. It removes sampling noise and catches points with unusual NTK values, such as corners. `"all"` requires
a finite pool; a generated sampler, such as `UniformDomainSampler`, raises `ValueError`. The cost of an update grows
with the number of rows, so keep random samples for large data pools and collocation terms.

By default, the Trainer draws the diagnostic batches once per fit and reuses them for every update, so the weights
follow the training dynamics rather than sampling noise. Set `resample=True` to draw new rows for every update,
which covers more of the domain over time. Terms whose `kernel_size` is `"all"` use the same rows either way.

Diagnostic batches depend only on `TrainState.balancer_key` and, with `resample=True`, the global step of the update.
A resumed fit therefore draws the same batches as an uninterrupted run.

The method is motivated by the empirical NTK analysis and adaptive PINN loss weighting in
[Wang, Yu, and Perdikaris (2022)](https://doi.org/10.1016/j.jcp.2021.110768). PhiJAX uses the mean-diagonal rule
`lambda_i = mean_j(mu_j) / mu_i`. It supports moving-average smoothing and processes one named stream at a time to
limit peak memory.

::: phijax.balancers.exact_ntk_trace

::: phijax.balancers.ExactNTKBalancer

```python
balancer = ExactNTKBalancer(
    module.loss_names,
    update_every_n_steps=100,
    kernel_size={"pde": 256, "boundary": "all", "initial": "all"},
    kernel_chunk_size=1,
)
```

The `kernel_size` keys are objective batch keys, the names of the training source's batches, which may differ from
loss names.

See [Creating a custom loss balancer](../guides/balancers.md) for implementation and testing requirements.

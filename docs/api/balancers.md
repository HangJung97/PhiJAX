# Loss balancers

A balancer combines ordered unweighted objective losses. It remains independent of `PhiModule`, allowing the same
module and objective to be paired with static, gradient-norm, exact-NTK, or custom weighting.

## Shared state and update contract

`BalancerState.weights` is aligned with objective `loss_names`. The `traces` field stores the latest adaptive
diagnostic: parameter-gradient norms for `GradNormBalancer`, or mean pointwise diagonal NTKs for `ExactNTKBalancer`.

`BalancerUpdatePlan` contains a compiled functional update and optional diagnostic batch sizes. A `None` batch-size
mapping reuses the current training batches.

::: phijax.balancers.BalancerState

::: phijax.balancers.BalancerUpdatePlan

::: phijax.balancers.AdaptiveBalancer

## Static weighting

::: phijax.balancers.StaticLossBalancer

```yaml
factory:
  _target_: phijax.balancers.StaticLossBalancer
  weights:
    initial/u: 1.0
    pde/burgers: 1.0
```

Assembly injects `loss_names`; do not duplicate them in configuration.

## Gradient-norm weighting

This policy balances inverse parameter-gradient magnitudes. It is not the training-rate-aware GradNorm multitask
algorithm. Losses are differentiated individually to limit retained intermediate state.

::: phijax.balancers.GradNormBalancer

## Exact NTK weighting

Exact NTK balancing computes a pointwise squared parameter-Jacobian norm, averages it for each residual stream, and
normalizes loss weights by those means. Named streams are evaluated sequentially. `kernel_chunk_size=1` minimizes peak
memory; `None` fully vectorizes over samples.

The method is motivated by the empirical NTK analysis and adaptive PINN loss weighting in
[Wang, Yu, and Perdikaris (2022)](https://doi.org/10.1016/j.jcp.2021.110768). PhiJAX uses the mean-diagonal rule
`lambda_i = mean_j(mu_j) / mu_i`, applies optional moving-average smoothing, and computes one named residual stream at
a time to limit peak memory.

::: phijax.balancers.exact_ntk_trace

::: phijax.balancers.ExactNTKBalancer

See [Creating a custom loss balancer](../guides/balancers.md) for implementation and testing requirements.

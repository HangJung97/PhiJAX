# PhiJAX documentation

PhiJAX is a typed JAX framework for physics-informed neural networks. It supplies reusable runtime and numerical
contracts while leaving executable entrypoints, Hydra configuration trees, datasets, and domain applications in the
projects that own them.

## Start here

- [Data](guides/datasets.md): implement a `PhiDataModule`, immutable host pools, and explicit-key batch sources.
- [Equations and objectives](guides/objectives.md): compose differentiable residuals into named scalar losses.
- [Loss balancers](guides/balancers.md): implement static or adaptive functional loss weighting.
- [Models](api/models.md): use MLP, Modified MLP, PirateNet, or adapt a custom Flax NNX architecture.
- [Hooks and lifecycle](api/hooks.md): choose module and callback extension points from the exact Trainer call order.
- [API overview](api/index.md): supported imports, common types, conventions, and stability policy.
- [API map](api-reference.md): focused reference pages generated from source docstrings.

The separate `phijax-hydra-template` repository provides runnable Hydra entrypoints, project configuration groups, and
a small Burgers example. Applications built from that template depend on PhiJAX rather than placing their code inside
the framework wheel.

## Runtime ownership

```text
Application DataModule ──> host batch source ──> Trainer device placement
                                                       │
Model factory ──> InitializedModel ──> PhiModule ──────┼──> TrainingPlan ──> compiled update
                                                       │
Objective ──> named losses ──> LossBalancer ───────────┘
```

The Trainer owns lifecycle dispatch, placement, checkpoint restoration, callbacks, logging, and teardown. Model,
optimizer, balancer, and PRNG state remain explicit functional values. Neither the Trainer nor the core numerical
contracts read Hydra configuration.

## Beta stability

PhiJAX `0.1.0b1` supports the names documented in package `__all__` declarations. Prediction artifacts use schema
version 2. Checkpoints include a versioned manifest and restore only within a compatible PhiJAX major/minor line.

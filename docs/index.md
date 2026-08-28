# PhiJAX documentation

PhiJAX is a typed JAX framework for physics-informed neural networks (PINNs). It provides reusable tools for numerical
code and training. New applications can build on PhiJAX while keeping their datasets, domain-specific logic, and
runnable workflows in a separate project.

## New to PhiJAX?

Follow these pages in order:

1. [Installation](getting-started/installation.md): install CPU or NVIDIA GPU support and verify the selected JAX
   device.
2. [Heat-equation quickstart](getting-started/quickstart.md): run a complete PINN with no external dataset.
3. [Core concepts](getting-started/concepts.md): understand model state, DataModules, objectives, balancers, and the
   Trainer boundary.

## Extend an application

- [Data](guides/datasets.md): implement a `PhiDataModule`, immutable host pools, and explicit-key batch sources.
- [Equations and objectives](guides/objectives.md): compose differentiable residuals into named scalar losses.
- [Loss balancers](guides/balancers.md): implement static or adaptive functional loss weighting.
- [Models](api/models.md): use MLP, Modified MLP, PirateNet, or adapt a custom Flax NNX architecture.
- [Hooks and lifecycle](api/hooks.md): choose module and callback extension points from the exact Trainer call order.
- [Troubleshooting](guides/troubleshooting.md): diagnose accelerator selection, memory, compilation, and lifecycle
  issues.

Use the [API overview](api/index.md) for supported imports and conventions, or the
[API map](api-reference.md) to find generated signatures and source docstrings.

The separate `phijax-hydra-template` repository provides runnable Hydra entrypoints, config groups, and a small
Burgers example. Projects created from the template use PhiJAX as a dependency.

## Runtime ownership

```text
Application DataModule ──> host batch source ──> Trainer device placement
                                                       │
Model factory ──> InitializedModel ──> PhiModule ──────┼──> TrainingPlan ──> compiled update
                                                       │
Objective ──> named losses ──> LossBalancer ───────────┘
```

The Trainer runs hooks, places data, restores checkpoints, logs metrics, and cleans up resources. Model, optimizer,
balancer, and PRNG state remain explicit. Neither the Trainer nor the numerical APIs read Hydra configuration.

## Beta stability

PhiJAX `0.1.0b1` supports the names documented in package `__all__` declarations. Prediction artifacts use schema
version 2. Checkpoints include a versioned manifest and restore only within a compatible PhiJAX major/minor line.

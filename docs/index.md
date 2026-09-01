# PhiJAX documentation

PhiJAX is a JAX framework for physics-informed neural networks (PINNs). It provides reusable tools for numerical
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

- [Training and prediction](guides/training.md): fit, resume, predict, and choose between the common and explicit APIs.
- [Data](guides/datasets.md): implement a `PhiDataModule`, immutable host pools, and explicit-key batch sources.
- [Equations and objectives](guides/objectives.md): compose differentiable residuals into named scalar losses.
- [Loss balancers](guides/balancers.md): implement static or adaptive functional loss weighting.
- [Randomness and reproducibility](guides/reproducibility.md): understand PRNG streams, sampling, and checkpoint
  continuation.
- [Models](api/models.md): use MLP, Modified MLP, PirateNet, or adapt a custom Flax NNX architecture.
- [Hooks and lifecycle](api/hooks.md): choose module and callback extension points from the exact Trainer call order.
- [Troubleshooting](guides/troubleshooting.md): diagnose accelerator selection, memory, compilation, and lifecycle
  issues.

Use the [API reference](api/index.md) for supported imports, generated signatures, accepted options, return values,
and errors.

The separate [PhiJAX Hydra template](https://github.com/HangJung97/phijax-hydra-template) provides runnable
entrypoints, config groups, and a small Burgers example. Projects created from the template use PhiJAX as a dependency.

## Runtime ownership

```text
Model factory + objective ---> PhiModule blueprint ----+
DataModule -------------------> named batches ---------+---> Trainer.fit() ---> FitResult ---> Trainer.predict()
Optimizer + seed --------------------------------------+
LossBalancer + update policy (optional) ---------------+
```

The Trainer initializes and binds the model, creates explicit optimizer and PRNG state, compiles updates, runs hooks,
places data, restores checkpoints, logs metrics, and cleans up resources. Neither the Trainer nor the numerical APIs
read Hydra configuration.

## Beta stability

PhiJAX supports the names documented in package `__all__` declarations. Prediction artifacts use schema version 2.
Checkpoints include a versioned manifest and restore only within a compatible PhiJAX major/minor line.

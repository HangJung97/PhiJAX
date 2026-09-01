<div align="center">

# PhiJAX

[![Code Quality](https://github.com/HangJung97/PhiJAX/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/HangJung97/PhiJAX/actions/workflows/code-quality-main.yaml)
[![Tests](https://github.com/HangJung97/PhiJAX/actions/workflows/tests.yaml/badge.svg)](https://github.com/HangJung97/PhiJAX/actions/workflows/tests.yaml)
[![Documentation](https://github.com/HangJung97/PhiJAX/actions/workflows/docs.yaml/badge.svg)](https://github.com/HangJung97/PhiJAX/actions/workflows/docs.yaml)
[![Codecov](https://codecov.io/gh/HangJung97/PhiJAX/graph/badge.svg)](https://codecov.io/gh/HangJung97/PhiJAX)
<br>
[![Python](https://img.shields.io/pypi/pyversions/phijax?color=blue&logo=python&logoColor=white)](https://pypi.org/project/phijax/)
[![PyPI](https://img.shields.io/pypi/v/phijax?include_prereleases)](https://pypi.org/project/phijax/)
<br>
[![License](https://img.shields.io/github/license/HangJung97/PhiJAX?color=blue)](LICENSE)

</div>

PhiJAX is a typed [JAX](https://docs.jax.dev/) framework for physics-informed neural networks (PINNs). It provides the
core tools for training PINNs: functional state, explicit-key sampling, reusable equations and objectives, adaptive
loss balancing, callbacks, logging, checkpointing, and prediction artifacts. Its `Trainer` and `PhiModule` lifecycles
are inspired by Lightning.

For a configuration-first project layout, start from
[`phijax-hydra-template`](https://github.com/HangJung97/phijax-hydra-template) and customize its entrypoints, Hydra
configs, and application DataModules.

> **Beta API:** PhiJAX is under active development. Breaking changes may occur between major versions and during the
> beta period before the stable release. The supported public API is documented in package `__all__` declarations.
> Checkpoints can be restored by compatible PhiJAX releases from the same major and minor version.

[Quickstart](https://hangjung97.github.io/PhiJAX/getting-started/quickstart/) |
[Guides](https://hangjung97.github.io/PhiJAX/guides/datasets/) |
[API reference](https://hangjung97.github.io/PhiJAX/api/) |
[Changelog](CHANGELOG.md) |
[Contributing](CONTRIBUTING.md)

## Why PhiJAX?

JAX provides automatic differentiation, vectorization, compilation, and accelerator support. PhiJAX adds the
structure needed to build reproducible, maintainable PINN experiments:

- selective coordinate derivatives and reusable PDE, boundary, and data-fidelity equations;
- named objective terms with static, gradient-norm, and exact-NTK loss balancing;
- standard MLP, Modified MLP, and adaptive-residual PirateNet architectures with a custom NNX adapter;
- explicit model, optimizer, balancer, and PRNG state for reproducible compiled updates;
- familiar, Lightning-inspired lifecycles for trainers, modules, DataModules, callbacks, loggers, and checkpoints; and
- scientific data preparation on the CPU, automatic batch placement by the Trainer, and prediction files with a
  stable, versioned format.

PhiJAX reduces boilerplate for training, prediction, logging, and checkpointing without hiding JAX transformations or
functional state. Hydra support is optional, and the core Trainer API does not depend on a configuration framework.

## Installation

PhiJAX requires Python 3.12 or newer. Python 3.12, 3.13, and 3.14 are covered by CI.

```bash
pip install phijax
```

The default installation includes CPU-capable JAX. Select one mutually exclusive GPU extra when the matching NVIDIA
runtime is available:

```bash
pip install "phijax[cuda12]"
# or
pip install "phijax[cuda13]"
```

Optional experiment loggers are installed separately:

```bash
pip install "phijax[tensorboard]"
pip install "phijax[wandb]"
```

Extras can be combined when an environment needs GPU support and both logging integrations:

```bash
pip install "phijax[cuda13,wandb,tensorboard]"
```

Replace `cuda13` with `cuda12` for a CUDA 12 environment; do not install both CUDA extras together.

Experimental TPU environments can use `pip install "phijax[tpu]"`. The TPU, CUDA 12, and CUDA 13 extras are mutually
exclusive.

See the [installation guide](https://hangjung97.github.io/PhiJAX/getting-started/installation/) for environment
verification and development setup.

## Quickstart

The first runnable example solves a one-dimensional heat equation with initial, boundary, and PDE losses. It uses no
external data and runs on CPU after JAX completes its initial compilation:

```bash
git clone https://github.com/HangJung97/PhiJAX.git
cd PhiJAX
uv sync
JAX_PLATFORMS=cpu uv run --no-sync python examples/quickstart.py
```

`JAX_PLATFORMS=cpu` keeps the whole process strictly CPU-only. This is stronger than selecting
`Trainer(accelerator="cpu")`, which controls PhiJAX placement without changing JAX's process-wide default backend.

To run the same example on an available NVIDIA GPU:

```bash
uv sync --extra cuda13
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python examples/quickstart.py --accelerator gpu
```

Replace `cuda13` with `cuda12` when using the CUDA 12 extra. Disabling preallocation prevents JAX from reserving most
GPU memory at startup, which is helpful on shared or memory-constrained systems.

Read the [annotated quickstart](https://hangjung97.github.io/PhiJAX/getting-started/quickstart/) for the equation,
complete source, and an explanation of each runtime object.

## Core workflow

PhiJAX keeps each part of a PINN workflow explicit while assembling the common training path for you:

```text
Model factory + objective ---> PhiModule blueprint ----+
DataModule -------------------> named batches ---------+---> Trainer.fit() ---> FitResult ---> Trainer.predict()
Optimizer + seed --------------------------------------+
LossBalancer + update policy (optional) ---------------+
```

The Trainer initializes the model, splits independent PRNG streams, prepares data sources, compiles the update, moves
batches to devices, runs hooks, restores checkpoints, and cleans up resources. `FitResult` returns the bound module and
explicit `TrainState` for prediction or advanced workflows.

Application DataModules implement `setup()`, `train_batch_source()`, and optionally `predict_batch_source()`.
Prediction is skipped cleanly when a DataModule has no prediction source.

`Trainer` manages setup and cleanup automatically across fitting and prediction.

## Core APIs

- `Trainer.fit()` assembles the common application workflow; `fit_state()` exposes explicit state and plan control.
- `Trainer.predict()` consumes a `FitResult`; `predict_state()` supports checkpoint templates and custom state.
- `TrainingPlan` and `TrainState` remain public advanced contracts for custom compiled execution.
- `BasePhiModule` and `PhiModule` provide overridable fit and prediction hooks without owning optimizers or balancers.
- `PhiDataModule` owns host data and explicit-key batch sources. The Trainer owns device placement.
- `ModelFactory` and `InitializedModel` let any JAX architecture expose a pure apply callable, state, and summary.
- `LossBalancer` supports arbitrary JAX-compatible state and exposes diagnostics without a prescribed state layout.
- Derivative, equation, objective, callback, logger, evaluation, and artifact APIs can be reused across projects.
- Hydra instantiation helpers and OmegaConf resolvers are available under `phijax.integrations` without coupling the
  Trainer to Hydra.

Only names documented in package `__all__` declarations are supported public imports.

## Checkpoints and prediction artifacts

PhiJAX saves complete training state in versioned checkpoints and produces portable prediction artifacts for
downstream evaluation. Restoration raises an error when a checkpoint version is incompatible.

## Development

Install all development dependencies, then enable the repository hooks:

```bash
uv sync --group all
uv run --no-sync pre-commit install
```

Run the local validation suite before submitting a change:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync mkdocs build --strict
uv run --no-sync pre-commit run --all-files
```

See the [documentation](https://hangjung97.github.io/PhiJAX/) for the complete API and extension guides.

## Contributing

Contributions are welcome. Read the
[contribution guide](https://github.com/HangJung97/PhiJAX/blob/main/CONTRIBUTING.md) for the development workflow,
testing expectations, documentation requirements, and pull-request checklist.

## License

PhiJAX is distributed under the [Apache License 2.0](LICENSE).

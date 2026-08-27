# PhiJAX

[![Tests](https://github.com/HangJung97/PhiJAX/actions/workflows/tests.yaml/badge.svg)](https://github.com/HangJung97/PhiJAX/actions/workflows/tests.yaml)
[![Documentation](https://github.com/HangJung97/PhiJAX/actions/workflows/docs.yaml/badge.svg)](https://github.com/HangJung97/PhiJAX/actions/workflows/docs.yaml)
[![Python](https://img.shields.io/pypi/pyversions/phijax)](https://pypi.org/project/phijax/)
[![PyPI](https://img.shields.io/pypi/v/phijax)](https://pypi.org/project/phijax/)
[![License](https://img.shields.io/github/license/HangJung97/PhiJAX)](LICENSE)

PhiJAX is a typed [JAX](https://docs.jax.dev/) framework for physics-informed neural networks. It provides functional training state,
Lightning-inspired `Trainer` and `PhiModule` lifecycle contracts, explicit-key data sampling, reusable differential
equations and objectives, adaptive loss balancing, callbacks, logging, checkpointing, and prediction artifacts.

For a configuration-first project layout, start from
[`phijax-hydra-template`](https://github.com/HangJung97/phijax-hydra-template) and customize its entrypoints, Hydra
configs, and application DataModules.

> **Beta API:** PhiJAX `0.1.0b1` establishes the first supported public contracts. Minor API changes may still occur
> before `1.0`; checkpoint compatibility is enforced within the producing PhiJAX major/minor line.

## Why PhiJAX?

JAX provides powerful building blocks for automatic differentiation, vectorization, compilation, and accelerator
execution. PhiJAX builds on them with the structure needed to develop reproducible and maintainable PINN experiments:

- selective coordinate derivatives and reusable PDE, boundary, and data-fidelity equations;
- named objective terms with static, gradient-norm, and exact-NTK loss balancing;
- standard MLP, gated Modified MLP, and adaptive-residual PirateNet architectures with a custom NNX adapter;
- explicit model, optimizer, balancer, and PRNG state for reproducible compiled updates;
- familiar, Lightning-inspired lifecycles for trainers, modules, DataModules, callbacks, loggers, and checkpoints; and
- scientific data preparation on the CPU, automatic batch placement by the Trainer, and prediction files with a
  stable, versioned format.

PhiJAX preserves JAX's functional model and transparent transformations while taking care of the orchestration shared
by most PINN applications. Hydra is available as an optional integration, so the core Trainer API remains independent
of any configuration framework.

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

## Core workflow

Construct runtime objects in ordinary Python or through an external configuration system. The trainer owns source
preparation, device placement, lifecycle dispatch, teardown, and checkpoint restoration; the numerical update remains
a pure compiled function represented by `TrainingPlan`.

```python
import jax

from phijax import Trainer

# Select the runtime policy and accelerator.
trainer = Trainer(max_steps=1_000, accelerator="auto")

# Combine the model, optimizer, balancer, and PRNG state into one functional state.
initial_state = trainer.initialize_state(model_state, optimizer, balancer.initialize(), jax.random.key(0))

# Train with batches supplied and placed through the application DataModule.
result = trainer.fit(
    module,
    training_plan,
    initial_state,
    datamodule=data_module,
    sampling_key=jax.random.key(1),
    balancer_key=jax.random.key(2),
)

# Reuse the fitted in-memory state for prediction.
predictions = trainer.predict(module, result.state, datamodule=data_module)
```

Application DataModules implement `setup()`, `train_batch_source()`, and optionally `predict_batch_source()`.
Prediction is skipped cleanly when a DataModule has no prediction source.

`Trainer` also supports use as a context manager when an application wants deterministic cleanup at the end of a
custom orchestration scope.

## Supported contracts

- `Trainer`, `TrainingPlan`, and `TrainState` keep orchestration separate from compiled numerical state.
- `BasePhiModule` and `PhiModule` provide overridable fit and prediction hooks without owning optimizers or balancers.
- `PhiDataModule` owns host data and explicit-key batch-source construction while Trainer owns device placement.
- `InitializedModel` lets any JAX model factory return an apply callable, functional state, and optional summary.
- `LossBalancer` supports arbitrary JAX-compatible state and exposes diagnostics without a prescribed state layout.
- Selective derivative, equation, objective, callback, logger, evaluation, and artifact APIs are reusable across projects.
- Hydra instantiation helpers and OmegaConf resolvers are available under `phijax.integrations` without coupling the
  Trainer to Hydra.

Only names documented in package `__all__` declarations are supported public imports.

## Persistence

Checkpoints contain a manifest with the schema version, PhiJAX version, step, and state identifier. Restoration rejects
checkpoints from an incompatible PhiJAX major/minor line instead of silently loading an unsafe state layout.

Prediction artifacts use schema version 2 and store physical predictions, targets when available, model inputs,
output scales, masks, and application-provided metadata in a stable host-readable format.

## Development

Install the development and documentation dependency groups, then enable the repository hooks:

```bash
uv sync --group dev --group docs
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

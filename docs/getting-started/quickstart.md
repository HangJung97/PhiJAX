# Heat-equation quickstart

This example trains a small physics-informed neural network for the one-dimensional heat equation

$$
\frac{\partial u}{\partial t} - 0.1\frac{\partial^2 u}{\partial x^2} = 0,
\qquad
u(0,x) = \sin(\pi x),
\qquad
u(t,0) = u(t,1) = 0.
$$

Its analytical solution is

$$
u(t,x) = \exp\!\left(-0.1\pi^2 t\right)\sin(\pi x).
$$

The example needs no downloaded data and runs on CPU in a few seconds after JAX finishes its initial compilation.

## Run it

From a source checkout:

```bash
uv sync
JAX_PLATFORMS=cpu uv run --no-sync python examples/quickstart.py
```

The environment variable makes the entire process strictly CPU-only. `--accelerator cpu` selects CPU for Trainer-owned
state and batches, but does not change the default backend used by arrays created before `Trainer.fit()`.

The final line reports the completed optimizer steps, fitting time, relative L2 error, and maximum absolute error on
an ordered grid. The demo computes both errors with PhiJAX's `regression_metrics()` function. Values may vary across
JAX platforms. The default seed makes repeated runs on the same platform deterministic.

Use fewer steps while checking an installation:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python examples/quickstart.py --max-steps 5
```

Disable terminal progress when redirecting output or running under a batch scheduler:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python examples/quickstart.py --no-progress-bar
```

The script defaults to CPU and `32-true` precision. Select another available accelerator or precision mode explicitly:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python examples/quickstart.py --accelerator gpu --precision bf16-mixed
```

Available accelerators are `auto`, `cpu`, `gpu`, and `tpu`. Available precision modes are `64-true`, `32-true`,
`16-true`, `bf16-true`, `16-mixed`, and `bf16-mixed`. Use `64-true` only when the selected JAX build supports x64.
Disabling preallocation prevents JAX from reserving most GPU memory at startup. This is useful on shared or
memory-constrained GPUs; see [Troubleshooting](../guides/troubleshooting.md#cuda-runs-out-of-memory-during-startup).

## What the example demonstrates

1. `HeatDataModule.setup("fit")` creates finite initial-condition, boundary, and collocation pools on the host.
2. A partial `build_mlp()` factory retains architecture options until the Trainer supplies its model key, precision,
   and normalization statistics.
3. `CompositeObjective.from_equations()` turns two supervised equations and the heat residual into named losses and
   infers their batch keys.
4. `PhiModule` combines the lazy model factory and objective without owning optimizer or balancer state.
5. `Trainer.fit()` creates equal static loss weights, initializes `TrainState`, compiles the update, requests named
   batches, and places them on the selected device.
6. `RichModelSummary` prints the initialized architecture, and `RichProgressBar` displays fit and prediction progress
   with a refresh interval of 10 steps. Without explicit replacements, the Trainer uses `ModelSummary` and
   `TQDMProgressBar`; the `enable_model_summary` and `enable_progress_bar` flags disable the corresponding display.
7. `Trainer.predict()` consumes the returned `FitResult` and assembles finite prediction chunks.
8. `regression_metrics()` compares the in-memory predictions with the analytical reference values.

The equation uses `value_and_jacobian()` for `du/dt` and `hessian_diagonal()` for only `d2u/dx2`; it does not construct
the unused full Hessian.

## Try other callbacks

Callbacks are regular Python objects passed in dispatch order. For example, add learning-rate monitoring and early
stopping alongside the progress display:

```python
from phijax.callbacks import EarlyStopping, LearningRateMonitor, RichModelSummary, RichProgressBar

learning_rate = optax.exponential_decay(1.0e-3, transition_steps=1_000, decay_rate=0.9)
callbacks = (
    RichModelSummary(),
    LearningRateMonitor(learning_rate, log_key_prefix="train/"),
    EarlyStopping(monitor="train/loss", patience=2_000),
    RichProgressBar(total=max_steps, refresh_rate=10),
)
trainer = Trainer(max_steps=max_steps, callbacks=callbacks, logger=True)
optimizer = optax.adam(learning_rate)
```

`RichModelSummary` prints the initialized architecture, `LearningRateMonitor` contributes metrics, `EarlyStopping` may
request a clean stop, and `RichProgressBar` replaces the default progress display. See
[Callbacks](../api/callbacks.md) for lifecycle hooks, checkpointing, model summaries, and prediction writing.

`LearningRateMonitor` requires a configured logger. The default `logger=True` writes to
`phijax_logs/version_N`; use `tensorboard --logdir phijax_logs` when the TensorBoard extra is installed.

## Complete source

The source below is executed by the documentation smoke test, so it stays aligned with the public API.

```{.python}
--8<-- "examples/quickstart.py"
```

## Next steps

- [Core concepts](concepts.md) explains how the objects fit together.
- [Building a DataModule](../guides/datasets.md) develops the data layer in detail.
- [Building equations and objectives](../guides/objectives.md) explains residual groups and equation composition.
- [Creating a loss balancer](../guides/balancers.md) covers fixed and adaptive weighting policies.
- [Models](../api/models.md) shows how to select or implement a network architecture.

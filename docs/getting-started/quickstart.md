# Heat-equation quickstart

This example trains a small physics-informed neural network for the one-dimensional heat equation

\[
\\frac{\\partial u}{\\partial t} - 0.1\\frac{\\partial^2 u}{\\partial x^2} = 0,
\\qquad
u(0,x) = \\sin(\\pi x),
\\qquad
u(t,0) = u(t,1) = 0.
\]

Its analytical solution is

\[
u(t,x) = \\exp!\\left(-0.1\\pi^2 t\\right)\\sin(\\pi x).
\]

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
2. `build_mlp()` returns a pure application callable and explicit Flax NNX state.
3. `ResidualTerm` turns two supervised equations and the heat residual into three named losses.
4. `StaticLossBalancer` combines those losses without becoming part of `PhiModule`.
5. `TrainingPlan` declares the named batches needed by the compiled update.
6. `Trainer.fit()` requests batches from the DataModule, places them on the selected device, and runs the update.
7. `Trainer.predict()` reuses the fitted state and assembles finite prediction chunks.
8. `regression_metrics()` compares the in-memory predictions with the analytical reference values.

The equation uses `value_and_jacobian()` for `du/dt` and `hessian_diagonal()` for only `d2u/dx2`; it does not construct
the unused full Hessian.

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

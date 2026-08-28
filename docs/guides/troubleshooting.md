# Troubleshooting

## Useful JAX environment settings

Set runtime environment variables before Python starts so they take effect before JAX initializes a backend.

| Variable                        | Typical value | Purpose                                                        |
| ------------------------------- | ------------- | -------------------------------------------------------------- |
| `JAX_PLATFORMS`                 | `cpu`         | Restrict backend initialization and make CPU the default.      |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false`       | Allocate GPU memory as needed instead of reserving most of it. |
| `JAX_DEFAULT_MATMUL_PRECISION`  | `highest`     | Set process-wide dot and convolution precision.                |

Use `JAX_PLATFORMS` with an `S`; JAX deprecates the older `JAX_PLATFORM_NAME` option. The official
[JAX configuration reference](https://docs.jax.dev/en/latest/config_options.html) lists all supported settings.

PhiJAX can scope matrix-multiplication precision to training and prediction:

```python
trainer = Trainer(max_steps=1_000, matmul_precision="highest")
```

When both settings are present, an explicit Trainer value takes precedence during `fit()` and `predict()`. PhiJAX
restores `JAX_DEFAULT_MATMUL_PRECISION` afterward. Set `matmul_precision=None` to preserve the environment setting
throughout Trainer execution. See the [JAX matrix-multiplication precision guide](https://docs.jax.dev/en/latest/201/precision.html)
for the accuracy and performance trade-offs.

## JAX reports an NVIDIA GPU but falls back to CPU

The default PhiJAX installation contains CPU-capable JAX. If a machine has an NVIDIA GPU but no CUDA-enabled JAX
plugin, JAX warns and continues on CPU.

Install exactly one matching accelerator extra:

```bash
python -m pip install "phijax[cuda12]"
# or
python -m pip install "phijax[cuda13]"
```

For an intentional CPU run, select the backend before Python starts:

```bash
JAX_PLATFORMS=cpu python your_experiment.py
```

## JAX logs that the TPU backend is unavailable

Backend discovery may report that `libtpu` is unavailable on a non-TPU machine. This is informational when the
selected Trainer accelerator is CPU or GPU. Confirm the active backend with:

```bash
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

## CUDA runs out of memory during startup

JAX normally preallocates most GPU memory to reduce fragmentation. On a shared or memory-constrained GPU, disable
preallocation before starting Python:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python your_experiment.py
```

Alternatively, reserve a smaller fraction:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python your_experiment.py
```

These settings trade predictable allocation for lower initial memory use. They must be set before JAX initializes its
backend. See the official [JAX GPU memory allocation guide](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
for allocator behavior and additional options.

## The first training step is much slower

The first call includes JAX tracing and XLA compilation. Compare steady-state step time only after compilation. New
array shapes, dtypes, or PyTree structures can trigger another compilation, so training batch sources should keep
those properties fixed between steps.

## A DataModule reports that another stage is active

One DataModule cannot own `fit` and `predict` simultaneously. Pass it to `Trainer.fit()` or `Trainer.predict()` and let
the Trainer prepare and tear down the stage. When manually calling `prepare_stage()`, pair it with
`teardown_stage()` before starting a different stage.

## Prediction returns `None`

This is expected when `predict_batch_source()` returns `None`. Prediction hooks are skipped cleanly for a fit-only
DataModule. Implement both `predict_batch_source()` and `prediction_pool()` when prediction is required.

## A checkpoint cannot be restored

PhiJAX checkpoints include a version manifest and are restored only within a compatible PhiJAX major/minor line.
Verify that the model-state structure, optimizer, balancer, precision policy, and installed PhiJAX version match the
checkpoint. Use weights-only restoration when intentionally changing optimizer or balancer policy.

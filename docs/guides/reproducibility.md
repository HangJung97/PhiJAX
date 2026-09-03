# Randomness and reproducibility

PhiJAX follows JAX's explicit-key model for pseudorandom numbers. `Trainer.fit()` accepts either a Python integer or
one unbatched JAX key:

```python
result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=42,
)
```

The equivalent typed-key form is:

```python
import jax

result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=jax.random.key(42),
)
```

Passing a seed to the Trainer does not modify Python's or NumPy's global random state. If application-side data
preparation uses either global generator, seed it separately or pass an explicit generator into the DataModule.

See [Pseudorandom numbers](https://docs.jax.dev/en/latest/101/random.html) for the reasoning behind JAX's functional
key model.

## Independent streams

The common `fit()` path splits the root key once in a stable order:

```text
root key
   |
   +---> model
   +---> runtime
   +---> sampling
   +---> balancer
```

Each stream has one responsibility:

| Stream     | Ownership and purpose                                                                                   |
| ---------- | ------------------------------------------------------------------------------------------------------- |
| `model`    | Passed once to the model factory for parameter and variable initialization                              |
| `runtime`  | Stored as `TrainState.rng_key` and advanced for custom stochastic model execution                       |
| `sampling` | Stored as `TrainState.sampling_key` and passed to the DataModule's training batch source                |
| `balancer` | Stored as `TrainState.balancer_key` and used for fixed adaptive-balancer diagnostic batches when needed |

Keeping these streams separate means that changing collocation sampling does not reuse the model-initialization key.
It also keeps adaptive-balancer diagnostics independent from ordinary training batches.

The standard PhiJAX architectures are deterministic unless an application explicitly implements stochastic model
execution. Advanced training plans can carry that behavior through `TrainState.rng_key`.

## Deterministic batch sampling

`NamedBatchSource` treats a training batch as a function of its root sampling key and global optimizer step:

```python
step_key = jax.random.fold_in(sampling_key, step)
```

It then splits `step_key` across named samplers in declaration order. Repeating the same step with the same source
configuration returns the same samples. The ordering is part of the reproducibility contract: adding or reordering a
sampler intentionally changes the derived sampler keys.

A finite-pool DataModule can preserve this behavior with the standard constructor:

```python
def train_batch_source(self, batch_keys, key):
    return NamedBatchSource.from_pools(
        self.pools,
        self.batch_sizes,
        key,
        names=batch_keys,
    )
```

Custom continuous samplers should also derive each step from the supplied key instead of reading a module-level or
global random generator.

## Checkpoint continuation

A full checkpoint contains `rng_key`, `sampling_key`, `balancer_key`, and the completed global step alongside model,
optimizer, and balancer state. PhiJAX restores that state before asking the DataModule to construct its training
source. The next source call therefore folds the restored sampling key with the restored step.

Under the same code, data, source ordering, and numerical configuration, resuming from step `N` produces the same
step-`N` training batch as an uninterrupted run. Using `weights_only=True` is different: it restores model weights but
keeps the fresh run's optimizer, balancer, PRNG, and step state.

## Deterministic GPU execution

Explicit keys make random choices repeatable, but GPU execution and compile-time autotuning can still select
non-deterministic operations or kernels. Launch a deterministic GPU experiment with both XLA options set before
Python imports JAX:

```bash
XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops --xla_gpu_autotune_level=0" \
  uv run --no-sync python train.py
```

`--xla_gpu_exclude_nondeterministic_ops` restricts execution to deterministic GPU implementations.
`--xla_gpu_autotune_level=0` disables timing-based kernel autotuning between compilations. These process-wide options
must be set before JAX initializes its backend, so the Trainer does not attempt to set them during construction. See
the [OpenXLA determinism guide](https://openxla.org/xla/determinism) and
[JAX compiler flag documentation](https://docs.jax.dev/en/latest/201/controlling-xla.html) for details.

Deterministic implementations can be slower. Compilation can also fail when XLA cannot lower an operation
deterministically.

## What a seed does not guarantee

An equal seed makes PhiJAX's key derivation and sampling repeatable. It does not promise bitwise-identical trained
parameters after changing:

- accelerator type or device count;
- precision or matrix-multiplication policy;
- JAX, XLA, model, optimizer, or sampler implementation;
- batch sizes, sampler declaration order, or training length; or
- application data and preprocessing.

Floating-point kernels may use different reduction orders across platforms. Compare metrics within appropriate
tolerances when validating CPU, GPU, and TPU runs.

## Explicit state API

`fit_state()` accepts a fully initialized `TrainState`. Use `initialize_train_state()` to create the three persistent
streams from one key, or provide `sampling_key` and `balancer_key` explicitly when an integration already owns the
split:

```python
state = initialize_train_state(
    model_state,
    optimizer,
    balancer.initialize(),
    runtime_key,
    sampling_key=sampling_key,
    balancer_key=balancer_key,
)
```

Do not reuse one key for unrelated random operations. Split or fold keys at the point where a new independent stream
is required.

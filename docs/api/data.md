# Data

PhiJAX keeps application data construction on the host and accelerator placement in the trainer. A DataModule owns
immutable pools and sampling policy; it does not need to know which device the trainer selected.

## DataModule lifecycle

```text
prepare_data()
setup("fit" | "predict")
train_batch_source(...) | predict_batch_source()
Trainer prepares source and places batches
teardown("fit" | "predict")
```

`prepare_data` may generate or obtain shared artifacts. Process-local NumPy pools belong in `setup`.
`prepare_stage(stage)` makes this pair idempotent until `teardown_stage(stage)` releases the application lifecycle.
The factory prepares the initial stage before model normalization statistics are requested; the Trainer then owns
source retrieval and teardown. Returning `None` from `predict_batch_source()` skips prediction. Available prediction
sources must be finite and repeatable; training sources may sample indefinitely using the global optimizer step.

::: phijax.data.PhiDataModule

## Host pools

`inputs` and `targets` are rank-two NumPy arrays with a shared sample axis. Unsupervised pools use a zero-width target
array. Array-valued `aux` fields share the sample axis. `reference_shape` and `flat_index` reconstruct dense predictions.

::: phijax.data.HostPool

::: phijax.data.input_statistics

::: phijax.data.reconstruct_predictions

## Prediction artifacts

Artifact serialization remains reusable independently of the callback lifecycle. The canonical NPZ schema stores
physical arrays, applied output scales, schema metadata, and safe application metadata. Optional MATLAB arrays use the
same physical value space and may use application-owned field names.

::: phijax.data.save_prediction_artifact

::: phijax.data.load_prediction_artifact

::: phijax.data.PredictionArtifact

::: phijax.data.to_matlab_prediction_arrays

## Samplers

| Name             | Source                           | Typical PINN use                         |
| ---------------- | -------------------------------- | ---------------------------------------- |
| `random_rows`    | Finite aligned pool              | Initial, boundary, or supervised samples |
| `uniform_domain` | Continuous rectangular bounds    | Fresh PDE collocation coordinates        |
| `space_time`     | Fixed spatial mesh + fresh times | Time-dependent PDE collocation           |

`prepare(device)` moves persistent arrays once. `sample(key, batch_size)` remains explicitly keyed and returns a fixed
leading dimension.

::: phijax.data.BatchSampler

::: phijax.data.RandomRowSampler

::: phijax.data.UniformDomainSampler

::: phijax.data.SpaceTimeSampler

::: phijax.data.create_sampler

::: phijax.data.samplers.sample_pool

## Batch sources

`NamedBatchSource` folds its root key with the global step and splits keys in sampler declaration order. This preserves
sampling reproducibility after checkpoint restoration. The `"all"` batch-size policy is limited to finite row sources.

`ChunkedPredictionSource` pads its final host batch and supplies a Boolean `mask`; `Trainer.predict` removes padded rows
before output assembly.

::: phijax.data.TrainingBatchSource

::: phijax.data.PredictionBatchSource

::: phijax.data.NamedBatchSource

::: phijax.data.ChunkedPredictionSource

## Array IO and declarative pools

`load_arrays` supports NPZ, MATLAB, and HDF5 files. Classic MATLAB files are decoded by SciPy, while `mat73` restores
logical MATLAB arrays from v7.3 HDF5 storage; generic HDF5 input retains its native storage order. `get_array` accepts
slash-delimited nested keys. Declarative pool construction is useful for direct field mapping; subclass
`PhiDataModule` when generation or sampling is application specific.

::: phijax.data.load_arrays

::: phijax.data.get_array

::: phijax.data.build_array_pools

## Host transforms

Reusable transforms operate only on NumPy arrays during host-side data construction. They derive statistics from
finite values, preserve non-finite entries, and accept explicit statistics when one training-derived transform must be
reused for prediction. Application-specific physical nondimensionalization remains in the application DataModule.

::: phijax.data.log_compress

::: phijax.data.minmax_scale

::: phijax.data.standardize

::: phijax.data.scale_by_max

See [Building an application DataModule](../guides/datasets.md) for a complete implementation.

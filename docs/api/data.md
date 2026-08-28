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

`prepare_data` may generate or obtain shared artifacts. Build process-local NumPy pools in `setup`.
`prepare_stage(stage)` runs these methods at most once until `teardown_stage(stage)` releases the stage. The factory
prepares the initial stage before requesting model normalization statistics. The Trainer then retrieves sources and
handles teardown. Return `None` from `predict_batch_source()` to skip prediction. Prediction sources must be finite and
repeatable; training sources may sample indefinitely from the global optimizer step.

::: phijax.data.PhiDataModule

## Host pools

`inputs` and `targets` are rank-two NumPy arrays with a shared sample axis. Unsupervised pools use a zero-width target
array. Array-valued `aux` fields share the sample axis. `reference_shape` and `flat_index` reconstruct dense predictions.

::: phijax.data.HostPool

::: phijax.data.input_statistics

::: phijax.data.reconstruct_predictions

## Prediction artifacts

Prediction artifacts can be saved and loaded without callbacks. The canonical NPZ stores physical arrays, output
scales, schema information, and safe application metadata. Optional MATLAB files store the same physical values and
may use application-specific field names.

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

`NamedBatchSource` folds its root key with the global step, then splits keys in sampler declaration order. A restored
checkpoint therefore continues the same sampling sequence. The `"all"` batch-size policy works only with finite row
sources.

`ChunkedPredictionSource` pads its final host batch and supplies a Boolean `mask`; `Trainer.predict` removes padded rows
before output assembly.

::: phijax.data.TrainingBatchSource

::: phijax.data.PredictionBatchSource

::: phijax.data.NamedBatchSource

::: phijax.data.ChunkedPredictionSource

## Array IO and declarative pools

`load_arrays` supports NPZ, MATLAB, and HDF5 files. SciPy reads classic MATLAB files. `mat73` restores MATLAB's logical
array order from v7.3 HDF5 files, while generic HDF5 input keeps its stored order. `get_array` accepts slash-delimited
nested keys. Use declarative pool construction for direct field mapping. Subclass `PhiDataModule` for custom generation
or sampling.

::: phijax.data.load_arrays

::: phijax.data.get_array

::: phijax.data.build_array_pools

## Host transforms

Reusable transforms operate on NumPy arrays while building host data. They derive statistics from finite values and
preserve non-finite entries. They also accept fixed statistics, so prediction can reuse values derived from training
data. Keep application-specific physical scaling in the application DataModule.

::: phijax.data.log_compress

::: phijax.data.minmax_scale

::: phijax.data.standardize

::: phijax.data.scale_by_max

See [Building an application DataModule](../guides/datasets.md) for a complete implementation.

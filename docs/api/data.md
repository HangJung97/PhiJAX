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
`prepare_stage(stage)` runs these methods at most once until `teardown_stage(stage)` releases the stage. The Trainer
prepares fitting before requesting optional model normalization statistics, then retrieves sources and handles
teardown. The base `input_statistics()` returns `None`; override it to opt into input normalization. The Trainer stores
the returned statistics in `TrainState`, so a full resume and prediction from a checkpoint use the statistics the model
was trained with. See [input normalization statistics](checkpointing.md#input-normalization-statistics). Override
`input_periods()` to declare the physical period of periodic input axes; see
[periodic features](models/layers.md#periodic-features). Return `None` from
`predict_batch_source()` to skip prediction. Prediction sources must be finite and repeatable; training sources may
sample indefinitely from the global optimizer step.

### Multi-process runs

In a run with several JAX processes, the Trainer prepares data in three steps:

1. Global process zero calls `prepare_data()`. No other process calls it.
2. Every process waits at `Strategy.barrier()` until process zero has finished.
3. Every process calls `setup()` and builds its own pools.

`prepare_data()` therefore runs once per run, and no process reads a file that is still being written. A single-process
run is unaffected: process zero prepares the data, and the barrier returns immediately.

Write prepared files to storage that every host can read, such as a shared filesystem. If the files are on
process zero's local disk, `setup()` fails on the other hosts because they cannot read them. Keep `prepare_data()` free
of state that `setup()` needs, such as attributes, because other processes never run it.

Lightning defaults to `prepare_data_per_node=True`, which runs `prepare_data()` on one process per machine. PhiJAX
always uses global process zero instead. JAX usually runs one process per host, so a per-machine rule would make
every process prepare the same files.

::: phijax.data.PhiDataModule

## Host pools

`inputs` is a rank-two NumPy array. Omit `targets` for an automatically created zero-width target array. `aux` and
`metadata` default to immutable empty mappings. When omitted, `reference_shape` becomes `(sample_count,)` and
`flat_index` becomes `arange(sample_count)`. Array-valued `aux` fields share the sample axis.

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

`NamedBatchSource.from_pools()` is the concise path for aligned finite datasets. It creates ordered
`RandomRowSampler` instances and validates requested pool names and batch-size policies.

Define one batch-size policy per pool in the DataModule, then reuse the objective's inferred order:

```python
self.batch_sizes = {"initial": "all", "boundary": 128, "pde": 4096}


def train_batch_source(self, batch_keys, key):
    return NamedBatchSource.from_pools(self.pools, self.batch_sizes, key, names=batch_keys)
```

An integer samples that many rows on every optimizer step. The `"all"` policy uses every row and is available only
for finite pools.

`ChunkedPredictionSource` pads its final host batch and supplies a Boolean `mask`; prediction removes padded rows
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

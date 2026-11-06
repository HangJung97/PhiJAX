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

`prepare_data` may generate or obtain shared artifacts. Build process-local NumPy pools in `setup`. The Trainer runs
this lifecycle once per `fit()` or `predict()` call, and always calls `teardown()`, even after a failed `setup()`. It
sets up fitting before requesting optional model normalization statistics, then retrieves sources. The base `input_statistics()` returns `None`; override it to opt into input normalization. The Trainer stores
the returned statistics in `TrainState`, so a full resume and prediction from a checkpoint use the statistics the model
was trained with. See [input normalization statistics](checkpointing.md#input-normalization-statistics). Override
`input_periods()` to declare the physical period of periodic input axes; see
[periodic features](models/layers.md#periodic-features). Return `None` from
`predict_batch_source()` to skip prediction. Prediction sources must be finite and repeatable; training sources may
sample indefinitely from the global optimizer step.

### Default sources

Most DataModules only build pools. Pass the source sizes to `PhiDataModule.__init__()` and override `setup()`:

| Hook                                  | Default                                                                                        |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `train_batch_source(batch_keys, key)` | Random rows from the pool named by each batch key, sized by `batch_sizes`                      |
| `prediction_pool()`                   | The pool named `predict`, or `None`                                                            |
| `predict_batch_source()`              | `batch_sizes["predict"]` rows per device from `prediction_pool()`, or `None` without that pool |

`batch_sizes["predict"]` may be `"all"`, which predicts the whole pool in one chunk split over the devices. This gives
one compiled shape and no chunk loop, but the whole grid must fit in device memory in one forward pass. In training,
`"all"` instead gives every device the full pool.

A default source raises an error when a size is missing or a batch key names no pool. Override the hook for custom
sampling, such as a [`SpaceTimeSampler`](#phijax.data.SpaceTimeSampler).

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

A pool is a set of rows. `inputs` is a rank-two NumPy array. Omit `targets` for an automatically created zero-width
target array. `aux` and `metadata` default to immutable empty mappings. When omitted, `reference_shape` becomes
`(sample_count,)` and `flat_index` becomes `arange(sample_count)`. Array-valued `aux` fields share the sample axis.

[`HostPool.from_grid()`](#phijax.data.HostPool.from_grid) builds a pool from dense grid arrays shaped
`[*grid, features]`. An optional Boolean `mask` with shape `grid` keeps only part of the grid, such as a masked
physical domain. The pool records the grid in `reference_shape` and the kept positions in `flat_index`, so
[`reconstruct_predictions()`](#phijax.data.reconstruct_predictions) returns `[*grid, outputs]` with `NaN` outside the
mask.

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

| Sampler                | Source                           | Typical PINN use                         |
| ---------------------- | -------------------------------- | ---------------------------------------- |
| `RandomRowSampler`     | Finite aligned pool              | Initial, boundary, or supervised samples |
| `UniformDomainSampler` | Continuous rectangular bounds    | Fresh PDE collocation coordinates        |
| `SpaceTimeSampler`     | Fixed spatial mesh + fresh times | Time-dependent PDE collocation           |

Construct samplers directly in `train_batch_source()`.

`prepare(device)` moves persistent arrays once. `sample(key, batch_size)` remains explicitly keyed and returns a fixed
leading dimension.

::: phijax.data.BatchSampler

::: phijax.data.RandomRowSampler

::: phijax.data.UniformDomainSampler

::: phijax.data.SpaceTimeSampler

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

An integer samples that many rows per device on every optimizer step. The `"all"` policy uses every row, is available
only for finite pools, and places the whole pool on every device. See [batch sizes](strategies.md#batch-sizes).

`ChunkedPredictionSource` pads its final host batch to `batch_size`. Each batch carries a Boolean `valid_predict_mask`
that is `True` for pool rows and `False` for padding, and prediction removes padded rows before output assembly. The
name is reserved: a prediction pool with an auxiliary array named `valid_predict_mask` raises `ValueError`.

### Custom batch sources

A custom source implements [`TrainingBatchSource`](#phijax.data.TrainingBatchSource) or
[`PredictionBatchSource`](#phijax.data.PredictionBatchSource). To support per-device batch sizes on
several devices, also define `for_devices(count)`, returning a copy whose integer sizes are multiplied by `count`. The
Trainer passes this process's device count for training sources and the total device count for prediction sources.
Without the method, a training source's batches are the rows of the whole process and must divide by its device count.
See [device strategies](strategies.md).

::: phijax.data.TrainingBatchSource

::: phijax.data.PredictionBatchSource

::: phijax.data.NamedBatchSource

::: phijax.data.ChunkedPredictionSource

## Array IO

`load_arrays` supports NPZ, MATLAB, and HDF5 files. NPZ needs only NumPy. MATLAB and HDF5 files require the `io` extra,
`pip install "phijax[io]"`. SciPy reads classic MATLAB files. `mat73` restores MATLAB's logical
array order from v7.3 HDF5 files, while generic HDF5 input keeps its stored order. `get_array` accepts slash-delimited
nested keys. Build pools from the loaded arrays with NumPy and `HostPool` or `HostPool.from_grid()`.

::: phijax.data.load_arrays

::: phijax.data.get_array

## Host transforms

Reusable transforms operate on NumPy arrays while building host data. They derive statistics from finite values and
preserve non-finite entries. They also accept fixed statistics, so prediction can reuse values derived from training
data. Keep application-specific physical scaling in the application DataModule.

::: phijax.data.log_compress

::: phijax.data.minmax_scale

::: phijax.data.standardize

::: phijax.data.scale_by_max

See [Building an application DataModule](../guides/datasets.md) for a complete implementation.

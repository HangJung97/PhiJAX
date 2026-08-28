# Building an application DataModule

This guide assumes you have run the [heat-equation quickstart](../getting-started/quickstart.md). It expands that
example to cover data setup, storage, sampling, normalization, prediction, and testing.

The core data API is deliberately small. `PhiDataModule` defines the training and prediction lifecycle. Each
application defines its datasets, pool names, sampling policy, and constructor options.

This follows two principles:

- framework code should not need to understand an application's files, geometry, targets, or boundary extraction; and
- application APIs should expose experiment choices rather than the mechanics of a universal data schema.

`phijax.data` provides immutable pools, array IO, optional pool builders, device placement, samplers, and prediction
chunking. An application DataModule uses only the pieces it needs.

## The `PhiDataModule` contract

An application subclasses `phijax.data.PhiDataModule`, implements setup and training-source construction, and may
provide prediction hooks:

```python
class PhiDataModule(ABC):
    def prepare_data(self) -> None: ...

    @abstractmethod
    def setup(self, stage: DataStage) -> None: ...

    @abstractmethod
    def train_batch_source(
        self,
        batch_keys: tuple[str, ...],
        key: jax.Array,
    ) -> TrainingBatchSource: ...

    def predict_batch_source(self) -> PredictionBatchSource | None: ...

    def prediction_pool(self) -> HostPool | None: ...
```

The remaining lifecycle and normalization methods are:

- `prepare_stage(stage)`, for idempotently running `prepare_data()` and `setup(stage)`;
- `teardown_stage(stage)`, for releasing the active stage exactly once;
- `prepare_data()`, for generating or downloading an artifact before loading it;
- `prediction_pool()`, for declaring the ordered host pool represented by prediction chunks;
- `normalization_pools()`, for selecting the coordinates used to derive network input mean and standard deviation;
- `input_statistics()`, for overriding pool-derived normalization with exact continuous-distribution statistics; and
- `teardown(stage)`, for releasing application-owned resources.

Use NumPy in `prepare_data()` and host-pool builders so they do not initialize JAX. DataModules create host-backed
samplers and prediction chunks. The Trainer prepares sampler state on the selected device and places or shards each
batch before compiled execution.

Pass the DataModule to `Trainer.fit()` so the Trainer can request, place, and iterate its training source. Pass it to
`Trainer.predict()` for the same prediction lifecycle. Return `None` from `predict_batch_source()` when no prediction
data exists. The Trainer will skip prediction callbacks and module hooks.

Like Lightning, `setup(stage)` assigns process-local data state instead of returning it. PhiJAX stores this state in
the DataModule's host-side `pools` mapping. The source hooks are not called `dataloader` because they do not imply
PyTorch datasets or worker processes.

## Reusable host transforms

Application DataModules can compose NumPy-only transforms from `phijax.data` while constructing pools:

```python
from phijax.data import log_compress, minmax_scale, scale_by_max, standardize

power_weight = log_compress(power, percentile=99.0, dynamic_range=40.0)
normalized_temperature = minmax_scale(temperature, minimum=train_min, maximum=train_max)
standardized_signal = standardize(signal, mean=train_mean, std=train_std)
bounded_confidence = scale_by_max(confidence, percentile=99.5)
```

You can derive `minimum`, `maximum`, `mean`, and `std` from training data once and reuse them for prediction. These
generic transforms do not replace physical nondimensionalization. Keep characteristic physical scales explicit in the
application DataModule.

## `HostPool` fields

Every pool is an immutable `HostPool`:

| Field             | Required shape or meaning                                                          |
| ----------------- | ---------------------------------------------------------------------------------- |
| `inputs`          | Rank-two `[samples, input_features]` NumPy array                                   |
| `targets`         | Rank-two `[samples, target_features]`; use `[samples, 0]` for an unsupervised pool |
| `aux`             | Sample-wise arrays such as weights, normals, periods, or material coefficients     |
| `metadata`        | Structural information that is not transferred into compiled training              |
| `reference_shape` | Dense grid dimensions used to reconstruct flat predictions                         |
| `flat_index`      | Integer indices mapping pool rows into the flattened reference grid                |

`HostPool` copies and freezes its arrays. Every NumPy array in `aux` must share the leading input row count.

Batch-stream names are part of the application API. Every objective `batch_key` must be handled by
`train_batch_source()` and have a batch-size policy. A stream may select a finite pool or generate coordinates from a
continuous domain:

```text
ResidualTerm(batch_key="pde")
              |
              +--> DataModule training source
              +--> per-stream batch-size policy
```

## Pool size and batch size

Keep construction sizes separate from runtime batch sizes:

```python
pde_size = 16_384
batch_size = {"pde": 4_096}
```

These values mean:

```text
pde_size          = finite candidate coordinates stored in the host pool
batch_size["pde"] = candidate rows evaluated during one optimizer step
```

For a continuously generated sampler such as `uniform_domain`, no finite `pde_size` exists. In that case,
`batch_size["pde"]` is the number of newly generated coordinates per optimizer step.

An application DataModule can expose a `pde_sampling` constructor option without changing its objective. In `fixed`
mode, the DataModule builds `pde_size` candidates and selects random rows. In `uniform` mode, it creates no
`pde` pool. Instead, it samples a fresh batch from the time-space bounds at each optimizer step. The explicit root key
and global step make this sequence reproducible after resuming.

For independent uniform coordinates on intervals `[a, b]`, normalization does not require a surrogate finite pool.
An application can override `PhiDataModule.input_statistics()` with the exact values
`mean = (a + b) / 2` and `std = (b - a) / sqrt(12)`.

Prediction batch size is a chunking policy, not a prediction-grid size. The grid size comes from `reference_shape`;
`batch_size["predict"]` only limits the number of rows evaluated at once.

## Application DataModule pattern

A DataModule should make four application decisions explicit:

- which immutable host pools exist for fitting and prediction;
- which sampler serves each objective `batch_key`;
- how large each training batch and prediction chunk is; and
- which coordinates define model-input normalization.

The [executable heat-equation quickstart](https://github.com/HangJung97/PhiJAX/blob/main/examples/quickstart.py)
implements this complete pattern. Use it as a starting point, then replace its analytic coordinates and targets with
your application's geometry and data.

Applications may factor host-only construction into a separate `data.py`. This keeps geometry and file parsing
independently testable while the application DataModule remains responsible for lifecycle policy.

## Construct the DataModule

Once lifecycle decisions live in the application class, construction contains only meaningful experiment parameters:

```python
data_module = HeatDataModule(
    seed=42,
    initial_size=256,
    boundary_size=256,
    pde_size=16_384,
    predict_shape=(101, 256),
    batch_size={
        "initial": "all",
        "boundary": 128,
        "pde": 4_096,
        "predict": 4_096,
    },
)
```

The Trainer only requires a `PhiDataModule`. It does not need to know how pools or sources were created.

## Reusing array IO inside an application

Applications backed by NPZ, MATLAB, or HDF5 artifacts can call these helpers from `setup()`:

- `load_arrays()` loads `.npz`, classic `.mat`, HDF5-backed `.mat`, `.h5`, `.hdf`, and `.hdf5` artifacts;
- `get_array()` selects nested fields with slash-delimited keys; and
- `build_array_pools()` assembles coordinate grids, slices, source fields, constants, auxiliary fields, and finite
  uniformly sampled pools.

These are construction utilities, not a framework DataModule. The application still chooses the file keys, pool names,
objective mappings, and samplers.

An application may use `build_array_pools()` internally while exposing only meaningful choices such as `data_path`,
`pde_sampling`, `pde_size`, and `batch_size` through its constructor or project settings.

## Sampler choices

PhiJAX provides three immutable explicit-key samplers:

| Class                  | Explicit name    | Behavior                                                                |
| ---------------------- | ---------------- | ----------------------------------------------------------------------- |
| `RandomRowSampler`     | `random_rows`    | Select aligned rows from a finite pool; supports the exact `all` policy |
| `UniformDomainSampler` | `uniform_domain` | Generate fresh continuous coordinates from bounds                       |
| `SpaceTimeSampler`     | `space_time`     | Combine fixed spatial rows with freshly generated temporal coordinates  |

The data package separates three tasks. `phijax.data.samplers` selects or generates a batch.
`phijax.data.sources` supplies named batches during training or prediction. `phijax.data.batching` defines shared
batch-size and prediction-layout rules. Public names are re-exported from `phijax.data`.

An application may construct these classes directly or expose a short-name option and call `create_sampler()`. Sampler
configuration is application policy; it is not required in every data config.

`NamedBatchSource` starts with host-backed sampler definitions. Before fitting, the Trainer moves persistent candidate
arrays, bounds, templates, and the root key to the Strategy's root device. Sampling then stays on the device. The
Trainer applies final precision conversion and data-parallel sharding to each batch.

The source folds its root key with the global optimizer step. Repeating one step is deterministic, and a resumed run
continues the same sampling sequence from its restored global step.

`ChunkedPredictionSource` is finite and can be iterated more than once. It keeps the full prediction pool on the host,
then slices and pads one fixed-size chunk at a time. `source.pool` exposes the original pool for dense reconstruction.
The Trainer transfers one chunk at a time and copies valid outputs back to the host.

The default `PredictionWriter` receives the pool and joined outputs through `PredictionContext`. Keep
`return_predictions=True` when using it. A streaming callback can instead write each batch from
`on_predict_batch_end`. Only set `return_predictions=False` for this streaming case.

## Testing an application DataModule

At minimum, test:

- valid `fit` and `predict` setup stages;
- pool names, coordinate order, target order, shapes, and dtypes;
- deterministic pool generation for equal seeds;
- initial and boundary values;
- finite sampler alignment and exact `all` behavior;
- generated sampler bounds when used;
- prediction padding and dense reconstruction order;
- empirical or exact input-statistics policy; and
- invalid sizes, bounds, stages, and prediction chunk sizes.

Use synthetic CPU-sized fixtures. Ordinary tests must not download data, require a GPU, or generate the full numerical
reference artifact.

Continue with [Building equations and objectives](objectives.md) to connect these batch names to scalar losses.

For Hydra-based application assembly, see the
[PhiJAX Hydra template](https://github.com/HangJung97/phijax-hydra-template) and the
[configuration integration API](../api/configuration.md).

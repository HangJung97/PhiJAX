# Building an application DataModule

PhiJAX keeps its core data abstraction deliberately small. `PhiDataModule` defines the lifecycle required by training
and prediction, while each application owns its dataset construction, pool names, sampler policy, and meaningful
configuration fields.

This follows two principles:

- framework code should not need to understand an application's files, geometry, targets, or boundary extraction; and
- application configuration should describe experiment choices rather than the mechanics of a universal data schema.

The reusable `phijax.data` package still provides immutable pools, array IO, declarative array-building helpers, device
placement, finite and generated samplers, and prediction chunking. An application DataModule composes only the helpers
it needs.

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
- `normalization_pools()`, for selecting the coordinates used to derive network input mean and standard deviation; and
- `input_statistics()`, for overriding pool-derived normalization with exact continuous-distribution statistics; and
- `teardown(stage)`, for releasing application-owned resources.

`prepare_data()` and host-pool builders should use NumPy and avoid initializing JAX. DataModules construct host-backed
sampler definitions and prediction chunks; the Trainer prepares persistent sampler state on its Strategy's root device
and places or shards every batch before compiled execution.

Calling `Trainer.fit(module, training_plan, state, datamodule=data_module, sampling_key=key)` lets the Trainer request,
place, and iterate the training source. Calling `Trainer.predict(module, state, datamodule=data_module)` does the same
for prediction. Return `None` from `predict_batch_source()` when an application has no prediction data; callbacks and
module prediction hooks are then skipped.

Like Lightning, `setup(stage)` assigns process-local data state rather than returning it. PhiJAX stores that state as
the explicit `pools` mapping on the host-side DataModule. Unlike Lightning, the two source hooks are not called
`dataloader` because they do not imply PyTorch datasets, worker processes, or `torch.utils.data.DataLoader` behavior.

## Reusable host transforms

Application DataModules can compose NumPy-only transforms from `phijax.data` while constructing pools:

```python
from phijax.data import log_compress, minmax_scale, scale_by_max, standardize

power_weight = log_compress(power, percentile=99.0, dynamic_range=40.0)
normalized_temperature = minmax_scale(temperature, minimum=train_min, maximum=train_max)
standardized_signal = standardize(signal, mean=train_mean, std=train_std)
bounded_confidence = scale_by_max(confidence, percentile=99.5)
```

Statistics supplied through `minimum`, `maximum`, `mean`, and `std` may be derived from training data once and reused
for prediction. These generic transforms do not replace physical nondimensionalization: characteristic length,
velocity, time, pressure, or material scales remain explicit application rules in the DataModule.

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
model.objective.terms.<term>.batch_key
                    │
                    ├── finite HostPool or generated-domain sampler
                    └── data.batch_size key
```

## Pool size and batch size

Keep construction sizes separate from runtime batch sizes:

```yaml
pde_sampling: fixed
pde_size: 16384

batch_size:
  pde: 4096
```

These values mean:

```text
pde_size           = finite candidate coordinates stored in the host pool
batch_size.pde     = candidate rows evaluated during one optimizer step
```

For a continuously generated sampler such as `uniform_domain`, no finite `pde_size` exists. In that case,
`batch_size.pde` is the number of newly generated coordinates per optimizer step.

For example, an application DataModule can switch policies without changing its objective:

```bash
python -m my_project.train experiment=heat_static_1d data.pde_sampling=uniform
```

In `fixed` mode, it constructs `pde_size` candidates and selects random rows. In `uniform` mode, it does not construct
a `pde` host pool; instead, it samples a reproducible fresh batch from the time-space bounds for every global optimizer
step. Its explicit root key and restored global step ensure resumed training follows the same coordinate sequence.

For independent uniform coordinates on intervals `[a, b]`, normalization does not require a surrogate finite pool.
An application can override `PhiDataModule.input_statistics()` with the exact values
`mean = (a + b) / 2` and `std = (b - a) / sqrt(12)`.

Prediction batch size is a chunking policy, not a prediction-grid size. The grid size comes from `reference_shape`;
`batch_size.predict` only limits the number of rows evaluated at once.

## Example application DataModule

The following compact heat-equation module owns finite initial, boundary, PDE, and prediction pools:

```python
from collections.abc import Mapping

import jax
import numpy as np

from phijax.data import (
    ChunkedPredictionSource,
    HostPool,
    NamedBatchSource,
    PhiDataModule,
    RandomRowSampler,
)
from phijax.data.datamodule import DataStage


class HeatDataModule(PhiDataModule):
    """Own heat-equation data construction and batching."""

    def __init__(
        self,
        batch_size: Mapping[str, int | str],
        *,
        seed: int = 42,
        initial_size: int = 256,
        boundary_size: int = 256,
        pde_size: int = 16384,
        predict_shape: tuple[int, int] = (101, 256),
    ) -> None:
        """Store application data policy.

        Args:
            batch_size: Per-pool training and prediction policies.
            seed: PDE candidate-pool seed.
            initial_size: Initial-condition candidate count.
            boundary_size: Boundary candidate count.
            pde_size: Interior candidate count.
            predict_shape: Dense `(time, space)` prediction shape.
        """
        super().__init__()
        self.batch_size = dict(batch_size)
        self.seed = seed
        self.initial_size = initial_size
        self.boundary_size = boundary_size
        self.pde_size = pde_size
        self.predict_shape = predict_shape

    def setup(self, stage: DataStage) -> None:
        """Construct and store immutable application pools.

        Args:
            stage: Requested `fit` or `predict` stage.

        """
        if stage not in ("fit", "predict"):
            raise ValueError("Heat data stage must be `fit` or `predict`.")
        generator = np.random.default_rng(self.seed)
        initial_x = np.linspace(0.0, 1.0, self.initial_size, dtype=np.float32)
        initial_inputs = np.column_stack((np.zeros_like(initial_x), initial_x))
        initial_targets = np.sin(np.pi * initial_x)[:, None].astype(np.float32)

        boundary_t = np.linspace(0.0, 1.0, self.boundary_size, dtype=np.float32)
        boundary_x = np.where(np.arange(self.boundary_size) % 2 == 0, 0.0, 1.0).astype(np.float32)
        boundary_inputs = np.column_stack((boundary_t, boundary_x))

        pde_inputs = generator.uniform(0.0, 1.0, (self.pde_size, 2)).astype(np.float32)
        times = np.linspace(0.0, 1.0, self.predict_shape[0], dtype=np.float32)
        positions = np.linspace(0.0, 1.0, self.predict_shape[1], dtype=np.float32)
        prediction_inputs = np.stack(np.meshgrid(times, positions, indexing="ij"), axis=-1).reshape(-1, 2)

        self.pools = {
            "initial": self._pool(initial_inputs, initial_targets),
            "boundary": self._pool(boundary_inputs, np.zeros((self.boundary_size, 1), dtype=np.float32)),
            "pde": self._pool(pde_inputs, np.zeros((self.pde_size, 0), dtype=np.float32)),
            "predict": self._pool(
                prediction_inputs,
                np.zeros((prediction_inputs.shape[0], 0), dtype=np.float32),
                reference_shape=self.predict_shape,
            ),
        }

    def train_batch_source(
        self,
        batch_keys: tuple[str, ...],
        key: jax.Array,
    ) -> NamedBatchSource:
        """Build deterministic finite-row training samplers.

        Args:
            batch_keys: Objective batch keys.
            key: Explicit root sampling key.

        Returns:
            Unprepared global-step-indexed named batch source.
        """
        pools = self._require_setup("fit")
        samplers = {name: RandomRowSampler(pools[name].fields()) for name in batch_keys}
        sizes = {name: self.batch_size[name] for name in batch_keys}
        return NamedBatchSource(samplers, sizes, key)

    def predict_batch_source(self) -> ChunkedPredictionSource:
        """Build a lazy host-backed prediction source.

        Returns:
            Re-iterable padded prediction source.
        """
        size = self.batch_size["predict"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("Prediction batch size must be a positive integer.")
        return ChunkedPredictionSource(self.prediction_pool(), size)

    def prediction_pool(self) -> HostPool:
        """Return the dense ordered prediction pool.

        Returns:
            Host pool carrying dense reconstruction metadata.
        """
        return self._require_setup("predict")["predict"]

    def normalization_pools(self) -> Mapping[str, HostPool]:
        """Use interior candidates to derive input statistics.

        Returns:
            Interior PDE pool mapping.
        """
        pools = self._require_setup("fit")
        return {"pde": pools["pde"]}

    @staticmethod
    def _pool(
        inputs: np.ndarray,
        targets: np.ndarray,
        *,
        reference_shape: tuple[int, ...] | None = None,
    ) -> HostPool:
        """Construct one indexed heat-equation pool.

        Args:
            inputs: Coordinate rows.
            targets: Aligned target rows.
            reference_shape: Optional dense reconstruction shape.

        Returns:
            Immutable host pool.
        """
        row_count = inputs.shape[0]
        return HostPool(
            inputs=inputs,
            targets=targets,
            aux={},
            metadata={"coordinate_names": ("t", "x")},
            reference_shape=reference_shape or (row_count,),
            flat_index=np.arange(row_count, dtype=np.int64),
        )
```

Applications may factor host-only construction into a separate `data.py`. This keeps geometry and file parsing
independently testable while the application DataModule remains responsible for lifecycle policy.

## Hydra configuration

Once lifecycle decisions live in the application class, its config contains only meaningful experiment parameters:

```yaml
_target_: my_project.applications.heat.HeatDataModule

seed: ${seed}
initial_size: 256
boundary_size: 256
pde_size: 16384
predict_shape: [101, 256]

batch_size:
  initial: all
  boundary: 128
  pde: 4096
  predict: 4096
```

Temporary comparisons remain direct:

```bash
python -m my_project.train experiment=heat_static_1d \
  data.pde_size=32768 data.batch_size.pde=8192
```

The factory verifies that the configured object implements `PhiDataModule`. The Trainer does not know how any
pool was created; it only recognizes the generic `prepare(device)` source contract and places each resulting batch.

## Reusing array IO inside an application

Applications backed by NPZ, MATLAB, or HDF5 artifacts can call these helpers from `setup()`:

- `load_arrays()` loads `.npz`, classic `.mat`, HDF5-backed `.mat`, `.h5`, `.hdf`, and `.hdf5` artifacts;
- `get_array()` selects nested fields with slash-delimited keys; and
- `build_array_pools()` assembles coordinate grids, slices, source fields, constants, auxiliary fields, and finite
  uniformly sampled pools.

These are construction utilities, not a framework DataModule. The application remains responsible for deciding which
file keys matter, how pool names map to objectives, and which sampler each pool uses.

An application may use `build_array_pools()` internally while exposing only meaningful choices such as `data_path`,
`pde_sampling`, `pde_size`, and `batch_size` through its project configuration.

## Sampler choices

PhiJAX provides three immutable explicit-key samplers:

| Class                  | Explicit name    | Behavior                                                                |
| ---------------------- | ---------------- | ----------------------------------------------------------------------- |
| `RandomRowSampler`     | `random_rows`    | Select aligned rows from a finite pool; supports the exact `all` policy |
| `UniformDomainSampler` | `uniform_domain` | Generate fresh continuous coordinates from bounds                       |
| `SpaceTimeSampler`     | `space_time`     | Combine fixed spatial rows with freshly generated temporal coordinates  |

The implementation keeps three responsibilities separate: `phijax.data.samplers` defines how one batch is selected
or generated, `phijax.data.sources` delivers named batches over training or prediction, and `phijax.data.batching`
contains shared batch-size and prediction-layout policies. Public classes and functions are re-exported from
`phijax.data`.

An application may construct these classes directly or expose a short-name option and call `create_sampler()`. Sampler
configuration is application policy; it is not required in every data config.

`NamedBatchSource` initially owns host-backed sampler definitions. Before fitting, the Trainer prepares persistent
candidate arrays, domain bounds, templates, and the root key once on the Strategy's process-local root device. Sampling
and generated-coordinate construction then remain device-side. The Trainer still applies final precision conversion
and data-parallel sharding to every produced batch.

The source folds its root key with the global optimizer step. Repeating one step is deterministic, and a resumed run
continues the same sampling sequence from its restored global step.

`ChunkedPredictionSource` is a finite re-iterable source. It keeps the full prediction pool on the host, slices and pads
one fixed-size chunk at a time, and exposes the original pool through `source.pool` for dense reconstruction. The
Trainer transfers each yielded chunk immediately before prediction, avoiding eager placement of a large prediction
grid. Following Lightning's prediction loop, PhiJAX copies each valid output chunk back to the host before retaining
it. The default `PredictionWriter` receives this pool and the final concatenated outputs through `PredictionContext`.
Keep `return_predictions=True` when using that writer. A custom streaming callback may instead write every batch from
`on_predict_batch_end`; use `return_predictions=False` only in that streaming case to avoid retaining the complete
output.

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

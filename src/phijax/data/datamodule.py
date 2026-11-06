from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal

import jax
from numpy.typing import NDArray

from phijax.data.batching import BatchSize
from phijax.data.pools import HostPool
from phijax.data.sources import ChunkedPredictionSource, NamedBatchSource, PredictionBatchSource, TrainingBatchSource

type DataStage = Literal["fit", "predict"]


class PhiDataModule(ABC):
    """Define a Lightning-inspired host data lifecycle for PhiJAX applications.

    Application subclasses construct immutable pools in :meth:`setup`. The default :meth:`train_batch_source` samples
    those pools with `batch_sizes`, and the default :meth:`predict_batch_source` chunks the `predict` pool with
    `batch_sizes["predict"]`. Override either hook for custom sampling. The module is mutable host-side orchestration
    state; model, optimizer, balancer, and PRNG state used by compiled training remain explicit and functional.

    The Trainer runs the lifecycle: :meth:`prepare_data` on global process zero, then :meth:`setup` and
    :meth:`teardown` around each `fit` or `predict` task.

    Attributes:
        pools: Immutable application pools populated by :meth:`setup`.
    """

    def __init__(self, *, batch_sizes: Mapping[str, BatchSize] | None = None) -> None:
        """Initialize an empty pool mapping and the default source sizes.

        Args:
            batch_sizes: Optional per-device batch size keyed by pool name, used by the default
                :meth:`train_batch_source` and :meth:`predict_batch_source`. The `predict` entry sets the prediction
                chunk size. Finite pools also accept `"all"`.
        """
        self.pools: Mapping[str, HostPool] = {}
        self.batch_sizes = None if batch_sizes is None else dict(batch_sizes)

    def prepare_data(self) -> None:
        """Prepare external artifacts before application pools are constructed.

        The base implementation performs no work. Subclasses may generate or download files. The Trainer calls this
        method only on global process zero, and every process waits for it before :meth:`setup`. Write files to storage
        every host can read, and do not assign attributes that :meth:`setup` needs: other processes never run this
        method. Runtime pools belong in :meth:`setup`, which runs on every process.
        """
        return None

    @abstractmethod
    def setup(self, stage: DataStage) -> None:
        """Populate immutable application pools for one execution stage.

        Args:
            stage: Requested `fit` or `predict` lifecycle stage.
        """

    def train_batch_source(
        self,
        batch_keys: tuple[str, ...],
        key: jax.Array,
    ) -> TrainingBatchSource:
        """Build a deterministic step-indexed training batch source.

        The default draws random rows from the pool named by each batch key, with the size from `batch_sizes`.

        Args:
            batch_keys: Objective batch keys in stable declaration order.
            key: Explicit root sampling key.

        Returns:
            Unprepared step-indexed source that the Trainer places before sampling.

        Raises:
            RuntimeError: If :meth:`setup` has not populated any pools.
            ValueError: If `batch_sizes` is `None`.
            KeyError: If a batch key names no pool or has no batch size.
        """
        pools = self._require_setup("fit")
        if self.batch_sizes is None:
            raise ValueError(
                f"`{type(self).__name__}` needs `batch_sizes` for the default `train_batch_source()`. Pass "
                "`batch_sizes` to `PhiDataModule.__init__()`, or override `train_batch_source()`."
            )
        missing_pools = [name for name in batch_keys if name not in pools]
        if missing_pools:
            raise KeyError(f"Batch keys {missing_pools} name no pool; available pools are {list(pools)}.")
        missing_sizes = [name for name in batch_keys if name not in self.batch_sizes]
        if missing_sizes:
            raise KeyError(f"Batch keys {missing_sizes} have no entry in `batch_sizes`.")
        return NamedBatchSource.from_pools(pools, self.batch_sizes, key, names=batch_keys)

    def predict_batch_source(self) -> PredictionBatchSource | None:
        """Build an optional finite ordered host-backed prediction batch source.

        The default chunks :meth:`prediction_pool` into `batch_sizes["predict"]` rows per device. With `"all"`, one
        chunk holds every row and is split over the devices.

        Returns:
            Re-iterable source whose batches the Trainer places before prediction, or `None` when this DataModule does
            not provide prediction data.

        Raises:
            ValueError: If a prediction pool exists but `batch_sizes` has no `predict` entry, or that entry is not a
                positive integer or `"all"`.
        """
        pool = self.prediction_pool()
        if pool is None:
            return None
        if self.batch_sizes is None or "predict" not in self.batch_sizes:
            raise ValueError(
                f"`{type(self).__name__}` needs `batch_sizes['predict']` for the default `predict_batch_source()`. "
                "Pass it to `PhiDataModule.__init__()`, or override `predict_batch_source()`."
            )
        return ChunkedPredictionSource(pool, self.batch_sizes["predict"])

    def prediction_pool(self) -> HostPool | None:
        """Return the optional ordered host pool represented by prediction batches.

        Returns:
            The pool named `predict`, or `None` when no such pool is set up.
        """
        return self.pools.get("predict")

    def normalization_pools(self) -> Mapping[str, HostPool]:
        """Select host pools used to derive shared model input statistics.

        The default excludes the prediction source's pool so a dense evaluation grid does not dominate
        training-coordinate statistics. Applications may override this policy when one pool best represents the
        physical domain.

        Returns:
            Non-empty training-oriented host-pool mapping.

        Raises:
            ValueError: If excluding the prediction pool leaves no pools.
        """
        prediction = self.prediction_pool()
        selected = dict(self.pools)
        if prediction is not None:
            selected = {name: pool for name, pool in selected.items() if pool is not prediction}
        if not selected:
            raise ValueError("Input normalization requires at least one non-prediction pool.")
        return selected

    def input_statistics(self) -> tuple[NDArray[Any], NDArray[Any]] | None:
        """Return optional model-input normalization statistics.

        The base implementation disables input normalization. Applications can opt in by overriding this method and
        returning empirical statistics from :func:`phijax.data.input_statistics` or exact statistics for a continuous
        sampling distribution.

        The Trainer stores the statistics in the training state. A full resume and prediction from a checkpoint use the
        stored statistics; fresh and weights-only runs use this method. For transfer learning with the source model's
        statistics, return :func:`phijax.training.load_input_statistics` here.

        Returns:
            Per-coordinate mean and safely positive standard deviation arrays, or `None` to skip normalization.
        """
        return None

    def input_periods(self) -> Mapping[int, float] | None:
        """Return optional physical periods of periodic input axes.

        Models with periodic features embed each declared axis as `cos(2*pi*x/T)` and `sin(2*pi*x/T)` of the physical
        coordinate, independently of input normalization. Return a period computed from the data, such as a measured
        cardiac cycle, or a fixed one, such as `2*pi` for an angle. A model's `periodic_features_kwargs['periods']`
        overrides or adds axes.

        The Trainer stores the periods the model applies in the training state. A full resume and prediction from a
        checkpoint use the stored periods; fresh and weights-only runs use this method.

        Returns:
            Physical period keyed by input axis index, or `None` when no axis is periodic.
        """
        return None

    def teardown(self, stage: DataStage) -> None:
        """Release application resources after one lifecycle stage.

        Args:
            stage: Completed `fit` or `predict` lifecycle stage.
        """
        del stage
        return None

    def _require_setup(self, stage: DataStage) -> Mapping[str, HostPool]:
        """Return populated pools or report a missing lifecycle call.

        Args:
            stage: Stage whose dataloader or statistics were requested.

        Returns:
            Populated immutable host-pool mapping.

        Raises:
            RuntimeError: If :meth:`setup` has not populated any pools.
        """
        if not self.pools:
            raise RuntimeError(f"Data pools are unavailable. Call `setup('{stage}')` first.")
        return self.pools


__all__ = ["DataStage", "PhiDataModule"]

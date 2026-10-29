from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal

import jax
from numpy.typing import NDArray

from phijax.data.pools import HostPool
from phijax.data.sources import PredictionBatchSource, TrainingBatchSource

type DataStage = Literal["fit", "predict"]


class PhiDataModule(ABC):
    """Define a Lightning-inspired host data lifecycle for PhiJAX applications.

    Application subclasses construct immutable pools in :meth:`setup` and expose JAX-compatible sources through
    :meth:`train_batch_source` and :meth:`predict_batch_source`. The module is mutable host-side orchestration state;
    model, optimizer, balancer, and PRNG state used by compiled training remain explicit and functional.

    Attributes:
        pools: Immutable application pools populated by :meth:`setup`.
        prepared_stage: Currently prepared lifecycle stage, or `None` outside a Trainer task.
    """

    def __init__(self) -> None:
        """Initialize an empty application pool mapping."""
        self.pools: Mapping[str, HostPool] = {}
        self.prepared_stage: DataStage | None = None

    def prepare_data(self) -> None:
        """Prepare external artifacts before application pools are constructed.

        The base implementation performs no work. Subclasses may generate or download files but should not assign
        process-local runtime state here. Runtime pools belong in :meth:`setup`.
        """
        return None

    @abstractmethod
    def setup(self, stage: DataStage) -> None:
        """Populate immutable application pools for one execution stage.

        Args:
            stage: Requested `fit` or `predict` lifecycle stage.
        """

    def prepare_stage(self, stage: DataStage) -> None:
        """Prepare data and set up one stage exactly once until teardown.

        Args:
            stage: Requested `fit` or `predict` lifecycle stage.

        Raises:
            BaseException: Re-raises preparation or setup errors after best-effort application teardown.
        """
        if self.prepared_stage == stage:
            return
        if self.prepared_stage is not None:
            active = self.prepared_stage
            raise RuntimeError(f"DataModule stage `{active}` is still active; tear it down before `{stage}`.")
        try:
            self.prepare_data()
            self.setup(stage)
            self.prepared_stage = stage
        except BaseException:
            self.teardown(stage)
            raise

    def teardown_stage(self, stage: DataStage) -> None:
        """Tear down one prepared stage and clear its lifecycle marker.

        Args:
            stage: Completed `fit` or `predict` lifecycle stage.
        """
        if self.prepared_stage != stage:
            return
        try:
            self.teardown(stage)
        finally:
            self.prepared_stage = None

    @abstractmethod
    def train_batch_source(
        self,
        batch_keys: tuple[str, ...],
        key: jax.Array,
    ) -> TrainingBatchSource:
        """Build a deterministic step-indexed training batch source.

        Args:
            batch_keys: Objective batch keys in stable declaration order.
            key: Explicit root sampling key.

        Returns:
            Unprepared step-indexed source that the Trainer places before sampling.
        """

    def predict_batch_source(self) -> PredictionBatchSource | None:
        """Build an optional finite ordered host-backed prediction batch source.

        Returns:
            Re-iterable source whose batches the Trainer places before prediction, or `None` when this DataModule does
            not provide prediction data.
        """
        return None

    def prediction_pool(self) -> HostPool | None:
        """Return the optional ordered host pool represented by prediction batches.

        Returns:
            Prediction pool carrying reconstruction metadata and host targets, or `None` when prediction is absent.
        """
        return None

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

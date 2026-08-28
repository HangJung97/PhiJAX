import jax
import numpy as np
import pytest

from phijax.data import HostPool, PhiDataModule
from phijax.data.datamodule import DataStage
from phijax.data.samplers import RandomRowSampler
from phijax.data.sources import ChunkedPredictionSource, NamedBatchSource


def _pool(offset: float = 0.0) -> HostPool:
    """Build one compact immutable pool.

    Args:
        offset: Constant added to every coordinate.

    Returns:
        Four-row host pool.
    """
    inputs = np.arange(8, dtype=np.float32).reshape(4, 2) + offset
    return HostPool(
        inputs=inputs,
        targets=np.zeros((4, 0), dtype=np.float32),
        aux={},
        metadata={"coordinate_names": ("t", "x")},
        reference_shape=(2, 2),
        flat_index=np.arange(4),
    )


class _MinimalDataModule(PhiDataModule):
    """Implement the abstract contract for base lifecycle tests."""

    def setup(self, stage: DataStage) -> None:
        """Build and store separate training and prediction pools.

        Args:
            stage: Requested lifecycle stage.
        """
        del stage
        self.pools = {"train": _pool(), "predict": _pool(10.0)}

    def train_batch_source(
        self,
        batch_keys: tuple[str, ...],
        key: jax.Array,
    ) -> NamedBatchSource:
        """Build finite row samplers for requested batch keys.

        Args:
            batch_keys: Requested training pool names.
            key: Root sampler key.

        Returns:
            Unprepared step-indexed batch source.
        """
        pools = self._require_setup("fit")
        samplers = {name: RandomRowSampler(pools[name].fields()) for name in batch_keys}
        return NamedBatchSource(samplers, dict.fromkeys(batch_keys, 2), key)

    def predict_batch_source(self) -> ChunkedPredictionSource:
        """Build a lazy host-backed source for the prediction pool.

        Returns:
            Two-row fixed prediction source.
        """
        return ChunkedPredictionSource(self.prediction_pool(), 2)

    def prediction_pool(self) -> HostPool:
        """Return the fixture prediction pool.

        Returns:
            Dense host prediction pool.
        """
        return self._require_setup("predict")["predict"]


class _FailingDataModule(_MinimalDataModule):
    """Raise during setup while recording best-effort lifecycle cleanup."""

    def __init__(self) -> None:
        """Initialize the base lifecycle and teardown record."""
        super().__init__()
        self.torn_down_stage: DataStage | None = None

    def setup(self, stage: DataStage) -> None:
        """Raise one synthetic setup error.

        Args:
            stage: Requested lifecycle stage.

        Raises:
            RuntimeError: Always.
        """
        del stage
        raise RuntimeError("synthetic setup failure")

    def teardown(self, stage: DataStage) -> None:
        """Record cleanup after failed setup.

        Args:
            stage: Failed lifecycle stage.
        """
        self.torn_down_stage = stage


class _FitOnlyDataModule(PhiDataModule):
    """Provide training data without implementing prediction hooks."""

    def setup(self, stage: DataStage) -> None:
        """Populate one training pool.

        Args:
            stage: Requested lifecycle stage.
        """
        del stage
        self.pools = {"train": _pool()}

    def train_batch_source(self, batch_keys: tuple[str, ...], key: jax.Array) -> NamedBatchSource:
        """Build random-row sources for the requested training pools.

        Args:
            batch_keys: Requested pool names.
            key: Root sampling key.

        Returns:
            Named random-row batch source.
        """
        pools = self._require_setup("fit")
        samplers = {name: RandomRowSampler(pools[name].fields()) for name in batch_keys}
        return NamedBatchSource(samplers, dict.fromkeys(batch_keys, 2), key)


def test_phi_data_module_is_a_minimal_abstract_lifecycle() -> None:
    """Verify the public base cannot be instantiated without application lifecycle methods."""
    with pytest.raises(TypeError):
        PhiDataModule()  # type: ignore[abstract]


def test_phi_data_module_default_lifecycle_selects_non_prediction_pools() -> None:
    """Verify default preparation, normalization selection, teardown, and application delegation."""
    data_module = _MinimalDataModule()
    assert data_module.prepare_data() is None
    data_module.prepare_stage("fit")
    prepared_pools = data_module.pools
    data_module.prepare_stage("fit")
    assert data_module.pools is prepared_pools
    assert data_module.prepared_stage == "fit"
    with pytest.raises(RuntimeError, match="still active"):
        data_module.prepare_stage("predict")
    assert tuple(data_module.normalization_pools()) == ("train",)
    mean, std = data_module.input_statistics()
    np.testing.assert_allclose(mean, np.asarray([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(std, np.sqrt(5.0))
    assert data_module.train_batch_source(("train",), jax.random.key(1))(0)["train"]["inputs"].shape == (2, 2)
    assert len(data_module.predict_batch_source()) == 2
    assert data_module.teardown_stage("fit") is None
    assert data_module.prepared_stage is None
    data_module.prepare_stage("predict")
    assert data_module.prepared_stage == "predict"
    data_module.teardown_stage("predict")


def test_phi_data_module_supports_fit_without_prediction_data() -> None:
    """Verify prediction hooks are optional and all fit pools contribute to normalization."""
    data_module = _FitOnlyDataModule()
    data_module.prepare_stage("fit")

    assert data_module.predict_batch_source() is None
    assert data_module.prediction_pool() is None
    assert tuple(data_module.normalization_pools()) == ("train",)


def test_phi_data_module_sources_require_setup() -> None:
    """Verify source construction reports a missing setup lifecycle call."""
    data_module = _MinimalDataModule()
    with pytest.raises(RuntimeError, match=r"setup\('fit'\)"):
        data_module.train_batch_source(("train",), jax.random.key(1))
    with pytest.raises(RuntimeError, match=r"setup\('predict'\)"):
        data_module.predict_batch_source()


def test_phi_data_module_tears_down_after_stage_setup_failure() -> None:
    """Verify lifecycle preparation performs best-effort cleanup and leaves no active stage."""
    data_module = _FailingDataModule()

    with pytest.raises(RuntimeError, match="synthetic setup failure"):
        data_module.prepare_stage("fit")

    assert data_module.torn_down_stage == "fit"
    assert data_module.prepared_stage is None

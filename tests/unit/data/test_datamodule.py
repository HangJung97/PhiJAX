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


def test_phi_data_module_hooks_select_non_prediction_pools() -> None:
    """Verify default preparation, optional normalization, teardown, and application delegation."""
    data_module = _MinimalDataModule()
    assert data_module.prepare_data() is None
    data_module.setup("fit")

    assert tuple(data_module.normalization_pools()) == ("train",)
    assert data_module.input_statistics() is None
    assert data_module.input_periods() is None
    assert data_module.train_batch_source(("train",), jax.random.key(1))(0)["train"]["inputs"].shape == (2, 2)
    assert len(data_module.predict_batch_source()) == 2
    assert data_module.teardown("fit") is None


def test_phi_data_module_supports_fit_without_prediction_data() -> None:
    """Verify prediction hooks are optional and all fit pools contribute to normalization."""
    data_module = _FitOnlyDataModule()
    data_module.setup("fit")

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


class _DefaultDataModule(PhiDataModule):
    """Use only the default source hooks."""

    def setup(self, stage: DataStage) -> None:
        """Populate two training pools and one prediction pool.

        Args:
            stage: Requested lifecycle stage.
        """
        del stage
        self.pools = {"a": _pool(), "b": _pool(5.0), "predict": _pool(10.0)}


def test_phi_data_module_default_sources_use_constructor_sizes() -> None:
    """Verify the default sources sample with `batch_sizes`, whose `predict` entry chunks the `predict` pool."""
    data_module = _DefaultDataModule(batch_sizes={"a": 3, "b": "all", "predict": 3})
    data_module.setup("fit")

    source = data_module.train_batch_source(("b", "a"), jax.random.key(0))
    batch = source(0)
    prediction = data_module.predict_batch_source()

    assert isinstance(source, NamedBatchSource)
    assert tuple(batch) == ("b", "a")
    assert batch["a"]["inputs"].shape == (3, 2)
    np.testing.assert_array_equal(np.sort(batch["b"]["inputs"], axis=0), _pool(5.0).inputs)
    assert data_module.prediction_pool() is data_module.pools["predict"]
    assert isinstance(prediction, ChunkedPredictionSource)
    assert prediction.batch_size == 3
    assert len(prediction) == 2
    assert tuple(data_module.normalization_pools()) == ("a", "b")


def test_phi_data_module_default_sources_report_missing_sizes_and_keys() -> None:
    """Verify each default source names what it is missing."""
    data_module = _DefaultDataModule()
    data_module.setup("fit")
    with pytest.raises(ValueError, match="needs `batch_sizes`"):
        data_module.train_batch_source(("a",), jax.random.key(0))
    with pytest.raises(ValueError, match=r"needs `batch_sizes\['predict'\]`"):
        data_module.predict_batch_source()

    data_module = _DefaultDataModule(batch_sizes={"a": 2})
    data_module.setup("fit")
    with pytest.raises(KeyError, match=r"\['c'\] name no pool"):
        data_module.train_batch_source(("a", "c"), jax.random.key(0))
    with pytest.raises(KeyError, match=r"\['b'\] have no entry in `batch_sizes`"):
        data_module.train_batch_source(("a", "b"), jax.random.key(0))
    with pytest.raises(ValueError, match=r"needs `batch_sizes\['predict'\]`"):
        data_module.predict_batch_source()

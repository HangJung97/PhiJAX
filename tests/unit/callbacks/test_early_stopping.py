import jax.numpy as jnp
import pytest

from phijax.callbacks import EarlyStopping, TrainerContext


def test_early_stopping_tracks_improvement_and_patience() -> None:
    """Verify the callback stops after the configured number of non-improving updates."""
    callback = EarlyStopping(monitor="loss", patience=1, mode="min", min_delta=0.1)
    assert callback.on_train_batch_end(TrainerContext(None, 1, {"loss": jnp.asarray(2.0)})) is False
    assert callback.on_train_batch_end(TrainerContext(None, 2, {"loss": jnp.asarray(1.95)})) is False
    assert callback.on_train_batch_end(TrainerContext(None, 3, {"loss": jnp.asarray(1.96)})) is True


def test_early_stopping_rejects_missing_or_non_scalar_metrics() -> None:
    """Verify monitored values must exist and contain exactly one element."""
    callback = EarlyStopping(monitor="loss")
    with pytest.raises(KeyError, match="loss"):
        callback.on_train_batch_end(TrainerContext(None, 1, {}))
    with pytest.raises(ValueError, match="scalar"):
        callback.on_train_batch_end(TrainerContext(None, 1, {"loss": jnp.ones(2)}))

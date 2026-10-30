from __future__ import annotations

from collections.abc import Iterable, Mapping, Sized
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import numpy as np
from jax.experimental import multihost_utils

from phijax.callbacks import PredictionContext
from phijax.core import BasePhiModule, PhiModuleContext
from phijax.training.lifecycle import TaskLifecycle

if TYPE_CHECKING:
    from phijax.training.strategies import Strategy
    from phijax.training.trainer import Trainer


class _PredictionLoop:
    """Run the finite host prediction loop for one Trainer."""

    def __init__(self, trainer: Trainer) -> None:
        """Bind the loop to its owning Trainer.

        Args:
            trainer: Host runtime providing callbacks, placement, and logging.
        """
        self._trainer = trainer

    def run(
        self,
        module: BasePhiModule,
        model_state: Any,
        batches: Iterable[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
        return_predictions: bool = True,
    ) -> np.ndarray | None:
        """Run fixed-size prediction batches with callback lifecycle dispatch.

        Callbacks run before matching module hooks. Batch-end hooks receive each valid, unpadded output, while epoch-end
        and predict-end hooks receive the final concatenated host array when collection is enabled.

        With several devices, every batch is padded so its rows divide over all devices, and each process predicts an
        equal share. The outputs are gathered so every process receives the complete batch output as a host array in
        the original row order, for any batch size.

        Args:
            module: Application module exposing :meth:`phijax.core.BasePhiModule.predict_step`.
            model_state: Explicit restored model parameter PyTree.
            batches: Finite iterable of prediction batches, optionally containing a Boolean `valid_predict_mask`. In a
                multi-process run, every process must yield the same batches.
            metadata: Optional immutable task metadata exposed to prediction callbacks.
            return_predictions: Whether to collect and concatenate batch predictions on the host.

        Returns:
            Concatenated host predictions in batch iteration order, or `None` when collection is disabled.

        Raises:
            ValueError: If `batches` is empty.
            BaseException: Re-raises prediction or callback failures after cleanup.
        """
        trainer = self._trainer
        model_state = trainer.strategy.place_state(model_state)
        task_metadata = dict(metadata or {})
        prediction_pool = getattr(batches, "pool", None)
        total_batches = len(batches) if isinstance(batches, Sized) else None
        context = PredictionContext(
            outputs=None,
            batch_index=None,
            metadata=task_metadata,
            total_batches=total_batches,
            pool=prediction_pool,
            is_global_zero=trainer.strategy.is_global_zero,
        )
        module_context = PhiModuleContext(step=0, metrics={})
        lifecycle = TaskLifecycle(
            trainer.callbacks,
            module,
            trainer.logger,
            is_global_zero=trainer.strategy.is_global_zero,
        )
        outputs: list[np.ndarray] = []
        batch_count = 0
        # Pad and gather whenever batches are split over devices, so any batch size divides evenly.
        multi_device = len(trainer.strategy.devices) > 1
        try:
            lifecycle.setup()
            for callback in trainer.callbacks:
                callback.on_predict_start(context)
            module.on_predict_start(model_state, context)
            for callback in trainer.callbacks:
                callback.on_predict_epoch_start(context)
            module.on_predict_epoch_start(model_state, context)
            for batch_index, batch in enumerate(batches):
                share = _process_share(batch, trainer.strategy) if multi_device else None
                placed_batch = trainer.prepare_batch(batch if share is None else share.batch)
                context = PredictionContext(
                    outputs=None,
                    batch=placed_batch,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    metadata=task_metadata,
                    pool=prediction_pool,
                    is_global_zero=trainer.strategy.is_global_zero,
                )
                for callback in trainer.callbacks:
                    callback.on_predict_batch_start(context)
                module.on_predict_batch_start(model_state, context)
                prediction = module.predict_step(model_state, placed_batch)
                if share is not None:
                    prediction = _gather_rows(prediction, trainer.strategy.place_batch(share.rows), share.valid)
                elif (valid_rows := placed_batch.get("valid_predict_mask")) is not None:
                    prediction = prediction[valid_rows]
                batch_count += 1
                context = PredictionContext(
                    outputs=prediction,
                    batch=placed_batch,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    metadata=task_metadata,
                    pool=prediction_pool,
                    is_global_zero=trainer.strategy.is_global_zero,
                )
                for callback in trainer.callbacks:
                    callback.on_predict_batch_end(context)
                module.on_predict_batch_end(model_state, context)
                if return_predictions:
                    # Release accelerator outputs after every batch instead of retaining device buffers for the pass.
                    outputs.append(np.asarray(jax.device_get(prediction)))
            if batch_count == 0:
                raise ValueError("Prediction requires at least one batch.")
            combined = np.concatenate(outputs, axis=0) if return_predictions else None
            context = PredictionContext(
                outputs=combined,
                batch_index=None,
                total_batches=total_batches,
                metadata=task_metadata,
                pool=prediction_pool,
                is_global_zero=trainer.strategy.is_global_zero,
            )
            for callback in trainer.callbacks:
                callback.on_predict_epoch_end(context)
            module.on_predict_epoch_end(model_state, context)
            for callback in trainer.callbacks:
                callback.on_predict_end(context)
            module.on_predict_end(model_state, context)
            lifecycle.finalize("success")
            return combined
        except BaseException as error:
            lifecycle.handle_exception(error, context, module_context)
            lifecycle.finalize("failed")
            raise
        finally:
            lifecycle.teardown()


class _ProcessShare(NamedTuple):
    """Hold one process's share of a host prediction batch.

    Attributes:
        batch: This process's rows of every non-scalar leaf.
        rows: Ids of those rows in the padded batch.
        valid: Boolean validity of the unpadded batch rows.
    """

    batch: dict[str, Any]
    rows: np.ndarray
    valid: np.ndarray


def _process_share(batch: Mapping[str, Any], strategy: Strategy) -> _ProcessShare:
    """Select this process's equal share of a host prediction batch.

    The batch is padded so its rows divide evenly over every device of every process. Padded rows are invalid.

    Args:
        batch: Host batch with the same leading row count for every non-scalar leaf, identical on every process.
        strategy: Strategy defining the process and device layout.

    Returns:
        This process's padded rows, their ids, and the validity of the unpadded rows.
    """
    row_count = next(np.shape(leaf)[0] for leaf in jax.tree.leaves(batch) if np.ndim(leaf) > 0)
    local_devices = sum(device.process_index == strategy.process_index for device in strategy.devices)
    # Round each process's share up to a whole number of rows per local device.
    share_rows = -(-row_count // (strategy.process_count * local_devices)) * local_devices
    start = strategy.process_index * share_rows
    padding = share_rows * strategy.process_count - row_count

    def take(value: Any) -> Any:
        """Pad one leaf with zero rows and keep this process's share.

        Args:
            value: Batch leaf.

        Returns:
            Scalars unchanged, or this process's padded rows.
        """
        array = np.asarray(value)
        if array.ndim == 0:
            return array
        padded = np.pad(array, ((0, padding), *(((0, 0),) * (array.ndim - 1))))
        return padded[start : start + share_rows]

    valid = np.asarray(batch.get("valid_predict_mask", np.ones(row_count, dtype=bool)), dtype=bool)
    rows = np.arange(start, start + share_rows, dtype=np.int32)
    return _ProcessShare(jax.tree.map(take, dict(batch)), rows, valid)


def _gather_rows(prediction: jax.Array, rows: jax.Array, valid: np.ndarray) -> np.ndarray:
    """Gather a row-sharded prediction from every device and process and restore the original row order.

    Args:
        prediction: Global prediction array sharded along its leading row axis.
        rows: Global row ids placed with the same sharding as `prediction`.
        valid: Boolean validity of the unpadded batch rows.

    Returns:
        Valid predictions in batch row order, identical on every process.
    """
    values, ids = multihost_utils.process_allgather((prediction, rows), tiled=True)
    # The mesh may order processes' shards differently from their host order, so place rows by their ids.
    ordered = np.empty_like(np.asarray(values))
    ordered[np.asarray(ids)] = np.asarray(values)
    return ordered[: valid.size][valid]

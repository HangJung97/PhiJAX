from __future__ import annotations

from collections.abc import Iterable, Mapping, Sized
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from phijax.callbacks import PredictionContext
from phijax.core import BasePhiModule, PhiModuleContext
from phijax.training.lifecycle import TaskLifecycle

if TYPE_CHECKING:
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

        Args:
            module: Application module exposing :meth:`phijax.core.BasePhiModule.predict_step`.
            model_state: Explicit restored model parameter PyTree.
            batches: Finite iterable of prediction batches, optionally containing a Boolean `mask`.
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
        try:
            lifecycle.setup()
            for callback in trainer.callbacks:
                callback.on_predict_start(context)
            module.on_predict_start(model_state, context)
            for callback in trainer.callbacks:
                callback.on_predict_epoch_start(context)
            module.on_predict_epoch_start(model_state, context)
            for batch_index, batch in enumerate(batches):
                placed_batch = trainer.prepare_batch(batch)
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
                mask = placed_batch.get("mask")
                if mask is not None:
                    prediction = prediction[mask]
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

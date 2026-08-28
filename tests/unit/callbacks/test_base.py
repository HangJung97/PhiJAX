from phijax.callbacks import Callback, PostprocessingContext, PredictionContext, TrainerContext


def test_callback_default_hooks_accept_each_supported_lifecycle() -> None:
    """Verify the shared callback contract provides no-op hooks for every supported task lifecycle."""
    callback = Callback()
    trainer_context = TrainerContext(state=None, step=0, metrics={})
    prediction_context = PredictionContext(outputs=None, batch_index=None, metadata={})
    postprocessing_context = PostprocessingContext(value=None, metadata={})

    callback.setup()
    callback.on_fit_start(trainer_context)
    callback.on_train_batch_start(trainer_context)
    assert callback.on_train_batch_end(trainer_context) is False
    assert callback.training_metrics(trainer_context) == {}
    callback.on_fit_end(trainer_context)
    callback.on_predict_start(prediction_context)
    callback.on_predict_epoch_start(prediction_context)
    callback.on_predict_batch_start(prediction_context)
    callback.on_predict_batch_end(prediction_context)
    callback.on_predict_epoch_end(prediction_context)
    callback.on_predict_end(prediction_context)
    callback.on_postprocessing_start(postprocessing_context)
    callback.on_postprocessing_end(postprocessing_context)
    callback.on_exception(RuntimeError("test"), trainer_context)
    callback.teardown()
    assert prediction_context.batch is None
    assert prediction_context.total_batches is None


def test_training_package_does_not_export_callback_api() -> None:
    """Verify callbacks have one canonical public package rather than training compatibility aliases."""
    import phijax.training as training

    assert not hasattr(training, "Callback")
    assert not hasattr(training, "EarlyStopping")
    assert not hasattr(training, "TrainerContext")

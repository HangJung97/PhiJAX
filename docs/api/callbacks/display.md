# Display callbacks

The Trainer supplies plain terminal displays by default. Rich callbacks replace them when an application wants
formatted tables and progress output.

## Model summaries

`Trainer(enable_model_summary=True)` adds `ModelSummary` when no summary callback is supplied. Pass
`RichModelSummary` to replace it, or set `enable_model_summary=False` for no automatic summary. A Trainer rejects
multiple summary callbacks and prints summaries only on global rank zero.

::: phijax.callbacks.ModelSummary

::: phijax.callbacks.RichModelSummary

## Progress bars

When no progress callback is supplied, `Trainer(enable_progress_bar=True)` adds `TQDMProgressBar`. Set
`enable_progress_bar=False` for quiet library, test, or batch-scheduler execution.

All progress callbacks provide `enable()`, `disable()`, and `is_enabled` for temporary runtime control. Most
applications should use the Trainer option instead of managing callback state directly.

::: phijax.callbacks.ProgressBar

::: phijax.callbacks.TQDMProgressBar

Supplying `RichProgressBar`, which extends `ProgressBar`, replaces TQDM. A Trainer accepts at most one progress
callback, avoiding duplicate output. Both implementations refresh device metrics only at their configured interval.

By default, both displays show `train/loss`, every `train/loss/<name>`, every `train/weight/<name>`, and the primary
logger version as `v_num`. Learning rates and other diagnostics still reach loggers without entering the display.
Override `ProgressBar.get_metrics()` to change standard fields, or pass `metric_names` for an exact ordered selection.
During prediction, the callbacks use `PredictionContext.total_batches` without transferring prediction values to the
host.

::: phijax.callbacks.RichProgressBarTheme

::: phijax.callbacks.RichProgressBar

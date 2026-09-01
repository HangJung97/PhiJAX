# Prediction callbacks

Prediction callbacks observe placed batches and model outputs on the Python host. Use them for streaming or artifact
writing rather than changing the numerical prediction step.

## Prediction writer

`PredictionWriter` writes the canonical NPZ after joining host outputs and can also write a MATLAB file. A Trainer
accepts at most one prediction writer. Projects may include it only for prediction runs or reuse the same Trainer and
in-memory state after fitting.

The writer receives application metadata through `PredictionContext`. `SIGTERM` ends the active task after cleanup;
project entrypoints decide whether a later prediction task should run.

::: phijax.callbacks.PredictionWriter

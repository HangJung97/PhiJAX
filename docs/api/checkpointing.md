# Checkpointing

PhiJAX stores complete functional training state for exact resumption and also supports model-only restoration for
transfer learning.

## ModelCheckpoint callback

`ModelCheckpoint` delegates storage to `CheckpointIO`; the default backend uses Orbax. A Trainer accepts only one
checkpoint callback. The callback opens its backend when fitting starts and closes it during teardown. Custom backends
must support repeated `open()` and `close()` calls and must be able to reopen for later tasks.

The callback owns retention through `save_top_k`, as in Lightning. The backend never deletes checkpoints on its own.

| Configuration                              | Retained ranked checkpoints           |
| ------------------------------------------ | ------------------------------------- |
| `monitor=None, save_top_k=1` (default)     | The latest checkpoint                 |
| `monitor=None, save_top_k=-1`              | Every checkpoint                      |
| `monitor=None, save_top_k=0`               | None; periodic saves are skipped      |
| `monitor="train/loss", save_top_k=k`       | The best `k` by the metric and `mode` |
| `monitor="step", mode="max", save_top_k=k` | The latest `k` checkpoints            |

Without `monitor`, only `-1`, `0`, and `1` are valid, as in Lightning. `step` is always available as a monitor.

`save_last=True` keeps the terminal state outside the ranking. The next terminal checkpoint replaces it. Public
`best_model_path`, `best_model_score`, and `last_model_path` attributes expose the resolved results. Without `monitor`,
`best_model_path` points to the latest ranked checkpoint and `best_model_score` is `None`. Checkpoints also persist
callback state, including ranking and learning-rate bookkeeping.

A checkpoint leaves storage only after the next checkpoint write has committed. An interrupted asynchronous write
therefore never removes the only valid checkpoint.

::: phijax.callbacks.ModelCheckpoint

## Checkpoint IO protocol

::: phijax.callbacks.CheckpointIO

## Orbax implementation

::: phijax.training.OrbaxCheckpointIO

With asynchronous saving, teardown waits for pending writes and closes Orbax before `Trainer.fit()` returns. The
backend reopens when the same Trainer starts another fit or restore task. Applications do not need to call
`Trainer.close()`.

## Configure saving

Save the state at a fixed interval and keep the latest three checkpoints:

```python
from phijax.callbacks import ModelCheckpoint
from phijax.training import OrbaxCheckpointIO

checkpoint_io = OrbaxCheckpointIO(output_directory / "checkpoints", enable_async_checkpointing=True)
checkpoint_callback = ModelCheckpoint(
    checkpoint_io,
    every_n_steps=5_000,
    monitor="step",
    mode="max",
    save_top_k=3,
    save_last=True,
)
```

Omit `monitor`, `mode`, and `save_top_k` to keep only the latest checkpoint.

Monitor any scalar metric available after module and callback metric collection to keep the best checkpoints:

```python
checkpoint_callback = ModelCheckpoint(
    checkpoint_io,
    every_n_steps=1_000,
    monitor="train/loss",
    mode="min",
    save_top_k=3,
    save_last=True,
)
```

Equal scores do not displace an already retained checkpoint. When a better checkpoint enters a full top-k set, the
deterministic worst entry is removed through the backend. The checkpoint at `last_model_path` is never removed by
top-k ranking. Callback state is stored under stable callback identifiers and restored before fit-start hooks.

Resuming with a different set of callbacks follows Lightning:

- Callbacks without stored state, such as callbacks added since the checkpoint was written, start fresh.
- Stored state for a callback the Trainer no longer has is skipped with a `UserWarning`. Stateless callbacks, such as
  progress bars, store no state and never trigger the warning.
- A callback that rejects its stored state raises instead of applying a partial restoration.

Identifiers combine the callback class with its position among callbacks of the same class. Keep callbacks of the same
class in the same order when resuming.

## Restore behavior

::: phijax.training.restore_checkpoint

| Mode                | Restored state                                                           | Input statistics |
| ------------------- | ------------------------------------------------------------------------ | ---------------- |
| Full resume         | Model, optimizer, balancer, all PRNG streams, step, and precision fields | Checkpoint       |
| `weights_only=True` | Model state only; all other template state remains fresh                 | Target           |
| `ckpt_path=None`    | Target is returned unchanged                                             | Target           |

Training accepts `ckpt_path=None` for fresh initialization. Standalone prediction requires a checkpoint root. An
explicit `step` chooses an exact committed step; `None` selects the latest.

`Trainer.fit(..., ckpt_path=...)` restores the full state by default. Set `weights_only=True` to load only model
weights, or set `ckpt_path="last"` to use the latest checkpoint from the configured callback. `predict_state()`
restores the checkpoint's model state and input statistics into the supplied state template when a path is provided.
The template must have the same model, optimizer, and balancer structure. Use `restore_checkpoint()` directly only when
managing state outside the Trainer.

### Input normalization statistics

[`TrainState`](training.md#phijax.training.TrainState) stores the statistics returned by
[`PhiDataModule.input_statistics()`](data.md#phijax.data.PhiDataModule.input_statistics) as
[`InputStatistics`](#phijax.types.InputStatistics), so checkpoints keep the normalization the model was trained with.
The statistics are never differentiated or updated by the optimizer. `TrainState.input_periods` records the
physical periods the model applies with the same rules, except that periods configured in
`periodic_features_kwargs` always win; see [periodic features](models/layers.md#periodic-features).

| Task                                   | Statistics used                                          |
| -------------------------------------- | -------------------------------------------------------- |
| `Trainer.fit()` without a checkpoint   | Current DataModule                                       |
| `Trainer.fit()` full resume            | Checkpoint; the module is rebound before training starts |
| `Trainer.fit(..., weights_only=True)`  | Current DataModule (the transfer target)                 |
| `Trainer.predict_state(ckpt_path=...)` | Checkpoint; the module is rebound before prediction      |

Transfer learning uses the target statistics by default. To start from the source model's statistics, or to mix them
per coordinate, return [`load_input_statistics()`](#phijax.training.load_input_statistics) from the target DataModule:

```python
from phijax.data import input_statistics
from phijax.training import load_input_statistics


# Subclass the application's existing DataModule and override only its statistics.
class TransferDataModule(ApplicationDataModule):
    def input_statistics(self):
        source = load_input_statistics("runs/source/checkpoints")
        # Coordinates are ordered `[x, y, t]`: keep the source spatial statistics, normalize time with target data.
        mean, std = input_statistics(self.normalization_pools())
        mean[:-1], std[:-1] = source.mean[:-1], source.std[:-1]
        return mean, std
```

`Trainer.fit_state()` trains a plan that is already compiled for the module, so it raises `ValueError` when the
restored statistics differ from the module's. Rebuild the module and plan with the checkpoint statistics, or use
`Trainer.fit()`.

::: phijax.types.InputStatistics

::: phijax.training.load_input_statistics

### Checkpoint schema

PhiJAX 0.2 checkpoints use schema version 3, which adds `TrainState.input_statistics`. Schema-2 checkpoints, written by
PhiJAX 0.2.0b5 and earlier, still restore. They do not contain statistics, so the restored state keeps the template's
statistics, which the Trainer takes from the current DataModule, and PhiJAX emits a `UserWarning`. PhiJAX 0.1
checkpoints remain readable by 0.1; 0.2 rejects them rather than partially restoring incompatible state.

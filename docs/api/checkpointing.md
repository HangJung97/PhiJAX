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
top-k ranking. Callback state is stored under stable callback identifiers and restored before fit-start hooks. A changed callback set or malformed state raises clearly instead of applying a
partial restoration.

## Restore behavior

::: phijax.training.restore_checkpoint

| Mode                | Restored state                                                           |
| ------------------- | ------------------------------------------------------------------------ |
| Full resume         | Model, optimizer, balancer, all PRNG streams, step, and precision fields |
| `weights_only=True` | Model state only; all other template state remains fresh                 |
| `ckpt_path=None`    | Target is returned unchanged                                             |

Training accepts `ckpt_path=None` for fresh initialization. Standalone prediction requires a checkpoint root. An
explicit `step` chooses an exact committed step; `None` selects the latest.

`Trainer.fit(..., ckpt_path=...)` restores the full state by default. Set `weights_only=True` to load only model
weights, or set `ckpt_path="last"` to use the latest checkpoint from the configured callback. `predict_state()` loads
model weights into the supplied state template when a path is provided. The template must have a compatible model
structure. Use `restore_checkpoint()` directly only when managing state outside the Trainer.

PhiJAX 0.2 checkpoints use schema version 2 because `TrainState` stores separate sampling and balancing keys. PhiJAX
0.1 checkpoints remain readable by 0.1; 0.2 rejects them rather than partially restoring incompatible state.

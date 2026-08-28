# Checkpointing

PhiJAX stores complete functional training state for exact resumption and also supports model-only restoration for
transfer learning.

## Checkpoint IO protocol

::: phijax.callbacks.CheckpointIO

## Orbax implementation

::: phijax.training.OrbaxCheckpointIO

With asynchronous saving, teardown waits for pending writes and closes Orbax before `Trainer.fit()` returns. The
backend reopens when the same Trainer starts another fit or restore task. Applications do not need to call
`Trainer.close()`. Set `max_to_keep=None` to retain every saved step.

## Restore behavior

::: phijax.training.restore_checkpoint

| Mode                | Restored state                                                      |
| ------------------- | ------------------------------------------------------------------- |
| Full resume         | Model, optimizer, balancer, RNG, step, and precision-scaling fields |
| `weights_only=True` | Model state only; all other template state remains fresh            |
| `ckpt_path=None`    | Target is returned unchanged                                        |

Training accepts `ckpt_path: null` for fresh initialization. Standalone prediction requires a checkpoint root. An
explicit `step` chooses an exact committed step; `None` selects the latest.

`Trainer.fit(..., ckpt_path=...)` restores the full state by default. Set `weights_only=True` to load only model
weights. `Trainer.predict(..., ckpt_path=...)` always loads model weights into the supplied state template. The template
must have a compatible model structure. Use `restore_checkpoint` directly only when managing state outside the
Trainer.

```yaml
model_checkpoint:
  enabled: true
  _target_: phijax.callbacks.ModelCheckpoint
  every_n_steps: 5000
  save_last: true
  checkpoint_io:
    _target_: phijax.training.OrbaxCheckpointIO
    directory: ${paths.output_dir}/checkpoints
    max_to_keep: 3
    enable_async_checkpointing: true
```

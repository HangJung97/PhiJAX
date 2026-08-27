# Checkpointing

PhiJAX stores complete functional training state for exact resumption and also supports model-only restoration for
transfer learning.

## Checkpoint IO protocol

::: phijax.callbacks.CheckpointIO

## Orbax implementation

::: phijax.training.OrbaxCheckpointIO

With asynchronous saving enabled, `Trainer.close` and callback teardown wait for pending writes. `max_to_keep=None`
retains every committed integer step.

## Restore behavior

::: phijax.training.restore_checkpoint

| Mode                | Restored state                                                      |
| ------------------- | ------------------------------------------------------------------- |
| Full resume         | Model, optimizer, balancer, RNG, step, and precision-scaling fields |
| `weights_only=True` | Model state only; all other template state remains fresh            |
| `ckpt_path=None`    | Target is returned unchanged                                        |

Training accepts `ckpt_path: null` for fresh initialization. Standalone prediction requires a checkpoint root. A
an explicit `step` chooses an exact committed step; `None` selects the latest.

`Trainer.fit(..., ckpt_path=...)` performs full-state restoration by default and supports
`weights_only=True`. `Trainer.predict(..., ckpt_path=...)` always restores only model weights into the supplied state
template. A project entrypoint therefore composes a structurally compatible fresh state and delegates restoration
to the trainer. `restore_checkpoint` remains available for lower-level workflows that intentionally manage state
outside `Trainer`.

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

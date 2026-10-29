# Monitoring callbacks

Monitoring callbacks inspect completed training metrics on the Python host. They do not run inside the compiled
update.

## Early stopping

`EarlyStopping` follows Lightning's callback. It checks the monitored metric in `on_train_metrics()`, so it can monitor
metrics logged by the module with `self.log()` and metrics contributed by other callbacks. When it stops, it sets
`Trainer.should_stop` and logs the reason.

| Option                | Behavior                                                                             |
| --------------------- | ------------------------------------------------------------------------------------ |
| `patience`            | Stop once `wait_count`, the number of checks without improvement, reaches it         |
| `min_delta`           | Smallest change over `best_score` that counts as an improvement                      |
| `mode`                | `"min"` for losses, `"max"` for scores where larger values are better                |
| `check_finite`        | Stop immediately on a NaN or infinite value; enabled by default                      |
| `check_every_n_steps` | Check only at global steps divisible by this interval, avoiding a host sync per step |

Checkpoints store `wait_count`, `best_score`, and `stopped_step`, so a resumed fit stops at the same step as an
uninterrupted one. They also store `monitor`, `mode`, and `patience`:

- Resuming with a different `monitor` or `mode` skips the stored progress with a warning, because a best score of
  another metric is not comparable.
- Resuming with a different `patience` keeps the configured value and warns, so `patience` can be changed on resume.

`mode` uses the public `MonitorMode` alias.

::: phijax.callbacks.EarlyStopping

## Learning-rate monitoring

`LoggingInterval` represents the explicit `"step"` and `"epoch"` interval values. Passing `None` follows the Trainer
logging cadence.

`LearningRateMonitor` evaluates the Optax schedule at the optimizer count for each completed update. It sends
`optimizer/lr-Adam` to configured loggers when `optimizer_name="Adam"`. `logging_interval="step"` evaluates every step,
while `logging_interval="epoch"` evaluates once at the end of fitting. The default `None` follows
`trainer.log_every_n_steps` and always records the final rate. Unlike a Lightning scheduler, a raw Optax schedule has
no `interval` field. PhiJAX therefore avoids evaluating the schedule on steps that will not be logged.

Like Lightning, this callback requires an experiment logger. Fitting raises before requesting the first batch when
`LearningRateMonitor` is enabled with `logger=False`, `logger=None`, or an empty logger collection. The default
`logger=True` is valid.

Optimizer names identify learning-rate series, while `log_momentum` and `log_weight_decay` add `-momentum` and
`-weight_decay` suffixes. PhiJAX requires `optimizer_name` because Optax transformations do not retain their factory
name. Metrics use the `optimizer/` group by default; set `log_key_prefix` to replace it or pass `None` to disable
grouping. Optional momentum and weight-decay values must be supplied because Optax transformations do not expose
inspectable parameter groups.

::: phijax.callbacks.LearningRateMonitor

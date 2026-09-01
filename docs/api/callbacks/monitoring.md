# Monitoring callbacks

Monitoring callbacks inspect completed training metrics on the Python host. They do not run inside the compiled
update.

## Early stopping

`mode` accepts `"min"` or `"max"`, represented by the public `MonitorMode` alias. Use `"min"` for losses and `"max"`
for scores where larger values are better.

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

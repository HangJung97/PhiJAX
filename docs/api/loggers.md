# Experiment loggers

All backends implement the same hyperparameter, scalar metric, artifact, and finalization contract.
`LoggerCollection` broadcasts identical events to each configured backend.

Logger construction does not open files, import optional SDKs, or start remote runs. The Trainer creates logger
resources when a task starts and releases them when it ends. Calling a built-in logger directly remains convenient:
its first logging operation calls `setup()` automatically.

`Trainer(logger=True)` enables local logging by default. It uses `TensorBoardLogger` when TensorBoard is installed and
falls back to `CSVLogger` otherwise. Runs are stored below
`<default_root_dir>/phijax_logs/version_N/`, including `hparams.yaml` and scalar metrics. Set `logger=False` or
`logger=None` to disable logging, or pass one logger or an iterable of loggers explicitly.

```python
trainer = Trainer(max_steps=10_000, default_root_dir="outputs")
```

Inspect TensorBoard runs with:

```bash
tensorboard --logdir outputs/phijax_logs
```

Loggers expose `name`, `version`, and `log_dir`. Progress callbacks display the primary version as `v_num`; subclasses
may remove it by overriding `get_metrics()` and popping that key. Local and remote logging is owned by global rank
zero.

Pass a plain mapping to `Trainer.fit()` to record run settings before the first update:

```python
result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=0,
    hyperparameters={"model": {"width": 64}, "seed": 0},
)
```

Hydra projects can convert their composed config with `to_hyperparameters()` before passing it to `fit()`. The
Trainer performs the logger call after rank and task setup, so entrypoints do not need a separate logging utility.

Modules may select destinations with `self.log()` during `on_train_batch_end()`. This records device values without
writing them immediately. The Trainer exposes three read-only views of the same latest metric store:

- `callback_metrics` contains every module and callback value, including arrays;
- `logged_metrics` contains scalar values selected for experiment loggers;
- `progress_bar_metrics` contains scalar values selected for terminal display.

Logger conversion still follows `log_every_n_steps`. Progress callbacks transfer their selected values only at their
own refresh interval.

::: phijax.training.ExperimentLogger

::: phijax.training.LoggerCollection

::: phijax.training.ConsoleLogger

::: phijax.training.CSVLogger

::: phijax.training.TensorBoardLogger

::: phijax.training.WandbLogger

## Optional dependencies

```bash
uv sync --extra tensorboard
uv sync --extra cuda13 --extra wandb
```

W&B authenticates through `wandb login` or the `WANDB_API_KEY` environment variable. Do not interpolate credentials
into project configuration because resolved configurations are commonly printed and saved.

```bash
WANDB_API_KEY="your-key" python -m my_project.train logger=wandb
```

Root `tags` are forwarded to compatible loggers:

```bash
python -m my_project.train logger=wandb tags='[baseline,gpu]'
```

## Custom logger lifecycle

A custom `ExperimentLogger` should follow three rules:

- Keep `__init__()` free of filesystem, network, and optional-dependency side effects.
- Create resources in an idempotent `setup()` method.
- Make `finalize()` safe before setup, after partial setup, and when called more than once.

Only global rank zero calls setup, logging methods, artifact methods, and finalization. Logger objects are still
constructed on every rank so callbacks see the same `has_logger` value.

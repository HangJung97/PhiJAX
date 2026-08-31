# Experiment loggers

All backends implement the same hyperparameter, scalar metric, artifact, and finalization contract.
`LoggerCollection` broadcasts identical events to each configured backend.

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

# Experiment loggers

All backends implement the same hyperparameter, scalar metric, artifact, and finalization contract.
`LoggerCollection` broadcasts identical events to each configured backend.

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

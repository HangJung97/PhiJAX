# Choose an optimizer

Create an optimizer, such as `optax.adam()`, and pass it to
[`Trainer.fit()`](../api/trainer.md#phijax.training.Trainer.fit) to choose how model parameters are updated.

PhiJAX stores the optimizer state in
[`TrainState`](../api/training.md#phijax.training.TrainState), saves it in full checkpoints, and restores it when
resuming training. Weights-only loading starts with fresh optimizer state.

Keep optimizer configuration in your application. The
[`PhiModule`](../api/module.md#phijax.core.PhiModule) defines the model and objective independently of the optimizer.
Custom optimizers must follow the Optax interface, support JAX transformations, and preserve the model's parameter
PyTree structure.

## Use an Optax optimizer

Optax is installed with PhiJAX. With the module, DataModule, and Trainer from the
[training workflow](training.md#run-the-common-workflow), create an Adam optimizer and pass it to `fit()`:

```python
import optax

optimizer = optax.adam(learning_rate=1.0e-3)
result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=0,
)
```

A learning-rate schedule can be supplied through the optimizer's `learning_rate` argument. See
[Logging and monitoring](logging.md) for recording the learning rate alongside training metrics.

## Use the optional SOAP optimizer

The [SOAP_JAX compatibility fork](https://github.com/HangJung97/SOAP_JAX) provides an Optax optimizer that works with
PhiJAX's explicit NNX model state. Pass `soap_jax.soap()` directly to `Trainer.fit()`; no adapter is required.
PhiJAX does not install SOAP. Keep the dependency and optimizer configuration in your downstream project.

Install the validated fork revision from your application directory, then commit its `pyproject.toml` and `uv.lock`:

```bash
uv add "soap-jax @ git+https://github.com/HangJung97/SOAP_JAX.git@3c3aafe6b8fd477bf634020ec5d44f1431175ab6"
```

The fork retains the package name `soap-jax` and version `0.2.2`. Pin the Git revision to identify the compatibility
fixes; the version number alone does not distinguish it from upstream.

Using the same module, DataModule, and Trainer as above:

```python
from soap_jax import soap

optimizer = soap(
    learning_rate=1.0e-3,
    precondition_frequency=5,
    precondition_1d=False,
    weight_decay=0.0,
)
result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=0,
)
```

`precondition_frequency` controls how often SOAP refreshes its preconditioning basis. `precondition_1d=False` leaves
vector parameters, such as biases, without matrix preconditioning. These settings are an example, not tuned PINN
defaults. The first optimizer call initializes preconditioners and returns zero parameter updates. PhiJAX still
advances its training step, sampling sequence, and step-based callbacks for that call.

For a project using Hydra, place this mapping in its optimizer config group:

```yaml
_target_: soap_jax.soap
learning_rate: 1.0e-3
precondition_frequency: 5
precondition_1d: false
weight_decay: 0.0
```

Pass that config to
[`instantiate_optimizer()`](../api/configuration.md#phijax.integrations.hydra.instantiate_optimizer), then pass the
returned transformation to `Trainer.fit()`. See [Configuration integrations](../api/configuration.md) for project-owned
configuration assembly.

### Resume SOAP training

Use the ordinary [checkpoint workflow](training.md#resume-or-load-weights) for checkpoints created with this fork. For exact
resumption, retain the model structure, SOAP revision and configuration, dependency versions, and execution backend.
Full checkpoints include SOAP's moments and preconditioners as part of the optimizer state.

The fork changes the optimizer-state layout from upstream SOAP_JAX `v0.2.2`. Existing upstream full-state checkpoints
require explicit migration or `weights_only=True` with a fresh optimizer. Loading weights alone starts a new
optimization trajectory; it does not exactly resume the old run. See the fork's
[checkpoint migration notes](https://github.com/HangJung97/SOAP_JAX/blob/3c3aafe6b8fd477bf634020ec5d44f1431175ab6/COMPATIBILITY.md#checkpoint-compatibility).

## SOAP compatibility validation

The pinned SOAP fork was tested with PhiJAX `0.2.0b4` at revision `6b4ba66`, Python 3.12.11, JAX 0.11.1, Flax 0.12.9,
Optax 0.2.8, and Orbax 0.12.4 on CPU. The compiled MLP test restored a checkpoint after seven steps and matched all
state arrays and metrics for eight further steps. The heat PINN test matched 16 uninterrupted steps against seven
steps followed by restoration and nine resumed steps through `Trainer.fit()`.

These checks establish compatibility and exact continuation for the tested setup, not converged PDE accuracy or
cross-device reproducibility. The fork maintains the regression tests and
[CPU/CUDA validation instructions](https://github.com/HangJung97/SOAP_JAX/blob/3c3aafe6b8fd477bf634020ec5d44f1431175ab6/COMPATIBILITY.md#validation).
Repeat the integration tests when changing the fork revision or dependency versions.

## Next steps

- [Training and prediction](training.md)
- [Checkpointing](../api/checkpointing.md)
- [Configuration integrations](../api/configuration.md)
- [Loss balancers](balancers.md)

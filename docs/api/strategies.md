# Device strategies

A strategy decides where the training state and batches live. The [`Trainer`](trainer.md#phijax.training.Trainer)
builds a [`SingleDeviceStrategy`](#phijax.training.SingleDeviceStrategy) or a
[`DataParallelStrategy`](#phijax.training.DataParallelStrategy) from `accelerator` and `devices`, so most code never
creates a strategy directly. `Accelerator` and `DeviceSelection` describe the accepted values.

| Selected devices | Strategy               | Behavior                                                |
| ---------------- | ---------------------- | ------------------------------------------------------- |
| One              | `SingleDeviceStrategy` | State and batches on one CPU, GPU, or TPU device        |
| Several          | `DataParallelStrategy` | State replicated on every device, batches split by rows |

`DataParallelStrategy` is synchronous data parallelism, like Lightning's DDP. Each device computes gradients on its
rows of the batch, the gradients are averaged over all devices, and every copy of the state receives the same update.
The mesh is one-dimensional and built with `jax.make_mesh` and `AxisType.Auto`, so `jax.jit` inserts the gradient
all-reduce. Multi-process support is experimental and does not provide full Lightning DDP parity.

## Run on several GPUs of one machine

Select more than one device. No distributed setup is needed:

```python
trainer = Trainer(max_steps=20_000, accelerator="gpu", devices="auto")  # or devices=4, or devices=[0, 2]
trainer.fit(module, datamodule=datamodule, optimizer=optimizer, seed=0)
```

The batch sizes requested by the DataModule are the total rows per step. They must be divisible by the device count.
Prediction chunks can have any size: `Trainer.predict()` pads each chunk to a multiple of the device count, splits it
over the devices, and drops the padded rows from the returned predictions.

## Run on several machines

Start one process per machine. Each process calls
[`initialize_distributed()`](#phijax.training.initialize_distributed) before any JAX computation or device query, then
runs the same training code:

```python
from phijax import Trainer
from phijax.training import initialize_distributed

initialize_distributed()  # Detects Slurm, Open MPI, Kubernetes, and Cloud TPU environments.

trainer = Trainer(max_steps=20_000, accelerator="gpu", devices="auto")
trainer.fit(module, datamodule=datamodule, optimizer=optimizer, seed=0)
```

Outside a detected environment, pass the coordinator explicitly:
`initialize_distributed("host0:1234", num_processes=2, process_id=rank)`. Under Slurm, launch with
`srun python train.py`.

Several machines change four rules:

| Setting         | Meaning with several processes                                                                 |
| --------------- | ---------------------------------------------------------------------------------------------- |
| `devices`       | Devices per process, like Lightning's `devices` per node. `devices=4` on two machines gives 8. |
| Batch sizes     | Rows per process. The global batch is the batch size times the process count.                  |
| `seed`          | Must be identical on every process. Fitting raises `ValueError` if the starting state differs. |
| Data and output | `prepare_data()` files and checkpoints must be on storage every machine can read.              |

Each process samples its own rows, and logging, progress bars, and prediction writing run on global process zero
only. `Trainer.predict()` runs on every process and returns the complete predictions on each of them.

## How multi-process placement works

- **Devices.** Every process selects the same local positions from `devices`, and the mesh spans the selected devices
  of all processes. `devices=1` on two processes gives a two-device mesh with one device per process.
- **State.** Every process builds the full training state, and `place_state()` combines the copies into one replicated
  global array. It first compares one SHA-256 digest per state leaf across processes and names any leaves that differ.
- **Sampling.** [`Strategy.process_key()`](#phijax.training.Strategy.process_key) folds the process index into the
  stored sampling and balancer keys. Processes draw different rows, while the checkpointed keys stay identical, so a
  resumed fit draws the same batches.
- **Prediction.** Every process builds the same prediction batches. As on one machine, each batch is padded so its
  rows divide over all devices, and each process predicts an equal share. The outputs are gathered with
  `multihost_utils.process_allgather()` and put back in batch order, so every process receives the same complete
  predictions, and `PredictionWriter` writes them once from global process zero. Like Lightning's distributed
  prediction, this spreads the work over every device; unlike Lightning, no per-rank files need merging.
- **Data preparation.** [`Strategy.barrier()`](#phijax.training.Strategy.barrier) makes every process wait until
  global process zero finishes `PhiDataModule.prepare_data()`. See [multi-process runs](data.md#multi-process-runs).

Every process must run the same compiled steps in the same order, so every process also stops after the same step:

| Stop request                                   | Result on every process                                   |
| ---------------------------------------------- | --------------------------------------------------------- |
| `trainer.should_stop`, such as `EarlyStopping` | Stops after the same step                                 |
| `Ctrl+C` (`SIGINT`)                            | Returns `FitResult(interrupted=True)` after the same step |
| `SIGTERM` (preemption)                         | Saves a checkpoint at JAX's sync step and exits with 143  |

A request on one process is enough, as in Lightning DDP. After each step, one small collective combines
`should_stop` and `SIGINT` across processes; single-process runs skip it. `SIGTERM` is left to JAX's preemption
service, which tells every process and picks a shared sync step that PhiJAX reads with
`multihost_utils.reached_preemption_sync_point()`. `ModelCheckpoint` saves that step even without `save_on_exception`.

## Testing

To try several processes on one CPU machine, give each process two CPU devices and start two processes. Each process
calls `initialize_distributed("localhost:1234", 2, process_id)` with its own `process_id`:

```bash
export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2
python train.py 0 & python train.py 1 & wait
```

PhiJAX CI runs such a two-process CPU fit, including checkpoint resume. Validate the intended GPU or multi-host
environment before a long run.

## Common mistakes

- **Different seeds or configs per process.** Fitting raises `ValueError` naming the state leaves that differ.
- **Prepared files on a local disk.** `setup()` fails on the other machines because they cannot read the files.
- **A process fails mid-step.** An error inside one process's step, such as running out of memory, is not a stop
  request. The other processes wait at the next collective until the job is cancelled.
- **Calling JAX before `initialize_distributed()`.** Initialization fails once a JAX backend has started.

::: phijax.training.Strategy

::: phijax.training.SingleDeviceStrategy

::: phijax.training.DataParallelStrategy

::: phijax.training.create_strategy

::: phijax.training.initialize_distributed

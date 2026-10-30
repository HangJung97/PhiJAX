# Device strategies

`Accelerator` and `DeviceSelection` describe the values accepted by `Trainer` and `create_strategy()`.

[`SingleDeviceStrategy`](#phijax.training.SingleDeviceStrategy) places complete state and batches on one selected CPU,
GPU, or TPU device. [`DataParallelStrategy`](#phijax.training.DataParallelStrategy) replicates state and shards
non-scalar batch leaves along their leading dimension. Batch sizes must be divisible by the number of local shards.

## Experimental distributed execution

`DataParallelStrategy` provides experimental synchronous data parallelism. It does not provide full Lightning DDP
parity. Call `initialize_distributed()` before any JAX backend operation for multi-process use. Every process must run
the same compiled steps and collectives in the same order. Logging and checkpoint writes remain restricted to global
rank zero.

`Strategy.barrier(name)` blocks until every JAX process reaches it; a single-process run returns immediately. The
Trainer uses it so that `PhiDataModule.prepare_data()` runs once, on global process zero, before any process runs
`setup()`. See [multi-process runs](data.md#multi-process-runs).

Ordinary CI covers single-process CPU behavior. Validate the intended multi-device or multi-host environment before a
long run.

::: phijax.training.Strategy

::: phijax.training.SingleDeviceStrategy

::: phijax.training.DataParallelStrategy

::: phijax.training.create_strategy

::: phijax.training.initialize_distributed

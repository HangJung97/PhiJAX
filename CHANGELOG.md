# Changelog

All notable changes to PhiJAX are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and PhiJAX follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0b1] - 2026-08-28

### Added

- Concise `Trainer.fit()` and `Trainer.predict()` APIs that initialize models, state, balancing, batch routing, and
  compiled plans from a module blueprint, DataModule, optimizer, and seed.
- `ModelFactory`, balancer-owned adaptive update plans, objective equation composition, finite-pool batch-source
  construction, and safe `HostPool` defaults.
- Independent persistent PRNG streams for model execution, sampling, and adaptive-balancer diagnostics.
- Structured training diagnostics with host-side `PhiModule.log()` control for loggers and progress displays.
- Automatic TQDM progress, plain model summaries, default TensorBoard or CSV logging, and versioned run directories.
- Monitored top-k checkpoints with persistent callback state and experimental TPU installation support.

### Changed

- Renamed the explicit execution APIs to `Trainer.fit_state()` and `Trainer.predict_state()`.
- Changed `PhiModule` into an immutable model-factory blueprint; the bound module is returned as `FitResult.module`.
- Made DataModule input normalization opt-in through `input_statistics()`.
- Changed `ResidualTerm` to accept its residual function first and infer its default NTK stream from equation metadata.
- Raised the checkpoint schema to version 2 for the expanded `TrainState` layout.
- Made `LearningRateMonitor` require a configured experiment logger before fitting begins.
- Moved adaptive-balancer scheduling into the Trainer host loop and removed unconditional per-step transfers of
  `TrainState.step`.
- Split module contracts into `phijax.core` and moved fit, prediction, logging, and signal orchestration into private
  loops and connectors while keeping the top-level API unchanged.
- Moved adaptive cadence and diagnostic sampling settings into adaptive-balancer constructors.
- Replaced the adaptive `skip_first_step` flag with cadence-anchored `update_start_step` scheduling.

### Removed

- Removed separate sampling and balancer key arguments from Trainer methods; both now live in `TrainState`.
- Removed `resume_latest()` in favor of `fit(..., ckpt_path="last")`.
- Removed the low-level `with_balancer_updates()` helper; advanced schedules now belong to `TrainingPlan`.
- Removed `BalancerUpdateConfig` and `BalancerUpdateSchedule`; `TrainingPlan` now accepts one `BalancerUpdatePlan`.
- Removed the Hydra `balancer.factory` and `balancer.update` layers; balancer groups are directly instantiable.
- Removed `MetricRoute`; modules now customize metric destinations with `self.log()` from `on_train_batch_end()`.
- Removed the `phijax.module` deep import path; use top-level imports or `phijax.core` for custom module contracts.

## [0.1.0b1] - 2026-08-28

### Added

- Initial beta API for building, fitting, and evaluating physics-informed neural networks with JAX.
- Lightning-inspired `Trainer`, `PhiModule`, `DataModule`, callback, logger, and checkpoint lifecycles.
- Explicit model, optimizer, loss-balancer, PRNG, and training state for compiled updates.
- Selective coordinate derivatives and reusable PDE, boundary, data-fidelity, and objective components.
- Static, normalized-static, gradient-norm, and exact-NTK loss balancing.
- MLP, Modified MLP, and PirateNet architectures with a common model initialization contract.
- CPU, CUDA 12, and CUDA 13 installation options, configurable precision, and centralized device placement.
- Versioned checkpoints and portable prediction artifacts for downstream evaluation.
- Optional Hydra integration and Weights & Biases and TensorBoard logging.

[0.1.0b1]: https://github.com/HangJung97/PhiJAX/releases/tag/v0.1.0b1
[0.2.0b1]: https://github.com/HangJung97/PhiJAX/releases/tag/v0.2.0b1
[unreleased]: https://github.com/HangJung97/PhiJAX/compare/v0.2.0b1...HEAD

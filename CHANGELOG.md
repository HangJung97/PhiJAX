# Changelog

All notable changes to PhiJAX are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and PhiJAX follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added public type aliases for supported accelerator, device, precision, monitoring, activation, and initialization
  options.

### Changed

- Clarified generated API signatures and option tables, and hid full implementation source from reference pages by
  default.
- Reorganized the documentation into task-oriented guides and focused API pages for the Trainer, state, precision,
  and device strategies.
- Added custom-model and logging guides, cross-referenced public APIs, and checks for navigable pages, valid local
  links, and non-duplicated class documentation.
- Added configurable source links to generated class, function, and method references without embedding full source
  listings. Documentation built from `main` links to source available on GitHub.
- Restructured the `PhiModule` and `Trainer` references to lead with common workflows and group configuration,
  extension, lifecycle, and runtime details before the generated API.

## [0.2.0b3] - 2026-09-01

### Added

- Added a `target.name` OmegaConf resolver for deriving display labels from Hydra `_target_` paths.

### Breaking changes

- Made `LearningRateMonitor` require an explicit optimizer name and changed its default metric group from `train/` to
  `optimizer/`, using Lightning-style keys such as `optimizer/lr-Adam`.

### Changed

- Made the Trainer report its precision and accelerator environment automatically when constructed on global rank
  zero.
- Expanded the balancer documentation with the exact metrics, meanings, update behavior, and default destinations for
  static, gradient-norm, and exact-NTK balancing.

## [0.2.0b2] - 2026-09-01

### Added

- Added `to_hyperparameters()` for converting composed Hydra configs before passing them to `Trainer.fit()`.

### Breaking changes

- Removed callback and logger `enabled` configuration options. Omit a service entry to disable it.
- Removed `Trainer.set_logger()`. Construct loggers first and pass them through `Trainer(logger=...)` or
  `instantiate_trainer(..., logger=...)`; `instantiate_loggers()` no longer accepts a Trainer.
- Renamed the experiment logger method `log_hyperparameters()` to the Lightning-style `log_hyperparams()`.

### Changed

- Made CSV, TensorBoard, and W&B logger resource setup lazy, idempotent, and restricted to global rank zero during
  Trainer tasks. Direct built-in logger use still starts resources automatically on first use.

## [0.2.0b1] - 2026-08-31

### Added

- Concise `Trainer.fit()` and `Trainer.predict()` APIs that initialize models, state, balancing, batch routing, and
  compiled plans from a module blueprint, DataModule, optimizer, and seed.
- `ModelFactory`, balancer-owned adaptive update plans, objective equation composition, finite-pool batch-source
  construction, and safe `HostPool` defaults.
- Independent persistent PRNG streams for model execution, sampling, and adaptive-balancer diagnostics.
- Structured training diagnostics with host-side `PhiModule.log()` control for loggers and progress displays.
- Automatic TQDM progress, plain model summaries, default TensorBoard or CSV logging, and versioned run directories.
- Monitored top-k checkpoints with persistent callback state and experimental TPU installation support.

### Breaking changes

- Renamed the explicit execution APIs to `Trainer.fit_state()` and `Trainer.predict_state()`.
- Changed `PhiModule` into an immutable model-factory blueprint; the bound module is returned as `FitResult.module`.
- Changed `ResidualTerm` to accept its residual function first and infer its default NTK stream from equation metadata.
- Raised the checkpoint schema to version 2 for the expanded `TrainState` layout.
- Moved adaptive cadence and diagnostic sampling settings into adaptive-balancer constructors.
- Replaced the adaptive `skip_first_step` flag with cadence-anchored `update_start_step` scheduling.
- Removed separate sampling and balancer key arguments from Trainer methods; both now live in `TrainState`.
- Removed `resume_latest()` in favor of `fit(..., ckpt_path="last")`.
- Removed the low-level `with_balancer_updates()` helper; advanced schedules now belong to `TrainingPlan`.
- Removed `BalancerUpdateConfig` and `BalancerUpdateSchedule`; `TrainingPlan` now accepts one `BalancerUpdatePlan`.
- Removed the Hydra `balancer.factory` and `balancer.update` layers; balancer groups are directly instantiable.
- Removed `MetricRoute`; modules now customize metric destinations with `self.log()` from `on_train_batch_end()`.
- Removed the `phijax.module` deep import path; use top-level imports or `phijax.core` for custom module contracts.

### Changed

- Made DataModule input normalization opt-in through `input_statistics()`.
- Made `LearningRateMonitor` require a configured experiment logger before fitting begins.
- Moved adaptive-balancer scheduling into the Trainer host loop and removed unconditional per-step transfers of
  `TrainState.step`.
- Split module contracts into `phijax.core` and moved fit, prediction, logging, and signal orchestration into private
  loops and connectors while keeping the top-level API unchanged.

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
[0.2.0b2]: https://github.com/HangJung97/PhiJAX/releases/tag/v0.2.0b2
[0.2.0b3]: https://github.com/HangJung97/PhiJAX/releases/tag/v0.2.0b3
[unreleased]: https://github.com/HangJung97/PhiJAX/compare/v0.2.0b3...HEAD

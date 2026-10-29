# Changelog

Best viewed [here](https://hangjung97.github.io/PhiJAX/changelog/).

All notable changes to PhiJAX are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and PhiJAX follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ModelCheckpoint` accepts `monitor="step"` to rank checkpoints by training step.
- The terminal checkpoint from `save_last` or `save_on_exception` is replaced by the next one, including across
  resumed fits, instead of accumulating.
- `TrainState.input_statistics` stores the model's input normalization statistics as `InputStatistics`, so checkpoints
  keep the normalization a model was trained with.
- `load_input_statistics()` reads the statistics stored in a checkpoint. Return it from a DataModule's
  `input_statistics()` to start transfer learning from the source statistics.
- `BasePhiModule.input_statistics`, `input_periods`, and `bind_inputs()` let the Trainer rebind a module to restored
  statistics and periods.
- `PhiDataModule.input_periods()` declares physical periods of periodic input axes, and `TrainState.input_periods`
  stores them with checkpoints. Resume and prediction use the stored periods; transfer learning uses the target
  DataModule's; `periodic_features_kwargs['periods']` overrides both.
- `ExactNTKBalancer(kernel_size=...)` accepts `"all"` to compute the NTK on every row of a finite pool, and a mapping
  that sets the sample count per objective batch key, such as `{"pde": 1024, "boundary": "all"}`.
- `ExactNTKBalancer(resample=True)` draws new diagnostic batches for every update instead of once per fit. Custom
  balancers request the same behavior with `BalancerUpdatePlan(resample=True)`.
- `Trainer.should_stop` lets callbacks request a clean stop, as in Lightning. Returning `True` from
  `on_train_batch_end()` still works and sets the flag.
- `EarlyStopping` accepts `check_finite` and `check_every_n_steps`, and stores its progress in checkpoints, so a
  resumed fit stops at the same step as an uninterrupted one. Progress stored for a different `monitor` or `mode` is
  skipped with a warning, and a changed `patience` keeps the configured value with a warning.

### Breaking changes

- `ModelCheckpoint` now owns checkpoint retention, following Lightning's `ModelCheckpoint`:
    - `OrbaxCheckpointIO` no longer accepts `max_to_keep` and never deletes checkpoints on its own.
    - `save_top_k` defaults to `1`, so `ModelCheckpoint(checkpoint_io)` keeps only the latest checkpoint instead of
      every checkpoint. Pass `save_top_k=-1` to keep every checkpoint.
    - Without `monitor`, `save_top_k` must be `-1`, `0`, or `1`.
    - To keep the latest `k` checkpoints, replace `max_to_keep=k` with
      `ModelCheckpoint(..., monitor="step", mode="max", save_top_k=k)`. See the
      [migration guide](https://hangjung97.github.io/PhiJAX/getting-started/migration-0.2/#checkpoint-retention).
- `Trainer(max_steps=...)` is now the global optimizer step at which training stops, as in Lightning. A fit resumed
  from step `N` runs `max_steps - N` updates instead of `max_steps` more, and logs and progress bars follow the global
  step. Resume with `max_steps = N + extra` to train `extra` more steps. See the
  [migration guide](https://hangjung97.github.io/PhiJAX/getting-started/migration-0.2/#global-max_steps).
- `EarlyStopping` follows Lightning's callback: it checks metrics in `on_train_metrics()`, so it sees module-logged and
  callback metrics, and stops once `wait_count >= patience` instead of `> patience`. `best` and `bad_checks` are
  renamed `best_score` and `wait_count`, and a NaN or infinite metric stops training immediately by default. See the
  [migration guide](https://hangjung97.github.io/PhiJAX/getting-started/migration-0.2/#early-stopping).
- Periodic features use physical periods of the physical coordinate, so input normalization no longer affects periodic
  axes. `PeriodicFeatures` takes `periods={axis: T}` instead of `axes` and `frequencies`, model factories receive
  `input_periods`, and `bind_input_statistics()` is now `bind_inputs()`. See the
  [migration guide](https://hangjung97.github.io/PhiJAX/getting-started/migration-0.2/#periodic-features-in-physical-units).

### Changed

- Checkpoints use schema version 3. Schema-2 checkpoints from 0.2.0b5 still restore, using the current DataModule's
  input statistics with a warning. PhiJAX 0.2.0b5 cannot read schema-3 checkpoints.
- `Trainer.predict_state(..., ckpt_path=...)` restores the full checkpoint and uses its model state and input
  statistics, instead of restoring model weights only.
- The Trainer runs `ModelCheckpoint` after all other callbacks, as in Lightning, so each checkpoint stores the state
  every other callback reached at that step.

### Fixed

- Fixed a crash with `FileNotFoundError` when monitored top-k retention and Orbax `max_to_keep` deleted the same
  checkpoints. Ranking now always matches the stored checkpoints, and restored ranking state drops steps that are no
  longer stored.
- Top-k displacement no longer deletes the checkpoint at `last_model_path`.
- `ModelCheckpoint` deletes a displaced checkpoint only after the next checkpoint write has committed, so an
  interrupted asynchronous write never removes the only valid checkpoint.
- `OrbaxCheckpointIO.delete` now ignores steps that are not stored.
- `build_array_pools` now raises for a field stored in a different axis order from its declared `axes`, instead of
  silently reshaping it. Single-value coordinates, such as one time slice, now load.
- A full resume and prediction from a checkpoint now use the input statistics the model was trained with. Previously
  they were recomputed from the current DataModule, which silently changed the model if the data had changed.
  Transfer learning with `weights_only=True` still uses the current DataModule's statistics.
- Resuming with a different set of callbacks no longer raises. Following Lightning, added callbacks start fresh, and
  stored state for a removed stateful callback is skipped with a warning. Toggling stateless callbacks, such as the
  progress bar or model summary, neither fails nor warns.
- `OrbaxCheckpointIO` no longer makes Orbax log `Item "default" ... could not be restored` on every checkpoint
  metadata read, such as each restore.
- `spherical_navier_stokes` now preserves the sign of `sin(th)` when clamping it, so polar angles outside `[0, pi]`
  give correct residuals.
- Fidelity and boundary equations now raise `ValueError` for out-of-range `output_indices` or `target_indices`
  instead of producing NaN residuals.
- Pools built by `build_array_pools` now name every output channel, so prediction artifacts can be saved for
  vector-valued targets (`v_0`, `v_1`, ...) and for pools without targets.
- A resumed fit now draws the same exact-NTK diagnostic batches as an uninterrupted run. Previously the batches
  depended on the step the fit resumed from.
- A failing `on_exception`, logger finalization, or teardown hook no longer replaces the original training or
  prediction error, and no longer skips the remaining cleanup. The failure is logged and attached to the original
  error as a note.

## [0.2.0b5] - 2026-09-11

### Added

- Added `no_slip_boundary` with configurable velocity components and residual/output streams for stationary and moving
  walls. ([#39](https://github.com/HangJung97/PhiJAX/pull/39))
- Added configurable weight, projection, and period batch keys to fidelity equations, and a configurable normals key to
  `free_slip_boundary`. ([#38](https://github.com/HangJung97/PhiJAX/pull/38))

### Breaking changes

- `phase_wrapped_fidelity` now uses `batch["projection"]` when present. Set `projection_key=None` to preserve direct
  supervision when that field contains unrelated data. ([#38](https://github.com/HangJung97/PhiJAX/pull/38))

### Changed

- Updated the optional W&B dependency to `>=0.30,<0.31`. ([#37](https://github.com/HangJung97/PhiJAX/pull/37))
- Added an optimizer guide covering Optax, the optional SOAP_JAX compatibility fork, and checkpoint restoration.
  ([#40](https://github.com/HangJung97/PhiJAX/pull/40))
- Clarified installation workflows with uv and pip. ([#36](https://github.com/HangJung97/PhiJAX/pull/36))

### Fixed

- Log all term losses before loss-balancer weights for consistent metric ordering.
  ([#35](https://github.com/HangJung97/PhiJAX/pull/35))

## [0.2.0b4] - 2026-09-03

### Added

- Added public type aliases for supported accelerator, device, precision, monitoring, activation, and initialization
  options. ([#25](https://github.com/HangJung97/PhiJAX/pull/25))

### Breaking changes

- Removed the inert `Trainer.deterministic` option. Deterministic GPU experiments must configure XLA before JAX
  initializes. ([#30](https://github.com/HangJung97/PhiJAX/pull/30))

### Changed

- Reorganized the documentation into task-oriented guides and focused API reference pages.
  ([#26](https://github.com/HangJung97/PhiJAX/pull/26), [#27](https://github.com/HangJung97/PhiJAX/pull/27),
  [#28](https://github.com/HangJung97/PhiJAX/pull/28))

## [0.2.0b3] - 2026-09-01

### Added

- Added a `target.name` OmegaConf resolver for deriving display labels from Hydra `_target_` paths. ([#22](https://github.com/HangJung97/PhiJAX/pull/22))

### Breaking changes

- Made `LearningRateMonitor` require an explicit optimizer name and changed its default metric group from `train/` to
  `optimizer/`, using Lightning-style keys such as `optimizer/lr-Adam`. ([#22](https://github.com/HangJung97/PhiJAX/pull/22))

### Changed

- Made the Trainer report its precision and accelerator environment automatically when constructed on global rank
  zero. ([#21](https://github.com/HangJung97/PhiJAX/pull/21))
- Expanded the balancer documentation with the exact metrics, meanings, update behavior, and default destinations for
  static, gradient-norm, and exact-NTK balancing. ([#23](https://github.com/HangJung97/PhiJAX/pull/23))

## [0.2.0b2] - 2026-09-01

### Added

- Added `to_hyperparameters()` for converting composed Hydra configs before passing them to `Trainer.fit()`. ([#19](https://github.com/HangJung97/PhiJAX/pull/19))

### Breaking changes

- Removed callback and logger `enabled` configuration options. Omit a service entry to disable it. ([#16](https://github.com/HangJung97/PhiJAX/pull/16))
- Removed `Trainer.set_logger()`. Construct loggers first and pass them through `Trainer(logger=...)` or
  `instantiate_trainer(..., logger=...)`; `instantiate_loggers()` no longer accepts a Trainer. ([#19](https://github.com/HangJung97/PhiJAX/pull/19))
- Renamed the experiment logger method `log_hyperparameters()` to the Lightning-style `log_hyperparams()`. ([#19](https://github.com/HangJung97/PhiJAX/pull/19))

### Changed

- Made CSV, TensorBoard, and W&B logger resource setup lazy, idempotent, and restricted to global rank zero during
  Trainer tasks. Direct built-in logger use still starts resources automatically on first use. ([#19](https://github.com/HangJung97/PhiJAX/pull/19))

## [0.2.0b1] - 2026-08-31

### Added

- Concise `Trainer.fit()` and `Trainer.predict()` APIs that initialize models, state, balancing, batch routing, and
  compiled plans from a module blueprint, DataModule, optimizer, and seed. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- `ModelFactory`, balancer-owned adaptive update plans, objective equation composition, finite-pool batch-source
  construction, and safe `HostPool` defaults. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Independent persistent PRNG streams for model execution, sampling, and adaptive-balancer diagnostics. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Structured training diagnostics with host-side `PhiModule.log()` control for loggers and progress displays. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Automatic TQDM progress, plain model summaries, default TensorBoard or CSV logging, and versioned run directories.
  ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Monitored top-k checkpoints with persistent callback state and experimental TPU installation support. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))

### Breaking changes

- Renamed the explicit execution APIs to `Trainer.fit_state()` and `Trainer.predict_state()`. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Changed `PhiModule` into an immutable model-factory blueprint; the bound module is returned as `FitResult.module`.
  ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Changed `ResidualTerm` to accept its residual function first and infer its default NTK stream from equation metadata.
  ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Raised the checkpoint schema to version 2 for the expanded `TrainState` layout. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Moved adaptive cadence and diagnostic sampling settings into adaptive-balancer constructors. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Replaced the adaptive `skip_first_step` flag with cadence-anchored `update_start_step` scheduling. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Removed separate sampling and balancer key arguments from Trainer methods; both now live in `TrainState`. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Removed `resume_latest()` in favor of `fit(..., ckpt_path="last")`. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Removed the low-level `with_balancer_updates()` helper; advanced schedules now belong to `TrainingPlan`. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Removed `BalancerUpdateConfig` and `BalancerUpdateSchedule`; `TrainingPlan` now accepts one `BalancerUpdatePlan`.
  ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Removed the Hydra `balancer.factory` and `balancer.update` layers; balancer groups are directly instantiable. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Removed `MetricRoute`; modules now customize metric destinations with `self.log()` from `on_train_batch_end()`.
  ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Removed the `phijax.module` deep import path; use top-level imports or `phijax.core` for custom module contracts.
  ([#15](https://github.com/HangJung97/PhiJAX/pull/15))

### Changed

- Made DataModule input normalization opt-in through `input_statistics()`. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Made `LearningRateMonitor` require a configured experiment logger before fitting begins. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Moved adaptive-balancer scheduling into the Trainer host loop and removed unconditional per-step transfers of
  `TrainState.step`. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))
- Split module contracts into `phijax.core` and moved fit, prediction, logging, and signal orchestration into private
  loops and connectors while keeping the top-level API unchanged. ([#15](https://github.com/HangJung97/PhiJAX/pull/15))

## [0.1.0b1] - 2026-08-28

### Added

- Initial beta API for building, fitting, and evaluating physics-informed neural networks with JAX. ([#1](https://github.com/HangJung97/PhiJAX/pull/1))
- Lightning-inspired `Trainer`, `PhiModule`, `DataModule`, callback, logger, and checkpoint lifecycles. ([#1](https://github.com/HangJung97/PhiJAX/pull/1))
- Explicit model, optimizer, loss-balancer, PRNG, and training state for compiled updates. ([#1](https://github.com/HangJung97/PhiJAX/pull/1))
- Selective coordinate derivatives and reusable PDE, boundary, data-fidelity, and objective components. ([#1](https://github.com/HangJung97/PhiJAX/pull/1))
- Static, normalized-static, gradient-norm, and exact-NTK loss balancing. ([#1](https://github.com/HangJung97/PhiJAX/pull/1))
- MLP, Modified MLP, and PirateNet architectures with a common model initialization contract. ([#1](https://github.com/HangJung97/PhiJAX/pull/1))
- CPU, CUDA 12, and CUDA 13 installation options, configurable precision, and centralized device placement. ([#1](https://github.com/HangJung97/PhiJAX/pull/1))
- Versioned checkpoints and portable prediction artifacts for downstream evaluation. ([#1](https://github.com/HangJung97/PhiJAX/pull/1))
- Optional Hydra integration and Weights & Biases and TensorBoard logging. ([#1](https://github.com/HangJung97/PhiJAX/pull/1))

[0.1.0b1]: https://github.com/HangJung97/PhiJAX/releases/tag/v0.1.0b1
[0.2.0b1]: https://github.com/HangJung97/PhiJAX/releases/tag/v0.2.0b1
[0.2.0b2]: https://github.com/HangJung97/PhiJAX/releases/tag/v0.2.0b2
[0.2.0b3]: https://github.com/HangJung97/PhiJAX/releases/tag/v0.2.0b3
[0.2.0b4]: https://github.com/HangJung97/PhiJAX/releases/tag/v0.2.0b4
[0.2.0b5]: https://github.com/HangJung97/PhiJAX/releases/tag/v0.2.0b5
[unreleased]: https://github.com/HangJung97/PhiJAX/compare/v0.2.0b5...HEAD

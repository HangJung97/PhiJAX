# PhiJAX API reference

These focused pages generate signatures from PhiJAX's Google-style source docstrings.

## Core computation

- [PhiModule](api/module.md): module ownership, prediction, metrics, and lifecycle hooks.
- [Models](api/models.md): initialization contracts, MLP architecture, layers, features, and summaries.
- [Derivatives](api/derivatives.md): selective coordinate Jacobian and Hessian-diagonal utilities.
- [Equations](api/equations.md): fidelity, boundary, Burgers, Cartesian, polar, and spherical residuals.
- [Objectives](api/objectives.md): residual groups, scalar reduction, protocols, and composition.
- [Loss balancers](api/balancers.md): generic state, diagnostics, static, gradient-norm, and exact-NTK contracts.

## Data and runtime

- [Data](api/data.md): DataModules, immutable pools, samplers, batch sources, placement, IO, and artifacts.
- [Trainer and state](api/training.md): compiled steps, precision, device strategies, plans, and lifecycle ownership.
- [Callbacks](api/callbacks.md): contexts, prediction writing, stopping, progress, summaries, and checkpoints.
- [Checkpointing](api/checkpointing.md): Orbax persistence, manifests, complete resume, and weights-only loading.
- [Loggers](api/loggers.md): console, CSV, TensorBoard, W&B, and logger collections.
- [Evaluation](api/evaluation.md): evaluator contracts, metric primitives, and host-only output writers.

## Integrations and utilities

- [Configuration](api/configuration.md): optional Hydra factories and OmegaConf resolvers.
- [Utilities](api/utilities.md): reproducibility, rank-aware logging, formatting, and task finalization.
- [API overview](api/index.md): stability policy, conventions, and recommended imports.

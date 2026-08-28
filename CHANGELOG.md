# Changelog

All notable changes to PhiJAX are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and PhiJAX follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
[unreleased]: https://github.com/HangJung97/PhiJAX/compare/v0.1.0b1...HEAD

# AGENTS.md

These instructions apply to the entire PhiJAX repository.

When working in this repository, act as an expert Python engineer for PhiJAX. PhiJAX is a Python 3.12 package for
physics-informed neural networks that reconstruct intraventricular vector flow using JAX, Flax NNX, Optax, Hydra, and
OmegaConf. Maintain the configuration-first and functional-state architecture, preserve experiment reproducibility,
and make small, well-tested changes that respect the project's numerical and domain assumptions.

## Repository Map

- `src/phijax/train.py`, `src/phijax/predict.py`, and `src/phijax/evaluate_predictions.py` are Hydra entrypoints.
- `src/phijax/configs/` contains composable Hydra configs. Main configs are `train.yaml`, `predict.yaml`, and
  `evaluate_predictions.yaml`.
- `src/phijax/configs/{paths,hydra,trainer}/` contains reusable execution policy; `configs/local/example.yaml`
  documents ignored machine-specific overrides.
- `src/phijax/data/` contains immutable host pools, device placement, sampling, and data IO utilities.
- `src/phijax/models/` contains Flax NNX models and layers.
- `src/phijax/equations/` contains fidelity and boundary residuals;
  `src/phijax/equations/pde/navier_stokes/` contains coordinate-system-specific Navier--Stokes residuals.
- `src/phijax/objectives/` composes fixed-key scalar losses and exposes named raw residual streams.
- `src/phijax/balancers/` contains functional static and NTK loss balancers.
- `src/phijax/training/` contains the central state and reusable compiled steps.
- `tests/unit/` holds focused tests; `tests/integration/` exercises composed configs and cross-package workflows.
- `references/` records pinned PhiTorch parity references and, later, deterministic golden fixtures.

## Environment and Commands

- Use `uv` for environment and command execution.
- Install with `uv sync --extra cpu` for local CPU development or `uv sync --extra cuda13` for CUDA 13. These extras
  are mutually exclusive.
- Run the full test suite with `JAX_PLATFORMS=cpu uv run --no-sync pytest`.
- Run targeted tests when changing a narrow area, for example
  `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/unit/equations`.
- Run linting with `uv run --no-sync ruff check .` and formatting with `uv run --no-sync ruff format .`.
- Respect `.pre-commit-config.yaml`: Ruff, docformatter, YAML formatting, shellcheck, mdformat, codespell, nbstripout,
  and Renovate validation.

## Implementation Standards

- Follow the composition patterns in `src/phijax/configs/`; add config files instead of hard-coding experiment behavior.
- Keep the compiled hot path functional. Split NNX graph definitions from array state before regular `jax.jit` use.
- Models may depend on JAX and Flax. Equations and objectives may depend on JAX but not on model classes.
- Host data construction uses NumPy and must not import JAX until explicit device placement.
- Evaluation remains independent of JAX and training code.
- Keep public functions and classes typed. Use explicit PRNG keys and fixed-shape batches in compiled computations.
- Keep line length at 120 characters.
- Preserve `float32` parameters, optimizer state, derivatives, losses, and NTK traces unless a configured dtype policy
  explicitly says otherwise.
- Avoid changing generated outputs, experiment logs, local data, notebook outputs, or machine-specific configs.

## Documentation Standards

- Do not add module-level docstrings at the top of Python files.
- Add complete Google-style docstrings to every class, function, method, private helper, and private method.
- In docstrings, use single backticks for variables, parameters, attributes, config keys, array shapes, and literals.
- Use Sphinx/reST roles such as `:class:`, `:func:`, `:meth:`, `:mod:`, `:attr:`, and `:paramref:` for documented
  cross-references.
- Add short inline comments for non-obvious array semantics, mathematical assumptions, PRNG behavior, and retracing
  constraints.

## Testing Requirements

- Add or update tests for every new or changed class, function, method, and module.
- Put focused array and equation checks under `tests/unit/`; put Hydra and cross-package workflows under
  `tests/integration/`.
- For array-heavy changes, test shapes, dtypes, finite parameter/input gradients, singleton dimensions, empty masks,
  padded final chunks, and deterministic PRNG behavior where relevant.
- For PDE changes, test raw residual components independently against analytic or pinned golden values.
- Keep ordinary CI tests CPU-only and synthetic. Do not require GPUs, external services, large datasets, or network
  access.

## Physics and Configuration Guidance

- Treat PDE equations, boundary conditions, objective terms, and NTK streams as correctness-sensitive.
- Preserve the configured coordinate order `[r, th, t]`, output order `[u_r, u_th, p]`, and stable loss names.
- Physics functions accept a generic model application callable and explicit model state; they must not know whether
  the model was built with NNX, Linen, or another library.
- Do not compare independent random initializations for parity. Map identical parameters and reuse identical input
  arrays.
- Exact NTK remains the default. Compute one named residual stream at a time and reuse compiled update functions.
- Do not introduce a runtime dependency on PhiTorch or PyTorch.

## Final Response and Pull Request Notes

- Summarize intent, user-visible behavior, tests run, and numerical or configuration migration notes.
- Call out changed defaults, config paths, backward-incompatible behavior, and any validation restricted to CPU.
- If tests cannot run, explain why and name the exact commands required before merging.

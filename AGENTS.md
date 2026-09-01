# AGENTS.md

These instructions apply to the entire PhiJAX repository.

When working in this repository, act as an expert Python engineer for PhiJAX. PhiJAX is a Python 3.12+ framework for
physics-informed neural networks using JAX, Flax NNX, Optax, and Orbax. Maintain its functional-state architecture,
preserve experiment reproducibility, and keep domain applications and executable project configuration outside the
framework package. Hydra and OmegaConf are supported integrations, not assumptions of the core Trainer.

## Repository Map

- `src/phijax/data/` contains the `PhiDataModule` lifecycle, immutable host pools, explicit-key batch sources,
  placement, host transforms, and IO.
- `src/phijax/models/` contains Flax NNX models and layers.
- `src/phijax/module.py` defines the overridable `BasePhiModule` contract and generic objective-backed `PhiModule`.
- `src/phijax/equations/` contains fidelity and boundary residuals;
  `src/phijax/equations/pde/navier_stokes/` contains coordinate-system-specific Navier--Stokes residuals.
- `src/phijax/derivatives.py` contains selective forward-mode coordinate derivative utilities used by PDE residuals.
- `src/phijax/objectives/` defines generic scalar-loss and raw-residual-stream contracts.
- `src/phijax/evaluation/` defines host-only evaluator, metric, and artifact-output contracts.
- `src/phijax/balancers/` contains functional static and NTK loss balancers.
- `src/phijax/callbacks/` contains shared lifecycle contracts and training, prediction, and postprocessing callbacks.
- `src/phijax/training/` contains the central state and reusable compiled steps.
- `src/phijax/integrations/hydra/` owns optional Hydra instantiation and training-plan assembly.
- `src/phijax/integrations/omegaconf.py` contains reusable resolver registration.
- `tests/unit/` holds focused tests; `tests/integration/` exercises framework-wide runtime workflows.
- `phijax-hydra-template` owns runnable entrypoints, Hydra configs, and the Burgers example.
- `phijax-iVFM` owns the two-dimensional IVFM application, configs, references, and evaluation workflows.

## Environment and Commands

- Use `uv` for environment and command execution.
- Install with `uv sync` for local CPU development or `uv sync --extra cuda13` for CUDA 13. CUDA extras are mutually
  exclusive.
- Run the full test suite with `JAX_PLATFORMS=cpu uv run --no-sync pytest`.
- Run targeted tests when changing a narrow area, for example
  `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/unit/equations`.
- Run linting with `uv run --no-sync ruff check .` and formatting with `uv run --no-sync ruff format .`.
- Respect `.pre-commit-config.yaml`: Ruff, docformatter, YAML formatting, shellcheck, mdformat, codespell, nbstripout,
  and Renovate validation.

## Implementation Standards

- Keep executable config trees in downstream projects. Framework integrations must accept project-owned config objects.
- Keep the compiled hot path functional. Split NNX graph definitions from array state before regular `jax.jit` use.
- Keep optimizer and balancer ownership outside `BasePhiModule`; module hooks must return explicit state, batch, or metric
  replacements to the trainer.
- Invoke callbacks before the corresponding `BasePhiModule` lifecycle hook to preserve Lightning-compatible ordering.
- Models may depend on JAX and Flax. Equations and objectives may depend on JAX but not on model classes.
- Application packages may depend on generic PhiJAX packages. The framework must not import application packages.
- Host data construction uses NumPy and must not import JAX until explicit device placement.
- Evaluation remains independent of JAX and training code.
- Keep public functions and classes typed. Use explicit PRNG keys and fixed-shape batches in compiled computations.
- Do not perform logging, filesystem IO, NumPy conversion, configuration access, or Python-side mutation inside JAX
  transformations. Keep input PyTree structure, shapes, and dtypes stable across compiled calls.
- Avoid data-dependent array shapes and Python control flow over traced values. Keep static configuration outside
  `jax.jit`, `jax.grad`, `jax.vmap`, and `jax.lax` transformed functions.
- Keep line length at 120 characters.
- Preserve `float32` parameters, optimizer state, derivatives, losses, and NTK traces unless a configured dtype policy
  explicitly says otherwise.
- Avoid changing generated outputs, experiment logs, local data, notebook outputs, or machine-specific configs.

## Documentation Standards

- Write for readers who understand Python and basic JAX but may be new to PhiJAX. Define PhiJAX-specific terms before
  relying on them.
- Prefer plain language, concrete verbs, and short sentences. Keep one main idea per sentence, and split paragraphs
  that mix separate concepts, decisions, or lifecycle stages.
- Lead with the user-visible purpose or outcome. Follow with implementation details only when they help the reader use,
  extend, or debug the API.
- Keep guides task-oriented and scannable. Use descriptive headings, short examples, tables for exact mappings, and
  diagrams only when they clarify relationships or ordering.
- Put testing guidance in a separate section near the end of a guide. Keep common mistakes, troubleshooting, and next
  steps after it. API reference pages should describe validation and errors instead of repository test procedures.
- Preserve technical precision while simplifying prose. Do not replace established JAX, PINN, mathematical, or public
  API terms with vague wording.
- Avoid filler, promotional language, unnecessary qualifiers, and long lists embedded in prose. Prefer a short list
  when readers need to compare three or more items.
- Keep code examples executable and aligned with the supported public API. Explain non-obvious state ownership, array
  shapes, coordinate order, PRNG behavior, and device placement close to the example that uses them.
- In Markdown, link the first meaningful mention of a public class, function, or method to its generated API anchor.
  Do not link every repeated mention. Keep plain backticks for parameters, attributes, config keys, array shapes, and
  literals.
- Document constructor inputs once under the initializer's `Args` section. Reserve a class-level `Attributes` section
  for public runtime values that are not constructor inputs, following the separation used in Lightning's API pages.
- Before adopting a framework convention, compare its primary documentation or source with mature projects such as
  JAX, Lightning, PyTorch, or MONAI. Adapt the useful parts to PhiJAX without claiming unsupported API parity.
- Do not add module-level docstrings at the top of Python files.
- Add complete Google-style docstrings to every class, function, method, private helper, and private method.
- In docstrings, use single backticks for variables, parameters, attributes, config keys, array shapes, and literals.
- Use Sphinx/reST roles such as `:class:`, `:func:`, `:meth:`, `:mod:`, `:attr:`, and `:paramref:` for documented
  cross-references.
- Add short inline comments for non-obvious array semantics, mathematical assumptions, PRNG behavior, and retracing
  constraints.

## Coding Style Guidelines

- Prefer concise, self-explanatory code. Minimize comments and do not restate behavior that is evident from nearby
  names, types, or control flow.
- Use comments for non-local context that cannot be inferred from the code itself, especially array semantics,
  mathematical assumptions, PRNG behavior, device placement, and JAX tracing or recompilation constraints.
- Do not introduce a trivial one- or two-line helper used only once unless it materially improves readability or
  isolates a meaningful JAX transformation, tracing boundary, or reusable contract. Keep simple operations at their
  call site.
- Prefer clear abstractions and explicit state. Declare class attributes and PyTree structure deliberately; do not
  create hidden runtime state through dynamic `setattr` and `getattr` patterns.
- Match existing naming, typing, configuration, and architectural patterns before introducing a new abstraction.
- Assume readers understand Python and the JAX ecosystem, but may not know the surrounding PhiJAX subsystem.
- When the 120-character limit forces awkward wrapping, first shorten names or introduce a descriptive local variable.
  Do not split a simple expression into a less readable shape merely to satisfy formatting.
- Use ASCII in newly added or rewritten code comments. Leave existing non-ASCII comments unchanged unless the task
  otherwise requires editing them.
- When several implementations are equally correct, choose the simpler and more concise one.

## Change and Tooling Discipline

- Treat public imports and integration contracts as user-facing APIs. When moving or renaming one, update package
  exports, documentation, tests, and affected downstream project usage in the same change.
- Keep callback, logger, optimizer, scheduler, and balancer options in their owning downstream project config groups.
- If a required command is unavailable, check the project `.venv` before using an alternative. Do not install or
  upgrade tools unless the user's request requires an environment change.
- Put temporary scripts, extracted diagnostics, and throwaway benchmarks under `/tmp`; do not leave scratch artifacts
  in source, test, documentation, data, or reference directories.
- Do not commit, amend, push, open a pull request, or modify remote state unless the user explicitly requests it.
- Before a requested commit, run the relevant tests and pre-commit checks. Record the literal validation commands in
  the proposed commit or pull-request test plan.
- Keep external integrations optional and lazily imported. Never place credentials in Hydra configs because resolved
  configurations may be printed and saved with the run.
- Restrict filesystem writes, console output, external logging, and checkpoint commits to global rank zero unless the
  implementation explicitly provides safe distributed semantics.
- Ensure resource-owning callbacks, loggers, DataModules, and checkpoint backends release resources after success,
  interruption, callback-requested stopping, and exceptions.

## Testing Requirements

- Add or update tests for every new or changed class, function, method, and module.
- Put focused array and equation checks under `tests/unit/`; put cross-component framework workflows under
  `tests/integration/`. Hydra composition and application workflows belong in downstream repositories.
- For array-heavy changes, test shapes, dtypes, finite parameter/input gradients, singleton dimensions, empty masks,
  padded final chunks, and deterministic PRNG behavior where relevant.
- For PDE changes, test raw residual components independently against analytic or pinned golden values.
- Keep ordinary CI tests CPU-only and synthetic. Do not require GPUs, external services, large datasets, or network
  access.

## Physics and Configuration Guidance

- Treat PDE equations, boundary conditions, objective terms, and NTK streams as correctness-sensitive.
- Do not encode application coordinate order, output semantics, loss names, or data formats in generic interfaces.
- Physics functions accept a generic model application callable and explicit model state; they must not know whether
  the model was built with NNX, Linen, or another library.
- Do not compare independent random initializations for parity. Map identical parameters and reuse identical input
  arrays.
- Exact NTK remains the default. Compute one named residual stream at a time and reuse compiled update functions.
- Do not introduce a runtime dependency on an application's external numerical reference.

## Final Response and Pull Request Notes

- Summarize intent, user-visible behavior, tests run, and numerical or configuration migration notes.
- Call out changed defaults, config paths, backward-incompatible behavior, and any validation restricted to CPU.
- If tests cannot run, explain why and name the exact commands required before merging.

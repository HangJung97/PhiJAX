# Contributing to PhiJAX

Thank you for helping improve PhiJAX. Contributions may include bug fixes, tests, documentation, reusable equations,
training features, and performance improvements.

## Set up the development environment

PhiJAX requires Python 3.12 or newer and uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/HangJung97/PhiJAX.git
cd PhiJAX
uv sync --group all
uv run --no-sync pre-commit install
```

The aggregate `all` group includes the `dev` and `docs` groups. `dev` installs testing, linting, typing, and pre-commit
tools, while `docs` installs MkDocs and its documentation plugins. For local GPU development, add the extra matching
the installed CUDA runtime:

```bash
uv sync --group all --extra cuda12
# or
uv sync --group all --extra cuda13
```

The CUDA extras are mutually exclusive. Ordinary tests must remain CPU-compatible even when a change also adds GPU
coverage.

## Make a focused change

Create a branch from the target development branch and keep each pull request focused on one behavior. Follow the
existing architecture and public contracts described in `AGENTS.md`.

When changing a public import, config path, callback hook, artifact schema, or checkpoint layout, update the related
tests and documentation in the same pull request. Mark backward-incompatible changes clearly in the pull-request
description.

## Test the change

Run the smallest relevant tests while developing, followed by the complete CPU suite:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest tests/unit/<area>
JAX_PLATFORMS=cpu uv run --no-sync pytest --cov=phijax --cov-report=term-missing
```

New or changed behavior requires tests. Keep ordinary CI tests synthetic and independent of GPUs, network services,
and external datasets. Physics changes should test individual residual components against analytic or pinned values.

## Run quality checks

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync mkdocs build --strict
uv run --no-sync pre-commit run --all-files
```

Pre-commit runs automatically on future commits after `pre-commit install`. Some hooks can modify files; review those
changes and rerun the command until every hook passes.

## Submit a pull request

Before submitting:

1. Explain the motivation and user-visible behavior.
2. Link related issues when applicable.
3. List breaking changes, changed defaults, and migration steps.
4. Record the validation commands you ran.
5. Note checks that require GPU or multi-host infrastructure and were not run locally.

Do not include generated documentation sites, experiment logs, local datasets, credentials, or machine-specific
configuration in a pull request.

# Installation

PhiJAX requires Python 3.12 or newer. The default installation supports CPU execution and is sufficient for the
quickstart, documentation examples, and ordinary tests.

We recommend [uv](https://docs.astral.sh/uv/); install it by following the
[official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
PhiJAX and the [quickstart example](quickstart.md) also work with pip alone in a standard Python virtual environment.

## CPU

Inside your own [uv project](https://docs.astral.sh/uv/guides/projects/), add PhiJAX as a dependency:

```bash
uv add phijax
```

For a new project, first run `uv init my-pinn --python 3.12` and `cd my-pinn`.
`uv add` updates the project's dependencies, lockfile, and virtual environment.

For pip, create and activate a virtual environment first (macOS and Linux):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install phijax
```

Verify the installation and check which device JAX selected:

```bash
# With uv
uv run python -c "import jax, phijax; print(phijax.__version__); print(jax.devices())"

# Or with pip in the activated venv
python -c "import jax, phijax; print(phijax.__version__); print(jax.devices())"
```

## NVIDIA GPU

Install one extra that matches the available NVIDIA driver and CUDA runtime:

```bash
# With uv (choose one)
uv add "phijax[cuda12]"
# or
uv add "phijax[cuda13]"

# Or with pip in the activated venv (choose one)
python -m pip install "phijax[cuda12]"
# or
python -m pip install "phijax[cuda13]"
```

The extras are mutually exclusive. Confirm that JAX can see the accelerator before starting a long experiment:

```bash
# With uv
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"

# Or with pip in the activated venv
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Refer to the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for supported platforms,
drivers, and accelerator-specific limitations.

## Experimental TPU

Install the mutually exclusive TPU extra in a TPU runtime:

```bash
uv add "phijax[tpu]"
# Or with pip in the activated venv
python -m pip install "phijax[tpu]"
```

TPU support is experimental. Device discovery and placement follow JAX, but PhiJAX does not run real TPU jobs in its
ordinary CI. Do not combine `tpu` with `cuda12` or `cuda13`, and validate the intended topology before a long run.

## Optional experiment loggers

TensorBoard and Weights & Biases remain optional:

```bash
uv add "phijax[tensorboard]"
uv add "phijax[wandb]"

# Or with pip in the activated venv
python -m pip install "phijax[tensorboard]"
python -m pip install "phijax[wandb]"
```

Extras can be combined:

```bash
uv add "phijax[cuda13,wandb,tensorboard]"
# Or with pip in the activated venv
python -m pip install "phijax[cuda13,wandb,tensorboard]"
```

## Development installation

Clone the repository and install all testing and documentation tools with `uv`:

```bash
git clone https://github.com/HangJung97/PhiJAX.git
cd PhiJAX
uv sync --group all
uv run --no-sync pre-commit install
```

Continue with the [heat-equation quickstart](quickstart.md), or see [Troubleshooting](../guides/troubleshooting.md) if
JAX selects an unexpected backend.

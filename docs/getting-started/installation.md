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

## Optional features

Hydra configuration and MATLAB or HDF5 files need optional extras:

| Extra   | Installs                               | Needed for                                                                 |
| ------- | -------------------------------------- | -------------------------------------------------------------------------- |
| `hydra` | `hydra-core`, `omegaconf`, `rootutils` | `phijax.integrations`, `task_wrapper()`, `extras()`, `pre_hydra_routine()` |
| `io`    | `h5py`, `mat73`, `scipy`               | `.mat` and HDF5 files in `load_arrays()`, `save_mat=True` artifacts        |

```bash
uv add "phijax[hydra,io]"
# Or with pip in the activated venv
python -m pip install "phijax[hydra,io]"
```

Without the extra, these features raise `ModuleNotFoundError` naming the extra to install. Training, NPZ files, and
prediction artifacts need neither. The [Hydra template](https://github.com/HangJung97/phijax-hydra-template) installs
both.

## Optional experiment loggers

TensorBoard and Weights & Biases remain optional:

```bash
uv add "phijax[tensorboard]"
uv add "phijax[wandb]"

# Or with pip in the activated venv
python -m pip install "phijax[tensorboard]"
python -m pip install "phijax[wandb]"
```

Extras can be combined. `all` installs every optional feature: `hydra`, `io`, `tensorboard`, and `wandb`. It contains
no accelerator, so add at most one:

```bash
uv add "phijax[all,cuda13]"
# Or with pip in the activated venv
python -m pip install "phijax[all,cuda13]"
```

## Development installation

Clone the repository and install all testing and documentation tools with `uv`. The `all` dependency group holds
repository tools and is separate from the `all` extra. Its test tools include the `hydra` and `io` extras:

```bash
git clone https://github.com/HangJung97/PhiJAX.git
cd PhiJAX
uv sync --group all
uv run --no-sync pre-commit install
```

Continue with the [heat-equation quickstart](quickstart.md), or see [Troubleshooting](../guides/troubleshooting.md) if
JAX selects an unexpected backend.

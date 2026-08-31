# Installation

PhiJAX requires Python 3.12 or newer. The default installation supports CPU execution and is sufficient for the
quickstart, documentation examples, and ordinary tests.

## CPU

```bash
python -m pip install phijax
```

Verify the installation and check which device JAX selected:

```bash
python -c "import jax, phijax; print(phijax.__version__); print(jax.devices())"
```

## NVIDIA GPU

Install one extra that matches the available NVIDIA driver and CUDA runtime:

```bash
python -m pip install "phijax[cuda12]"
# or
python -m pip install "phijax[cuda13]"
```

The extras are mutually exclusive. Confirm that JAX can see the accelerator before starting a long experiment:

```bash
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Refer to the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for supported platforms,
drivers, and accelerator-specific limitations.

## Experimental TPU

Install the mutually exclusive TPU extra in a TPU runtime:

```bash
python -m pip install "phijax[tpu]"
```

TPU support is experimental. Device discovery and placement follow JAX, but PhiJAX does not run real TPU jobs in its
ordinary CI. Do not combine `tpu` with `cuda12` or `cuda13`, and validate the intended topology before a long run.

## Optional experiment loggers

TensorBoard and Weights & Biases remain optional:

```bash
python -m pip install "phijax[tensorboard]"
python -m pip install "phijax[wandb]"
```

Extras can be combined:

```bash
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

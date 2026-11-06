import argparse
from collections.abc import Mapping
from functools import partial
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from phijax import PhiModule, Trainer, hessian_diagonal, value_and_jacobian
from phijax.callbacks import RichModelSummary, RichProgressBar
from phijax.data import BatchSize, HostPool, PhiDataModule, input_statistics
from phijax.data.datamodule import DataStage
from phijax.equations import base_data_fidelity, residual_equation
from phijax.evaluation import regression_metrics
from phijax.models import build_mlp
from phijax.objectives import CompositeObjective
from phijax.training import Accelerator, PrecisionName
from phijax.types import ArrayMapping, ModelApply, ResidualOutput, ResidualStream


@residual_equation(names=("heat",))
def heat_equation(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    diffusivity: float = 0.1,
    stream: ResidualStream = "residual",
) -> ResidualOutput:
    """Evaluate `du/dt - diffusivity * d2u/dx2 = 0`.

    Args:
        model_apply: Pure explicit-state model application callable.
        model_state: Differentiable model parameter PyTree.
        batch: Arrays containing rank-two `[t, x]` coordinates under `inputs`.
        diffusivity: Positive heat-diffusion coefficient.
        stream: Requested equation representation. The objective requests only `residual` from this equation.

    Returns:
        Residuals with shape `[samples]`.

    Raises:
        ValueError: If the diffusivity or input shape is invalid.
    """
    if diffusivity <= 0.0:
        raise ValueError("`diffusivity` must be positive.")
    inputs = batch["inputs"]
    if inputs.ndim != 2 or inputs.shape[-1] != 2:
        raise ValueError("Heat-equation inputs must have shape `[samples, 2]` with columns `[t, x]`.")

    def solutions(t: jax.Array, x: jax.Array) -> jax.Array:
        """Evaluate the scalar temperature at one coordinate.

        Args:
            t: Scalar time coordinate.
            x: Scalar spatial coordinate.

        Returns:
            Model output 0 at the point `[t, x]`.
        """
        return model_apply(model_state, jnp.stack((t, x)))[0]

    def residual(t: jax.Array, x: jax.Array) -> jax.Array:
        """Evaluate the heat equation at one coordinate.

        Args:
            t: Scalar time coordinate.
            x: Scalar spatial coordinate.

        Returns:
            Scalar residual `du/dt - diffusivity * d2u/dx2`.
        """
        _, (du_dt,) = value_and_jacobian(solutions, 0)(t, x)
        (d2u_dx2,) = hessian_diagonal(solutions, 1)(t, x)
        return du_dt - diffusivity * d2u_dx2

    return jax.vmap(residual)(inputs[:, 0], inputs[:, 1])  # [samples]


class HeatDataModule(PhiDataModule):
    """Provide initial, boundary, collocation, and prediction coordinates."""

    def __init__(
        self,
        *,
        initial_size: int = 64,
        boundary_size: int = 32,
        pde_shape: tuple[int, int] = (16, 32),
        predict_shape: tuple[int, int] = (21, 51),
        batch_sizes: Mapping[str, BatchSize] | None = None,
    ) -> None:
        """Store finite source sizes.

        Args:
            initial_size: Number of initial-condition coordinates.
            boundary_size: Number of time coordinates sampled on each boundary edge.
            pde_shape: Number of candidate coordinates along the time and space axes.
            predict_shape: Number of ordered prediction coordinates along the time and space axes.
            batch_sizes: Optional per-pool batch sizes; `predict` sets the prediction chunk size. Finite pools also
                accept `"all"`.
        """
        defaults = {"initial": 32, "boundary": 32, "pde": 64, "predict": 128}
        super().__init__(batch_sizes=defaults if batch_sizes is None else batch_sizes)
        self.initial_size = initial_size
        self.boundary_size = boundary_size
        self.pde_shape = pde_shape
        self.predict_shape = predict_shape

    def setup(self, stage: DataStage) -> None:
        """Construct immutable pools for one Trainer stage.

        Args:
            stage: Requested `fit` or `predict` stage.

        Raises:
            ValueError: If `stage` is unsupported.
        """
        if stage == "fit":
            initial_x = np.linspace(0.0, 1.0, self.initial_size, dtype=np.float32)
            initial_inputs = np.column_stack((np.zeros_like(initial_x), initial_x))
            initial_targets = np.sin(np.pi * initial_x)[:, None].astype(np.float32)

            boundary_t = np.linspace(0.0, 1.0, self.boundary_size, dtype=np.float32)
            left = np.column_stack((boundary_t, np.zeros_like(boundary_t)))
            right = np.column_stack((boundary_t, np.ones_like(boundary_t)))
            boundary_inputs = np.concatenate((left, right), axis=0)

            pde_t = np.linspace(0.0, 1.0, self.pde_shape[0], dtype=np.float32)
            pde_x = np.linspace(0.0, 1.0, self.pde_shape[1], dtype=np.float32)
            pde_inputs = np.stack(np.meshgrid(pde_t, pde_x, indexing="ij"), axis=-1)
            self.pools = {
                "initial": HostPool(initial_inputs, initial_targets),
                "boundary": HostPool(boundary_inputs, np.zeros((len(boundary_inputs), 1), dtype=np.float32)),
                "pde": HostPool.from_grid(pde_inputs),
            }
            return
        if stage == "predict":
            times = np.linspace(0.0, 1.0, self.predict_shape[0], dtype=np.float32)
            positions = np.linspace(0.0, 1.0, self.predict_shape[1], dtype=np.float32)
            # Coordinates have shape `[time, position, 2]`; the pool keeps this grid for reconstruction.
            inputs = np.stack(np.meshgrid(times, positions, indexing="ij"), axis=-1)
            targets = np.exp(-0.1 * np.pi**2 * inputs[..., :1]) * np.sin(np.pi * inputs[..., 1:])
            self.pools = {"predict": HostPool.from_grid(inputs, targets.astype(np.float32))}
            return
        raise ValueError(f"Unsupported stage: {stage}")

    def input_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute normalization statistics from the finite training pools.

        Returns:
            Per-coordinate mean and safely positive standard deviation arrays.
        """
        return input_statistics(self.normalization_pools())


def run_quickstart(
    max_steps: int = 500,
    *,
    accelerator: Accelerator = "cpu",
    precision: PrecisionName = "32-true",
    enable_progress_bar: bool = True,
) -> float:
    """Train the heat PINN and return its maximum prediction error.

    Args:
        max_steps: Positive number of optimizer updates.
        accelerator: `auto`, `cpu`, `gpu`, or `tpu` backend selected by the Trainer.
        precision: Canonical PhiJAX precision mode used for model construction and training.
        enable_progress_bar: Whether to display Rich fit and prediction progress.

    Returns:
        Maximum absolute error against the analytical solution on the prediction grid.
    """
    # Print the initialized network architecture once before training starts.
    callbacks = (RichModelSummary(),)
    if enable_progress_bar:
        # Refresh the optional Rich display every 10 optimizer steps.
        callbacks += (RichProgressBar(refresh_rate=10),)
    trainer = Trainer(
        max_steps=max_steps,
        accelerator=accelerator,
        precision=precision,
        log_every_n_steps=100,
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks,
    )
    data_module = HeatDataModule()
    model = partial(
        build_mlp,
        input_dim=2,
        output_dim=1,
        hidden=(32, 32),
        activation="tanh",
        input_norm=True,
    )
    objective = CompositeObjective.from_equations(
        {
            "initial": base_data_fidelity,
            "boundary": base_data_fidelity,
            "pde": heat_equation,
        }
    )
    module = PhiModule(model, objective, name="Heat PINN")
    optimizer = optax.adam(learning_rate=1.0e-3)

    training_started = perf_counter()
    result = trainer.fit(module, datamodule=data_module, optimizer=optimizer, seed=0)
    jax.block_until_ready(result.state)
    training_time = perf_counter() - training_started
    predictions = trainer.predict(result, datamodule=data_module)

    if predictions is None:
        raise RuntimeError("The quickstart DataModule did not produce prediction batches.")
    prediction_pool = data_module.prediction_pool()
    if prediction_pool is None:
        raise RuntimeError("The quickstart DataModule did not retain its prediction pool.")
    assert prediction_pool.targets is not None
    metrics = regression_metrics(np.asarray(predictions), prediction_pool.targets)
    maximum_error = metrics["max_absolute_error"]
    relative_l2_error = metrics["relative_l2_error"]
    print(
        f"Finished {result.iterations} steps in {training_time:.2f}s; relative L2 error: {relative_l2_error:.3e}; "
        f"maximum absolute error: {maximum_error:.3e}"
    )
    return maximum_error


def main() -> None:
    """Parse the training length and run the quickstart example."""
    parser = argparse.ArgumentParser(description="Train a small one-dimensional heat-equation PINN.")
    parser.add_argument("--max-steps", type=int, default=500, help="Number of optimizer updates.")
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu", "tpu"),
        default="cpu",
        help="JAX accelerator backend. Defaults to CPU for a safe first run.",
    )
    parser.add_argument(
        "--precision",
        choices=("64-true", "32-true", "16-true", "bf16-true", "16-mixed", "bf16-mixed"),
        default="32-true",
        help="Model and training precision mode.",
    )
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable fit and prediction progress output.")
    arguments = parser.parse_args()
    run_quickstart(
        arguments.max_steps,
        accelerator=arguments.accelerator,
        precision=arguments.precision,
        enable_progress_bar=not arguments.no_progress_bar,
    )


if __name__ == "__main__":
    main()

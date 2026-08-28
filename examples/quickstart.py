import argparse
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from phijax import PhiModule, Trainer, TrainingPlan, hessian_diagonal, value_and_jacobian
from phijax.balancers import StaticLossBalancer
from phijax.data import ChunkedPredictionSource, HostPool, NamedBatchSource, PhiDataModule, RandomRowSampler
from phijax.data.datamodule import DataStage
from phijax.equations import base_data_fidelity, residual_equation
from phijax.evaluation import regression_metrics
from phijax.models import build_mlp
from phijax.objectives import CompositeObjective, ResidualTerm
from phijax.training import configure_precision
from phijax.types import ArrayMapping, ModelApply, ResidualGroups, ResidualStream


@residual_equation(names=("heat",))
def heat_equation(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    diffusivity: float = 0.1,
    stream: ResidualStream = "residual",
) -> ResidualGroups:
    """Evaluate `du/dt - diffusivity * d2u/dx2 = 0`.

    Args:
        model_apply: Pure explicit-state model application callable.
        model_state: Differentiable model parameter PyTree.
        batch: Arrays containing rank-two `[t, x]` coordinates under `inputs`.
        diffusivity: Positive heat-diffusion coefficient.
        stream: Requested equation representation; only `residual` is supported.

    Returns:
        One residual group with shape `[samples, 1]`.

    Raises:
        ValueError: If the stream, diffusivity, or input shape is invalid.
    """
    if stream != "residual":
        raise ValueError("The heat equation supports only the `residual` stream.")
    if diffusivity <= 0.0:
        raise ValueError("`diffusivity` must be positive.")
    inputs = batch["inputs"]
    if inputs.ndim != 2 or inputs.shape[-1] != 2:
        raise ValueError("Heat-equation inputs must have shape `[samples, 2]` with columns `[t, x]`.")

    def scalar_prediction(state: Any, time: jax.Array, position: jax.Array) -> jax.Array:
        """Select the scalar temperature prediction at one coordinate.

        Args:
            state: Differentiable model parameter PyTree.
            time: Scalar time coordinate.
            position: Scalar spatial coordinate.

        Returns:
            Scalar network prediction.
        """
        return model_apply(state, jnp.stack((time, position)))[0]

    value_and_time_derivative = value_and_jacobian(scalar_prediction, (1,))
    second_spatial_derivative = hessian_diagonal(scalar_prediction, 2)

    def point_residual(point: jax.Array) -> jax.Array:
        """Evaluate the heat equation at one coordinate.

        Args:
            point: Coordinate vector ordered as `[t, x]`.

        Returns:
            Scalar residual `du/dt - diffusivity * d2u/dx2`.
        """
        _, (du_dt,) = value_and_time_derivative(model_state, point[0], point[1])
        (d2u_dx2,) = second_spatial_derivative(model_state, point[0], point[1])
        return du_dt - jnp.asarray(diffusivity, dtype=du_dt.dtype) * d2u_dx2

    residuals = jax.vmap(point_residual)(inputs)[:, None]
    return ((residuals,),)


class HeatDataModule(PhiDataModule):
    """Provide initial, boundary, collocation, and prediction coordinates."""

    def __init__(
        self,
        *,
        initial_size: int = 64,
        boundary_size: int = 32,
        pde_shape: tuple[int, int] = (16, 32),
        predict_shape: tuple[int, int] = (21, 51),
    ) -> None:
        """Store finite source sizes.

        Args:
            initial_size: Number of initial-condition coordinates.
            boundary_size: Number of time coordinates sampled on each boundary edge.
            pde_shape: Number of candidate coordinates along the time and space axes.
            predict_shape: Number of ordered prediction coordinates along the time and space axes.
        """
        super().__init__()
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
            pde_inputs = np.stack(np.meshgrid(pde_t, pde_x, indexing="ij"), axis=-1).reshape(-1, 2)
            self.pools = {
                "initial": _pool(initial_inputs, initial_targets),
                "boundary": _pool(
                    boundary_inputs,
                    np.zeros((boundary_inputs.shape[0], 1), dtype=np.float32),
                ),
                "pde": _pool(pde_inputs, np.zeros((pde_inputs.shape[0], 0), dtype=np.float32)),
            }
            return
        if stage == "predict":
            times = np.linspace(0.0, 1.0, self.predict_shape[0], dtype=np.float32)
            positions = np.linspace(0.0, 1.0, self.predict_shape[1], dtype=np.float32)
            inputs = np.stack(np.meshgrid(times, positions, indexing="ij"), axis=-1).reshape(-1, 2)
            targets = (np.exp(-0.1 * np.pi**2 * inputs[:, :1]) * np.sin(np.pi * inputs[:, 1:2])).astype(np.float32)
            self.pools = {"predict": _pool(inputs, targets, reference_shape=self.predict_shape)}
            return
        raise ValueError(f"Unsupported stage: {stage}")

    def train_batch_source(self, batch_keys: tuple[str, ...], key: jax.Array) -> NamedBatchSource:
        """Build deterministic finite-row training samplers.

        Args:
            batch_keys: Objective batch keys requested by the training plan.
            key: Explicit root sampling key.

        Returns:
            Step-indexed initial, boundary, and collocation batches.
        """
        pools = self._require_setup("fit")
        samplers = {name: RandomRowSampler(pools[name].fields()) for name in batch_keys}
        policies = {"initial": 32, "boundary": 32, "pde": 64}
        return NamedBatchSource(samplers, {name: policies[name] for name in batch_keys}, key)

    def predict_batch_source(self) -> ChunkedPredictionSource:
        """Build finite ordered prediction batches.

        Returns:
            Prediction source with at most 128 coordinates per batch.
        """
        pool = self.prediction_pool()
        if pool is None:
            raise RuntimeError("Prediction data is unavailable before the `predict` stage is prepared.")
        return ChunkedPredictionSource(pool, 128)

    def prediction_pool(self) -> HostPool | None:
        """Return the ordered pool represented by prediction batches.

        Returns:
            Prediction pool containing the analytical solution, or `None` during fitting.
        """
        return self.pools.get("predict")


def _pool(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    reference_shape: tuple[int, ...] | None = None,
) -> HostPool:
    """Create one immutable heat-equation host pool.

    Args:
        inputs: Rank-two coordinate array.
        targets: Rank-two target array with the same row count.
        reference_shape: Optional dense reconstruction shape.

    Returns:
        Pool with coordinate and reconstruction metadata.
    """
    size = inputs.shape[0]
    return HostPool(
        inputs=inputs,
        targets=targets,
        aux={},
        metadata={"coordinate_names": ("t", "x")},
        reference_shape=reference_shape or (size,),
        flat_index=np.arange(size),
    )


def run_quickstart(
    max_steps: int = 500,
    *,
    accelerator: str = "cpu",
    precision: str = "32-true",
) -> float:
    """Train the heat PINN and return its maximum prediction error.

    Args:
        max_steps: Positive number of optimizer updates.
        accelerator: `auto`, `cpu`, `gpu`, or `tpu` backend selected by the Trainer.
        precision: Canonical PhiJAX precision mode used for model construction and training.

    Returns:
        Maximum absolute error against the analytical solution on the prediction grid.
    """
    configure_precision(precision)
    trainer = Trainer(
        max_steps=max_steps,
        accelerator=accelerator,
        precision=precision,
        log_every_n_steps=100,
    )
    trainer.print_environment_info()
    data_module = HeatDataModule()
    data_module.prepare_stage("fit")
    input_mean, input_std = data_module.input_statistics()

    # Keep parameter initialization, Trainer state, and batch sampling on independent PRNG streams.
    model_key, state_key, sampling_key = jax.random.split(jax.random.key(0), 3)
    initialized = build_mlp(
        model_key,
        input_dim=2,
        output_dim=1,
        input_mean=input_mean,
        input_std=input_std,
        hidden=(32, 32),
        activation="tanh",
        input_norm=True,
        precision=trainer.precision,
    )
    objective = CompositeObjective(
        {
            "initial": ResidualTerm(
                residual_fn=base_data_fidelity,
                batch_key="initial",
                ntk_stream="output",
            ),
            "boundary": ResidualTerm(
                residual_fn=base_data_fidelity,
                batch_key="boundary",
                ntk_stream="output",
            ),
            "pde": ResidualTerm(residual_fn=heat_equation, batch_key="pde"),
        }
    )
    module = PhiModule(initialized.apply, objective, name="Heat PINN", model_summary=initialized.summary)
    balancer = StaticLossBalancer(module.loss_names)
    optimizer = optax.adam(learning_rate=1.0e-3)
    state = trainer.initialize_state(initialized.state, optimizer, balancer.initialize(), state_key)
    batch_keys = ("initial", "boundary", "pde")
    plan = TrainingPlan(trainer.compile_train_step(module, balancer, optimizer), batch_keys)

    training_started = perf_counter()
    result = trainer.fit(module, plan, state, datamodule=data_module, sampling_key=sampling_key)
    jax.block_until_ready(result.state)
    training_time = perf_counter() - training_started
    predictions = trainer.predict(module, result.state, datamodule=data_module)

    if predictions is None:
        raise RuntimeError("The quickstart DataModule did not produce prediction batches.")
    prediction_pool = data_module.prediction_pool()
    if prediction_pool is None:
        raise RuntimeError("The quickstart DataModule did not retain its prediction pool.")
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
    arguments = parser.parse_args()
    run_quickstart(
        arguments.max_steps,
        accelerator=arguments.accelerator,
        precision=arguments.precision,
    )


if __name__ == "__main__":
    main()

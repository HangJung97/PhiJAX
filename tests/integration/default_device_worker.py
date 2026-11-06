import json
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from phijax import PhiModule, Trainer
from phijax.callbacks import Callback, PredictionContext, TrainerContext
from phijax.data import HostPool, PhiDataModule
from phijax.data.datamodule import DataStage
from phijax.equations import base_data_fidelity
from phijax.models import build_mlp
from phijax.objectives import CompositeObjective
from phijax.training import SingleDeviceStrategy


class _LineDataModule(PhiDataModule):
    """Provide one small supervised pool and a prediction pool through the default sources."""

    def __init__(self) -> None:
        """Sample four rows per step and predict in chunks of four rows."""
        super().__init__(batch_sizes={"data": 4, "predict": 4})

    def setup(self, stage: DataStage) -> None:
        """Build the pool for one stage.

        Args:
            stage: Requested `fit` or `predict` stage.
        """
        inputs = np.linspace(0.0, 1.0, 8, dtype=np.float32)[:, None]
        pool = HostPool(inputs, 2.0 * inputs)
        self.pools = {"data": pool} if stage == "fit" else {"predict": pool}


class _ImplicitArrayDevices(Callback):
    """Record the device id of an array created without explicit placement inside Trainer hooks."""

    def __init__(self) -> None:
        """Start with no recorded devices."""
        self.devices: dict[str, int] = {}

    def on_train_batch_start(self, context: TrainerContext) -> None:
        """Record the default device during a training step.

        Args:
            context: Current trainer context.
        """
        del context
        self.devices["fit"] = next(iter(jnp.zeros(()).devices())).id

    def on_predict_batch_start(self, context: PredictionContext) -> None:
        """Record the default device during prediction.

        Args:
            context: Current prediction context.
        """
        del context
        self.devices["predict"] = next(iter(jnp.zeros(()).devices())).id


def main(output: Path) -> None:
    """Fit and predict on the second CPU device and write the device ids of implicitly created arrays.

    Args:
        output: JSON result path.
    """
    recorder = _ImplicitArrayDevices()
    trainer = Trainer(
        max_steps=2,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[1]),
        callbacks=(recorder,),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    module = PhiModule(
        partial(build_mlp, input_dim=1, output_dim=1, hidden=(4,)),
        CompositeObjective.from_equations({"data": base_data_fidelity}),
    )
    datamodule = _LineDataModule()
    result = trainer.fit(module, datamodule=datamodule, optimizer=optax.sgd(1.0e-2), seed=0)
    trainer.predict(result, datamodule=datamodule)
    state_devices = {device.id for leaf in jax.tree.leaves(result.state.model_state) for device in leaf.devices()}
    results = {
        **recorder.devices,
        "after": next(iter(jnp.zeros(()).devices())).id,
        "state": sorted(state_devices),
    }
    output.write_text(json.dumps(results))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

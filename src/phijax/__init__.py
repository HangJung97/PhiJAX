from importlib.metadata import version
from typing import TYPE_CHECKING

from phijax._lazy import lazy_exports

if TYPE_CHECKING:
    from phijax.balancers import LossBalancer
    from phijax.core import BasePhiModule, PhiModule, PhiModuleContext
    from phijax.data import PhiDataModule
    from phijax.derivatives import hessian_diagonal, value_and_jacobian
    from phijax.metrics import TrainingOutput
    from phijax.models import InitializedModel, ModelFactory
    from phijax.training import FitResult, Trainer
    from phijax.training.assembly import build_training_plan
    from phijax.training.plans import TrainingPlan
    from phijax.training.state import TrainState
    from phijax.types import InputStatistics

    DataModule = PhiDataModule

__version__ = version("phijax")

__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "BasePhiModule": "phijax.core",
        "DataModule": "phijax.data.datamodule:PhiDataModule",
        "FitResult": "phijax.training",
        "InitializedModel": "phijax.models",
        "InputStatistics": "phijax.types",
        "LossBalancer": "phijax.balancers",
        "ModelFactory": "phijax.models",
        "PhiDataModule": "phijax.data",
        "PhiModule": "phijax.core",
        "PhiModuleContext": "phijax.core",
        "TrainState": "phijax.training.state",
        "Trainer": "phijax.training",
        "TrainingOutput": "phijax.metrics",
        "TrainingPlan": "phijax.training.plans",
        "build_training_plan": "phijax.training.assembly",
        "hessian_diagonal": "phijax.derivatives",
        "value_and_jacobian": "phijax.derivatives",
    },
)

__all__ = [
    "BasePhiModule",
    "DataModule",
    "FitResult",
    "InitializedModel",
    "InputStatistics",
    "LossBalancer",
    "ModelFactory",
    "PhiDataModule",
    "PhiModule",
    "PhiModuleContext",
    "TrainState",
    "Trainer",
    "TrainingOutput",
    "TrainingPlan",
    "__version__",
    "build_training_plan",
    "hessian_diagonal",
    "value_and_jacobian",
]

from importlib.metadata import version

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

__version__ = version("phijax")
DataModule = PhiDataModule

__all__ = [
    "BasePhiModule",
    "DataModule",
    "FitResult",
    "InitializedModel",
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

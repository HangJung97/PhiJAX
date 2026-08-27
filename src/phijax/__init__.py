from importlib.metadata import version

from phijax.balancers import LossBalancer
from phijax.data import PhiDataModule
from phijax.derivatives import hessian_diagonal, value_and_jacobian
from phijax.models import InitializedModel
from phijax.module import BasePhiModule, PhiModule, PhiModuleContext
from phijax.training.plans import BalancerUpdateSchedule, TrainingPlan
from phijax.training.state import TrainState
from phijax.training.trainer import FitResult, Trainer

__version__ = version("phijax")
DataModule = PhiDataModule

__all__ = [
    "BalancerUpdateSchedule",
    "BasePhiModule",
    "DataModule",
    "FitResult",
    "InitializedModel",
    "LossBalancer",
    "PhiDataModule",
    "PhiModule",
    "PhiModuleContext",
    "TrainState",
    "Trainer",
    "TrainingPlan",
    "__version__",
    "hessian_diagonal",
    "value_and_jacobian",
]

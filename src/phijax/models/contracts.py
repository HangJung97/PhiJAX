from dataclasses import dataclass
from typing import Any

from phijax.types import ModelApply, ModelSummaryFunction


@dataclass(frozen=True, slots=True)
class InitializedModel:
    """Bundle a pure model application with its explicit initialized state.

    Attributes:
        apply: Pure callable accepting explicit model state and input arrays.
        state: Initialized differentiable model-state PyTree.
        summary: Optional callable rendering a summary for the explicit state.
    """

    apply: ModelApply
    state: Any
    summary: ModelSummaryFunction | None = None


__all__ = ["InitializedModel"]

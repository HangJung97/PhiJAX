from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import jax

from phijax.types import ModelApply, ModelSummaryFunction

if TYPE_CHECKING:
    from phijax.training.precision import PrecisionPolicy


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


class ModelFactory(Protocol):
    """Build an explicit-state model after Trainer-owned data preparation.

    Model factories receive normalization statistics and precision through keyword arguments so applications can bind
    architecture options independently with :class:`functools.partial`.
    """

    def __call__(
        self,
        *,
        key: jax.Array,
        input_mean: jax.typing.ArrayLike | None,
        input_std: jax.typing.ArrayLike | None,
        precision: str | PrecisionPolicy,
    ) -> InitializedModel:
        """Initialize a pure model application and its explicit state.

        Args:
            key: Model-parameter initialization key.
            input_mean: Optional per-coordinate normalization mean supplied by the DataModule.
            input_std: Optional per-coordinate normalization standard deviation supplied by the DataModule.
            precision: Trainer precision policy.

        Returns:
            Initialized pure model contract.
        """
        ...


__all__ = ["InitializedModel", "ModelFactory"]

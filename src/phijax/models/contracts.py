from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import jax

from phijax.types import InputPeriods, ModelApply, ModelSummaryFunction, ResolvedInputPeriods

if TYPE_CHECKING:
    from phijax.training.precision import PrecisionName, PrecisionPolicy


@dataclass(frozen=True, slots=True)
class InitializedModel:
    """Bundle a pure model application with its explicit initialized state.

    Attributes:
        apply: Pure callable accepting explicit model state and input arrays.
        state: Initialized differentiable model-state PyTree.
        summary: Optional callable rendering a summary for the explicit state.
        input_periods: Physical periods the model applies, as sorted `(axis, period)` pairs, or `None`. Factories
            report the periods after applying their own overrides so the Trainer can store them.
    """

    apply: ModelApply
    state: Any
    summary: ModelSummaryFunction | None = None
    input_periods: ResolvedInputPeriods | None = None


class ModelFactory(Protocol):
    """Build an explicit-state model after Trainer-owned data preparation.

    Model factories receive normalization statistics, periods, and precision through keyword arguments so applications
    can bind architecture options independently with :class:`functools.partial`.
    """

    def __call__(
        self,
        *,
        key: jax.Array,
        input_mean: jax.typing.ArrayLike | None,
        input_std: jax.typing.ArrayLike | None,
        input_periods: InputPeriods | None,
        precision: PrecisionName | PrecisionPolicy,
    ) -> InitializedModel:
        """Initialize a pure model application and its explicit state.

        Args:
            key: Model-parameter initialization key.
            input_mean: Optional per-coordinate normalization mean supplied by the DataModule.
            input_std: Optional per-coordinate normalization standard deviation supplied by the DataModule.
            input_periods: Optional physical period per input axis, from the DataModule or a restored checkpoint.
                Factories without periodic features may ignore it.
            precision: Trainer precision policy.

        Returns:
            Initialized pure model contract.
        """
        ...


__all__ = ["InitializedModel", "ModelFactory"]

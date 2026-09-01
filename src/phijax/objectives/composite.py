from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from phijax.objectives.base import ObjectiveTerm
from phijax.objectives.terms import ResidualTerm
from phijax.types import ModelApply, NamedBatches, ResidualFunction


class CompositeObjective:
    """Compose an ordered mapping of independently configured objective terms.

    The term mapping and loss names are resolved on the host before JAX traces a training step. Mapping keys provide
    stable Hydra override paths, while each value may produce one or more scalar losses and raw residual streams.

    Attributes:
        loss_names: Flattened stable loss-name ordering across every term.
        batch_keys: Required DataModule batch names in first-declaration order.
    """

    def __init__(self, terms: Mapping[str, ObjectiveTerm]) -> None:
        """Initialize a composite objective.

        Args:
            terms: Non-empty ordered objective terms keyed by stable configuration names.

        Raises:
            TypeError: If an entry does not implement :class:`ObjectiveTerm`.
            ValueError: If no terms are supplied or loss names are empty or duplicated.
        """
        resolved_terms = dict(terms)
        if not resolved_terms:
            raise ValueError("`terms` must contain at least one objective term.")
        if any(not name or not name.strip() for name in resolved_terms):
            raise ValueError("Objective term configuration names must be non-empty.")
        if any(not isinstance(term, ObjectiveTerm) for term in resolved_terms.values()):
            raise TypeError("Every value in `terms` must implement `ObjectiveTerm`.")
        loss_names = tuple(name for term in resolved_terms.values() for name in term.loss_names)
        if not loss_names or len(set(loss_names)) != len(loss_names):
            raise ValueError("Objective term loss names must be non-empty and globally unique.")
        batch_keys = tuple(dict.fromkeys(key for term in resolved_terms.values() for key in term.batch_keys))
        if not batch_keys:
            raise ValueError("Objective terms must expose at least one batch key.")
        self.terms = resolved_terms
        self.loss_names = loss_names
        self.batch_keys = batch_keys

    @classmethod
    def from_equations(cls, equations: Mapping[str, ResidualFunction]) -> CompositeObjective:
        """Create one residual term per named equation.

        Mapping keys route DataModule batches and prefix equation-local loss names. Equation metadata selects the
        default derivative-balancing stream.

        Args:
            equations: Non-empty ordered mapping from batch names to decorated residual equations.

        Returns:
            Composite objective with inferred names, batch routing, and balancing streams.
        """
        return cls({name: ResidualTerm(equation, batch_key=name) for name, equation in equations.items()})

    def losses(
        self,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> dict[str, jax.Array]:
        """Evaluate and merge scalar losses from every configured term.

        Args:
            model_apply: Pure explicit-state model application callable.
            model_state: Differentiable model parameter PyTree.
            batches: Fixed-structure named input batches.

        Returns:
            Scalar losses in :attr:`loss_names` order.
        """
        losses: dict[str, jax.Array] = {}
        for term in self.terms.values():
            losses.update(term.losses(model_apply, model_state, batches))
        return {name: losses[name] for name in self.loss_names}

    def residual_stream(
        self,
        name: str,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> jax.Array:
        """Dispatch one named residual stream to its owning term.

        Args:
            name: One entry from :attr:`loss_names`.
            model_apply: Pure explicit-state model application callable.
            model_state: Differentiable model parameter PyTree.
            batches: Fixed-structure named residual batches.

        Returns:
            Raw output or equation residual array for `name`.

        Raises:
            KeyError: If no configured term owns `name`.
        """
        for term in self.terms.values():
            if name in term.loss_names:
                return term.residual_stream(name, model_apply, model_state, batches)
        raise KeyError(f"Unknown objective stream: {name}")

    def residual_streams(
        self,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> dict[str, jax.Array]:
        """Evaluate every configured raw residual stream.

        Args:
            model_apply: Pure explicit-state model application callable.
            model_state: Differentiable model parameter PyTree.
            batches: Fixed-structure named residual batches.

        Returns:
            Raw arrays in :attr:`loss_names` order.
        """
        return {name: self.residual_stream(name, model_apply, model_state, batches) for name in self.loss_names}

    def components(self, losses: Mapping[str, jax.Array]) -> jax.Array:
        """Stack scalar losses in stable configured order.

        Args:
            losses: Scalar loss arrays keyed by every name in :attr:`loss_names`.

        Returns:
            `float32` component vector aligned with :attr:`loss_names`.

        Raises:
            KeyError: If a required loss is absent.
        """
        return jnp.stack([losses[name] for name in self.loss_names]).astype(jnp.float32)


__all__ = ["CompositeObjective"]

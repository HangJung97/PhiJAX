from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from phijax.equations.metadata import get_default_ntk_stream, get_residual_names
from phijax.types import ModelApply, NamedBatches, ResidualFunction, ResidualGroup, ResidualGroups, ResidualStream


class ResidualTerm:
    """Reduce one configured grouped-residual function into named scalar losses.

    One outer residual group corresponds to one entry in :attr:`loss_names`. When explicit names are omitted, the
    equation's static local residual names are prefixed with :attr:`batch_key`. Every array inside a group contributes
    its own mean-squared value to that scalar loss. This preserves composite residuals such as phase-wrapped cosine
    and sine errors without requiring an equation-specific objective class.

    Attributes:
        loss_names: Stable names aligned with the residual groups returned by `residual_fn`.
    """

    def __init__(
        self,
        residual_fn: ResidualFunction,
        *,
        batch_key: str,
        names: Sequence[str] | None = None,
        ntk_stream: ResidualStream | None = None,
    ) -> None:
        """Initialize a generic residual term.

        Args:
            residual_fn: Configured equation callable returning one residual group per name.
            batch_key: Key selecting the equation batch from the named training batches.
            names: Optional non-empty unique scalar loss names ordered like the equation's residual groups. When
                omitted, names are formed as `batch_key/local_name` from metadata attached to `residual_fn` by
                :func:`phijax.equations.residual_equation`.
            ntk_stream: Optional stream override. When omitted, the equation metadata selects `"residual"` or
                `"output"`.

        Raises:
            TypeError: If `residual_fn` is not callable.
            ValueError: If names, inferred equation metadata, `batch_key`, or `ntk_stream` are invalid.
        """
        if not callable(residual_fn):
            raise TypeError("`residual_fn` must be callable.")
        resolved_stream = get_default_ntk_stream(residual_fn) if ntk_stream is None else ntk_stream
        if resolved_stream not in ("residual", "output"):
            raise ValueError("`ntk_stream` must be either 'residual' or 'output'.")
        resolved_batch_key = _validate_name(batch_key)
        if names is None:
            resolved_names = tuple(f"{resolved_batch_key}/{name}" for name in get_residual_names(residual_fn))
        else:
            resolved_names = tuple(_validate_name(name) for name in names)
            if not resolved_names or len(set(resolved_names)) != len(resolved_names):
                raise ValueError("`names` must contain unique non-empty loss names.")
        self.loss_names = resolved_names
        self.residual_fn = residual_fn
        self.batch_key = resolved_batch_key
        self.batch_keys = (resolved_batch_key,)
        self.ntk_stream: ResidualStream = resolved_stream

    def losses(
        self,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> dict[str, jax.Array]:
        """Evaluate and reduce the configured equation residual groups.

        Args:
            model_apply: Pure explicit-state model application callable.
            model_state: Differentiable model parameter PyTree.
            batches: Mapping containing :attr:`batch_key`.

        Returns:
            Scalar `float32` losses keyed by :attr:`loss_names`.
        """
        groups = self._groups(model_apply, model_state, batches, stream="residual")
        return {
            name: jnp.stack([jnp.mean(residual.astype(jnp.float32) ** 2) for residual in group]).sum()
            for name, group in zip(self.loss_names, groups, strict=True)
        }

    def residual_stream(
        self,
        name: str,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> jax.Array:
        """Return one two-dimensional stream for derivative-based balancing.

        Args:
            name: One configured scalar loss name.
            model_apply: Pure explicit-state model application callable.
            model_state: Differentiable model parameter PyTree.
            batches: Mapping containing :attr:`batch_key`.

        Returns:
            Residual matrix with samples on axis zero and flattened residual features on axis one.

        Raises:
            KeyError: If `name` does not belong to this term.
        """
        if name not in self.loss_names:
            raise KeyError(f"Unknown objective stream: {name}")
        groups = self._groups(model_apply, model_state, batches, stream=self.ntk_stream)
        return _merge_residual_group(groups[self.loss_names.index(name)])

    def _groups(
        self,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
        *,
        stream: ResidualStream,
    ) -> ResidualGroups:
        """Evaluate and validate residual groups for one configured stream.

        Args:
            model_apply: Pure explicit-state model application callable.
            model_state: Differentiable model parameter PyTree.
            batches: Mapping containing :attr:`batch_key`.
            stream: Equation representation requested from :attr:`residual_fn`.

        Returns:
            Validated groups aligned with :attr:`loss_names`.

        Raises:
            TypeError: If the equation does not return nested tuples of arrays.
            ValueError: If group counts, group contents, or residual ranks are invalid.
        """
        groups = self.residual_fn(model_apply, model_state, batches[self.batch_key], stream=stream)
        return _validate_residual_groups(groups, expected_count=len(self.loss_names))


def _validate_residual_groups(groups: ResidualGroups, *, expected_count: int) -> ResidualGroups:
    """Validate the static structure returned by an equation callable.

    Args:
        groups: Candidate nested residual groups.
        expected_count: Number of scalar loss names owned by the term.

    Returns:
        Unchanged validated residual groups.

    Raises:
        TypeError: If groups or their entries are not tuples.
        ValueError: If the group count, group contents, or residual ranks are invalid.
    """
    if not isinstance(groups, tuple) or any(not isinstance(group, tuple) for group in groups):
        raise TypeError("`residual_fn` must return a tuple of non-empty residual tuples.")
    if len(groups) != expected_count:
        raise ValueError(f"`residual_fn` returned {len(groups)} groups for {expected_count} configured loss names.")
    if any(not group for group in groups):
        raise ValueError("Every residual group must contain at least one array.")
    for group in groups:
        for residual in group:
            if not hasattr(residual, "ndim"):
                raise TypeError("Every residual group entry must be a JAX-compatible array.")
            if residual.ndim == 0:
                raise ValueError("Residual streams must retain a sample axis.")
    return groups


def _merge_residual_group(group: ResidualGroup) -> jax.Array:
    """Merge one residual group into a two-dimensional NTK stream.

    Args:
        group: Non-empty arrays sharing the same leading sample dimension.

    Returns:
        Matrix with shape `[samples, flattened_features]`.

    Raises:
        ValueError: If residual arrays do not share their leading sample dimension.
    """
    sample_count = group[0].shape[0]
    if any(residual.shape[0] != sample_count for residual in group[1:]):
        raise ValueError("Residual arrays in one loss group must share their leading sample dimension.")
    matrices = tuple(residual.reshape((sample_count, -1)) for residual in group)
    return matrices[0] if len(matrices) == 1 else jnp.concatenate(matrices, axis=-1)


def _validate_name(name: str) -> str:
    """Validate and preserve a configured name.

    Args:
        name: Candidate non-empty name.

    Returns:
        Unchanged validated name.

    Raises:
        ValueError: If `name` is empty or whitespace-only.
    """
    if not name or not name.strip():
        raise ValueError("Configured names must be non-empty.")
    return name


__all__ = ["ResidualTerm"]

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from phijax.balancers.base import BalancerState


class StaticLossBalancer:
    """Apply immutable scalar weights to a fixed sequence of loss names.

    Attributes:
        loss_names: Stable names defining component and weight ordering.
        initial_weights: Configured `float32` weight vector.
    """

    def __init__(
        self,
        loss_names: Sequence[str],
        *,
        weights: Mapping[str, float] | None = None,
        default_weight: float = 1.0,
    ) -> None:
        """Initialize stable loss ordering and configured weights.

        Args:
            loss_names: Unique non-empty sequence defining loss ordering.
            weights: Optional explicit weights keyed by loss name.
            default_weight: Weight assigned to names absent from `weights`.

        Raises:
            ValueError: If `loss_names` is empty or contains duplicates.
        """
        if not loss_names or len(set(loss_names)) != len(loss_names):
            raise ValueError("`loss_names` must be non-empty and unique.")
        configured = weights or {}
        self.loss_names = tuple(loss_names)
        self.initial_weights = jnp.asarray(
            [configured.get(name, default_weight) for name in self.loss_names],
            dtype=jnp.float32,
        )

    def initialize(self) -> BalancerState:
        """Create the fixed balancer state.

        Returns:
            State containing configured weights and zero diagnostic placeholders.
        """
        return BalancerState(weights=self.initial_weights, traces=jnp.zeros_like(self.initial_weights))

    def combine(
        self,
        losses: Mapping[str, jax.Array],
        state: BalancerState,
    ) -> jax.Array:
        """Combine named scalar losses in stable order.

        Args:
            losses: Scalar loss arrays keyed by every configured loss name.
            state: Current balancer state containing the fixed weights.

        Returns:
            Weighted scalar total.

        Raises:
            KeyError: If a configured loss name is absent from `losses`.
        """
        components = jnp.stack([losses[name] for name in self.loss_names]).astype(jnp.float32)
        return jnp.vdot(state.weights, components)

    def diagnostics(self, state: BalancerState) -> Mapping[str, jax.Array]:
        """Expose configured loss weights for logging.

        Args:
            state: Current fixed-weight state.

        Returns:
            Scalar weights keyed by loss name.
        """
        return {f"weight/{name}": state.weights[index] for index, name in enumerate(self.loss_names)}

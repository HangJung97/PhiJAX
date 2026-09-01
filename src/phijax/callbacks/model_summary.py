import logging

from phijax.callbacks.base import Callback, TrainerContext

log = logging.getLogger(__name__)


class ModelSummary(Callback):
    """Print a network summary at fit start using the module's configured summary provider."""

    def __init__(
        self,
        *,
        max_depth: int = -1,
        console_width: int = 120,
        compute_flops: bool = False,
        compute_vjp_flops: bool = False,
        rank_zero_only: bool = True,
        warn_if_unavailable: bool = True,
    ) -> None:
        """Initialize model-summary display policy.

        Args:
            max_depth: Maximum displayed module depth, or `-1` for every level.
            console_width: Positive Rich console width in characters.
            compute_flops: Whether to estimate forward-pass floating-point operations.
            compute_vjp_flops: Whether to estimate reverse-pass floating-point operations.
            rank_zero_only: Whether to suppress output outside global rank zero.
            warn_if_unavailable: Whether to warn when no summary can be rendered.

        Raises:
            ValueError: If `max_depth` is below `-1` or `console_width` is not positive.
        """
        if max_depth < -1:
            raise ValueError("`max_depth` must be `-1` or nonnegative.")
        if console_width < 1:
            raise ValueError("`console_width` must be positive.")
        self.max_depth = max_depth
        self.console_width = console_width
        self.compute_flops = compute_flops
        self.compute_vjp_flops = compute_vjp_flops
        self.rank_zero_only = rank_zero_only
        self.warn_if_unavailable = warn_if_unavailable

    def on_fit_start(self, context: TrainerContext) -> None:
        """Print the configured model summary before the first training batch.

        Args:
            context: Initial trainer context containing the module and explicit model state.
        """
        if self.rank_zero_only and not context.is_global_zero:
            return
        if context.module is None:
            if self.warn_if_unavailable:
                log.warning("Model summary skipped because the trainer context does not contain a module.")
            return
        summary = context.module.summarize_model(
            context.state.model_state,
            max_depth=self.max_depth,
            console_width=self.console_width,
            compute_flops=self.compute_flops,
            compute_vjp_flops=self.compute_vjp_flops,
        )
        if summary is None:
            if self.warn_if_unavailable:
                module_name = type(context.module).__name__
                log.warning(f"Model summary skipped because <{module_name}> has no summary provider.")
            return
        print(summary, flush=True)


class RichModelSummary(ModelSummary):
    """Select the Rich-capable model summary provider as an explicit display callback."""


__all__ = ["ModelSummary", "RichModelSummary"]

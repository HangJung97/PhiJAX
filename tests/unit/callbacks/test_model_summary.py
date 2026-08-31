from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from phijax.callbacks import ModelSummary, RichModelSummary, TrainerContext
from phijax.training import Trainer


def test_rich_model_summary_prints_module_summary_on_global_fit_start(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify the callback delegates explicit state and display policy to the application module.

    Args:
        capsys: Pytest fixture capturing the rendered summary.
    """
    module = Mock()
    module.summarize_model.return_value = "Model Summary"
    state = SimpleNamespace(model_state="parameters")
    callback = RichModelSummary(max_depth=2, console_width=100, compute_flops=True)

    callback.on_fit_start(TrainerContext(state, 0, {}, module=module, is_global_zero=True))

    assert capsys.readouterr().out == "Model Summary\n"
    module.summarize_model.assert_called_once_with(
        "parameters",
        max_depth=2,
        console_width=100,
        compute_flops=True,
        compute_vjp_flops=False,
    )


def test_rich_model_summary_respects_rank_and_missing_summary_provider(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify rank filtering is silent and unsupported modules produce a concise warning.

    Args:
        capsys: Pytest fixture capturing summary output.
        caplog: Pytest fixture capturing unsupported-module warnings.
    """
    module = Mock()
    module.summarize_model.return_value = "hidden"
    unsupported_module = Mock()
    unsupported_module.summarize_model.return_value = None
    state = SimpleNamespace(model_state="parameters")
    callback = RichModelSummary()

    callback.on_fit_start(TrainerContext(state, 0, {}, module=module, is_global_zero=False))
    callback.on_fit_start(TrainerContext(state, 0, {}, module=None, is_global_zero=True))
    callback.on_fit_start(TrainerContext(state, 0, {}, module=unsupported_module, is_global_zero=True))

    assert capsys.readouterr().out == ""
    module.summarize_model.assert_not_called()
    assert "does not contain a module" in caplog.text
    assert "has no summary provider" in caplog.text


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_depth": -2}, "max_depth"),
        ({"console_width": 0}, "console_width"),
    ],
)
def test_rich_model_summary_rejects_invalid_display_policy(kwargs: dict[str, int], match: str) -> None:
    """Verify invalid summary depth and console width fail during callback construction.

    Args:
        kwargs: Invalid callback keyword arguments.
        match: Expected error-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        RichModelSummary(**kwargs)


def test_trainer_treats_rich_summary_as_plain_summary_replacement() -> None:
    """Verify explicit Rich summaries suppress automatic plain summary insertion."""
    callback = RichModelSummary()
    trainer = Trainer(
        max_steps=1,
        callbacks=(callback,),
        enable_progress_bar=False,
        logger=False,
    )

    assert trainer.callbacks == (callback,)


def test_trainer_rejects_multiple_model_summaries() -> None:
    """Verify one fit task cannot render duplicate model summaries."""
    with pytest.raises(ValueError, match="Only one model-summary"):
        Trainer(
            max_steps=1,
            callbacks=(ModelSummary(), RichModelSummary()),
            enable_progress_bar=False,
            logger=False,
        )

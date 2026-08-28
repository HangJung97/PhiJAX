import io
import logging
import re

import pytest

from phijax.utils.pylogger import RankedLogger, get_colorlogger


def test_get_colorlogger_is_idempotent_and_formats_records() -> None:
    """Verify repeated setup reuses one PhiJAX handler and emits the Hydra-style fields."""
    logger = get_colorlogger("phijax.tests.colorlogger", logging.DEBUG)
    repeated = get_colorlogger("phijax.tests.colorlogger", logging.INFO)
    handlers = [handler for handler in logger.handlers if getattr(handler, "_phijax_colorlog", False)]
    stream = io.StringIO()
    handlers[0].setStream(stream)

    logger.info("configured")
    rendered = stream.getvalue()
    plain_rendered = re.sub(r"\x1b\[[0-9;]*m", "", rendered)

    assert repeated is logger
    assert len(handlers) == 1
    assert logger.level == logging.INFO
    assert logger.propagate is False
    assert "phijax.tests.colorlogger" in plain_rendered
    assert "INFO" in plain_rendered
    assert plain_rendered.endswith(" - configured\n")


@pytest.mark.parametrize(
    ("process_index", "rank_zero_only", "expected"),
    [(0, True, True), (1, True, False), (1, False, True)],
)
def test_ranked_logger_respects_rank_policy(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    process_index: int,
    rank_zero_only: bool,
    expected: bool,
) -> None:
    """Verify rank-zero filtering while preserving ordinary multi-process logging.

    Args:
        monkeypatch: Pytest fixture used to replace the JAX process index provider.
        caplog: Pytest fixture used to capture propagated standard-library records.
        process_index: Simulated JAX process index.
        rank_zero_only: Rank filtering policy supplied to the adapter.
        expected: Whether the test record should be emitted.
    """
    monkeypatch.setattr("phijax.utils.pylogger._process_index", lambda: process_index)
    logger_name = f"phijax.tests.ranked.{process_index}.{rank_zero_only}"
    logger = RankedLogger(logger_name, rank_zero_only=rank_zero_only)

    with caplog.at_level(logging.INFO, logger=logger_name):
        logger.info("rank-sensitive record")

    assert ("rank-sensitive record" in caplog.messages) is expected

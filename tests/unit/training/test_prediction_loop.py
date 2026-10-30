from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from phijax.training import Strategy
from phijax.training.loops import prediction_loop as prediction_loop_module


def _strategy(process_index: int) -> Strategy:
    """Build a two-process strategy layout with two devices per process.

    Args:
        process_index: Index of the simulated current process.

    Returns:
        Strategy stand-in exposing the process and device layout.
    """
    devices = tuple(SimpleNamespace(process_index=process) for process in (0, 0, 1, 1))
    return cast(Strategy, SimpleNamespace(process_index=process_index, process_count=2, devices=devices))


def test_process_share_pads_rows_and_splits_them_evenly() -> None:
    """Verify six rows split into four-row process shares with invalid padding and unchanged scalars."""
    batch = {
        "inputs": np.arange(6, dtype=np.float32)[:, None],
        "valid_predict_mask": np.asarray([True] * 5 + [False]),
        "scale": np.float32(2.0),
    }

    first = prediction_loop_module._process_share(batch, _strategy(0))
    second = prediction_loop_module._process_share(batch, _strategy(1))

    np.testing.assert_array_equal(first.batch["inputs"][:, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(second.batch["inputs"][:, 0], [4, 5, 0, 0])
    np.testing.assert_array_equal(second.batch["valid_predict_mask"], [True, False, False, False])
    np.testing.assert_array_equal(first.rows, [0, 1, 2, 3])
    np.testing.assert_array_equal(second.rows, [4, 5, 6, 7])
    np.testing.assert_array_equal(first.valid, batch["valid_predict_mask"])
    assert second.batch["scale"] == 2.0


def test_process_share_marks_every_row_valid_without_a_mask() -> None:
    """Verify batches without a padding mask treat every original row as valid."""
    share = prediction_loop_module._process_share({"inputs": np.zeros((3, 1))}, _strategy(0))

    np.testing.assert_array_equal(share.valid, [True, True, True])


def test_gather_rows_restores_row_order_and_drops_invalid_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify gathered rows are reordered by id, so a mesh that orders processes differently still yields pool order.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    # The simulated mesh lists process 1's rows before process 0's.
    gathered_values = np.asarray([[40.0], [50.0], [0.0], [0.0], [0.0], [10.0], [20.0], [30.0]])
    gathered_ids = np.asarray([4, 5, 6, 7, 0, 1, 2, 3])

    def allgather(tree: Any, tiled: bool) -> Any:
        """Return the simulated global prediction and row ids.

        Args:
            tree: This process's prediction and row ids.
            tiled: Must be `True` to concatenate shards along the row axis.

        Returns:
            Simulated gathered prediction and row ids.
        """
        assert tiled is True
        del tree
        return gathered_values, gathered_ids

    monkeypatch.setattr(prediction_loop_module.multihost_utils, "process_allgather", allgather)
    valid = np.asarray([True, True, True, True, True, False])

    outputs = prediction_loop_module._gather_rows(cast(Any, None), cast(Any, None), valid)

    np.testing.assert_array_equal(outputs[:, 0], [0.0, 10.0, 20.0, 30.0, 40.0])


def test_process_share_pads_a_single_process_batch_to_its_device_count() -> None:
    """Verify one process with four devices pads six rows to eight and keeps every row."""
    devices = tuple(SimpleNamespace(process_index=0) for _ in range(4))
    strategy = cast(Strategy, SimpleNamespace(process_index=0, process_count=1, devices=devices))

    share = prediction_loop_module._process_share({"inputs": np.arange(6, dtype=np.float32)[:, None]}, strategy)

    np.testing.assert_array_equal(share.batch["inputs"][:, 0], [0, 1, 2, 3, 4, 5, 0, 0])
    np.testing.assert_array_equal(share.rows, np.arange(8))
    np.testing.assert_array_equal(share.valid, [True] * 6)

import math
import sys

import pytest

from examples import quickstart


def test_heat_equation_quickstart_runs_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the documented heat PINN reports its runtime and completes on CPU.

    Args:
        capsys: Pytest fixture capturing quickstart output.
    """
    error = quickstart.run_quickstart(max_steps=1)
    output = capsys.readouterr().out

    assert math.isfinite(error)
    assert "Using 32-bit true precision" in output
    assert "GPU available:" in output
    assert "TPU available:" in output
    assert "MLP Summary" in output
    assert "steps in " in output


def test_heat_equation_quickstart_cli_forwards_runtime_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify command-line accelerator and precision options reach the runnable example.

    Args:
        monkeypatch: Pytest fixture used to isolate arguments and execution.
    """
    received: dict[str, object] = {}

    def record_run(max_steps: int, *, accelerator: str, precision: str, enable_progress_bar: bool) -> float:
        """Record parsed quickstart arguments without starting another training run.

        Args:
            max_steps: Parsed optimizer update count.
            accelerator: Parsed accelerator name.
            precision: Parsed precision mode.
            enable_progress_bar: Parsed progress display selection.

        Returns:
            Synthetic finite error.
        """
        received.update(
            max_steps=max_steps,
            accelerator=accelerator,
            precision=precision,
            enable_progress_bar=enable_progress_bar,
        )
        return 0.0

    monkeypatch.setattr(quickstart, "run_quickstart", record_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quickstart.py",
            "--max-steps",
            "3",
            "--accelerator",
            "auto",
            "--precision",
            "bf16-mixed",
            "--no-progress-bar",
        ],
    )

    quickstart.main()

    assert received == {
        "max_steps": 3,
        "accelerator": "auto",
        "precision": "bf16-mixed",
        "enable_progress_bar": False,
    }

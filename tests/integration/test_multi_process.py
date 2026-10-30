import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_PROCESS_COUNT = 2
_LOCAL_DEVICE_COUNT = 2


def _free_port() -> int:
    """Return a free localhost TCP port for the JAX coordinator.

    Returns:
        Port number that was free when checked.
    """
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        return probe.getsockname()[1]


@pytest.mark.slow
def test_data_parallel_fit_runs_across_processes(tmp_path: Path) -> None:
    """Verify two-process CPU fits sample per process, resume, predict, reject mismatched states, and stop together.

    Each worker process owns two CPU devices, so the data-parallel mesh spans four devices on two processes.

    Args:
        tmp_path: Directory shared by both workers for checkpoints and results.
    """
    worker = Path(__file__).with_name("multi_process_worker.py")
    address = f"localhost:{_free_port()}"
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={_LOCAL_DEVICE_COUNT}",
    }
    workers = [
        subprocess.Popen(
            [sys.executable, str(worker), address, str(_PROCESS_COUNT), str(index), str(tmp_path)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for index in range(_PROCESS_COUNT)
    ]
    for process in workers:
        output, _ = process.communicate(timeout=300)
        assert process.returncode == 0, output

    results = [json.loads((tmp_path / f"process_{index}.json").read_text()) for index in range(_PROCESS_COUNT)]
    # Every process sees the same replicated training outcome.
    assert results[0]["full"] == results[1]["full"]
    assert results[0]["resumed"] == results[1]["resumed"]
    full = results[0]["full"]
    assert full["mesh_size"] == _PROCESS_COUNT * _LOCAL_DEVICE_COUNT
    assert full["step"] == 4

    # The global batch stacks one four-row batch per process; distinct keys must give the processes different rows.
    for batch in full["batches"].values():
        rows = np.asarray(batch).reshape(_PROCESS_COUNT, -1)
        assert not np.array_equal(rows[0], rows[1])

    resumed = results[0]["resumed"]
    assert resumed["step"] == 4
    assert resumed["batches"] == {step: full["batches"][step] for step in ("2", "3")}
    assert resumed["parameters"] == full["parameters"]
    assert resumed["metrics"] == full["metrics"]

    # `devices=1` selects one device on each process.
    assert sorted(process for process, _ in results[0]["single_device_mesh"]) == [0, 1]
    # Prediction splits each chunk over both processes and gathers complete, ordered outputs on every process.
    assert len(full["prediction"]["values"]) == 10
    assert full["prediction"]["max_error"] < 1e-6
    assert results[0]["full"]["prediction"] == results[1]["full"]["prediction"]
    # Different seeds per process are rejected on every process instead of silently diverging.
    for result in results:
        assert "differs between JAX processes" in result["mismatch_error"]

    # A stop requested by process 1 alone stops every process after the same step instead of hanging the others.
    for result in results:
        assert result["should_stop"]["step"] == result["sigint"]["step"] == 2
        assert result["should_stop"]["stopped_early"]
        assert result["sigint"]["interrupted"]
    # `SIGTERM` goes through JAX's preemption service: every process saves the agreed step and exits with 143.
    assert results[0]["sigterm"] == results[1]["sigterm"]
    assert results[0]["sigterm"]["exit_code"] == 143
    assert results[0]["sigterm"]["saved_steps"][-1] > 2


@pytest.mark.slow
def test_single_process_prediction_pads_chunks_over_local_devices(tmp_path: Path) -> None:
    """Verify one process with four devices predicts six-row chunks, which do not divide evenly over the devices.

    Args:
        tmp_path: Directory for the worker's checkpoints and results.
    """
    worker = Path(__file__).with_name("multi_process_worker.py")
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=4"}
    completed = subprocess.run(
        [sys.executable, str(worker), "single-process", str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    result = json.loads((tmp_path / "single_process.json").read_text())
    assert result["mesh_size"] == 4
    assert len(result["prediction"]["values"]) == 10
    assert result["prediction"]["max_error"] < 1e-6

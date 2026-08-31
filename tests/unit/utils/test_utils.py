import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError

from phijax.training import WandbLogger
from phijax.utils import utils


def test_pre_hydra_routine_sets_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify pre-Hydra setup discovers PhiJAX and registers resolvers without changing directories."""
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    register_resolvers = Mock()
    monkeypatch.setattr(utils, "register_omegaconf_resolvers", register_resolvers)
    working_directory = Path.cwd()
    utils.pre_hydra_routine()
    assert Path(os.environ["PROJECT_ROOT"]) == Path(__file__).parents[3]
    assert Path.cwd() == working_directory
    register_resolvers.assert_called_once_with()


def test_extras_prints_config_and_can_disable_warnings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify configured optional utilities delegate to warning and Rich helpers."""
    printed: list[tuple[bool, bool]] = []

    def record_print(cfg: DictConfig, *, resolve: bool, save_to_file: bool) -> None:
        """Record Rich print options without writing a test artifact.

        Args:
            cfg: Composed test configuration.
            resolve: Whether interpolations would be resolved.
            save_to_file: Whether the tree would be persisted.
        """
        del cfg
        printed.append((resolve, save_to_file))

    monkeypatch.setattr(utils, "print_config_tree", record_print)
    with warnings.catch_warnings():
        warnings.resetwarnings()
        config = OmegaConf.create(
            {
                "extras": {"ignore_warnings": True, "print_config": True},
                "paths": {"output_dir": str(tmp_path)},
            }
        )
        utils.extras(config)
        assert warnings.filters[0][0] == "ignore"
    assert printed == [(True, True)]


def test_extras_accepts_missing_configuration(caplog: pytest.LogCaptureFixture) -> None:
    """Verify entrypoints may omit optional extras without failing."""
    utils.extras(OmegaConf.create({}))
    assert "optional entrypoint utilities are disabled" in caplog.text


def test_task_wrapper_preserves_result_and_metadata(tmp_path: Path) -> None:
    """Verify task wrapping preserves the callable name and return value."""

    @utils.task_wrapper
    def task(cfg: DictConfig) -> int:
        """Return a configured value.

        Args:
            cfg: Test task configuration.

        Returns:
            Configured integer result.
        """
        return int(cfg.value)

    config = OmegaConf.create({"value": 4, "paths": {"output_dir": str(tmp_path)}})
    assert task(config) == 4
    assert task.__name__ == "task"


def test_task_wrapper_resolves_assertions_before_calling_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify invalid deferred interpolations stop execution before the task begins."""
    called = False
    exception_log = Mock()
    monkeypatch.setattr(utils.log, "exception", exception_log)

    @utils.task_wrapper
    def task(cfg: DictConfig) -> None:
        """Record an unexpected task invocation.

        Args:
            cfg: Test task configuration.
        """
        nonlocal called
        del cfg
        called = True

    config = OmegaConf.create(
        {
            "_assert_": {"required": "${missing.value}"},
            "paths": {"output_dir": str(tmp_path)},
        }
    )
    with pytest.raises(InterpolationKeyError, match=r"missing\.value"):
        task(config)
    assert called is False
    exception_log.assert_called_once_with("Task execution failed.")


def test_task_wrapper_finishes_active_wandb_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify failed multirun tasks close a globally active W&B run before re-raising."""
    finish = Mock()
    wandb = SimpleNamespace(run=object(), finish=finish)
    monkeypatch.setattr(utils, "find_spec", lambda name: object() if name == "wandb" else None)
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    @utils.task_wrapper
    def task(cfg: DictConfig) -> None:
        """Raise after W&B initialization.

        Args:
            cfg: Test task configuration.

        Raises:
            RuntimeError: Always.
        """
        del cfg
        raise RuntimeError("failed run")

    config = OmegaConf.create({"paths": {"output_dir": str(tmp_path)}})
    with pytest.raises(RuntimeError, match="failed run"):
        task(config)
    finish.assert_called_once_with(exit_code=1)


def test_task_wrapper_marks_system_exit_as_interrupted(tmp_path: Path) -> None:
    """Verify trainer-propagated signal termination retains an interrupted resource status.

    Args:
        tmp_path: Temporary output directory fixture.
    """
    statuses: list[str] = []

    @utils.task_wrapper
    def task(cfg: DictConfig) -> None:
        """Register cleanup and simulate terminal trainer signal handling.

        Args:
            cfg: Test task configuration, unused by the synthetic task.

        Raises:
            SystemExit: Always, using the standard `SIGTERM` exit status.
        """
        del cfg
        utils.register_task_finalizer(statuses.append)
        raise SystemExit(143)

    with pytest.raises(SystemExit) as exception:
        task(OmegaConf.create({"paths": {"output_dir": str(tmp_path)}}))

    assert exception.value.code == 143
    assert statuses == ["interrupted"]


def test_task_wrapper_uses_registered_wandb_finalizer_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify a PhiJAX W&B logger flushes once and suppresses the global fallback."""
    wandb = SimpleNamespace(run=None)

    def finish_run(*, exit_code: int) -> None:
        """Record run completion and clear W&B's global active run.

        Args:
            exit_code: Terminal W&B exit code.
        """
        assert exit_code == 0
        wandb.run = None

    run = SimpleNamespace(finish=Mock(side_effect=finish_run))
    wandb.run = run
    wandb.init = Mock(return_value=run)
    wandb.finish = Mock()
    monkeypatch.setattr(utils, "find_spec", lambda name: object() if name == "wandb" else None)
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    @utils.task_wrapper
    def task(cfg: DictConfig) -> None:
        """Construct a task-owned W&B logger.

        Args:
            cfg: Test task configuration.
        """
        del cfg
        WandbLogger("phijax-tests", mode="disabled").setup()

    task(OmegaConf.create({"paths": {"output_dir": str(tmp_path)}}))
    run.finish.assert_called_once_with(exit_code=0)
    wandb.finish.assert_not_called()

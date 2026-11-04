import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from phijax._optional import missing_extra_error
from phijax.data import HostPool, load_arrays, save_prediction_artifact
from phijax.data import io as array_io

_HYDRA_PACKAGES = ("hydra", "omegaconf", "rootutils")
_IO_PACKAGES = ("h5py", "mat73", "scipy")


def _run_without(packages: tuple[str, ...], statement: str) -> dict[str, str]:
    """Run Python source in a fresh interpreter in which some top-level packages cannot be imported.

    Args:
        packages: Top-level package names whose import raises `ModuleNotFoundError`, as if they were not installed.
        statement: Python source that stores string results in a `results` dictionary.

    Returns:
        The `results` dictionary written by `statement`.
    """
    script = f"""
import importlib.abc, json, sys

class _Missing(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {packages!r}:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None

sys.meta_path.insert(0, _Missing())
results = {{}}
{statement}
print(json.dumps(results))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, check=True, text=True)
    return json.loads(result.stdout.splitlines()[-1])


def test_training_runs_without_the_hydra_extra() -> None:
    """Verify that the Trainer, built-in components, loggers, and core utilities need no Hydra package."""
    results = _run_without(
        _HYDRA_PACKAGES,
        "import phijax.balancers, phijax.callbacks, phijax.evaluation, phijax.models, phijax.objectives\n"
        "from phijax import PhiModule, Trainer\n"
        "from phijax.training import CSVLogger\n"
        "from phijax.utils import register_task_finalizer, to_plain_container\n"
        "results['plain'] = repr(to_plain_container({'a': 1}))\n"
        "results['finalizer'] = repr(register_task_finalizer(print))\n"
        "results['loaded'] = repr(sorted({'hydra', 'omegaconf', 'rootutils'} & set(sys.modules)))",
    )

    assert results == {"plain": "{'a': 1}", "finalizer": "False", "loaded": "[]"}


@pytest.mark.parametrize(
    ("module_name", "missing"),
    [
        ("phijax.integrations.hydra.factory", "hydra.utils"),
        ("phijax.integrations.omegaconf", "omegaconf"),
        ("phijax.utils.utils", "omegaconf"),
        ("phijax.utils.rich_utils", "omegaconf"),
    ],
)
def test_hydra_modules_name_the_missing_extra(monkeypatch: pytest.MonkeyPatch, module_name: str, missing: str) -> None:
    """Verify that each Hydra-only module raises `ModuleNotFoundError` with the `phijax[hydra]` install hint.

    Args:
        monkeypatch: Fixture that hides a package and re-imports the module, restoring `sys.modules` afterwards.
        module_name: Hydra-only PhiJAX module.
        missing: Package whose absence the module reports first.
    """
    # A `None` entry makes `import` raise `ModuleNotFoundError`, as if the package were not installed.
    monkeypatch.setitem(sys.modules, missing, None)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    with pytest.raises(ModuleNotFoundError, match=r"phijax\[hydra\]") as error:
        import_module(module_name)
    assert error.value.name == missing


def test_pre_hydra_routine_names_the_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `pre_hydra_routine()` names the `hydra` extra when `rootutils` is not installed.

    Args:
        monkeypatch: Fixture hiding `rootutils` for this test only.
    """
    from phijax.utils import pre_hydra_routine

    monkeypatch.setitem(sys.modules, "rootutils", None)

    with pytest.raises(ModuleNotFoundError, match=r"phijax\[hydra\]"):
        pre_hydra_routine()


def test_host_data_modules_import_without_the_io_extra() -> None:
    """Verify that importing data construction and NPZ IO loads no MATLAB or HDF5 package."""
    results = _run_without(
        _IO_PACKAGES,
        "from phijax.data import build_array_pools, load_arrays, save_prediction_artifact\n"
        "results['loaded'] = repr(sorted({'h5py', 'mat73', 'scipy'} & set(sys.modules)))",
    )

    assert results == {"loaded": "[]"}


@pytest.fixture
def without_io_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the `io` extra's packages unimportable for :mod:`phijax.data.io`.

    Args:
        monkeypatch: Fixture replacing the module importer used by the IO helpers.
    """

    def missing_io(name: str) -> ModuleType:
        """Raise like an uninstalled `io` package, and import every other module normally.

        Args:
            name: Module requested by the IO helpers.

        Returns:
            The imported module when it is not part of the `io` extra.

        Raises:
            ModuleNotFoundError: If `name` belongs to the `io` extra.
        """
        if name.split(".")[0] in _IO_PACKAGES:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return import_module(name)

    monkeypatch.setattr(array_io, "import_module", missing_io)


@pytest.mark.usefixtures("without_io_extra")
def test_npz_arrays_load_without_the_io_extra(tmp_path: Path) -> None:
    """Verify that NumPy archives need no optional package, while MATLAB and HDF5 files name the extra.

    Args:
        tmp_path: Directory for the synthetic data files.
    """
    np.savez(tmp_path / "data.npz", u=np.arange(3.0))
    (tmp_path / "data.mat").write_bytes(b"MATLAB")
    (tmp_path / "data.h5").write_bytes(b"HDF5")
    (tmp_path / "data.bin").write_bytes(b"unknown")

    np.testing.assert_array_equal(load_arrays(tmp_path / "data.npz")["u"], np.arange(3.0))
    for name in ("data.mat", "data.h5", "data.bin"):
        with pytest.raises(ModuleNotFoundError, match=r"phijax\[io\]"):
            load_arrays(tmp_path / name)


@pytest.mark.usefixtures("without_io_extra")
def test_matlab_sidecar_without_the_io_extra_writes_nothing(tmp_path: Path) -> None:
    """Verify that `save_mat=True` names the missing extra before any artifact file is written.

    Args:
        tmp_path: Destination directory that must stay empty.
    """
    pool = HostPool(
        inputs=np.asarray([[0.0], [1.0]], dtype=np.float32),
        targets=np.asarray([[1.0], [2.0]], dtype=np.float32),
        aux={},
        metadata={"coordinate_names": ("x",), "output_names": ("u",)},
        reference_shape=(2,),
        flat_index=np.arange(2),
    )

    with pytest.raises(ModuleNotFoundError, match=r"phijax\[io\]"):
        save_prediction_artifact(tmp_path / "prediction.npz", pool.targets, pool, save_mat=True)
    assert save_prediction_artifact(tmp_path / "prediction.npz", pool.targets, pool).is_file()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["prediction.npz"]


def test_missing_extra_error_keeps_the_package_name() -> None:
    """Verify that the error names the missing package and the extra that provides it."""
    error = missing_extra_error("io", ModuleNotFoundError("No module named 'h5py'", name="h5py"))

    assert error.name == "h5py"
    assert "`h5py` is not installed" in str(error)
    assert 'uv add "phijax[io]"' in str(error)

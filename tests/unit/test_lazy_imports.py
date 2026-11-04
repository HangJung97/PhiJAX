import importlib
import json
import subprocess
import sys
import types

import pytest

from phijax._lazy import lazy_exports

_LAZY_PACKAGES = ("phijax", "phijax.data", "phijax.evaluation", "phijax.utils")


def _modules_loaded_by(statement: str) -> list[str]:
    """Run an import statement in a fresh interpreter and list the top-level modules it loaded.

    Args:
        statement: Python source executed before the module list is read.

    Returns:
        Sorted top-level names of every module in `sys.modules` after `statement`.
    """
    script = f"import sys, json\n{statement}\nprint(json.dumps(sorted({{name.split('.')[0] for name in sys.modules}})))"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, check=True, text=True)
    return json.loads(result.stdout.splitlines()[-1])


def test_importing_phijax_loads_no_heavy_or_optional_dependency() -> None:
    """Verify that the root package imports none of its runtime dependencies until a name is used."""
    loaded = set(_modules_loaded_by("import phijax"))

    assert not loaded & {"jax", "jaxlib", "flax", "optax", "orbax", "omegaconf", "hydra", "rich", "scipy", "h5py"}


def test_host_data_and_evaluation_modules_load_no_jax() -> None:
    """Verify that NumPy-only data construction and evaluation stay independent of JAX."""
    loaded = set(
        _modules_loaded_by(
            "import phijax.data.pools, phijax.data.builders, phijax.data.artifacts, phijax.data.transforms\n"
            "from phijax.evaluation import RegressionEvaluator, regression_metrics"
        )
    )

    assert not loaded & {"jax", "jaxlib", "flax", "optax", "orbax"}


@pytest.mark.parametrize("package_name", _LAZY_PACKAGES)
def test_every_lazy_export_resolves_to_its_defining_object(package_name: str) -> None:
    """Verify that every name in a lazy package's `__all__` imports and appears in `dir()`.

    Args:
        package_name: Package whose initializer uses :func:`phijax._lazy.lazy_exports`.
    """
    package = importlib.import_module(package_name)

    for name in package.__all__:
        assert getattr(package, name) is not None
        assert name in dir(package)


def test_lazy_exports_cache_values_and_support_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify first-access import, caching on the package, aliases, and unknown-name errors.

    Args:
        monkeypatch: Fixture registering a temporary package in `sys.modules`.
    """
    package = types.ModuleType("lazy_package")
    monkeypatch.setitem(sys.modules, "lazy_package", package)
    get_attribute, list_attributes = lazy_exports(
        "lazy_package",
        {"dumps": "json", "encode": "json:dumps"},
    )

    assert get_attribute("encode") is json.dumps
    assert vars(package)["encode"] is json.dumps
    assert {"dumps", "encode"} <= set(list_attributes())
    with pytest.raises(AttributeError, match="module 'lazy_package' has no attribute 'missing'"):
        get_attribute("missing")

import pytest

from phijax.integrations.omegaconf import register_omegaconf_resolvers


@pytest.fixture(scope="session", autouse=True)
def register_test_omegaconf_resolvers() -> None:
    """Register PhiJAX resolvers before tests compose or resolve Hydra configurations."""
    register_omegaconf_resolvers()

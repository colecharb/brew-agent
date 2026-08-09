import pytest

from seed_fixture import load_seed_brews


@pytest.fixture(scope="session")
def seed_brews():
    return load_seed_brews()

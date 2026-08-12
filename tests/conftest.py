import pytest

from seed_fixture import load_seed_brews, seed_sql_path


@pytest.fixture(scope="session")
def seed_brews():
    try:
        return load_seed_brews()
    except FileNotFoundError:
        pytest.skip(
            f"seed dump not found at {seed_sql_path()} — "
            f"set BREW_AGENT_SEED_SQL to run the tests that need real data"
        )

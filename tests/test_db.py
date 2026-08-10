"""The data layer's time cutoff, against a recording query builder.

`test_agent.py` proves the agent hands the cutoff down. This proves the queries
actually carry it into PostgREST — which is the layer the leak lived in, and the
one a fake database cannot vouch for.
"""

from types import SimpleNamespace

import pytest

from brew_agent.db import BrewDatabase

CUTOFF = "2026-01-10T00:00:00+00:00"


def api_row(brew_id: str, when: str, grind: str = "500") -> dict:
    """A row in the shape `db.SELECT` returns."""
    return {
        "id": brew_id,
        "created_by": "user-1",
        "brewTimestamp": when,
        "profileCoffee": {"id": "bag-1", "coffeeId": "coffee-1", "coffee": None},
        "grinder_id": "g1",
        "brewer_id": "b1",
        "grindSetting": grind,
        "coffeeWeight": "15.00",
        "targetWeight": "250.00",
        "notes": "sour",
        "rating": 2,
    }


class FakeQuery:
    """Records every filter applied, then returns canned rows."""

    def __init__(self, log: list, rows: list) -> None:
        self._log = log
        self._rows = rows

    def _record(self, op, *args):
        self._log.append((op, *args))
        return self

    def select(self, *a, **k):
        return self._record("select")

    def eq(self, *a):
        return self._record("eq", *a)

    def in_(self, *a):
        return self._record("in_", *a)

    def gte(self, *a):
        return self._record("gte", *a)

    def lt(self, *a):
        return self._record("lt", *a)

    def is_(self, *a):
        return self._record("is_", *a)

    def order(self, *a, **k):
        return self._record("order", *a)

    def limit(self, *a):
        return self._record("limit", *a)

    def range(self, *a):
        return self._record("range", *a)

    def execute(self):
        return SimpleNamespace(data=self._rows)


class FakeClient:
    def __init__(self, rows_by_table: dict) -> None:
        self.log: list = []
        self._rows = rows_by_table

    def table(self, name: str) -> FakeQuery:
        self.log.append(("table", name))
        return FakeQuery(self.log, self._rows.get(name, []))


@pytest.fixture
def db():
    client = FakeClient(
        {
            "brew": [api_row("brew-0", "2026-01-05T00:00:00+00:00", "510")],
            "profile_coffee_public": [{"id": "bag-1"}],
        }
    )
    return BrewDatabase(client, "user-1"), client


def ops(client, name):
    return [entry for entry in client.log if entry[0] == name]


class TestHistoryQueriesCarryTheCutoff:
    def test_bean_history(self, db):
        database, client = db
        database.get_user_brews_with_bean("coffee-1", user_id="user-1", as_of=CUTOFF)
        assert ("lt", "brew_timestamp", CUTOFF) in client.log

    def test_gear_history(self, db):
        database, client = db
        database.get_user_brews_with_gear("g1", "b1", 2, as_of=CUTOFF)
        assert ("lt", "brew_timestamp", CUTOFF) in client.log

    def test_no_cutoff_means_no_filter(self, db):
        """Production has no future, so the filter is simply absent there."""
        database, client = db
        database.get_user_brews_with_gear("g1", "b1", 2)
        assert not ops(client, "lt")

    def test_the_cutoff_is_applied_before_the_limit(self, db):
        """Otherwise `limit` counts rows the caller can never see.

        A post-filter would ask for 20, get 20 including later brews, then trim
        to however few survive — quietly returning less history than requested.
        """
        database, client = db
        database.get_user_brews_with_gear("g1", "b1", 2, as_of=CUTOFF)
        names = [entry[0] for entry in client.log]
        assert names.index("lt") < names.index("limit")


class TestGetBrew:
    def test_returns_a_brew_at_the_cutoff(self, db):
        """The brew under diagnosis sits exactly on the boundary."""
        database, client = db
        client._rows["brew"] = [api_row("brew-1", CUTOFF)]
        assert database.get_brew("brew-1", as_of=CUTOFF).id == "brew-1"

    def test_returns_an_earlier_brew(self, db):
        database, _ = db
        assert database.get_brew("brew-0", as_of=CUTOFF).id == "brew-0"

    def test_refuses_a_later_brew(self, db):
        database, client = db
        client._rows["brew"] = [api_row("brew-future", "2026-01-20T00:00:00+00:00")]
        assert database.get_brew("brew-future", as_of=CUTOFF) is None

    def test_without_a_cutoff_any_brew_comes_back(self, db):
        database, client = db
        client._rows["brew"] = [api_row("brew-future", "2026-01-20T00:00:00+00:00")]
        assert database.get_brew("brew-future") is not None

    def test_missing_brew_is_none(self, db):
        database, client = db
        client._rows["brew"] = []
        assert database.get_brew("nope", as_of=CUTOFF) is None


def test_hidden_brews_are_excluded_everywhere(db):
    """Moderated rows must never reach the model or the eval."""
    database, client = db
    database.get_user_brews_with_gear("g1", "b1", 2, as_of=CUTOFF)
    assert ("is_", "hidden_at", "null") in client.log


def test_min_rating_uses_gte_which_drops_unrated(db):
    database, client = db
    database.get_user_brews_with_gear("g1", "b1", 3, as_of=CUTOFF)
    assert ("gte", "rating", 3) in client.log

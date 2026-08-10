"""The community rung: everyone else's brews, grouped by what they compare on.

The fake here filters for real rather than returning canned rows, because the
tiering and the dedupe between tiers are the parts most likely to be quietly
wrong — each relaxation is a superset of the tight query.
"""

from types import SimpleNamespace

import pytest

from brew_agent.db import COMMUNITY_TIERS, BrewDatabase
from brew_agent.models import Brew
from brew_agent.tools import (
    ALL_TOOLS,
    COMMUNITY_SYSTEM_PROMPT,
    COMMUNITY_TOOLS,
    GET_COMMUNITY_BREWS,
    SYSTEM_PROMPT,
    TIER_GUIDANCE,
    Toolbox,
)

# Filters name database columns; rows come back in the API's own shape.
COLUMNS = {
    "profile_coffee_id": lambda row: row["profileCoffee"]["id"],
    "brew_timestamp": lambda row: row["brewTimestamp"],
}


def value(row, column):
    return COLUMNS.get(column, lambda r: r.get(column))(row)


class FilteringQuery:
    """Applies the filters it is given to an in-memory list of rows."""

    def __init__(self, rows):
        self._rows = list(rows)
        self._limit = None

    def select(self, *a, **k):
        return self

    def is_(self, *a):
        return self

    def eq(self, column, wanted):
        self._rows = [r for r in self._rows if value(r, column) == wanted]
        return self

    def neq(self, column, unwanted):
        self._rows = [r for r in self._rows if value(r, column) != unwanted]
        return self

    def in_(self, column, wanted):
        self._rows = [r for r in self._rows if value(r, column) in set(wanted)]
        return self

    def gte(self, column, floor):
        self._rows = [
            r for r in self._rows if value(r, column) is not None
            and value(r, column) >= floor
        ]
        return self

    def lt(self, column, ceiling):
        self._rows = [r for r in self._rows if value(r, column) < ceiling]
        return self

    def order(self, column, desc=False):
        self._rows.sort(key=lambda r: value(r, column) or "", reverse=desc)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def execute(self):
        rows = self._rows if self._limit is None else self._rows[: self._limit]
        return SimpleNamespace(data=rows)


class FilteringClient:
    def __init__(self, brews, bags):
        self._brews = brews
        self._bags = bags

    def table(self, name):
        return FilteringQuery(self._brews if name == "brew" else self._bags)


def row(brew_id, user, coffee, grinder, brewer, rating=3, day=1):
    return {
        "id": brew_id,
        "created_by": user,
        "brewTimestamp": f"2026-01-{day:02d}T00:00:00+00:00",
        "profileCoffee": {"id": f"bag-{coffee}-{user}", "coffeeId": coffee, "coffee": None},
        "grinder_id": grinder,
        "brewer_id": brewer,
        "grindSetting": "500",
        "coffeeWeight": "15.00",
        "targetWeight": "250.00",
        "notes": "tasty",
        "rating": rating,
    }


def database(rows):
    bags = [{"id": r["profileCoffee"]["id"], "coffee_id": r["profileCoffee"]["coffeeId"]}
            for r in rows]
    # `_bag_ids_for_coffee` filters bags by coffee_id, so give the fake that column.
    return BrewDatabase(FilteringClient(rows, bags), "me")


CUTOFF = "2026-02-01T00:00:00+00:00"


def community(rows, **kwargs):
    kwargs.setdefault("coffee_id", "gesha")
    kwargs.setdefault("grinder_id", "z1")
    kwargs.setdefault("brewer_id", "v60")
    kwargs.setdefault("exclude_user", "me")
    kwargs.setdefault("as_of", CUTOFF)
    return database(rows).get_community_brews(**kwargs)


class TestTiering:
    def test_an_exact_match_lands_in_the_tight_tier(self):
        tiers = community([row("a", "someone", "gesha", "z1", "v60")])
        assert [b.id for b in tiers["same_coffee_same_setup"]] == ["a"]

    def test_the_users_own_brews_are_never_community(self):
        """Their own history is what the other two tools are for."""
        tiers = community([row("mine", "me", "gesha", "z1", "v60")])
        assert all(not brews for brews in tiers.values())

    def test_later_brews_are_excluded_by_the_cutoff(self):
        tiers = community([row("future", "someone", "gesha", "z1", "v60", day=20)],
                          as_of="2026-01-05T00:00:00+00:00")
        assert all(not brews for brews in tiers.values())

    def test_poorly_rated_brews_are_excluded(self):
        """A setting nobody scored well says nothing about whether it worked."""
        tiers = community([row("bad", "someone", "gesha", "z1", "v60", rating=0)])
        assert all(not brews for brews in tiers.values())

    def test_a_thin_tight_tier_widens_on_both_axes(self):
        rows = [
            row("exact", "u1", "gesha", "z1", "v60"),
            row("same-gear", "u2", "other-coffee", "z1", "v60"),
            row("same-coffee", "u3", "gesha", "ek43", "aeropress"),
        ]
        tiers = community(rows)
        assert [b.id for b in tiers["same_coffee_same_setup"]] == ["exact"]
        assert [b.id for b in tiers["same_setup_other_coffee"]] == ["same-gear"]
        assert [b.id for b in tiers["same_coffee_other_setup"]] == ["same-coffee"]

    def test_a_full_tight_tier_does_not_widen(self):
        """Enough exact matches means the looser groups are only noise."""
        rows = [row(f"e{i}", f"u{i}", "gesha", "z1", "v60") for i in range(4)]
        rows.append(row("same-gear", "other", "different-coffee", "z1", "v60"))
        tiers = community(rows, widen_below=3)
        assert len(tiers["same_coffee_same_setup"]) == 4
        assert tiers["same_setup_other_coffee"] == []

    def test_a_brew_never_appears_in_two_tiers(self):
        """Each relaxation is a superset, so without dedupe the tight rows repeat."""
        rows = [
            row("exact", "u1", "gesha", "z1", "v60"),
            row("same-gear", "u2", "other", "z1", "v60"),
        ]
        tiers = community(rows)
        seen = [b.id for tier in COMMUNITY_TIERS for b in tiers[tier]]
        assert len(seen) == len(set(seen))

    def test_every_tier_is_present_even_when_empty(self):
        tiers = community([])
        assert set(tiers) == set(COMMUNITY_TIERS)


class TestToolPayload:
    class FakeDB:
        def __init__(self, tiers):
            self._tiers = tiers
            self.calls = []

        def get_community_brews(self, **kwargs):
            self.calls.append(kwargs)
            return self._tiers

    @staticmethod
    def brew(bid):
        return Brew(
            id=bid,
            created_by="someone",
            brew_timestamp="2026-01-01T00:00:00+00:00",
            profile_coffee_id="bag",
            coffee_id="gesha",
            grind_setting="500",
        )

    def toolbox(self, tiers):
        db = self.FakeDB(tiers)
        return Toolbox(db), db

    def dispatch(self, tiers, viewer_id="me", **args):
        box, db = self.toolbox(tiers)
        args.setdefault("coffee_id", "gesha")
        args.setdefault("grinder_id", "z1")
        args.setdefault("brewer_id", "v60")
        payload, trace = box.dispatch(
            "get_community_brews", args, as_of=CUTOFF, viewer_id=viewer_id
        )
        return payload, trace, db

    def test_each_group_carries_what_transfers(self):
        """The failure mode is silent: a stranger's grind number looks usable."""
        tiers = {name: [] for name in COMMUNITY_TIERS}
        tiers["same_coffee_other_setup"] = [self.brew("x")]
        payload, _, _ = self.dispatch(tiers)
        group = payload["groups"][0]
        assert group["match"] == "same_coffee_other_setup"
        assert group["what_transfers"] == TIER_GUIDANCE["same_coffee_other_setup"]
        assert "do NOT" in group["what_transfers"]

    def test_empty_groups_are_omitted(self):
        tiers = {name: [] for name in COMMUNITY_TIERS}
        tiers["same_coffee_same_setup"] = [self.brew("x")]
        payload, _, _ = self.dispatch(tiers)
        assert [g["match"] for g in payload["groups"]] == ["same_coffee_same_setup"]
        assert payload["found"] == 1

    def test_the_viewer_is_excluded_by_the_harness_not_the_model(self):
        """The model cannot widen the community to include the user."""
        tiers = {name: [] for name in COMMUNITY_TIERS}
        _, _, db = self.dispatch(tiers, viewer_id="user-42", exclude_user="someone-else")
        assert db.calls[0]["exclude_user"] == "user-42"

    def test_a_missing_viewer_is_an_error_not_a_silent_widening(self):
        tiers = {name: [] for name in COMMUNITY_TIERS}
        payload, trace, db = self.dispatch(tiers, viewer_id=None)
        assert "error" in payload
        assert db.calls == []
        assert trace["error"]

    def test_the_row_count_spans_every_group(self):
        tiers = {name: [] for name in COMMUNITY_TIERS}
        tiers["same_coffee_same_setup"] = [self.brew("a")]
        tiers["same_coffee_other_setup"] = [self.brew("b"), self.brew("c")]
        _, trace, _ = self.dispatch(tiers)
        assert trace["row_count"] == 3


class TestTheRungIsOneCapabilityWide:
    """`agent` must stay exactly as measured, or the gap prices two things."""

    def test_the_community_arm_adds_exactly_one_tool(self):
        assert len(COMMUNITY_TOOLS) == len(ALL_TOOLS) + 1
        added = [t for t in COMMUNITY_TOOLS if t not in ALL_TOOLS]
        assert [t["name"] for t in added] == ["get_community_brews"]

    def test_the_agent_arm_keeps_its_own_tools(self):
        assert GET_COMMUNITY_BREWS not in ALL_TOOLS
        assert [t["name"] for t in ALL_TOOLS] == [
            "get_brew",
            "get_user_brews_with_bean",
            "get_user_brews_with_gear",
            "submit_recommendation",
        ]

    def test_the_community_prompt_actually_diverged(self):
        """It is built by substitution, which fails silently if the text moves."""
        assert COMMUNITY_SYSTEM_PROMPT != SYSTEM_PROMPT
        assert "this user's own history, not in generic" not in COMMUNITY_SYSTEM_PROMPT
        assert "get_community_brews" in COMMUNITY_SYSTEM_PROMPT

    def test_both_prompts_still_demand_a_number_on_the_same_dial(self):
        """The shared half has to stay shared, or the arms score differently."""
        for prompt in (SYSTEM_PROMPT, COMMUNITY_SYSTEM_PROMPT):
            assert "Express your answer as a number on that same dial." in prompt
            assert "submit_recommendation" in prompt

    @pytest.mark.parametrize("tier", COMMUNITY_TIERS)
    def test_every_tier_has_guidance(self, tier):
        assert TIER_GUIDANCE[tier]

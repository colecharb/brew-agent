"""Domain types.

Column names mirror `supabase/schema.sql` exactly. Two things about the schema
are load-bearing everywhere downstream:

- `brew.grind_setting` is `text`, not a number. Each grinder has its own scale,
  and even within one grinder the usable range depends on the brew method: the
  Z1 in this dataset reads in microns throughout, but espresso brewers sit at
  5-250 and filter brewers at 475-600. A grind number is only comparable to
  another number from the same (grinder, brewer) setup.
- `brew.rating` is a nullable smallint from 0-4 indexing a label array, not a
  star count. NULL means unrated, and `rating >= n` silently drops those rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

RATING_LABELS = ("Flawed", "Okay", "Good", "Great", "Superb")

_NUMERIC = re.compile(r"^\d+(\.\d+)?$")

# The parameters the agent may recommend changing. `primary_lever` is
# constrained to these plus "none".
LEVERS = ("grind_setting", "coffee_weight", "target_weight", "water_temp", "time")


def parse_grind(value: str | None) -> float | None:
    """Return a grind setting as a float, or None if it isn't plain numeric.

    Two of the 724 brews in this dataset use dotted forms like "1.8.2". They are
    excluded rather than guessed at.
    """
    if value is None:
        return None
    text = value.strip()
    return float(text) if _NUMERIC.match(text) else None


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    n = _num(value)
    return None if n is None else int(n)


@dataclass
class Brew:
    """One row of `public.brew`, with catalogue names resolved."""

    id: str
    created_by: str | None
    brew_timestamp: str

    profile_coffee_id: str | None
    coffee_id: str | None
    coffee_name: str | None = None
    roaster_name: str | None = None
    origin: str | None = None
    process: str | None = None

    grinder_id: str | None = None
    brewer_id: str | None = None
    burr_id: str | None = None
    filter_id: str | None = None
    grinder_name: str | None = None
    brewer_name: str | None = None
    burr_name: str | None = None
    filter_name: str | None = None

    grind_setting: str | None = None
    coffee_weight: float | None = None
    target_weight: float | None = None
    brew_weight: float | None = None
    water_temp: float | None = None
    time: int | None = None
    days_off_roast: int | None = None

    notes: str = ""
    recipe: str = ""
    rating: int | None = None
    rebrew_of: str | None = None

    @property
    def grind_value(self) -> float | None:
        return parse_grind(self.grind_setting)

    @property
    def ratio(self) -> float | None:
        """Brew ratio as water:coffee. There is no ratio column; it derives."""
        if self.coffee_weight and self.target_weight:
            return self.target_weight / self.coffee_weight
        return None

    @property
    def rating_label(self) -> str | None:
        if self.rating is None or not 0 <= self.rating < len(RATING_LABELS):
            return None
        return RATING_LABELS[self.rating]

    @property
    def setup(self) -> tuple[str | None, str | None]:
        """The (grinder, brewer) pair that makes grind numbers comparable."""
        return (self.grinder_id, self.brewer_id)

    @classmethod
    def from_api_row(cls, row: dict[str, Any]) -> "Brew":
        """Build from a PostgREST row using the select shape in `db.SELECT`."""
        bag = row.get("profileCoffee") or {}
        coffee = bag.get("coffee") or {}
        roaster = coffee.get("roaster") or {}

        def name_of(key: str) -> str | None:
            nested = row.get(key) or {}
            return nested.get("name")

        return cls(
            id=row["id"],
            created_by=row.get("created_by"),
            brew_timestamp=row["brewTimestamp"],
            profile_coffee_id=bag.get("id"),
            coffee_id=bag.get("coffeeId"),
            coffee_name=coffee.get("name"),
            roaster_name=roaster.get("name"),
            origin=coffee.get("origin"),
            process=coffee.get("process"),
            grinder_id=row.get("grinder_id"),
            brewer_id=row.get("brewer_id"),
            burr_id=row.get("burr_id"),
            filter_id=row.get("filter_id"),
            grinder_name=name_of("grinder"),
            brewer_name=name_of("brewer"),
            burr_name=name_of("burr"),
            filter_name=name_of("filter"),
            grind_setting=row.get("grindSetting"),
            coffee_weight=_num(row.get("coffeeWeight")),
            target_weight=_num(row.get("targetWeight")),
            brew_weight=_num(row.get("brewWeight")),
            water_temp=_num(row.get("waterTemp")),
            time=_int(row.get("time")),
            days_off_roast=_int(row.get("daysOffRoast")),
            notes=row.get("notes") or "",
            recipe=row.get("recipe") or "",
            rating=_int(row.get("rating")),
            rebrew_of=row.get("rebrewOf"),
        )

    def to_tool_result(self) -> dict[str, Any]:
        """The shape handed back to the model inside a tool result.

        Every row carries `created_by` and the gear names on purpose: `user_id`
        is optional on the history tools, so the model has to be able to see
        when a row belongs to somebody else on different gear. A grind number
        does not transfer across setups.
        """
        return {
            "brew_id": self.id,
            "created_by": self.created_by,
            "brewed_at": self.brew_timestamp,
            "coffee": {
                "coffee_id": self.coffee_id,
                "name": self.coffee_name,
                "roaster": self.roaster_name,
                "origin": self.origin,
                "process": self.process,
            },
            "gear": {
                "grinder_id": self.grinder_id,
                "grinder": self.grinder_name,
                "brewer_id": self.brewer_id,
                "brewer": self.brewer_name,
                "burr": self.burr_name,
                "filter": self.filter_name,
            },
            "params": {
                "grind_setting": self.grind_setting,
                "dose_g": self.coffee_weight,
                "target_yield_g": self.target_weight,
                "actual_yield_g": self.brew_weight,
                "ratio": round(self.ratio, 2) if self.ratio else None,
                "water_temp_c": self.water_temp,
                "time_seconds": self.time,
                "days_off_roast": self.days_off_roast,
            },
            "rating": self.rating,
            "rating_label": self.rating_label,
            "notes": self.notes,
            "recipe": self.recipe,
        }


@dataclass
class Recommendation:
    """The agent's answer: what to change on the next brew.

    A null field means "leave this alone". `grind_setting` must be expressed on
    the same dial as the brew being diagnosed — see the note in the module
    docstring about why a bare direction word is not enough.
    """

    grind_setting: str | None = None
    coffee_weight: float | None = None
    target_weight: float | None = None
    water_temp: float | None = None
    time: int | None = None
    primary_lever: str = "none"
    reasoning: str = ""
    # Populated by the harness, not the model.
    error: str | None = None

    @property
    def grind_value(self) -> float | None:
        return parse_grind(self.grind_setting)

    @property
    def changes_nothing(self) -> bool:
        return self.primary_lever == "none" and not any(
            getattr(self, lever) is not None for lever in LEVERS
        )

    @classmethod
    def from_tool_input(cls, data: dict[str, Any]) -> "Recommendation":
        grind = data.get("grind_setting")
        lever = data.get("primary_lever") or "none"
        if lever not in LEVERS and lever != "none":
            lever = "none"
        return cls(
            grind_setting=str(grind) if grind is not None else None,
            coffee_weight=_num(data.get("coffee_weight")),
            target_weight=_num(data.get("target_weight")),
            water_temp=_num(data.get("water_temp")),
            time=_int(data.get("time")),
            primary_lever=lever,
            reasoning=data.get("reasoning") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grind_setting": self.grind_setting,
            "coffee_weight": self.coffee_weight,
            "target_weight": self.target_weight,
            "water_temp": self.water_temp,
            "time": self.time,
            "primary_lever": self.primary_lever,
            "reasoning": self.reasoning,
            "error": self.error,
        }


@dataclass
class HoldoutPair:
    """An earlier brew and the brew the same user actually made next.

    `before.notes` is the complaint handed to the agent; `after` is the ground
    truth held out from it.
    """

    before: Brew
    after: Brew
    leaky: bool = False
    leak_phrase: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.before.id[:8]}-{self.after.id[:8]}"

    @property
    def user_id(self) -> str:
        # Filtered to non-null in pairs.build_pairs.
        return str(self.before.created_by)

    @property
    def rating_improved(self) -> bool | None:
        if self.before.rating is None or self.after.rating is None:
            return None
        return self.after.rating > self.before.rating

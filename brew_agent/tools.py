"""Tool definitions and dispatch.

Three data tools plus `submit_recommendation`, which is how the agent finishes.
Making the answer a tool call rather than parsed prose means a malformed answer
is retried by the model against the schema instead of by a regex here, and the
loop has an unambiguous termination signal.
"""

from __future__ import annotations

import time
from typing import Any

from .db import BrewDatabase
from .models import LEVERS

SUBMIT_TOOL = "submit_recommendation"


def _nullable(json_type: str, description: str) -> dict[str, Any]:
    """A field the model may leave unset, meaning "don't change this"."""
    return {
        "anyOf": [{"type": json_type}, {"type": "null"}],
        "description": description,
    }


DATA_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_brew",
        "description": (
            "Look up one brew by id. Returns its parameters (grind setting, "
            "dose, target yield, water temperature, brew time, days off roast), "
            "the coffee and gear it used, its 0-4 rating, and the user's "
            "free-text notes and recipe. Start here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "brew_id": {"type": "string", "description": "The brew's UUID."}
            },
            "required": ["brew_id"],
        },
    },
    {
        "name": "get_user_brews_with_bean",
        "description": (
            "Brew history for one coffee, most recent first. Use this to see how "
            "this particular coffee has behaved before — where the grind ended "
            "up, which ratios scored well, how it changed as it aged. "
            "coffee_id is the catalogue coffee, so this spans every bag of it. "
            "Pass user_id to stay within one person's history; omit it to see "
            "how everyone brews this coffee, but note that other people's grind "
            "numbers are on their own grinders and do not transfer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "coffee_id": {
                    "type": "string",
                    "description": "Catalogue coffee UUID, from get_brew.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Optional. Restrict to one user's brews.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows, default 20.",
                },
            },
            "required": ["coffee_id"],
        },
    },
    {
        "name": "get_user_brews_with_gear",
        "description": (
            "Well-rated brews on the same grinder-and-brewer setup, most recent "
            "first, filtered to rating >= min_rating (0=Flawed, 1=Okay, 2=Good, "
            "3=Great, 4=Superb). Use this to establish what a good brew looks "
            "like on this equipment — the grind range that works, typical ratio "
            "and time — so a recommendation is anchored to a real baseline "
            "rather than a generic one. Grinder and brewer together matter: the "
            "same grinder reads very differently for espresso and for filter. "
            "Pass user_id to stay within one person's history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "grinder_id": {"type": "string", "description": "Grinder UUID."},
                "brewer_id": {"type": "string", "description": "Brewer UUID."},
                "min_rating": {
                    "type": "integer",
                    "description": "Minimum rating, 0-4. Unrated brews are excluded.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Optional. Restrict to one user's brews.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows, default 20.",
                },
            },
            "required": ["grinder_id", "brewer_id", "min_rating"],
        },
    },
]


SUBMIT_RECOMMENDATION: dict[str, Any] = {
    "name": SUBMIT_TOOL,
    "description": (
        "Submit the adjustment for the next brew. Call this exactly once, when "
        "you are done investigating. Leave a field null to say 'do not change "
        "this'. Submitting every field null is a valid answer if the brew needs "
        "no change."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "grind_setting": _nullable(
                "string",
                "The grind setting to use next, as a number on this grinder's "
                "own dial — the same scale as the brew you are diagnosing. If "
                "that brew was ground at 500, answer with something like '485', "
                "not 'finer'. Null to leave the grind alone.",
            ),
            "coffee_weight": _nullable("number", "Dose in grams, or null."),
            "target_weight": _nullable(
                "number", "Target yield in grams, or null."
            ),
            "water_temp": _nullable(
                "number", "Water temperature in Celsius, or null."
            ),
            "time": _nullable("integer", "Total brew time in seconds, or null."),
            "primary_lever": {
                "type": "string",
                "enum": [*LEVERS, "none"],
                "description": (
                    "The single change that matters most. Users often move "
                    "several things at once, so say which one you are actually "
                    "betting on. Use 'none' if you are recommending no change."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Why, in two or three sentences. Cite what you found in the "
                    "history — the grind range that scored well on this setup, "
                    "what happened last time this coffee was brewed — rather "
                    "than restating general brewing theory."
                ),
            },
        },
        "required": [
            "grind_setting",
            "coffee_weight",
            "target_weight",
            "water_temp",
            "time",
            "primary_lever",
            "reasoning",
        ],
        "additionalProperties": False,
    },
}

ALL_TOOLS = [*DATA_TOOLS, SUBMIT_RECOMMENDATION]


SYSTEM_PROMPT = """You diagnose coffee brews. Given a brew and the drinker's \
complaint about how it tasted, recommend what to change on the next one.

Ground the recommendation in this user's own history, not in generic brewing \
advice. The tools let you pull up the brew itself, their previous brews with \
the same coffee, and their well-rated brews on the same grinder and brewer. A \
recommendation that lands inside the range they already brew well in is worth \
more than a textbook answer.

Grind settings are free text and every grinder has its own scale — some count \
up as they get finer, some count down, some read in microns. Never assume which \
way is which. Work in the numbers you actually observe for this grinder and \
brewer: if their good brews on this setup sit at 480-500 and this one was at \
515, that tells you where to go without needing to know what the numbers mean. \
Express your answer as a number on that same dial.

Be specific about size, not just direction. Look at how far this user actually \
moves between brews and stay in that neighbourhood — a change they would never \
make is not useful even if the direction is right.

When you have what you need, call submit_recommendation. Do not answer in prose."""


class Toolbox:
    """Dispatches the data tools and records what each call returned."""

    def __init__(self, db: BrewDatabase) -> None:
        self._db = db

    def dispatch(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], dict]:
        """Run one tool call. Returns (payload for the model, trace entry)."""
        started = time.monotonic()
        try:
            payload = self._run(name, args)
            error = None
        except Exception as exc:  # surfaced to the model, not raised
            payload = {"error": f"{type(exc).__name__}: {exc}"}
            error = payload["error"]

        rows = payload.get("brews")
        trace = {
            "tool": name,
            "arguments": args,
            "scoped_to_user": bool(args.get("user_id")),
            "row_count": len(rows) if isinstance(rows, list) else (0 if error else 1),
            "latency_ms": round((time.monotonic() - started) * 1000),
            "error": error,
            "returned": payload,
        }
        return payload, trace

    def _run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "get_brew":
            brew = self._db.get_brew(args["brew_id"])
            if brew is None:
                return {"error": f"No visible brew with id {args['brew_id']}."}
            return {"brew": brew.to_tool_result()}

        if name == "get_user_brews_with_bean":
            brews = self._db.get_user_brews_with_bean(
                coffee_id=args["coffee_id"],
                user_id=args.get("user_id"),
                limit=int(args.get("limit") or 20),
            )
            return {"brews": [b.to_tool_result() for b in brews]}

        if name == "get_user_brews_with_gear":
            brews = self._db.get_user_brews_with_gear(
                grinder_id=args["grinder_id"],
                brewer_id=args["brewer_id"],
                min_rating=int(args["min_rating"]),
                user_id=args.get("user_id"),
                limit=int(args.get("limit") or 20),
            )
            return {"brews": [b.to_tool_result() for b in brews]}

        return {"error": f"Unknown tool {name}."}

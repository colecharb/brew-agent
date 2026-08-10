"""Tool definitions and dispatch.

Three data tools plus `submit_recommendation`, which is how the agent finishes.
Making the answer a tool call rather than parsed prose means a malformed answer
is retried by the model against the schema instead of by a regex here, and the
loop has an unambiguous termination signal.
"""

from __future__ import annotations

import time
from typing import Any

from .db import COMMUNITY_TIERS, BrewDatabase
from .models import LEVERS

SUBMIT_TOOL = "submit_recommendation"


def nullable(json_type: str, description: str) -> dict[str, Any]:
    """A field the model may leave unset.

    Under `strict`, every property is required, so "leave this out" has to be
    expressible as a value. Documenting a plain string field as "empty when
    there is nothing to say" does not work: asked for a required string it has
    no content for, the model emits *something*, and what came back in practice
    was a fragment of the tool-call markup rather than "". Give it null instead.
    """
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
            "how everyone brews this coffee. Another person's grind number is "
            "comparable when their grinder and brewer match — those set the "
            "units and the regime — while ratio, water temperature and days "
            "off roast compare whatever the gear. Only brews made before the "
            "one you are diagnosing are returned, so an empty result means this "
            "is the first — not that the lookup failed."
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
            "Pass user_id to stay within one person's history. Only brews made "
            "before the one you are diagnosing are returned, so an empty result "
            "means there is no earlier baseline on this setup — say so rather "
            "than inventing one."
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
            "grind_setting": nullable(
                "string",
                "The grind setting to use next, as a number on this grinder's "
                "own dial — the same scale as the brew you are diagnosing. If "
                "that brew was ground at 500, answer with something like '485', "
                "not 'finer'. Null to leave the grind alone.",
            ),
            "coffee_weight": nullable("number", "Dose in grams, or null."),
            "target_weight": nullable(
                "number", "Target yield in grams, or null."
            ),
            "water_temp": nullable(
                "number", "Water temperature in Celsius, or null."
            ),
            "time": nullable("integer", "Total brew time in seconds, or null."),
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


# What each tier of community result is good for. Sent with the rows rather
# than left to be worked out, because the failure mode is silent: a grind
# number read off somebody else's grinder looks exactly like a usable one.
TIER_GUIDANCE = {
    "same_coffee_same_setup": (
        "Same coffee, same grinder and brewer. Everything here compares "
        "directly, grind number included."
    ),
    "same_setup_other_coffee": (
        "Same grinder and brewer, other coffees. Grind numbers are on this "
        "same dial, so read them as the range this equipment works in — not "
        "as a target for this particular coffee."
    ),
    "same_coffee_other_setup": (
        "Same coffee, different grinder or brewer. Ratio, water temperature "
        "and days off roast compare. The grind numbers do NOT — they are on "
        "another dial and mean nothing here."
    ),
}

GET_COMMUNITY_BREWS: dict[str, Any] = {
    "name": "get_community_brews",
    "description": (
        "Other people's well-rated brews of this coffee, on this equipment, or "
        "both. Use it when this user's own history is thin — a coffee they have "
        "brewed once, or gear they are new to — and to sanity-check a "
        "recommendation against how the same coffee behaves for everyone else. "
        "Results come back in up to three labelled groups, tightest first, "
        "because what transfers depends on how the brew matches: everything "
        "compares on the same coffee and setup, only the grind range compares "
        "on the same setup with a different coffee, and only ratio, "
        "temperature and days off roast compare on the same coffee with "
        "different gear. Each group says so, and every row carries its owner "
        "and gear. The tight group is often empty or a single brew, so the "
        "looser groups are the normal case rather than a fallback. The user's "
        "own brews are excluded — the other tools are for those. Only brews "
        "made before the one you are diagnosing are returned."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "coffee_id": {
                "type": "string",
                "description": "Catalogue coffee UUID, from get_brew.",
            },
            "grinder_id": {"type": "string", "description": "Grinder UUID."},
            "brewer_id": {"type": "string", "description": "Brewer UUID."},
            "min_rating": {
                "type": "integer",
                "description": (
                    "Minimum rating, 0-4. Default 2 (Good). Unrated brews are "
                    "excluded — a setting nobody scored says nothing about "
                    "whether it worked."
                ),
            },
            "limit": {"type": "integer", "description": "Max rows, default 20."},
        },
        "required": ["coffee_id", "grinder_id", "brewer_id"],
    },
}

COMMUNITY_TOOLS = [*DATA_TOOLS, GET_COMMUNITY_BREWS, SUBMIT_RECOMMENDATION]


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


COMMUNITY_SYSTEM_PROMPT = (
    SYSTEM_PROMPT.replace(
        "Ground the recommendation in this user's own history, not in generic "
        "brewing advice.",
        "Ground the recommendation in real brews rather than generic brewing "
        "advice — this user's own first, and everyone else's where their "
        "history runs out. get_community_brews is there for the coffee they "
        "have brewed once and the gear they are new to.",
    )
    + """

Other people's brews are evidence, but only about what their grouping says they \
are. A grind number from a different grinder is not a smaller or larger version \
of this user's number, it is a different scale entirely; ratio and temperature \
carry across gear untouched. Say which brews you leaned on."""
)


class Toolbox:
    """Dispatches the data tools and records what each call returned.

    `as_of` is a required argument of `dispatch`, not an option with a default.
    It is the one thing standing between the agent and the answer it is supposed
    to be predicting, and a default of `None` would make forgetting it silent.
    The caller has the brew in hand — the cutoff is `brew.brew_timestamp` — so
    there is never a reason not to pass it.
    """

    def __init__(self, db: BrewDatabase) -> None:
        self._db = db

    def dispatch(
        self,
        name: str,
        args: dict[str, Any],
        as_of: str,
        viewer_id: str | None = None,
    ) -> tuple[dict[str, Any], dict]:
        """Run one tool call. Returns (payload for the model, trace entry).

        `viewer_id` is whose brew is being diagnosed. Only the community tool
        needs it — to leave that person's own brews out of "everyone else" —
        and it refuses loudly when it is missing rather than quietly returning
        the user their own history relabelled as the community's.
        """
        started = time.monotonic()
        try:
            payload = self._run(name, args, as_of, viewer_id)
        except Exception as exc:  # surfaced to the model, not raised
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        # Read off the payload rather than set in the except branch: a tool that
        # returns an error without raising — an unknown name, a missing viewer —
        # is just as much a failed call, and the trace should say so.
        error = payload.get("error")

        rows = payload.get("brews")
        if rows is None and "groups" in payload:
            rows = [b for group in payload["groups"] for b in group["brews"]]
        trace = {
            "tool": name,
            "arguments": args,
            # Whether the model narrowed to one user. `get_brew` takes no
            # user_id at all, so it always reads false — that is the tool's
            # shape, not a cross-user read.
            "user_id_given": bool(args.get("user_id")),
            "as_of": as_of,
            "row_count": len(rows) if isinstance(rows, list) else (0 if error else 1),
            "latency_ms": round((time.monotonic() - started) * 1000),
            "error": error,
            "returned": payload,
        }
        return payload, trace

    def _run(
        self,
        name: str,
        args: dict[str, Any],
        as_of: str,
        viewer_id: str | None = None,
    ) -> dict[str, Any]:
        if name == "get_community_brews":
            if not viewer_id:
                return {
                    "error": (
                        "get_community_brews needs to know whose brew this is "
                        "so it can exclude them; the harness did not supply it."
                    )
                }
            tiers = self._db.get_community_brews(
                coffee_id=args["coffee_id"],
                grinder_id=args["grinder_id"],
                brewer_id=args["brewer_id"],
                exclude_user=viewer_id,
                min_rating=int(args.get("min_rating") or 2),
                limit=int(args.get("limit") or 20),
                as_of=as_of,
            )
            groups = [
                {
                    "match": tier,
                    "what_transfers": TIER_GUIDANCE[tier],
                    "brews": [b.to_tool_result() for b in tiers[tier]],
                }
                for tier in COMMUNITY_TIERS
                if tiers[tier]
            ]
            return {
                "groups": groups,
                "found": sum(len(group["brews"]) for group in groups),
            }

        if name == "get_brew":
            brew = self._db.get_brew(args["brew_id"], as_of=as_of)
            if brew is None:
                return {"error": f"No visible brew with id {args['brew_id']}."}
            return {"brew": brew.to_tool_result()}

        if name == "get_user_brews_with_bean":
            brews = self._db.get_user_brews_with_bean(
                coffee_id=args["coffee_id"],
                user_id=args.get("user_id"),
                limit=int(args.get("limit") or 20),
                as_of=as_of,
            )
            return {"brews": [b.to_tool_result() for b in brews]}

        if name == "get_user_brews_with_gear":
            brews = self._db.get_user_brews_with_gear(
                grinder_id=args["grinder_id"],
                brewer_id=args["brewer_id"],
                min_rating=int(args["min_rating"]),
                user_id=args.get("user_id"),
                limit=int(args.get("limit") or 20),
                as_of=as_of,
            )
            return {"brews": [b.to_tool_result() for b in brews]}

        return {"error": f"Unknown tool {name}."}

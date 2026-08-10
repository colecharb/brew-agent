"""The three rungs below the agent.

Each adds exactly one capability to the one before it, so the gap between any
two rungs prices that one thing:

| arm        | reads the note | picks the change | reads history |
|------------|----------------|------------------|---------------|
| `rules`    | keyword table  | fixed +/-5%      | no            |
| `classify` | **model**      | fixed +/-5%      | no            |
| `no_tools` | model          | **model**        | no            |
| `agent`    | model          | model            | **3 tools**   |

`rules` -> `classify` is the value of language understanding. `classify` ->
`no_tools` is the value of letting the model choose the size and the lever.
`no_tools` -> `agent` is the value of retrieval.

Keeping `rules` rather than replacing its vocabulary list with the classifier is
deliberate: it is the only rung with no model in it at all, and without it a gap
between keyword matching and a full recommender could not be attributed to
either cause.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from .config import MAX_TOKENS, ModelConfig
from .models import Brew, Recommendation
from .tools import SUBMIT_RECOMMENDATION, SUBMIT_TOOL, SYSTEM_PROMPT


@dataclass
class ArmResult:
    """One arm's answer for one brew, plus everything needed to audit it."""

    recommendation: Recommendation
    trace: dict[str, Any] = field(default_factory=dict)


def describe_brew(brew: Brew) -> str:
    """The brew as prose for a prompt. Mirrors what `get_brew` returns."""
    ratio = f"{brew.ratio:.1f}:1" if brew.ratio else "unknown"
    lines = [
        f"Coffee: {brew.coffee_name or 'unknown'}"
        + (f" ({brew.roaster_name})" if brew.roaster_name else ""),
        f"Grinder: {brew.grinder_name or 'unknown'}",
        f"Brewer: {brew.brewer_name or 'unknown'}",
        f"Grind setting: {brew.grind_setting or 'unknown'}",
        f"Dose: {brew.coffee_weight}g",
        f"Target yield: {brew.target_weight}g (ratio {ratio})",
    ]
    if brew.water_temp is not None:
        lines.append(f"Water temperature: {brew.water_temp}C")
    if brew.time is not None:
        lines.append(f"Brew time: {brew.time}s")
    if brew.days_off_roast is not None:
        lines.append(f"Days off roast: {brew.days_off_roast}")
    if brew.burr_name:
        lines.append(f"Burrs: {brew.burr_name}")
    if brew.filter_name:
        lines.append(f"Filter: {brew.filter_name}")
    if brew.rating is not None:
        lines.append(f"Rating given: {brew.rating}/4 ({brew.rating_label})")
    if brew.recipe.strip():
        lines.append(f"Recipe followed:\n{brew.recipe.strip()}")
    return "\n".join(lines)


def build_prompt(brew: Brew, complaint: str) -> str:
    return (
        f"Brew id {brew.id}\n\n"
        f"{describe_brew(brew)}\n\n"
        f"How it tasted, in the drinker's own words:\n{complaint.strip()}\n\n"
        f"What should change on the next brew?"
    )


# --- arm 1: static rule table ---------------------------------------------

# Classic under- and over-extraction vocabulary. Deliberately shallow: this arm
# exists to be beaten, and its job is to show how much of the score is available
# from keyword matching alone.
UNDER_EXTRACTED = (
    "sour", "sharp", "tart", "thin", "weak", "watery", "hollow", "empty",
    "salty", "grassy", "underdeveloped", "under extracted", "under-extracted",
    "underextracted", "lacking", "flat", "vegetal",
)
OVER_EXTRACTED = (
    "bitter", "harsh", "astringent", "drying", "dry finish", "tannic", "ashy",
    "burnt", "muddy", "over extracted", "over-extracted", "overextracted",
    "hollow bitter", "acrid", "chalky",
)

# Real grind moves in this dataset have a median of ~2% of the current setting
# and a p90 of ~20%. 5% sits inside that, so the baseline is not handicapped on
# magnitude — only on knowing which way to go.
GRIND_STEP = 0.05
TEMP_STEP = 2.0


def _mentions(text: str, terms: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [t for t in terms if re.search(rf"\b{re.escape(t)}", lowered)]


def run_rules(brew: Brew, complaint: str) -> ArmResult:
    """Keyword table, no model, no data.

    One assumption is baked in and worth naming: that a higher grind number
    means coarser. That holds for the micron readouts that dominate this
    dataset and for most grinder dials, but it is exactly the kind of guess the
    agent is supposed to avoid by reading the user's own history instead. Where
    the assumption is wrong, this arm will be confidently backwards — which is
    the point of having it.
    """
    started = time.monotonic()
    under = _mentions(complaint, UNDER_EXTRACTED)
    over = _mentions(complaint, OVER_EXTRACTED)

    rec = Recommendation(reasoning="No extraction keywords matched.")
    verdict = "none"

    # Both vocabularies present is genuinely ambiguous; a rule table has nothing
    # useful to say, so it says nothing rather than guessing.
    if under and not over:
        verdict = "under-extracted"
        rec = _apply_step(brew, finer=True, matched=under, verdict=verdict)
    elif over and not under:
        verdict = "over-extracted"
        rec = _apply_step(brew, finer=False, matched=over, verdict=verdict)
    elif under and over:
        verdict = "mixed"
        rec.reasoning = (
            f"Both under-extraction ({', '.join(under)}) and over-extraction "
            f"({', '.join(over)}) words present; no single-lever rule applies."
        )

    return ArmResult(
        recommendation=rec,
        trace={
            "arm": "rules",
            "verdict": verdict,
            "matched_under": under,
            "matched_over": over,
            "latency_ms": round((time.monotonic() - started) * 1000),
        },
    )


def _apply_step(
    brew: Brew, finer: bool, matched: list[str], verdict: str
) -> Recommendation:
    current = brew.grind_value
    rec = Recommendation(primary_lever="grind_setting")

    if current is not None:
        # Assumes higher == coarser; see the docstring.
        direction = -1 if finer else 1
        target = current * (1 + direction * GRIND_STEP)
        decimals = len(brew.grind_setting.split(".")[1]) if "." in (brew.grind_setting or "") else 0
        rec.grind_setting = f"{round(target, decimals):.{decimals}f}"
    else:
        rec.primary_lever = "none"

    if brew.water_temp is not None:
        rec.water_temp = brew.water_temp + (TEMP_STEP if finer else -TEMP_STEP)

    rec.reasoning = (
        f"Matched {verdict} keywords ({', '.join(matched)}). Static rule: go "
        f"{'finer' if finer else 'coarser'} by {GRIND_STEP:.0%} and "
        f"{'raise' if finer else 'lower'} the temperature."
    )
    return rec


# --- arm 2: model reads the note, the rule table does the arithmetic -------

UNDER, OVER, BOTH, NEITHER = (
    "under_extracted",
    "over_extracted",
    "both",
    "neither",
)

CLASSIFY_TOOL = "classify_taste"

CLASSIFY_TASTE: dict[str, Any] = {
    "name": CLASSIFY_TOOL,
    "description": "Record what the tasting note says about extraction.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [UNDER, OVER, BOTH, NEITHER],
                "description": (
                    "under_extracted when the cup was sour, sharp, thin, weak, "
                    "hollow, salty or grassy, was short on sweetness or body, "
                    "or ran fast. over_extracted when it was bitter, harsh, "
                    "astringent, drying, ashy or muddy, or ran slow. both when "
                    "the note genuinely describes each. neither only when the "
                    "note gives no evidence either way."
                ),
            },
            "evidence": {
                "type": "string",
                "description": (
                    "The words from the note that decided it, copied verbatim "
                    "rather than paraphrased. Empty for neither."
                ),
            },
        },
        "required": ["verdict", "evidence"],
        "additionalProperties": False,
    },
}

CLASSIFY_SYSTEM = """You read a coffee tasting note and say whether the cup was \
under-extracted or over-extracted.

Under-extraction means not enough came out of the grounds. The cup tastes sour, \
sharp, thin, weak, hollow, salty or grassy, and is short on sweetness and body. \
It shows up just as often in how the brew ran: water through too fast, a shot \
finishing early, a drawdown over sooner than it should have been.

Over-extraction means too much came out. The cup tastes bitter, harsh, \
astringent, drying, ashy or muddy, and can feel heavy and scoured at once. It \
shows up in a brew that ran slow: a choked or stalled shot, a drawdown that \
dragged.

Notes seldom use any of those words. Read for what the drinker meant, not for \
vocabulary. "Wants more sweetness", "a bit sharp", "didn't really sing" and \
"watery" are all describing the same under-extracted cup; "harsh finish", \
"drying", "a bit much" and "grippy" are all describing the same over-extracted \
one. How the brew ran is evidence exactly as much as how it tasted.

Reserve neither for notes that give you nothing to work with: purely positive, \
contentless ("Yes."), or about the occasion rather than the cup ("for the \
morning latte"). Indirect or partial evidence is still evidence — call what it \
points to. Reserve both for notes that genuinely describe each, not for ones you \
find hard to call.

Call classify_taste. Do not answer in prose."""


class ClassifyBaseline:
    """A model reads the note; the rule table still does the arithmetic.

    The prompt carries the note and **nothing else** — no grind setting, no
    dose, no gear. That is what keeps this a classifier rather than a second
    recommender, and it is why the arm cannot quietly turn into `no_tools`.

    The verdict then feeds the same `_apply_step` the keyword table uses, with
    the same 5% step, the same 2C, and the same "higher number is coarser"
    assumption. Every downstream variable is held constant, so the gap against
    `rules` is language understanding and nothing else.

    On abstention. `both` and `neither` both recommend nothing, and scoring
    counts silence as a miss, so between them they are the arm's only way to
    lose without being wrong. The first prompt here spent two sentences
    encouraging exactly that — "judge only what the note says about flavour",
    "saying so is the right answer rather than a failure" — and the arm duly
    returned `neither` on a note opening *"Shot pulled way too fast"*, which is
    an unambiguous under-extraction call that happens not to be a flavour word.
    The rung was measuring the brief, not the reader.

    So the brief now admits evidence about how the brew ran, says the textbook
    vocabulary will usually be absent, and scopes both escape hatches narrowly.
    What it deliberately does not do is mention the scoring: told that silence
    is penalised, the model would guess to protect a number rather than read a
    note, and the arm would stop measuring anything. The prompt describes the
    task; the metric stays the metric.
    """

    def __init__(self, client: anthropic.Anthropic, config: ModelConfig) -> None:
        self._client = client
        self._config = config

    def run(self, brew: Brew, complaint: str) -> ArmResult:
        started = time.monotonic()
        try:
            response = call_model(
                self._client,
                self._config,
                system=CLASSIFY_SYSTEM,
                messages=[{"role": "user", "content": complaint.strip()}],
                tools=[CLASSIFY_TASTE],
                force_tool=CLASSIFY_TOOL,
            )
        except Exception as exc:
            return ArmResult(
                recommendation=Recommendation(error=f"{type(exc).__name__}: {exc}"),
                trace={"arm": "classify", "error": str(exc)},
            )

        verdict, evidence = _read_verdict(response)
        if verdict == UNDER:
            rec = _apply_step(brew, finer=True, matched=[evidence], verdict=verdict)
        elif verdict == OVER:
            rec = _apply_step(brew, finer=False, matched=[evidence], verdict=verdict)
        else:
            # `both` and `neither` recommend nothing, exactly as the keyword
            # table does when its two lists collide.
            rec = Recommendation(
                reasoning=f"Classified {verdict}; no single-lever rule applies."
            )
            if verdict is None:
                rec.error = f"no {CLASSIFY_TOOL} call (stop_reason={response.stop_reason})"

        return ArmResult(
            recommendation=rec,
            trace={
                "arm": "classify",
                "model": self._config.model,
                "verdict": verdict,
                "evidence": evidence,
                # False means the model returned something that is not in the
                # note. Scores are unaffected; the trace's account of why is.
                "evidence_verbatim": evidence_is_verbatim(evidence, complaint),
                "stop_reason": response.stop_reason,
                "usage": usage_of(response),
                "latency_ms": round((time.monotonic() - started) * 1000),
                "recommendation": rec.to_dict(),
            },
        )


def _read_verdict(response: Any) -> tuple[str | None, str]:
    if response.stop_reason == "refusal":
        return None, ""
    for block in response.content:
        if block.type == "tool_use" and block.name == CLASSIFY_TOOL:
            return block.input.get("verdict"), block.input.get("evidence") or ""
    return None, ""


def evidence_is_verbatim(evidence: str, note: str) -> bool:
    """Is the quoted evidence actually a span of the note?

    One classify call came back with `"</antml\\u0903parameter>"` in this field:
    a malformed token fragment where a quote should have been. It changes no
    score — only `verdict` reaches `_apply_step` — but it means the trace
    misrepresents the reasoning, and a trace that cannot be trusted about the
    small things is not evidence about the large ones.

    Whitespace is normalised before comparing, since a model reflowing a quote
    across lines is a formatting difference rather than a fabrication. Anything
    else that is not in the note is reported as not verbatim, whether it is a
    paraphrase or a fragment of nothing at all.
    """
    needle = " ".join(evidence.split()).lower()
    if not needle:
        # Legitimately empty for `neither`, and nothing to check either way.
        return True
    return needle in " ".join(note.split()).lower()


# --- arm 3: one model call, no data ---------------------------------------

NO_TOOLS_SYSTEM = (
    SYSTEM_PROMPT.split("Ground the recommendation")[0].strip()
    + "\n\nYou have no access to this user's history. Work from the brew's "
    "parameters and the complaint alone.\n\nGrind settings are free text and "
    "every grinder has its own scale — some count up as they get finer, some "
    "count down, some read in microns. Express your answer as a number on the "
    "same dial as the brew you are given.\n\nCall submit_recommendation. Do not "
    "answer in prose."
)


class NoToolsBaseline:
    """A single model call with the brew and the complaint, and nothing else."""

    def __init__(self, client: anthropic.Anthropic, config: ModelConfig) -> None:
        self._client = client
        self._config = config

    def run(self, brew: Brew, complaint: str) -> ArmResult:
        started = time.monotonic()
        messages = [{"role": "user", "content": build_prompt(brew, complaint)}]
        try:
            response = call_model(
                self._client,
                self._config,
                system=NO_TOOLS_SYSTEM,
                messages=messages,
                tools=[SUBMIT_RECOMMENDATION],
                force_tool=SUBMIT_TOOL,
            )
        except Exception as exc:
            return ArmResult(
                recommendation=Recommendation(error=f"{type(exc).__name__}: {exc}"),
                trace={"arm": "no_tools", "error": str(exc)},
            )

        rec = extract_recommendation(response)
        return ArmResult(
            recommendation=rec,
            trace={
                "arm": "no_tools",
                "model": self._config.model,
                "stop_reason": response.stop_reason,
                "usage": usage_of(response),
                "latency_ms": round((time.monotonic() - started) * 1000),
                "recommendation": rec.to_dict(),
            },
        )


# --- shared model plumbing -------------------------------------------------


def call_model(
    client: anthropic.Anthropic,
    config: ModelConfig,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    force_tool: str | None = None,
) -> Any:
    """One Messages call.

    Claude Opus 5 rejects `temperature`/`top_p`/`top_k`, so behaviour is steered
    by prompt and `effort` only. Thinking is on by default and `max_tokens` caps
    thinking plus response together, hence the generous ceiling in config.
    """
    kwargs: dict[str, Any] = {
        "model": config.model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": messages,
        "tools": tools,
    }
    if config.effort:
        kwargs["output_config"] = {"effort": config.effort}
    if force_tool:
        kwargs["tool_choice"] = {"type": "tool", "name": force_tool}

    try:
        return client.messages.create(**kwargs)
    except anthropic.BadRequestError as exc:
        # Forcing a specific tool is not accepted in every model/thinking
        # combination. Losing the forcing is survivable — the prompt already
        # asks for the tool — so retry once rather than failing the call.
        if force_tool and "tool_choice" in str(exc).lower():
            kwargs.pop("tool_choice")
            return client.messages.create(**kwargs)
        raise


def extract_recommendation(response: Any) -> Recommendation:
    """Pull the submit_recommendation call out of a response."""
    if response.stop_reason == "refusal":
        return Recommendation(error="model declined the request")
    for block in response.content:
        if block.type == "tool_use" and block.name == SUBMIT_TOOL:
            return Recommendation.from_tool_input(block.input)
    return Recommendation(
        error=f"no {SUBMIT_TOOL} call (stop_reason={response.stop_reason})"
    )


def usage_of(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
    }

"""One shim, two vendors.

Every arm talks to a `Provider` rather than to an SDK, so the model under test
is an environment variable instead of a rewrite. `BREW_AGENT_MODEL` picks it;
the provider follows from the name unless `BREW_AGENT_PROVIDER` says otherwise.

Three things differ between vendors, and each of them is a place where a naive
port is wrong rather than merely ugly:

**The response.** Normalised into `ModelResponse` carrying the same
`type`/`text`/`name`/`input` blocks the arms already read, so `agent.py` and
`baselines.py` are unchanged in how they inspect an answer. Cohere returns tool
arguments as a JSON *string*; parsing it here is the difference between a
`Recommendation` and five nulls that score as an abstention.

**The transcript.** The loop keeps a neutral one and each provider renders it
natively, because the two disagree about who says what. Anthropic wants its own
assistant blocks handed back verbatim — thinking blocks carry signatures that
must survive the round trip, so this echoes the raw content rather than
rebuilding it from the normalised form. Cohere wants the pre-tool reasoning in
`tool_plan` and every tool result as its own `role: "tool"` message.

**The schema.** Cohere's structured-output subset has no `anyOf`, which is
exactly what `tools.nullable()` emits. Those fields are unwrapped to their real
type and dropped from `required`, so "leave this alone" is expressed by absence
rather than by null. Same meaning downstream: `Recommendation.from_tool_input`
reads every field with `.get`, so a missing key and a null one are already the
same answer.

One asymmetry is worth naming because it is silent rather than loud. Anthropic
can force a *named* tool; Cohere can only say `REQUIRED`, meaning "some tool".
Every forced call site here offers exactly one tool, so the two are equivalent —
and `_check_forcing` says so out loud the moment that stops being true, rather
than letting a forced `submit_recommendation` quietly become a forced anything.
"""

from __future__ import annotations

import json
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .config import ModelConfig

_warned: set[str] = set()
_warn_lock = threading.Lock()


def warn_once(message: str) -> None:
    """Print a warning the first time it happens, not once per call.

    Pairs run concurrently and a model-capability mismatch affects every call
    equally, so without this the same line would land a few hundred times and
    bury the results table.
    """
    with _warn_lock:
        if message in _warned:
            return
        _warned.add(message)
    print(f"warning: {message}", file=sys.stderr, flush=True)


# --- the normalised response ----------------------------------------------


@dataclass
class TextBlock:
    """Assistant prose. Shaped like Anthropic's because that is what the arms
    already read, and traces record it as `assistant_text`."""

    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    """One tool call. `input` is always a dict here, whatever the wire said."""

    name: str
    input: dict[str, Any]
    id: str = ""
    type: str = "tool_use"


@dataclass
class ModelResponse:
    """What every arm sees, whichever vendor answered.

    `raw` is kept so a provider can hand a vendor back its own assistant turn
    byte for byte instead of a reconstruction — see the module docstring.
    """

    stop_reason: str
    content: list[Any] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if b.type == "text").strip()

    @property
    def tool_calls(self) -> list[ToolUseBlock]:
        return [b for b in self.content if b.type == "tool_use"]


# --- the neutral transcript ------------------------------------------------
#
# A plain-text user turn is spelled the same on both wires, so it is stored in
# its native form and passed straight through. The other two are not, and are
# stored as intent — "the assistant said this", "the tools returned that" — for
# the provider to render.


def user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def assistant_message(response: ModelResponse) -> dict[str, Any]:
    return {"role": "assistant", "response": response}


def tool_results_message(results: list[dict[str, Any]]) -> dict[str, Any]:
    """`results` are `{id, content, is_error}` — content already serialised."""
    return {"role": "tool_results", "results": results}


class Provider(ABC):
    """A model, reachable. Subclasses own translation in both directions."""

    name: str = ""

    def __init__(self, client: Any) -> None:
        # Public so tests can inject a scripted client and then assert on the
        # native payload that actually went over the wire. Translation is the
        # risky half of this file; asserting on the neutral form would test
        # nothing.
        self.client = client

    @abstractmethod
    def complete(
        self,
        config: ModelConfig,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
    ) -> ModelResponse:
        """One call. `messages` is the neutral transcript."""

    @staticmethod
    def _check_forcing(force_tool: str | None, tools: list[dict[str, Any]]) -> None:
        """Only meaningful where a provider cannot force a tool *by name*."""
        if force_tool and len(tools) > 1:
            warn_once(
                f"forcing {force_tool} with {len(tools)} tools offered: this "
                f"provider can only require *some* tool, so the model may "
                f"answer with a different one."
            )


# --- Anthropic -------------------------------------------------------------


class AnthropicProvider(Provider):
    """Messages API.

    Claude Opus 5 rejects `temperature`/`top_p`/`top_k`, so behaviour is steered
    by prompt and `effort` only. Thinking is on by default and `max_tokens` caps
    thinking plus response together, hence the generous ceiling in config.
    """

    name = "anthropic"

    @classmethod
    def connect(cls, config: ModelConfig) -> "AnthropicProvider":
        import anthropic

        return cls(anthropic.Anthropic(api_key=config.api_key))

    def complete(
        self,
        config: ModelConfig,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
    ) -> ModelResponse:
        import anthropic

        kwargs: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "system": system,
            "messages": self._messages(messages),
            "tools": tools,
        }
        if config.effort:
            kwargs["output_config"] = {"effort": config.effort}
        if force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}

        try:
            return self._normalise(self.client.messages.create(**kwargs))
        except anthropic.BadRequestError as exc:
            detail = str(exc).lower()
            # Forcing a specific tool is not accepted in every model/thinking
            # combination. Losing the forcing is survivable — the prompt already
            # asks for the tool — so retry once rather than failing the call.
            if force_tool and "tool_choice" in detail:
                kwargs.pop("tool_choice")
                return self._normalise(self.client.messages.create(**kwargs))
            # Smaller models have no effort parameter at all, and rejecting it
            # would otherwise fail every pair identically. Dropping it is the
            # only way to run them, so run — but say so, because a run without
            # effort is not spending the same test-time compute as one with it.
            if "effort" in detail and "output_config" in kwargs:
                warn_once(
                    f"{config.model} rejected output_config.effort; continuing "
                    f"without it. This run is not comparable to one that set it."
                )
                kwargs.pop("output_config")
                return self._normalise(self.client.messages.create(**kwargs))
            raise

    @staticmethod
    def _messages(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
        native: list[dict[str, Any]] = []
        for entry in transcript:
            if entry["role"] == "assistant":
                # The vendor's own blocks, not a rebuild: extended thinking
                # blocks carry signatures, and a reconstructed turn without them
                # is rejected on the next call.
                native.append(
                    {"role": "assistant", "content": entry["response"].raw.content}
                )
            elif entry["role"] == "tool_results":
                native.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r["id"],
                                "content": r["content"],
                                "is_error": r["is_error"],
                            }
                            for r in entry["results"]
                        ],
                    }
                )
            else:
                native.append(entry)
        return native

    @staticmethod
    def _normalise(response: Any) -> ModelResponse:
        content: list[Any] = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(
                    ToolUseBlock(id=block.id, name=block.name, input=dict(block.input))
                )
        usage = getattr(response, "usage", None)
        return ModelResponse(
            stop_reason=response.stop_reason,
            content=content,
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            }
            if usage is not None
            else {},
            raw=response,
        )


# --- Cohere ----------------------------------------------------------------


# Cohere's finish reasons, mapped onto the names the arms already branch on.
# The v2 set is COMPLETE / STOP_SEQUENCE / MAX_TOKENS / TOOL_CALL / ERROR /
# TIMEOUT, with no equivalent of Anthropic's `refusal`. A Cohere model that
# declines therefore arrives as an ordinary turn carrying no tool call, and is
# recorded as `no submit_recommendation call (stop_reason=end_turn)` rather
# than as a refusal. That is the honest description of what came back, and the
# pair is scored the same either way.
COHERE_STOP_REASONS = {
    "COMPLETE": "end_turn",
    "TOOL_CALL": "tool_use",
    "MAX_TOKENS": "max_tokens",
    "STOP_SEQUENCE": "stop_sequence",
}

# Parameters this shim sends that an older model or SDK may reject outright.
# Each is dropped and retried once, because losing one degrades a run while
# failing the call loses it entirely.
COHERE_OPTIONAL_PARAMS = ("strict_tools", "tool_choice")

# What the run loses by dropping each of them, so a warning says what changed
# rather than only that something did.
COHERE_DEGRADED = {
    "strict_tools": "Answers are no longer guaranteed to match the schema.",
    "tool_choice": "The model is free to answer in prose instead.",
}


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field off either an SDK model or a plain dict.

    The Cohere SDK returns pydantic objects, but the scripted clients in the
    tests return dicts and older SDK versions have moved fields between the
    two. A shim that only understands one of them fails on the other in a way
    that looks like a model problem.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class CohereProvider(Provider):
    """Chat API v2.

    `strict_tools` is sent on every call rather than per tool, which is where
    Cohere puts it. It is what replaces the schema-retry the harness leans on
    elsewhere: a malformed answer never comes back to be re-asked, because the
    grammar will not emit one. Every tool here has at least one required
    parameter, which that mode demands.
    """

    name = "cohere"

    @classmethod
    def connect(cls, config: ModelConfig) -> "CohereProvider":
        import cohere

        return cls(cohere.ClientV2(api_key=config.api_key))

    def complete(
        self,
        config: ModelConfig,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
    ) -> ModelResponse:
        self._check_forcing(force_tool, tools)
        if config.effort:
            warn_once(
                f"effort is an Anthropic parameter and {config.model} is not "
                f"run with it. Cohere's reasoning models take a thinking token "
                f"budget instead; nothing here sets one."
            )

        kwargs: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": self._messages(system, messages),
            "tools": [self._tool(t) for t in tools],
            "strict_tools": True,
        }
        if force_tool:
            # Cohere requires *a* tool, never a named one. Equivalent here
            # because every forced call site offers exactly one.
            kwargs["tool_choice"] = "REQUIRED"

        return self._normalise(self._call(kwargs, config))

    def _call(self, kwargs: dict[str, Any], config: ModelConfig) -> Any:
        try:
            return self.client.chat(**kwargs)
        except Exception as exc:
            # Deliberately not catching a specific class: the SDK's exception
            # taxonomy has moved between versions, and this only fires when the
            # error names a parameter we sent, which no unrelated failure does.
            detail = str(exc).lower()
            for param in COHERE_OPTIONAL_PARAMS:
                if param in detail and param in kwargs:
                    warn_once(
                        f"{config.model} rejected {param}; continuing without "
                        f"it. {COHERE_DEGRADED[param]}"
                    )
                    retry = {k: v for k, v in kwargs.items() if k != param}
                    return self.client.chat(**retry)
            raise

    @staticmethod
    def _messages(system: str, transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Render the neutral transcript. The system prompt is a message here."""
        native: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for entry in transcript:
            if entry["role"] == "assistant":
                native.append(CohereProvider._assistant(entry["response"]))
            elif entry["role"] == "tool_results":
                # One message per result, and no `is_error` flag to set: a
                # failed call has to carry its own error in the payload. It
                # does — `Toolbox.dispatch` returns `{"error": ...}` as the
                # result body — so the model still reads what went wrong.
                native.extend(
                    {
                        "role": "tool",
                        "tool_call_id": r["id"],
                        "content": r["content"],
                    }
                    for r in entry["results"]
                )
            else:
                native.append(entry)
        return native

    @staticmethod
    def _assistant(response: ModelResponse) -> dict[str, Any]:
        calls = response.tool_calls
        if not calls:
            return {"role": "assistant", "content": response.text}
        return {
            "role": "assistant",
            # Cohere carries the reasoning that preceded a tool call here
            # rather than in `content`, and rejects an empty one.
            "tool_plan": response.text or "Continuing the investigation.",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.input),
                    },
                }
                for call in calls
            ],
        }

    @staticmethod
    def _tool(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": cohere_schema(tool["input_schema"]),
            },
        }

    @staticmethod
    def _normalise(response: Any) -> ModelResponse:
        message = _attr(response, "message")
        content: list[Any] = []

        # `tool_plan` is where the pre-tool reasoning lands; `content` holds the
        # prose of a plain answer. Traces record whichever arrived as the
        # assistant text for that step.
        text = "".join(
            _attr(block, "text", "") or ""
            for block in (_attr(message, "content") or [])
            if _attr(block, "type", "text") == "text"
        )
        plan = _attr(message, "tool_plan") or ""
        if text or plan:
            content.append(TextBlock(text=text or plan))

        for call in _attr(message, "tool_calls") or []:
            function = _attr(call, "function")
            name = _attr(function, "name") or ""
            raw_args = _attr(function, "arguments")
            content.append(
                ToolUseBlock(
                    id=_attr(call, "id") or "",
                    name=name,
                    input=_parse_arguments(raw_args, name),
                )
            )

        finish = _attr(response, "finish_reason") or ""
        return ModelResponse(
            stop_reason=COHERE_STOP_REASONS.get(finish, str(finish).lower()),
            content=content,
            usage=_cohere_usage(response),
            raw=response,
        )


def _parse_arguments(raw: Any, tool: str) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string. Refuse to guess at a bad one.

    An unparseable payload could be swallowed into `{}`, and for
    `submit_recommendation` that produces five nulls — which scores as an
    abstention and is indistinguishable from the model having no opinion. A
    silent zero is the one outcome this harness cannot afford, so this raises
    and the arm records the failure instead.
    """
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{tool} returned unparseable arguments: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{tool} returned non-object arguments: {raw!r}")
    return parsed


def _cohere_usage(response: Any) -> dict[str, int]:
    """Billed units where they exist, raw token counts otherwise.

    They differ, and the one worth recording is what the run actually cost.
    """
    usage = _attr(response, "usage")
    counts = _attr(usage, "billed_units") or _attr(usage, "tokens")
    if counts is None:
        return {}
    return {
        "input_tokens": int(_attr(counts, "input_tokens", 0) or 0),
        "output_tokens": int(_attr(counts, "output_tokens", 0) or 0),
    }


# Keywords Cohere's structured-output subset does not accept. `anyOf` is the
# one that matters: `tools.nullable()` is built out of it.
UNSUPPORTED_KEYWORDS = ("anyOf", "allOf", "oneOf", "additionalProperties")


def cohere_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a tool schema into the subset Cohere's strict mode accepts.

    A `nullable()` property — `anyOf: [{type: T}, {type: null}]` — becomes a
    plain `T` that is no longer required. The meaning survives intact: the
    Anthropic schema says "answer null to leave this alone", this one says
    "leave it out". Both arrive at `Recommendation.from_tool_input` as a
    missing value, because it reads every field with `.get`.

    Dropping them from `required` is not optional housekeeping. Under
    `strict_tools` a required property must be present, so leaving them in
    would force the model to invent a dose and a temperature for every brew —
    turning "change one thing" into "change everything", which no scoring
    column could tell from a real recommendation.
    """
    properties = schema.get("properties") or {}
    required = list(schema.get("required") or [])

    rewritten: dict[str, Any] = {}
    optional: list[str] = []
    for name, spec in properties.items():
        unwrapped = _unwrap_nullable(spec)
        if unwrapped is not spec:
            optional.append(name)
        rewritten[name] = unwrapped

    cleaned = {
        key: value
        for key, value in schema.items()
        if key not in UNSUPPORTED_KEYWORDS and key not in ("properties", "required")
    }
    return {
        **cleaned,
        "properties": rewritten,
        "required": [name for name in required if name not in optional],
    }


def _unwrap_nullable(spec: dict[str, Any]) -> dict[str, Any]:
    """`{anyOf: [{type: T}, {type: null}], description}` -> `{type: T, ...}`."""
    options = spec.get("anyOf")
    if not options:
        return spec
    concrete = [o for o in options if o.get("type") != "null"]
    if len(concrete) != 1:
        # Not the nullable() shape. Left alone rather than guessed at — it will
        # fail loudly at the API instead of silently meaning something else.
        return spec
    rest = {k: v for k, v in spec.items() if k != "anyOf"}
    return {**concrete[0], **rest}


PROVIDERS = {
    AnthropicProvider.name: AnthropicProvider,
    CohereProvider.name: CohereProvider,
}


def connect(config: ModelConfig) -> Provider:
    """The provider named by the config, with a live client."""
    return PROVIDERS[config.provider].connect(config)

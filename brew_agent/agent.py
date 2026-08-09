"""The tool-calling loop.

Standard shape: call the model, run whatever tools it asked for, hand the
results back, repeat. Two things worth pointing at:

- The loop ends when the model calls `submit_recommendation`, not on
  `end_turn`. A model that stops talking without answering is a failure, and
  this makes that visible instead of returning an empty recommendation.
- There is a hard iteration cap. On hitting it the loop makes one more call with
  `tool_choice` pinned to `submit_recommendation`, so a run that explores
  forever still produces a scoreable answer — flagged `hit_cap` in the trace so
  those pairs can be separated out later.

Every call is traced in full: tools called, in order, with their arguments and
the actual rows returned.
"""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic

from .baselines import (
    ArmResult,
    build_prompt,
    call_model,
    extract_recommendation,
    usage_of,
)
from .config import ModelConfig
from .models import Brew, Recommendation
from .tools import ALL_TOOLS, SUBMIT_RECOMMENDATION, SUBMIT_TOOL, SYSTEM_PROMPT, Toolbox


class BrewAgent:
    """Diagnoses a brew by reading the user's history through the data tools."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        config: ModelConfig,
        toolbox: Toolbox,
    ) -> None:
        self._client = client
        self._config = config
        self._toolbox = toolbox

    def run(self, brew: Brew, complaint: str) -> ArmResult:
        started = time.monotonic()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self._opening(brew, complaint)}
        ]
        trace: dict[str, Any] = {
            "arm": "agent",
            "model": self._config.model,
            "effort": self._config.effort,
            "brew_id": brew.id,
            "complaint": complaint,
            "iterations": [],
            "hit_cap": False,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

        recommendation: Recommendation | None = None
        for iteration in range(1, self._config.max_iterations + 1):
            try:
                response = call_model(
                    self._client,
                    self._config,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=ALL_TOOLS,
                )
            except Exception as exc:
                recommendation = Recommendation(error=f"{type(exc).__name__}: {exc}")
                trace["error"] = str(exc)
                break

            self._accumulate(trace, response)
            step: dict[str, Any] = {
                "n": iteration,
                "stop_reason": response.stop_reason,
                "assistant_text": _text_of(response),
                "tool_calls": [],
            }
            trace["iterations"].append(step)

            if response.stop_reason == "refusal":
                recommendation = Recommendation(error="model declined the request")
                break

            calls = [b for b in response.content if b.type == "tool_use"]
            submitted = next((c for c in calls if c.name == SUBMIT_TOOL), None)
            if submitted:
                step["tool_calls"].append(
                    {"tool": SUBMIT_TOOL, "arguments": submitted.input}
                )
                recommendation = Recommendation.from_tool_input(submitted.input)
                break

            if not calls:
                # Answered in prose despite being told not to. Nudge once rather
                # than throwing the pair away.
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Call {SUBMIT_TOOL} with your answer. Do not reply "
                            f"in prose."
                        ),
                    }
                )
                continue

            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {"role": "user", "content": self._run_tools(calls, step)}
            )

        if recommendation is None:
            trace["hit_cap"] = True
            recommendation = self._force_answer(messages, trace)

        trace["latency_ms"] = round((time.monotonic() - started) * 1000)
        trace["iterations_used"] = len(trace["iterations"])
        trace["recommendation"] = recommendation.to_dict()
        return ArmResult(recommendation=recommendation, trace=trace)

    # --- internals ---------------------------------------------------------

    @staticmethod
    def _opening(brew: Brew, complaint: str) -> str:
        # Deliberately not handed the brew's parameters — the agent has
        # get_brew and should use it. The no_tools baseline is the arm that
        # gets them for free.
        return (
            f"Diagnose brew {brew.id}.\n\n"
            f"How it tasted, in the drinker's own words:\n{complaint.strip()}\n\n"
            f"What should change on the next brew?"
        )

    def _run_tools(self, calls: list[Any], step: dict[str, Any]) -> list[dict]:
        """Execute every tool the model asked for and build the result blocks."""
        blocks = []
        for call in calls:
            payload, tool_trace = self._toolbox.dispatch(call.name, dict(call.input))
            step["tool_calls"].append(tool_trace)
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(payload, default=str),
                    "is_error": bool(tool_trace["error"]),
                }
            )
        return blocks

    def _force_answer(
        self, messages: list[dict[str, Any]], trace: dict[str, Any]
    ) -> Recommendation:
        """Out of iterations: demand the answer with the tools taken away."""
        messages.append(
            {
                "role": "user",
                "content": (
                    "You are out of investigation steps. Answer now with "
                    f"{SUBMIT_TOOL}, using what you already have."
                ),
            }
        )
        try:
            response = call_model(
                self._client,
                self._config,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=[SUBMIT_RECOMMENDATION],
                force_submit=True,
            )
        except Exception as exc:
            return Recommendation(error=f"{type(exc).__name__}: {exc}")

        self._accumulate(trace, response)
        trace["iterations"].append(
            {
                "n": len(trace["iterations"]) + 1,
                "stop_reason": response.stop_reason,
                "assistant_text": _text_of(response),
                "forced_submit": True,
                "tool_calls": [
                    {"tool": SUBMIT_TOOL, "arguments": block.input}
                    for block in response.content
                    if block.type == "tool_use"
                ],
            }
        )
        return extract_recommendation(response)

    @staticmethod
    def _accumulate(trace: dict[str, Any], response: Any) -> None:
        usage = usage_of(response)
        for key, value in usage.items():
            trace["usage"][key] = trace["usage"].get(key, 0) + value


def _text_of(response: Any) -> str:
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

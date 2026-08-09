"""Agent loop mechanics, driven by a scripted client. No network."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from brew_agent.agent import BrewAgent
from brew_agent.config import ModelConfig
from brew_agent.models import Brew
from brew_agent.tools import SUBMIT_TOOL, Toolbox


# --- scripted stand-ins ----------------------------------------------------


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    name: str
    input: dict
    id: str = "toolu_1"
    type: str = "tool_use"


@dataclass
class Usage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class FakeResponse:
    content: list
    stop_reason: str = "tool_use"
    usage: Usage = field(default_factory=Usage)


class FakeMessages:
    def __init__(self, script: list[FakeResponse]) -> None:
        self._script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("agent made more calls than the script allows")
        return self._script.pop(0)


class FakeClient:
    def __init__(self, script: list[FakeResponse]) -> None:
        self.messages = FakeMessages(script)


def a_brew(bid="brew-1", grind="500"):
    return Brew(
        id=bid,
        created_by="user-1",
        brew_timestamp="2026-01-01T00:00:00+00:00",
        profile_coffee_id="bag-1",
        coffee_id="coffee-1",
        coffee_name="Gesha",
        grinder_id="g1",
        brewer_id="b1",
        grinder_name="Z1",
        brewer_name="V60",
        grind_setting=grind,
        coffee_weight=15.0,
        target_weight=250.0,
        rating=1,
        notes="sour and thin",
    )


class FakeDatabase:
    """Stands in for BrewDatabase. `explode` makes a tool raise."""

    def __init__(self, explode: bool = False) -> None:
        self.explode = explode
        self.calls: list[tuple] = []

    def get_brew(self, brew_id):
        self.calls.append(("get_brew", brew_id))
        if self.explode:
            raise RuntimeError("connection reset")
        return a_brew(brew_id)

    def get_user_brews_with_bean(self, coffee_id, user_id=None, limit=20):
        self.calls.append(("bean", coffee_id, user_id))
        return [a_brew("brew-2", "495"), a_brew("brew-3", "505")]

    def get_user_brews_with_gear(
        self, grinder_id, brewer_id, min_rating, user_id=None, limit=20
    ):
        self.calls.append(("gear", grinder_id, brewer_id, min_rating, user_id))
        return [a_brew("brew-4", "490")]


SUBMIT_INPUT = {
    "grind_setting": "485",
    "coffee_weight": None,
    "target_weight": None,
    "water_temp": None,
    "time": None,
    "primary_lever": "grind_setting",
    "reasoning": "Their good brews on this setup sit at 485-495.",
}


def make_agent(script, explode=False, max_iterations=6):
    db = FakeDatabase(explode=explode)
    config = ModelConfig(
        api_key="test",
        model="claude-opus-5",
        effort=None,
        max_iterations=max_iterations,
    )
    agent = BrewAgent(FakeClient(script), config, Toolbox(db))
    return agent, db


# --- tests -----------------------------------------------------------------


def test_investigates_then_submits():
    script = [
        FakeResponse([ToolUseBlock("get_brew", {"brew_id": "brew-1"})]),
        FakeResponse(
            [ToolUseBlock("get_user_brews_with_gear", {
                "grinder_id": "g1", "brewer_id": "b1", "min_rating": 3,
                "user_id": "user-1",
            })]
        ),
        FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)], stop_reason="tool_use"),
    ]
    agent, db = make_agent(script)
    result = agent.run(a_brew(), "sour and thin")

    assert result.recommendation.grind_setting == "485"
    assert result.recommendation.primary_lever == "grind_setting"
    assert result.recommendation.error is None
    assert result.trace["hit_cap"] is False
    assert result.trace["iterations_used"] == 3
    assert [c[0] for c in db.calls] == ["get_brew", "gear"]


def test_trace_records_arguments_scoping_and_returned_rows():
    script = [
        FakeResponse([ToolUseBlock("get_user_brews_with_bean", {
            "coffee_id": "coffee-1", "user_id": "user-1",
        })]),
        FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)]),
    ]
    agent, _ = make_agent(script)
    trace = agent.run(a_brew(), "sour").trace

    call = trace["iterations"][0]["tool_calls"][0]
    assert call["tool"] == "get_user_brews_with_bean"
    assert call["arguments"]["coffee_id"] == "coffee-1"
    assert call["scoped_to_user"] is True
    assert call["row_count"] == 2
    # The actual rows, not just a count — a trace has to be inspectable.
    assert len(call["returned"]["brews"]) == 2
    assert call["returned"]["brews"][0]["params"]["grind_setting"] == "495"
    assert call["latency_ms"] >= 0


def test_unscoped_query_is_flagged_in_the_trace():
    """user_id is optional, so traces must show when a call went cross-user."""
    script = [
        FakeResponse([ToolUseBlock("get_user_brews_with_bean", {"coffee_id": "c"})]),
        FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)]),
    ]
    agent, _ = make_agent(script)
    trace = agent.run(a_brew(), "sour").trace
    assert trace["iterations"][0]["tool_calls"][0]["scoped_to_user"] is False


def test_hitting_the_cap_still_produces_an_answer():
    browsing = FakeResponse([ToolUseBlock("get_brew", {"brew_id": "brew-1"})])
    script = [browsing, browsing, FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)])]
    agent, _ = make_agent(script, max_iterations=2)
    result = agent.run(a_brew(), "sour")

    assert result.trace["hit_cap"] is True
    assert result.recommendation.grind_setting == "485"
    # The forced call pins tool_choice so an answer is unavoidable.
    assert agent._client.messages.calls[-1]["tool_choice"] == {
        "type": "tool",
        "name": SUBMIT_TOOL,
    }
    assert result.trace["iterations"][-1]["forced_submit"] is True


def test_prose_reply_is_nudged_rather_than_discarded():
    script = [
        FakeResponse([TextBlock("I think you should grind finer.")], stop_reason="end_turn"),
        FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)]),
    ]
    agent, _ = make_agent(script)
    result = agent.run(a_brew(), "sour")

    assert result.recommendation.grind_setting == "485"
    nudge = agent._client.messages.calls[1]["messages"][-1]
    assert SUBMIT_TOOL in nudge["content"]


def test_tool_failure_is_returned_to_the_model_not_raised():
    script = [
        FakeResponse([ToolUseBlock("get_brew", {"brew_id": "brew-1"})]),
        FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)]),
    ]
    agent, _ = make_agent(script, explode=True)
    result = agent.run(a_brew(), "sour")

    tool_result = agent._client.messages.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "connection reset" in tool_result["content"]
    # The run still finishes rather than taking the whole eval down.
    assert result.recommendation.grind_setting == "485"


def test_refusal_is_recorded_as_an_error():
    script = [FakeResponse([], stop_reason="refusal")]
    agent, _ = make_agent(script)
    result = agent.run(a_brew(), "sour")
    assert result.recommendation.error == "model declined the request"


def test_api_failure_does_not_take_down_the_run():
    class Exploding:
        calls: list = []

        def create(self, **kwargs):
            raise RuntimeError("503 overloaded")

    agent, _ = make_agent([])
    agent._client.messages = Exploding()
    result = agent.run(a_brew(), "sour")
    assert "503 overloaded" in str(result.recommendation.error)


def test_agent_is_not_handed_the_brew_parameters():
    """It has get_brew; the no_tools baseline is the arm that gets them free."""
    script = [FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)])]
    agent, _ = make_agent(script)
    agent.run(a_brew(grind="500"), "sour and thin")

    opening = agent._client.messages.calls[0]["messages"][0]["content"]
    assert "brew-1" in opening
    assert "sour and thin" in opening
    assert "500" not in opening
    assert "Gesha" not in opening


def test_usage_is_accumulated_across_iterations():
    script = [
        FakeResponse([ToolUseBlock("get_brew", {"brew_id": "brew-1"})]),
        FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)]),
    ]
    agent, _ = make_agent(script)
    usage = agent.run(a_brew(), "sour").trace["usage"]
    assert usage["input_tokens"] == 200
    assert usage["output_tokens"] == 100


def test_sampling_parameters_are_never_sent():
    """Claude Opus 5 rejects temperature/top_p/top_k with a 400."""
    script = [FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)])]
    agent, _ = make_agent(script)
    agent.run(a_brew(), "sour")
    sent = agent._client.messages.calls[0]
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in sent


@pytest.mark.parametrize("lever", ["grind_setting", "none", "bogus_lever"])
def test_primary_lever_is_validated(lever):
    payload = {**SUBMIT_INPUT, "primary_lever": lever}
    script = [FakeResponse([ToolUseBlock(SUBMIT_TOOL, payload)])]
    agent, _ = make_agent(script)
    rec = agent.run(a_brew(), "sour").recommendation
    assert rec.primary_lever == (lever if lever != "bogus_lever" else "none")

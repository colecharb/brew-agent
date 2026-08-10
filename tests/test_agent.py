"""Agent loop mechanics, driven by a scripted client. No network."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from brew_agent.agent import BrewAgent
from brew_agent.config import ModelConfig
from brew_agent.models import Brew
from brew_agent.tools import ALL_TOOLS, SUBMIT_TOOL, Toolbox


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


def a_brew(bid="brew-1", grind="500", when="2026-01-10T00:00:00+00:00"):
    return Brew(
        id=bid,
        created_by="user-1",
        brew_timestamp=when,
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


# The brew under diagnosis, one earlier brew, and the held-out later one. The
# last is the answer; no tool may ever return it.
DIAGNOSED = a_brew("brew-1", "500", "2026-01-10T00:00:00+00:00")
EARLIER = a_brew("brew-0", "510", "2026-01-05T00:00:00+00:00")
FUTURE = a_brew("brew-future", "480", "2026-01-20T00:00:00+00:00")


class FakeDatabase:
    """Stands in for BrewDatabase, and honours `as_of` the way the real one does.

    Honouring it matters: a fake that ignored the cutoff would let the
    regression test pass while the thing it guards stayed broken.
    """

    def __init__(self, explode: bool = False) -> None:
        self.explode = explode
        self.calls: list[tuple] = []
        self.brews = [EARLIER, DIAGNOSED, FUTURE]

    def _before(self, as_of):
        return [b for b in self.brews if as_of is None or b.brew_timestamp < as_of]

    def get_brew(self, brew_id, as_of=None):
        self.calls.append(("get_brew", brew_id, as_of))
        if self.explode:
            raise RuntimeError("connection reset")
        found = next((b for b in self.brews if b.id == brew_id), None)
        if found is None or (as_of and found.brew_timestamp > as_of):
            return None
        return found

    def get_user_brews_with_bean(self, coffee_id, user_id=None, limit=20, as_of=None):
        self.calls.append(("bean", coffee_id, user_id, as_of))
        return self._before(as_of)

    def get_user_brews_with_gear(
        self, grinder_id, brewer_id, min_rating, user_id=None, limit=20, as_of=None
    ):
        self.calls.append(("gear", grinder_id, brewer_id, min_rating, user_id, as_of))
        return self._before(as_of)


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
    trace = agent.run(DIAGNOSED, "sour").trace

    call = trace["iterations"][0]["tool_calls"][0]
    assert call["tool"] == "get_user_brews_with_bean"
    assert call["arguments"]["coffee_id"] == "coffee-1"
    assert call["user_id_given"] is True
    assert call["as_of"] == DIAGNOSED.brew_timestamp
    assert call["row_count"] == 1
    # The actual rows, not just a count — a trace has to be inspectable.
    assert call["returned"]["brews"][0]["params"]["grind_setting"] == "510"
    assert call["latency_ms"] >= 0


def test_trace_flags_when_no_user_was_given():
    """user_id is optional, so traces must show when a call went cross-user."""
    script = [
        FakeResponse([ToolUseBlock("get_user_brews_with_bean", {"coffee_id": "c"})]),
        FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)]),
    ]
    agent, _ = make_agent(script)
    trace = agent.run(DIAGNOSED, "sour").trace
    assert trace["iterations"][0]["tool_calls"][0]["user_id_given"] is False


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


class TestNoPeekingAtTheFuture:
    """The agent must not see brews made after the one it is diagnosing.

    This is the bug the first live run exposed: `get_user_brews_with_bean`
    returned the held-out next brew, and the agent read off what the user did
    instead of diagnosing anything. It scored "correct" and meant nothing.

    In production nothing later exists, so the cutoff is invisible there; in the
    eval the answer is sitting in the same table.
    """

    def _history_call(self, tool, args):
        script = [
            FakeResponse([ToolUseBlock(tool, args)]),
            FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)]),
        ]
        agent, db = make_agent(script)
        trace = agent.run(DIAGNOSED, "sour").trace
        return trace["iterations"][0]["tool_calls"][0], db

    @pytest.mark.parametrize(
        "tool,args",
        [
            ("get_user_brews_with_bean", {"coffee_id": "coffee-1"}),
            (
                "get_user_brews_with_gear",
                {"grinder_id": "g1", "brewer_id": "b1", "min_rating": 2},
            ),
        ],
    )
    def test_history_never_returns_a_later_brew(self, tool, args):
        call, _ = self._history_call(tool, args)
        returned = {b["brew_id"] for b in call["returned"]["brews"]}
        assert FUTURE.id not in returned
        assert returned == {EARLIER.id}

    def test_history_excludes_the_diagnosed_brew_itself(self):
        """It is not history, and the agent already has it from get_brew."""
        call, _ = self._history_call(
            "get_user_brews_with_bean", {"coffee_id": "coffee-1"}
        )
        assert DIAGNOSED.id not in {b["brew_id"] for b in call["returned"]["brews"]}

    def test_get_brew_refuses_an_id_from_the_future(self):
        script = [
            FakeResponse([ToolUseBlock("get_brew", {"brew_id": FUTURE.id})]),
            FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)]),
        ]
        agent, _ = make_agent(script)
        call = agent.run(DIAGNOSED, "sour").trace["iterations"][0]["tool_calls"][0]
        assert "No visible brew" in call["returned"]["error"]

    def test_get_brew_still_returns_the_brew_under_diagnosis(self):
        """The cutoff is inclusive for the brew itself, or nothing works."""
        script = [
            FakeResponse([ToolUseBlock("get_brew", {"brew_id": DIAGNOSED.id})]),
            FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)]),
        ]
        agent, _ = make_agent(script)
        call = agent.run(DIAGNOSED, "sour").trace["iterations"][0]["tool_calls"][0]
        assert call["returned"]["brew"]["brew_id"] == DIAGNOSED.id

    def test_the_cutoff_comes_from_the_diagnosed_brew(self):
        script = [
            FakeResponse([ToolUseBlock("get_user_brews_with_bean", {"coffee_id": "c"})]),
            FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)]),
        ]
        agent, db = make_agent(script)
        agent.run(DIAGNOSED, "sour")
        assert db.calls[0][-1] == DIAGNOSED.brew_timestamp

    def test_the_model_cannot_set_the_cutoff(self):
        """as_of is not a tool parameter, so a prompt cannot widen it."""
        for tool in ALL_TOOLS:
            assert "as_of" not in tool["input_schema"].get("properties", {})

    def test_dispatch_will_not_run_without_a_cutoff(self):
        """No default — omitting it must fail loudly, not silently leak."""
        with pytest.raises(TypeError):
            Toolbox(FakeDatabase()).dispatch("get_brew", {"brew_id": "brew-1"})

    def test_the_trace_records_the_horizon(self):
        script = [FakeResponse([ToolUseBlock(SUBMIT_TOOL, SUBMIT_INPUT)])]
        agent, _ = make_agent(script)
        assert agent.run(DIAGNOSED, "sour").trace["as_of"] == DIAGNOSED.brew_timestamp


@pytest.mark.parametrize("lever", ["grind_setting", "none", "bogus_lever"])
def test_primary_lever_is_validated(lever):
    payload = {**SUBMIT_INPUT, "primary_lever": lever}
    script = [FakeResponse([ToolUseBlock(SUBMIT_TOOL, payload)])]
    agent, _ = make_agent(script)
    rec = agent.run(a_brew(), "sour").recommendation
    assert rec.primary_lever == (lever if lever != "bogus_lever" else "none")

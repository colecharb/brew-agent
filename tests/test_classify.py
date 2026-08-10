"""The classifier arm, driven by a scripted client. No network."""

from dataclasses import dataclass, field

import pytest

from brew_agent.baselines import (
    BOTH,
    CLASSIFY_SYSTEM,
    CLASSIFY_TASTE,
    CLASSIFY_TOOL,
    NEITHER,
    OVER,
    UNDER,
    ClassifyBaseline,
    run_rules,
)
from brew_agent.config import ModelConfig
from brew_agent.models import Brew


@dataclass
class ToolUseBlock:
    name: str
    input: dict
    id: str = "toolu_1"
    type: str = "tool_use"


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class Usage:
    input_tokens: int = 20
    output_tokens: int = 8


@dataclass
class FakeResponse:
    content: list
    stop_reason: str = "tool_use"
    usage: Usage = field(default_factory=Usage)


class FakeMessages:
    def __init__(self, response, explode=False):
        self._response = response
        self._explode = explode
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._explode:
            raise RuntimeError("503 overloaded")
        return self._response


class FakeClient:
    def __init__(self, response=None, explode=False):
        self.messages = FakeMessages(response, explode)


CONFIG = ModelConfig(
    api_key="test", model="claude-opus-5", effort=None, max_iterations=6
)


def verdict_response(verdict, evidence="the words"):
    return FakeResponse(
        [ToolUseBlock(CLASSIFY_TOOL, {"verdict": verdict, "evidence": evidence})]
    )


def brew(grind="500", temp=93.0):
    return Brew(
        id="brew-1",
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
        water_temp=temp,
        time=180,
        rating=1,
    )


def classify(verdict, note="tasted a bit off", **kw):
    client = FakeClient(verdict_response(verdict))
    result = ClassifyBaseline(client, CONFIG).run(brew(**kw), note)
    return client, result


class TestVerdictMapping:
    def test_under_extraction_goes_finer(self):
        _, result = classify(UNDER)
        assert result.recommendation.grind_value == 475.0
        assert result.recommendation.water_temp == 95.0
        assert result.recommendation.primary_lever == "grind_setting"

    def test_over_extraction_goes_coarser(self):
        _, result = classify(OVER)
        assert result.recommendation.grind_value == 525.0
        assert result.recommendation.water_temp == 91.0

    @pytest.mark.parametrize("verdict", [BOTH, NEITHER])
    def test_ambiguous_and_empty_verdicts_change_nothing(self, verdict):
        _, result = classify(verdict)
        assert result.recommendation.changes_nothing
        assert result.recommendation.error is None

    def test_the_arithmetic_is_identical_to_the_rule_table(self):
        """The rung isolates language understanding, not the step size.

        Same brew, same verdict reached two different ways, byte-identical
        numbers out. If this ever diverges, the rules-to-classify delta stops
        measuring one variable.
        """
        _, classified = classify(UNDER, note="anything at all")
        keyword = run_rules(brew(), "sour and thin")
        assert classified.recommendation.grind_setting == (
            keyword.recommendation.grind_setting
        )
        assert classified.recommendation.water_temp == keyword.recommendation.water_temp


class TestPromptIsolation:
    def test_only_the_note_is_sent(self):
        """The classifier must not see the brew — that is the whole rung."""
        client, _ = classify(UNDER, note="Sour and thin, watery body.")
        sent = client.messages.calls[0]
        content = sent["messages"][0]["content"]

        assert content == "Sour and thin, watery body."
        for leaked in ("500", "Gesha", "Z1", "V60", "15.0", "250"):
            assert leaked not in content
        assert CLASSIFY_TOOL in str(sent["tools"])
        assert sent["tool_choice"] == {"type": "tool", "name": CLASSIFY_TOOL}

    def test_no_sampling_parameters_are_sent(self):
        client, _ = classify(UNDER)
        for banned in ("temperature", "top_p", "top_k"):
            assert banned not in client.messages.calls[0]

    def test_only_the_classify_tool_is_offered(self):
        """No submit_recommendation — it cannot choose its own adjustment."""
        client, _ = classify(UNDER)
        tools = client.messages.calls[0]["tools"]
        assert [t["name"] for t in tools] == [CLASSIFY_TOOL]


class TestTheBrief:
    """Regression pins for the abstention bug.

    `classify` first scored below the keyword table it exists to beat. Its
    brief said "judge only what the note says about flavour" and spent two more
    sentences framing abstention as virtuous, so the arm returned `neither` on a
    note opening *"Shot pulled way too fast"* — an unambiguous under-extraction
    call that happens to contain no flavour word at all. The rung was measuring
    the brief rather than the reader.

    Asserting on prompt text is ordinarily a poor test. Here it is the only
    artefact available offline, and the specific wording is what broke, so these
    pin the wording.
    """

    def test_the_brief_admits_evidence_beyond_taste(self):
        brief = CLASSIFY_SYSTEM.lower()
        assert "how the brew ran" in brief
        assert "too fast" in brief and "ran slow" in brief

    def test_the_schema_admits_evidence_beyond_taste(self):
        verdict = CLASSIFY_TASTE["input_schema"]["properties"]["verdict"]
        assert "ran fast" in verdict["description"].lower()
        assert "ran slow" in verdict["description"].lower()

    def test_judgement_is_not_restricted_to_flavour(self):
        assert "only what the note says about flavour" not in CLASSIFY_SYSTEM.lower()

    def test_both_ways_out_are_scoped_narrowly(self):
        """`both` and `neither` both recommend nothing.

        They are between them the arm's only way to lose without being wrong, so
        widening either one is how this regresses.
        """
        brief = CLASSIFY_SYSTEM.lower()
        assert "reserve neither" in brief
        assert "reserve both" in brief

    def test_the_scoring_is_never_disclosed_to_the_classifier(self):
        """Told that silence is penalised, it would guess to protect a number.

        The prompt describes the task. The metric stays outside it — otherwise
        the arm optimises the eval instead of reading the note, and the rung
        measures nothing.
        """
        brief = CLASSIFY_SYSTEM.lower()
        for tell in ("miss", "penal", "score", "accuracy", "counts against"):
            assert tell not in brief, tell


class TestDegradation:
    def test_an_api_error_scores_as_a_miss_rather_than_crashing(self):
        client = FakeClient(explode=True)
        result = ClassifyBaseline(client, CONFIG).run(brew(), "sour")
        assert "503 overloaded" in str(result.recommendation.error)
        assert result.recommendation.changes_nothing

    def test_a_refusal_is_recorded(self):
        client = FakeClient(FakeResponse([], stop_reason="refusal"))
        result = ClassifyBaseline(client, CONFIG).run(brew(), "sour")
        assert result.recommendation.error
        assert result.trace["verdict"] is None

    def test_a_prose_reply_is_recorded_as_an_error(self):
        client = FakeClient(FakeResponse([TextBlock("Sounds under-extracted.")]))
        result = ClassifyBaseline(client, CONFIG).run(brew(), "sour")
        assert CLASSIFY_TOOL in str(result.recommendation.error)

    def test_non_numeric_grind_leaves_the_dial_alone(self):
        _, result = classify(UNDER, grind="1.8.2")
        assert result.recommendation.grind_setting is None
        assert result.recommendation.primary_lever == "none"

    def test_missing_temperature_yields_no_temperature_advice(self):
        _, result = classify(UNDER, temp=None)
        assert result.recommendation.water_temp is None
        assert result.recommendation.grind_value is not None


def test_trace_records_the_verdict_and_evidence():
    _, result = classify(OVER, note="Bitter and drying")
    assert result.trace["arm"] == "classify"
    assert result.trace["verdict"] == OVER
    assert result.trace["evidence"] == "the words"
    assert result.trace["usage"]["input_tokens"] == 20

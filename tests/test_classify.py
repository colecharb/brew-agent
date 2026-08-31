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
from brew_agent.providers import AnthropicProvider


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
    provider="anthropic",
    api_key="test",
    model="claude-opus-5",
    effort=None,
    max_iterations=6,
    max_tokens=16000,
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


def classify(verdict, note="tasted a bit off", evidence="the words", **kw):
    client = FakeClient(verdict_response(verdict, evidence))
    result = ClassifyBaseline(AnthropicProvider(client), CONFIG).run(brew(**kw), note)
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
        result = ClassifyBaseline(AnthropicProvider(client), CONFIG).run(brew(), "sour")
        assert "503 overloaded" in str(result.recommendation.error)
        assert result.recommendation.changes_nothing

    def test_a_refusal_is_recorded(self):
        client = FakeClient(FakeResponse([], stop_reason="refusal"))
        result = ClassifyBaseline(AnthropicProvider(client), CONFIG).run(brew(), "sour")
        assert result.recommendation.error
        assert result.trace["verdict"] is None

    def test_a_prose_reply_is_recorded_as_an_error(self):
        client = FakeClient(FakeResponse([TextBlock("Sounds under-extracted.")]))
        result = ClassifyBaseline(AnthropicProvider(client), CONFIG).run(brew(), "sour")
        assert CLASSIFY_TOOL in str(result.recommendation.error)

    def test_non_numeric_grind_leaves_the_dial_alone(self):
        _, result = classify(UNDER, grind="1.8.2")
        assert result.recommendation.grind_setting is None
        assert result.recommendation.primary_lever == "none"

    def test_a_coarse_dial_still_gets_a_real_move(self):
        """5% of 7 is 0.35, which a whole-number dial rounds back to 7.

        The arm would report `grind_setting: "7"` on a brew already at 7 — a
        confident recommendation that proposes nothing, scored as an abstention
        and indistinguishable from having had no opinion.
        """
        _, finer = classify(UNDER, grind="7")
        _, coarser = classify(OVER, grind="7")
        assert finer.recommendation.grind_setting == "6"
        assert coarser.recommendation.grind_setting == "8"

    def test_a_decimal_dial_expresses_5_percent_and_keeps_it(self):
        """7.0 has room for 6.65, so the floor must not fire here."""
        _, result = classify(UNDER, grind="7.0")
        assert result.recommendation.grind_setting == "6.6"

    def test_the_floor_respects_the_dial_s_own_precision(self):
        """5% of 0.5 is 0.025 — still too small for a one-decimal dial."""
        _, result = classify(UNDER, grind="0.5")
        assert result.recommendation.grind_setting == "0.4"

    def test_a_fine_dial_is_untouched_by_the_floor(self):
        """Where 5% is expressible, it is still exactly 5%."""
        _, result = classify(UNDER, grind="500")
        assert result.recommendation.grind_setting == "475"

    def test_missing_temperature_yields_no_temperature_advice(self):
        _, result = classify(UNDER, temp=None)
        assert result.recommendation.water_temp is None
        assert result.recommendation.grind_value is not None


class TestEvidenceIsCheckedAgainstTheNote:
    """A trace that misquotes the note cannot explain the score it sits next to.

    One live call returned `"</antml\\u0903parameter>"` in this field — a
    malformed fragment where a quote belonged. Nothing downstream reads
    `evidence`, so it cost no accuracy. It cost the trace its standing as
    evidence, which is the only reason the field exists.
    """

    def test_a_real_quote_passes(self):
        _, result = classify(OVER, note="Bitter and drying.", evidence="drying")
        assert result.trace["evidence_verbatim"] is True

    def test_case_and_reflowed_whitespace_are_not_fabrication(self):
        note = "Harsh finish,\n  really quite drying."
        _, result = classify(OVER, note=note, evidence="Harsh finish, really")
        assert result.trace["evidence_verbatim"] is True

    def test_a_fragment_of_nothing_is_flagged(self):
        _, result = classify(
            OVER, note="Bitter and drying.", evidence="</antmlःparameter>\n"
        )
        assert result.trace["evidence_verbatim"] is False

    def test_a_paraphrase_is_flagged(self):
        """Not a fabrication, but not a quote either, and the field asks for one."""
        _, result = classify(
            UNDER, note="Didn't really sing.", evidence="lacked sweetness"
        )
        assert result.trace["evidence_verbatim"] is False

    def test_empty_evidence_is_left_alone(self):
        """`neither` is documented as having none, so absence is not a fault."""
        _, result = classify(NEITHER, note="Yes.", evidence="")
        assert result.trace["evidence_verbatim"] is True

    def test_null_evidence_is_accepted(self):
        """The schema's way of saying "nothing to quote" — must not crash."""
        _, result = classify(NEITHER, note="Yes.", evidence=None)
        assert result.trace["evidence"] == ""
        assert result.trace["evidence_verbatim"] is True
        assert result.recommendation.error is None


class TestEvidenceMayBeNull:
    """Under `strict` every property is required, so "nothing to say" needs a
    value that means it.

    Documented as a plain required string that should be "empty for neither",
    the field came back holding a fragment of the tool-call markup on about
    half of all calls across two different models — and never once actually
    empty. Asked for a string it has no content for, the model emits something.
    """

    @staticmethod
    def _field(name: str) -> dict:
        return CLASSIFY_TASTE["input_schema"]["properties"][name]

    def test_evidence_accepts_null(self):
        assert {"type": "null"} in self._field("evidence")["anyOf"]

    def test_evidence_still_accepts_a_string(self):
        assert {"type": "string"} in self._field("evidence")["anyOf"]

    def test_the_verdict_itself_is_never_nullable(self):
        """It is the one field that always has an answer, and it drives scoring."""
        assert "anyOf" not in self._field("verdict")
        assert self._field("verdict")["type"] == "string"

    def test_the_description_no_longer_asks_for_an_empty_string(self):
        assert "empty for neither" not in self._field("evidence")["description"].lower()

    def test_junk_evidence_still_scores_off_the_verdict(self):
        """The check observes; it must not become a second gate on the answer.

        Only `verdict` reaches `_apply_step`, so a corrupted quote costs the
        arm nothing numerically — which is exactly why it went unnoticed. The
        reasoning string does differ, since it quotes what it was given.
        """
        clean = classify(UNDER, note="Sour.", evidence="Sour")[1].recommendation
        junk = classify(UNDER, note="Sour.", evidence="ःः")[1].recommendation
        for field in ("grind_setting", "water_temp", "primary_lever"):
            assert junk.to_dict()[field] == clean.to_dict()[field], field
        assert junk.error is None


def test_trace_records_the_verdict_and_evidence():
    _, result = classify(OVER, note="Bitter and drying")
    assert result.trace["arm"] == "classify"
    assert result.trace["verdict"] == OVER
    assert result.trace["evidence"] == "the words"
    assert result.trace["usage"]["input_tokens"] == 20

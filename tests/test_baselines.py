"""The rule table and prompt building — the parts that need no API key."""

import anthropic
import httpx
import pytest

from brew_agent import baselines
from brew_agent.baselines import build_prompt, call_model, describe_brew, run_rules
from brew_agent.config import ModelConfig
from brew_agent.models import Brew


def brew(grind="500", temp=93.0, dose=15.0, yield_g=250.0):
    return Brew(
        id="brew-1",
        created_by="user-1",
        brew_timestamp="2026-01-01T00:00:00+00:00",
        profile_coffee_id="bag-1",
        coffee_id="coffee-1",
        coffee_name="Gesha",
        roaster_name="Some Roaster",
        grinder_id="g1",
        brewer_id="b1",
        grinder_name="Z1",
        brewer_name="V60",
        grind_setting=grind,
        coffee_weight=dose,
        target_weight=yield_g,
        water_temp=temp,
        time=180,
        rating=1,
    )


class TestRuleTable:
    def test_under_extraction_goes_finer_and_hotter(self):
        result = run_rules(brew(grind="500"), "sour and thin, quite watery")
        rec = result.recommendation
        assert result.trace["verdict"] == "under-extracted"
        # Assumes higher == coarser, so finer means a smaller number.
        assert rec.grind_value == 475.0
        assert rec.water_temp == 95.0
        assert rec.primary_lever == "grind_setting"

    def test_over_extraction_goes_coarser_and_cooler(self):
        result = run_rules(brew(grind="500"), "bitter and drying on the finish")
        rec = result.recommendation
        assert result.trace["verdict"] == "over-extracted"
        assert rec.grind_value == 525.0
        assert rec.water_temp == 91.0

    def test_mixed_vocabulary_declines_to_guess(self):
        result = run_rules(brew(), "sour up front but bitter at the end")
        assert result.trace["verdict"] == "mixed"
        assert result.recommendation.changes_nothing

    def test_unmatched_complaint_changes_nothing(self):
        result = run_rules(brew(), "tastes like a nice cup of coffee")
        assert result.trace["verdict"] == "none"
        assert result.recommendation.changes_nothing

    @pytest.mark.parametrize(
        "setting,expected",
        [("500", "475"), ("4.1", "3.9"), ("22.0", "20.9"), ("1000", "950")],
    )
    def test_recommendation_keeps_the_dial_s_precision(self, setting, expected):
        result = run_rules(brew(grind=setting), "sour")
        assert result.recommendation.grind_setting == expected

    def test_step_size_is_within_observed_human_moves(self):
        """p90 of real grind moves is ~20%; the baseline must sit under that."""
        result = run_rules(brew(grind="500"), "sour")
        moved = abs(result.recommendation.grind_value - 500.0) / 500.0
        assert 0.01 < moved < 0.20

    def test_missing_temperature_yields_no_temperature_advice(self):
        result = run_rules(brew(temp=None), "sour and thin")
        assert result.recommendation.water_temp is None
        assert result.recommendation.grind_value is not None

    def test_non_numeric_grind_is_left_alone(self):
        result = run_rules(brew(grind="1.8.2"), "sour and thin")
        assert result.recommendation.grind_setting is None
        assert result.recommendation.primary_lever == "none"

    def test_prefix_matching_catches_inflections(self):
        """Terms match as prefixes on purpose, so noun forms are picked up."""
        assert run_rules(brew(), "noticeable sourness").trace["matched_under"] == [
            "sour"
        ]
        assert run_rules(brew(), "a lot of bitterness").trace["matched_over"] == [
            "bitter"
        ]
        # But only where the stem really is a prefix: "astringency" does not
        # start with "astringent", so it slips past. Another cost of the
        # keyword approach, and another reason this arm is the floor.
        assert run_rules(brew(), "real astringency").trace["verdict"] == "none"

    def test_prefix_matching_also_over_triggers(self):
        """The cost of prefix matching, asserted rather than hidden.

        "flat white" reads as the "flat" under-extraction cue and "sourdough" as
        "sour". This arm exists to show what keyword matching alone is worth, so
        its false positives belong in the measurement, not papered over.
        """
        assert run_rules(brew(), "made a flat white with it").trace["verdict"] == (
            "under-extracted"
        )
        assert run_rules(brew(), "sourdough on the side").trace["matched_under"] == [
            "sour"
        ]


class TestPrompt:
    def test_prompt_carries_the_dial_and_the_complaint(self):
        text = build_prompt(brew(grind="500"), "sour and thin")
        assert "Grind setting: 500" in text
        assert "sour and thin" in text
        assert "Z1" in text and "V60" in text
        assert "brew-1" in text

    def test_optional_fields_are_omitted_when_absent(self):
        text = describe_brew(brew(temp=None))
        assert "Water temperature" not in text
        assert "Brew time: 180s" in text

    def test_ratio_is_derived(self):
        assert "ratio 16.7:1" in describe_brew(brew(dose=15.0, yield_g=250.0))


def bad_request(message: str) -> anthropic.BadRequestError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(
        message, response=httpx.Response(400, request=request), body=None
    )


class RejectingMessages:
    """Rejects the first call with a given error, then accepts."""

    def __init__(self, error: anthropic.BadRequestError) -> None:
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise self._error
        return "ok"


class RejectingClient:
    def __init__(self, error: anthropic.BadRequestError) -> None:
        self.messages = RejectingMessages(error)


def call(client, effort=None, force_tool=None):
    config = ModelConfig(
        api_key="test", model="claude-haiku-4-5", effort=effort, max_iterations=6
    )
    return call_model(
        client, config, system="s", messages=[], tools=[], force_tool=force_tool
    )


class TestUnsupportedParametersDegradeRatherThanFail:
    """Smaller models reject parameters the frontier ones accept.

    Without this the eval fails every pair identically on a model-capability
    mismatch — an expensive way to learn that `effort` isn't universal.
    """

    @pytest.fixture(autouse=True)
    def _reset_warnings(self):
        baselines._warned.clear()

    def test_effort_is_dropped_and_the_call_retried(self):
        client = RejectingClient(
            bad_request("output_config.effort: unsupported on this model")
        )
        assert call(client, effort="high") == "ok"
        first, second = client.messages.calls
        assert first["output_config"] == {"effort": "high"}
        assert "output_config" not in second

    def test_dropping_effort_is_announced(self, capsys):
        client = RejectingClient(bad_request("output_config.effort: unsupported"))
        call(client, effort="high")
        assert "not comparable" in capsys.readouterr().err

    def test_the_warning_fires_once_across_many_calls(self, capsys):
        for _ in range(5):
            call(RejectingClient(bad_request("output_config.effort: bad")), effort="high")
        assert capsys.readouterr().err.count("warning:") == 1

    def test_a_forced_tool_is_still_dropped_on_its_own_error(self):
        client = RejectingClient(bad_request("tool_choice: not allowed here"))
        assert call(client, force_tool="submit_recommendation") == "ok"
        assert "tool_choice" not in client.messages.calls[1]

    def test_an_unrelated_bad_request_still_raises(self):
        """Only the two known-survivable rejections are absorbed."""
        client = RejectingClient(bad_request("messages: must not be empty"))
        with pytest.raises(anthropic.BadRequestError):
            call(client, effort="high")

    def test_nothing_is_dropped_when_effort_was_never_sent(self):
        client = RejectingClient(bad_request("output_config.effort: unsupported"))
        with pytest.raises(anthropic.BadRequestError):
            call(client)


def test_rule_table_survives_every_real_brew(seed_brews):
    """No crashes on real notes: empty strings, emoji, curly quotes, newlines."""
    for b in seed_brews:
        result = run_rules(b, b.notes)
        rec = result.recommendation
        if rec.grind_setting is not None:
            assert rec.grind_value is not None, f"unparseable output for {b.id}"

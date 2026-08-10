"""End-to-end runner over real seed data, with no network and no API key.

Exercises the whole command path — fetch, pair, sample, run, score, aggregate,
write traces and results — using the `rules` arm, which needs neither.
"""

import json

import pytest

from brew_agent.eval.run import main, print_report, run_eval


class SeedDatabase:
    """Stands in for BrewDatabase, serving the committed seed dump."""

    def __init__(self, brews):
        self._brews = brews
        self.user_id = "test-operator"

    def fetch_all_brews(self, max_rows: int = 5000):
        return self._brews[:max_rows]


@pytest.fixture
def result(tmp_path, seed_brews):
    return run_eval(
        SeedDatabase(seed_brews),
        names=["rules"],
        n=12,
        trace_root=tmp_path / "traces",
        output_root=tmp_path / "output",
    )


def test_produces_a_score_for_every_sampled_pair(result):
    arm = result["scores"]["rules"]
    assert arm.n == 12
    assert arm.errors == 0
    # The rule table must actually engage with real notes, not abstain on all.
    assert arm.recommended_nothing < 12


def test_headline_is_a_real_number_or_honestly_absent(result):
    arm = result["scores"]["rules"]
    assert arm.headline is None or 0.0 <= arm.headline <= 1.0
    assert arm.grind.considered <= arm.n
    assert arm.grind.correct + arm.grind.wrong + arm.grind.abstained == (
        arm.grind.considered
    )


def test_results_file_is_written_and_parses(result):
    payload = json.loads(result["output_path"].read_text())
    assert payload["run_id"] == result["run_id"]
    assert payload["sampled"] == 12
    assert payload["include_leaky"] is False
    assert set(payload["arms"]) == {"rules"}
    # 372 pairs survive the filter chain; 105 of them (28%) state the next
    # adjustment in the notes and are excluded by default.
    assert payload["funnel"]["something_changed"] == 372
    assert payload["funnel"]["leaky_excluded"] == 105
    assert payload["funnel"]["eligible"] == 267
    assert len(payload["pairs"]) == 12
    assert all(p["leaky"] is False for p in payload["pairs"])


def test_every_trace_is_inspectable(result):
    traces = sorted(result["trace_dir"].glob("*.json"))
    assert len(traces) == 12

    payload = json.loads(traces[0].read_text())
    # A trace has to answer "what was asked, what came back, how did it score"
    # without cross-referencing anything else.
    assert payload["input"]["complaint"]
    assert payload["input"]["brew"]["params"]["grind_setting"]
    assert payload["held_out_next_brew"]["params"]["grind_setting"]
    assert payload["recommendation"]["primary_lever"] in {
        "grind_setting",
        "coffee_weight",
        "target_weight",
        "water_temp",
        "time",
        "none",
    }
    assert payload["score"]["grind"]
    assert payload["trace"]["arm"] == "rules"


def test_traces_never_land_in_the_repo_by_default(result, tmp_path):
    """Traces carry other users' notes; they must stay under the given root."""
    assert result["trace_dir"].is_relative_to(tmp_path)
    assert result["output_path"].is_relative_to(tmp_path)


def test_report_prints_without_blowing_up(result, capsys):
    print_report(result)
    out = capsys.readouterr().out
    assert "eligible pairs" in out
    assert "rules" in out
    # The table has to show abstention next to accuracy, or a quiet arm reads
    # as a wrong one.
    for column in ("ok", "wrong", "quiet", "when improved", "held"):
        assert column in out


def test_leaky_pairs_are_opt_in(tmp_path, seed_brews):
    strict = run_eval(
        SeedDatabase(seed_brews), ["rules"], n=400,
        trace_root=tmp_path / "a", output_root=tmp_path / "a",
    )
    loose = run_eval(
        SeedDatabase(seed_brews), ["rules"], n=400, include_leaky=True,
        trace_root=tmp_path / "b", output_root=tmp_path / "b",
    )
    assert loose["stats"].eligible > strict["stats"].eligible
    assert strict["stats"].leaky_excluded > 0


def test_unknown_arm_is_rejected_before_connecting():
    """Fails on the argument, not on a missing database connection."""
    with pytest.raises(SystemExit) as exc:
        main(["--arms", "telepathy"])
    assert exc.value.code == 2

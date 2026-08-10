"""End-to-end runner over real seed data, with no network and no API key.

Exercises the whole command path — fetch, pair, sample, run, score, aggregate,
write traces and results — using the `rules` arm, which needs neither.
"""

import json

import pytest

from brew_agent.baselines import run_rules
from brew_agent.eval.pairs import REDACT, build_pairs
from brew_agent.eval.run import Runner, _run_one_pair, main, print_report, run_eval


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
    assert payload["leak_mode"] == "redact"
    assert set(payload["arms"]) == {"rules"}
    # 372 pairs survive the filter chain; 105 state an adjustment, and
    # redacting rather than excluding keeps all but the 5 that were nothing
    # else.
    assert payload["funnel"]["something_changed"] == 372
    assert payload["funnel"]["leaky_detected"] == 105
    assert payload["funnel"]["redacted_to_nothing"] == 5
    assert payload["funnel"]["eligible"] == 367
    assert len(payload["pairs"]) == 12


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


def test_the_three_leak_modes_are_all_runnable(tmp_path, seed_brews):
    def run(mode, tag):
        return run_eval(
            SeedDatabase(seed_brews), ["rules"], n=400, leak_mode=mode,
            trace_root=tmp_path / tag, output_root=tmp_path / tag,
        )

    redacted, excluded, raw = run("redact", "a"), run("exclude", "b"), run("raw", "c")

    # Redaction keeps nearly everything; exclusion pays 100 pairs for the same
    # protection; raw keeps the contamination on purpose.
    assert raw["stats"].eligible == 372
    assert redacted["stats"].eligible == 367
    assert excluded["stats"].eligible == 267


def test_traces_show_what_was_redacted(result):
    """A redaction has to be auditable, not taken on trust."""
    payloads = [json.loads(p.read_text()) for p in result["trace_dir"].glob("*.json")]
    assert all("redacted_out" in p["input"] for p in payloads)
    # complaint + redacted_out reconstructs the note, so the raw text is not
    # duplicated into the trace.
    assert all("raw_notes" not in p["input"] for p in payloads)


class TestDiagnosableSubset:
    """The second table: only the pairs where a right answer exists.

    Many notes are "Yes." or "For Clemi's latte" — the grind moved for reasons
    never written down. Scoring those pulls every arm toward the same middle.
    """

    @staticmethod
    def _labelled(seed_brews, tmp_path, tag="d"):
        from brew_agent.eval.labels import NoteLabel

        # Stand in for the labelling pass: call a note diagnosable when it has
        # any extraction vocabulary in it.
        labels = {
            b.id: NoteLabel(
                states_adjustment=False,
                adjustment_quotes=[],
                has_complaint=any(
                    w in b.notes.lower() for w in ("sour", "bitter", "thin", "harsh")
                ),
            )
            for b in seed_brews
        }
        return run_eval(
            SeedDatabase(seed_brews), ["rules"], n=40, labels=labels,
            trace_root=tmp_path / tag, output_root=tmp_path / tag,
        )

    def test_absent_without_labels(self, result):
        assert not result.get("diagnosable")

    def test_present_and_smaller_with_labels(self, seed_brews, tmp_path):
        labelled = self._labelled(seed_brews, tmp_path)
        assert labelled["diagnosable"]
        overall = labelled["scores"]["rules"]
        subset = labelled["diagnosable"]["rules"]
        assert 0 < subset.n < overall.n

    def test_written_to_the_results_file(self, seed_brews, tmp_path):
        labelled = self._labelled(seed_brews, tmp_path, tag="e")
        payload = json.loads(labelled["output_path"].read_text())
        assert set(payload["arms_diagnosable_only"]) == {"rules"}
        assert payload["funnel"]["labelled"] is True

    def test_both_tables_print(self, seed_brews, tmp_path, capsys):
        print_report(self._labelled(seed_brews, tmp_path, tag="f"))
        out = capsys.readouterr().out
        assert "All sampled pairs" in out
        assert "note describes a taste problem" in out


class TestPairsRunConcurrently:
    """Pairs go out in parallel. None of the numbers may depend on that."""

    @staticmethod
    def _run(seed_brews, tmp_path, tag, concurrency):
        return run_eval(
            SeedDatabase(seed_brews),
            names=["rules"],
            n=16,
            trace_root=tmp_path / f"t{tag}",
            output_root=tmp_path / f"o{tag}",
            concurrency=concurrency,
        )

    def test_serial_and_parallel_agree(self, seed_brews, tmp_path):
        serial = self._run(seed_brews, tmp_path, "s", 1)
        parallel = self._run(seed_brews, tmp_path, "p", 8)
        assert (
            serial["scores"]["rules"].to_dict()
            == parallel["scores"]["rules"].to_dict()
        )

    def test_the_results_file_stays_in_sample_order(self, seed_brews, tmp_path):
        """Completion order is nondeterministic; the artefact must not be."""
        serial = json.loads(self._run(seed_brews, tmp_path, "s", 1)["output_path"].read_text())
        parallel = json.loads(self._run(seed_brews, tmp_path, "p", 8)["output_path"].read_text())
        assert [p["pair_id"] for p in serial["pairs"]] == [
            p["pair_id"] for p in parallel["pairs"]
        ]

    def test_each_result_stays_with_its_own_pair(self, seed_brews, tmp_path):
        """The real hazard of the fan-out: an answer filed against a neighbour.

        `rules` is deterministic, so every trace can be recomputed from the pair
        it claims to be about and checked against what was written.
        """
        result = self._run(seed_brews, tmp_path, "x", 8)
        pairs = {p.id: p for p in build_pairs(seed_brews, leak_mode=REDACT)[0]}

        traces = list(result["trace_dir"].glob("rules-*.json"))
        assert len(traces) == 16
        for path in traces:
            payload = json.loads(path.read_text())
            pair = pairs[payload["pair_id"]]
            assert path.name == f"rules-{pair.id}.json"
            assert payload["input"]["brew_id"] == pair.before.id
            expected = run_rules(pair.before, pair.complaint).recommendation
            assert payload["recommendation"]["grind_setting"] == expected.grind_setting
            assert payload["recommendation"]["reasoning"] == expected.reasoning


def test_an_exploding_arm_is_scored_rather_than_losing_the_pair(seed_brews):
    """One arm failing must not discard the other arms' work on that pair."""

    def boom(pair):
        raise RuntimeError("connection reset by peer")

    pair = build_pairs(seed_brews, leak_mode=REDACT)[0][0]
    answers = _run_one_pair(
        [
            Runner("rules", lambda p: run_rules(p.before, p.complaint)),
            Runner("boom", boom),
            Runner("rules2", lambda p: run_rules(p.before, p.complaint)),
        ],
        pair,
    )

    assert [name for name, _, _ in answers] == ["rules", "boom", "rules2"]
    scores = {name: score for name, _, score in answers}
    assert "connection reset by peer" in scores["boom"].error
    assert scores["rules"].error is None and scores["rules2"].error is None


class TestMisquotedEvidenceIsReported:
    """Counted per run, so "was that a one-off?" stops needing a trace grep."""

    def test_silent_when_no_arm_quotes_the_note(self, result, capsys):
        # `rules` has no evidence field at all, so there is nothing to warn about.
        assert result["evidence_not_verbatim"] == {}
        print_report(result)
        assert "quoted evidence absent" not in capsys.readouterr().out

    def test_it_reaches_the_results_file(self, result):
        written = json.loads(result["output_path"].read_text())
        assert written["evidence_not_verbatim"] == {}

    def test_reported_when_an_arm_misquotes(self, result, capsys):
        print_report({**result, "evidence_not_verbatim": {"classify": 3}})
        out = capsys.readouterr().out
        assert "classify quoted evidence absent from the note on 3 pair(s)" in out
        assert "Scores are unaffected" in out


def test_unknown_arm_is_rejected_before_connecting():
    """Fails on the argument, not on a missing database connection."""
    with pytest.raises(SystemExit) as exc:
        main(["--arms", "telepathy"])
    assert exc.value.code == 2

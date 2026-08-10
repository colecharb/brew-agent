"""Bucketing the classify arm's quoted evidence. Reads traces, no network."""

import json

import pytest

from brew_agent.eval.audit_evidence import audit, bucket, main


class TestBucket:
    """Each bucket implies a different fix, so the boundaries matter."""

    NOTE = "Shot pulled way too fast. Sour, drying — didn't really sing."

    def test_a_real_substring_is_verbatim(self):
        assert bucket("Sour, drying", self.NOTE) == "verbatim"

    def test_reflowed_whitespace_is_still_verbatim(self):
        assert bucket("Shot pulled\n  way too fast", self.NOTE) == "verbatim"

    def test_dropped_punctuation_is_its_own_bucket(self):
        """The check is stricter than the field deserves here, not the model."""
        assert bucket("Sour drying", self.NOTE) == "punctuation"

    def test_stitched_spans_are_reordered(self):
        """Every word is in the note; the model joined two separate phrases."""
        assert bucket("sour fast", self.NOTE) == "reordered"

    def test_a_summary_is_a_paraphrase(self):
        """Most words come from the note, joined by words that don't."""
        assert bucket("pulled too fast and sour", self.NOTE) == "paraphrase"

    def test_words_from_nowhere_are_unrelated(self):
        assert bucket("bitter ashy overcooked", self.NOTE) == "unrelated"

    def test_a_stray_note_word_does_not_rescue_a_summary(self):
        """One shared word out of four is the model's vocabulary, not the note's."""
        assert bucket("sour and under-extracted", self.NOTE) == "unrelated"

    def test_the_malformed_fragment_is_unrelated(self):
        """The anomaly that prompted the check in the first place."""
        assert bucket("</antmlःparameter>\n", self.NOTE) == "unrelated"

    def test_punctuation_only_is_unrelated_not_empty(self):
        """Empty means the model declined to quote; '>>' means it broke."""
        assert bucket(">>", self.NOTE) == "unrelated"

    def test_nothing_quoted_is_empty(self):
        assert bucket("", self.NOTE) == "empty"
        assert bucket("   ", self.NOTE) == "empty"

    def test_case_never_decides_a_bucket(self):
        assert bucket("SOUR, DRYING", self.NOTE) == "verbatim"


def write_trace(directory, pair_id, evidence, complaint, arm="classify", verdict=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{arm}-{pair_id}.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "pair_id": pair_id,
                "input": {"complaint": complaint},
                "trace": {"arm": arm, "evidence": evidence, "verdict": verdict},
            }
        )
    )


class TestAudit:
    def test_counts_every_trace_once(self, tmp_path):
        run = tmp_path / "run"
        write_trace(run, "a", "Sour, thin", "Sour, thin body.")
        write_trace(run, "b", "sour thin", "Sour, thin body.")
        write_trace(run, "c", "", "Yes.")
        counts, _, _ = audit(run, "classify")
        assert counts["verbatim"] == 1
        assert counts["punctuation"] == 1
        assert counts["empty"] == 1
        assert sum(counts.values()) == 3

    def test_other_arms_are_ignored(self, tmp_path):
        run = tmp_path / "run"
        write_trace(run, "a", "Sour", "Sour and thin.")
        write_trace(run, "b", "Sour", "Sour and thin.", arm="agent")
        counts, _, _ = audit(run, "classify")
        assert sum(counts.values()) == 1

    def test_a_trace_with_no_evidence_field_is_empty_not_a_crash(self, tmp_path):
        """`agent` traces and error traces carry no evidence at all."""
        run = tmp_path / "run"
        (run).mkdir()
        (run / "classify-x.json").write_text(
            json.dumps({"pair_id": "x", "input": {}, "trace": {}})
        )
        counts, _, _ = audit(run, "classify")
        assert counts["empty"] == 1

    def test_examples_are_collected_for_inspection(self, tmp_path):
        run = tmp_path / "run"
        write_trace(run, "a", "bitter ashy", "Sour and thin.")
        _, examples, _ = audit(run, "classify")
        assert examples["unrelated"] == [("bitter ashy", "Sour and thin.")]


class TestCrossTabAgainstVerdict:
    """The cross-tab is what tells a bad model from a bad schema.

    A field the schema documents as empty for one verdict, that is never empty
    for it, is being filled because it cannot be left unfilled.
    """

    def test_buckets_are_split_by_verdict(self, tmp_path):
        run = tmp_path / "run"
        write_trace(run, "a", "Sour", "Sour and thin.", verdict="under_extracted")
        write_trace(run, "b", "</antml parameter>", "Yes.", verdict="neither")
        write_trace(run, "c", "</antml parameter>", "Lovely.", verdict="neither")
        _, _, by_verdict = audit(run, "classify")
        assert by_verdict[("under_extracted", "verbatim")] == 1
        assert by_verdict[("neither", "unrelated")] == 2
        assert by_verdict[("neither", "empty")] == 0

    def test_a_missing_verdict_is_labelled_not_dropped(self, tmp_path):
        run = tmp_path / "run"
        write_trace(run, "a", "Sour", "Sour and thin.")
        _, _, by_verdict = audit(run, "classify")
        assert by_verdict[("no verdict", "verbatim")] == 1

    def test_the_cross_tab_is_printed_per_run(self, tmp_path, capsys):
        run = tmp_path / "run"
        write_trace(run, "a", "</antml parameter>", "Yes.", verdict="neither")
        main([str(run)])
        out = capsys.readouterr().out
        assert "bucket by verdict" in out
        assert "neither" in out


class TestCommand:
    def test_reports_and_compares_two_runs(self, tmp_path, capsys):
        first, second = tmp_path / "r1", tmp_path / "r2"
        write_trace(first, "a", "Sour, thin", "Sour, thin body.")
        write_trace(second, "a", "tasted bad", "Sour, thin body.")

        assert main([str(first), str(second)]) == 0
        out = capsys.readouterr().out
        assert "r1" in out and "r2" in out
        assert "verbatim" in out and "unrelated" in out
        # The examples come from the last run given, and quote both sides.
        assert "tasted bad" in out and "Sour, thin body." in out

    def test_a_directory_with_no_traces_fails_loudly(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert main([str(empty)]) == 1
        assert "no classify-*.json traces" in capsys.readouterr().err

    def test_a_missing_directory_fails_loudly(self, tmp_path, capsys):
        assert main([str(tmp_path / "absent")]) == 2
        assert "not a directory" in capsys.readouterr().err

    def test_examples_can_be_suppressed(self, tmp_path, capsys):
        run = tmp_path / "run"
        write_trace(run, "a", "tasted bad", "Sour, thin body.")
        main([str(run), "--examples", "0"])
        assert "Examples from" not in capsys.readouterr().out

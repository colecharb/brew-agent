"""The cached note labeller, driven by a scripted client. No network."""

import json
import threading
from dataclasses import dataclass, field

import pytest

from brew_agent.config import ModelConfig
from brew_agent.eval.labels import (
    LABEL_TOOL,
    NoteLabel,
    label_brews,
    load_cache,
    quote_spans,
    save_cache,
)
from brew_agent.eval.pairs import REDACT, build_pairs, redact_leaks
from brew_agent.models import Brew


@dataclass
class ToolUseBlock:
    name: str
    input: dict
    id: str = "toolu_1"
    type: str = "tool_use"


@dataclass
class Usage:
    input_tokens: int = 10
    output_tokens: int = 5


@dataclass
class FakeResponse:
    content: list
    stop_reason: str = "tool_use"
    usage: Usage = field(default_factory=Usage)


class FakeMessages:
    def __init__(self, label: dict, explode: bool = False) -> None:
        self._label = label
        self._explode = explode
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._explode:
            raise RuntimeError("503 overloaded")
        return FakeResponse([ToolUseBlock(LABEL_TOOL, self._label)])


class FakeClient:
    def __init__(self, label: dict, explode: bool = False) -> None:
        self.messages = FakeMessages(label, explode)


CONFIG = ModelConfig(
    api_key="test", model="claude-opus-5", effort=None, max_iterations=6
)

CLEAN = {"states_adjustment": False, "adjustment_quotes": [], "has_complaint": True}


def note_brew(bid: str, notes: str) -> Brew:
    return Brew(
        id=bid,
        created_by="user-1",
        brew_timestamp="2026-01-01T00:00:00+00:00",
        profile_coffee_id="bag-1",
        coffee_id="coffee-1",
        notes=notes,
    )


class TestQuoteSpans:
    def test_locates_a_verbatim_quote(self):
        notes = "Juicy and sweet. Might push finer next round."
        label = NoteLabel(True, ["Might push finer next round."], True)
        (start, end), = quote_spans(notes, label)
        assert notes[start:end] == "Might push finer next round."

    def test_skips_a_quote_it_cannot_find(self):
        """A model may normalise whitespace or a curly apostrophe."""
        label = NoteLabel(True, ["a phrase that isn't in the note"], True)
        assert quote_spans("Juicy and sweet.", label) == []

    def test_ignores_blank_quotes(self):
        assert quote_spans("Juicy.", NoteLabel(True, ["", "   "], True)) == []


class TestCache:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "notes.json"
        labels = {"brew-1": NoteLabel(True, ["push finer"], True)}
        save_cache(path, labels)
        assert load_cache(path) == labels

    def test_missing_cache_is_empty(self, tmp_path):
        assert load_cache(tmp_path / "absent.json") == {}

    def test_holds_no_note_text(self, tmp_path):
        """Only ids, booleans and quotes — the corpus stays out of the repo."""
        path = tmp_path / "notes.json"
        save_cache(path, {"brew-1": NoteLabel(False, [], True)})
        payload = json.loads(path.read_text())
        assert set(payload["brew-1"]) == {
            "states_adjustment",
            "adjustment_quotes",
            "has_complaint",
        }


class TestLabelBrews:
    def test_labels_each_note_once_and_caches(self, tmp_path):
        path = tmp_path / "notes.json"
        client = FakeClient(CLEAN)
        brews = [note_brew("a", "Sour and thin"), note_brew("b", "Bitter")]

        first = label_brews(client, CONFIG, brews, path, progress=False)
        assert len(first) == 2
        assert len(client.messages.calls) == 2

        # Second run is free — that is the point of the cache.
        second = label_brews(FakeClient(CLEAN), CONFIG, brews, path, progress=False)
        assert second == first

    def test_skips_empty_notes(self, tmp_path):
        client = FakeClient(CLEAN)
        label_brews(
            client, CONFIG, [note_brew("a", "   ")], tmp_path / "n.json", progress=False
        )
        assert client.messages.calls == []

    def test_only_the_note_is_sent(self, tmp_path):
        client = FakeClient(CLEAN)
        label_brews(
            client, CONFIG, [note_brew("a", "Sour and thin")],
            tmp_path / "n.json", progress=False,
        )
        sent = client.messages.calls[0]
        assert sent["messages"][0]["content"] == "Sour and thin"
        assert sent["tool_choice"] == {"type": "tool", "name": LABEL_TOOL}

    def test_a_failed_call_fails_closed(self, tmp_path):
        """A labelling error must not quietly mark a note clean."""
        client = FakeClient(CLEAN, explode=True)
        labels = label_brews(
            client, CONFIG, [note_brew("a", "Sour")], tmp_path / "n.json", progress=False
        )
        assert labels["a"].states_adjustment is True


class PerNoteClient:
    """Answers per note, and can be told to fail on one of them."""

    class Messages:
        def __init__(self, outer):
            self._outer = outer
            self.calls: list[dict] = []
            self._lock = threading.Lock()

        def create(self, **kwargs):
            note = kwargs["messages"][0]["content"]
            with self._lock:
                self.calls.append(kwargs)
            if note == self._outer.fails_on:
                raise RuntimeError("503 overloaded")
            return FakeResponse(
                [
                    ToolUseBlock(
                        LABEL_TOOL,
                        {
                            "states_adjustment": note.startswith("leak"),
                            "adjustment_quotes": [],
                            "has_complaint": True,
                        },
                    )
                ]
            )

    def __init__(self, fails_on: str | None = None):
        self.fails_on = fails_on
        self.messages = self.Messages(self)


class TestConcurrency:
    """Labelling runs in parallel; correctness must not depend on that."""

    @staticmethod
    def _corpus(n=40):
        return [
            note_brew(f"b{i}", ("leak note " if i % 3 == 0 else "taste note ") + str(i))
            for i in range(n)
        ]

    def test_every_note_is_labelled_exactly_once(self, tmp_path):
        brews = self._corpus()
        client = PerNoteClient()
        labels = label_brews(
            client, CONFIG, brews, tmp_path / "n.json", progress=False, concurrency=8
        )
        assert len(labels) == len(brews)
        assert len(client.messages.calls) == len(brews)

    def test_each_label_lands_against_its_own_note(self, tmp_path):
        """The real risk of parallelism: answers attached to the wrong brew."""
        brews = self._corpus()
        labels = label_brews(
            PerNoteClient(), CONFIG, brews, tmp_path / "n.json",
            progress=False, concurrency=8,
        )
        for brew in brews:
            expected = brew.notes.startswith("leak")
            assert labels[brew.id].states_adjustment is expected, brew.id

    def test_one_failure_does_not_lose_the_rest(self, tmp_path):
        brews = self._corpus()
        doomed = brews[7]
        labels = label_brews(
            PerNoteClient(fails_on=doomed.notes), CONFIG, brews,
            tmp_path / "n.json", progress=False, concurrency=8,
        )
        assert len(labels) == len(brews)
        # Fails closed, and the other 39 are unaffected.
        assert labels[doomed.id].states_adjustment is True
        assert labels[brews[8].id].states_adjustment is brews[8].notes.startswith("leak")

    def test_the_cache_survives_the_parallel_run(self, tmp_path):
        path = tmp_path / "n.json"
        brews = self._corpus()
        label_brews(PerNoteClient(), CONFIG, brews, path, progress=False, concurrency=8)

        # A second pass makes no calls at all.
        client = PerNoteClient()
        again = label_brews(client, CONFIG, brews, path, progress=False, concurrency=8)
        assert client.messages.calls == []
        assert len(again) == len(brews)

    def test_serial_and_parallel_agree(self, tmp_path):
        brews = self._corpus(12)
        serial = label_brews(
            PerNoteClient(), CONFIG, brews, tmp_path / "s.json",
            progress=False, concurrency=1,
        )
        parallel = label_brews(
            PerNoteClient(), CONFIG, brews, tmp_path / "p.json",
            progress=False, concurrency=8,
        )
        assert serial == parallel


class TestIntegrationWithPairs:
    def test_labeller_spans_redact_what_the_regex_missed(self):
        """The two detectors compose rather than competing."""
        notes = "Lovely and sweet. I reckon the burrs want opening up a hair."
        # The regex has no idea about "want opening up a hair".
        assert redact_leaks(notes)[1] == []

        label = NoteLabel(True, ["I reckon the burrs want opening up a hair."], True)
        kept, removed = redact_leaks(notes, quote_spans(notes, label))
        assert kept == "Lovely and sweet."
        assert removed == ["I reckon the burrs want opening up a hair."]

    def test_labels_are_optional(self, seed_brews):
        """Without labels, results are exactly what they were before."""
        unlabelled, stats = build_pairs(seed_brews, leak_mode=REDACT)
        assert stats.labelled is False
        assert stats.eligible == 367
        assert all(p.diagnosable is None for p in unlabelled)

    def test_labels_can_only_add_leaks(self, seed_brews):
        """A label marking a note clean never un-redacts a regex match."""
        clean_labels = {b.id: NoteLabel(False, [], True) for b in seed_brews}
        _, stats = build_pairs(seed_brews, leak_mode=REDACT, labels=clean_labels)
        assert stats.leaky_detected == stats.leaky_regex_only == 105
        assert stats.leaky_labeller_added == 0

    def test_labeller_additions_are_counted(self, seed_brews):
        everything_leaks = {b.id: NoteLabel(True, [], True) for b in seed_brews}
        _, stats = build_pairs(seed_brews, leak_mode=REDACT, labels=everything_leaks)
        assert stats.leaky_detected > stats.leaky_regex_only
        assert stats.leaky_labeller_added == stats.leaky_detected - 105

    def test_diagnosable_flows_through(self, seed_brews):
        labels = {b.id: NoteLabel(False, [], b.rating == 0) for b in seed_brews}
        pairs, _ = build_pairs(seed_brews, leak_mode=REDACT, labels=labels)
        assert {p.diagnosable for p in pairs} == {True, False}
        for pair in pairs:
            assert pair.diagnosable == (pair.before.rating == 0)

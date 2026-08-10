"""Cached model labelling of tasting notes.

The regex in `pairs.py` catches the phrasings we found by reading the corpus.
Paraphrase will always outrun it, and without a second opinion there is no way
to say what the residual leak rate is — which matters, because a missed leak
helps only the arms that can read prose. That asymmetry is the whole reason this
exists.

Two labels per note:

- `states_adjustment` — does it name a change to make, in either grammar? The
  plan ("might push finer") and the verdict ("too coarse") both count. Character
  spans come back too, since redaction needs them.
- `has_complaint` — does it describe a taste problem at all? Many notes are
  *"Yes."* or *"For Clemi's latte"*; scoring an arm on those measures nothing,
  so `run.py` reports a second row restricted to the ones that do.

Labels are cached to `labels/notes.json` and keyed by brew id, so the eval stays
deterministic and a rerun is free. The cache holds ids, booleans, and offsets
only — never note text — so no user data lands on disk here beyond what
`traces/` already holds, and it is gitignored regardless.

This is shared infrastructure, not an arm: the same labels gate every arm
identically.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..baselines import call_model
from ..config import ModelConfig
from ..models import Brew

LABEL_TOOL = "label_note"

# Notes are independent, so labelling is embarrassingly parallel. Eight in
# flight turns ~30 minutes of sequential calls into a few minutes. Lower it with
# BREW_AGENT_LABEL_CONCURRENCY if your account's rate limits complain — the SDK
# retries 429s on its own, but there is no point provoking them.
DEFAULT_CONCURRENCY = int(os.environ.get("BREW_AGENT_LABEL_CONCURRENCY", "8"))

LABEL_SCHEMA: dict[str, Any] = {
    "name": LABEL_TOOL,
    "description": "Record what a coffee tasting note contains.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "states_adjustment": {
                "type": "boolean",
                "description": (
                    "True if the note names a change to a brewing parameter — "
                    "grind, temperature, ratio, dose, or time. Both grammars "
                    "count: a plan ('might push finer', 'needs to be coarser') "
                    "and a verdict on the brew in hand ('too fine', 'wasn't "
                    "grinding coarse enough'). False for notes that only "
                    "describe taste, however strongly ('want more body', "
                    "'sour and thin'), and for remarks about equipment upkeep "
                    "such as cleaning or purging the grinder."
                ),
            },
            "adjustment_quotes": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exact substrings of the note that state the adjustment, "
                    "copied verbatim so they can be located. Empty when "
                    "states_adjustment is false."
                ),
            },
            "has_complaint": {
                "type": "boolean",
                "description": (
                    "True if the note describes something wrong or wanted "
                    "different about how the coffee tasted. False for purely "
                    "positive notes, contentless ones ('Yes.'), and notes about "
                    "circumstance rather than flavour ('for Clemi's latte')."
                ),
            },
        },
        "required": ["states_adjustment", "adjustment_quotes", "has_complaint"],
        "additionalProperties": False,
    },
}

SYSTEM = """You label coffee tasting notes for an evaluation harness.

The harness asks a model to diagnose a brew from its tasting note. A note that \
already names the adjustment gives the answer away, so those spans get removed \
before anything reads the note. Your job is to find them.

The distinction that matters: naming a taste outcome is fine, naming a \
parameter change is not. "Wants more body" describes the problem. "Needs to be \
finer" is the answer. Grade the note, not the coffee, and quote spans exactly as \
they appear."""


@dataclass
class NoteLabel:
    states_adjustment: bool = False
    adjustment_quotes: list[str] = field(default_factory=list)
    has_complaint: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoteLabel":
        return cls(
            states_adjustment=bool(data.get("states_adjustment")),
            adjustment_quotes=[str(q) for q in data.get("adjustment_quotes") or []],
            has_complaint=bool(data.get("has_complaint")),
        )


def load_cache(path: Path) -> dict[str, NoteLabel]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {key: NoteLabel.from_dict(value) for key, value in raw.items()}


def save_cache(path: Path, labels: dict[str, NoteLabel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({k: asdict(v) for k, v in sorted(labels.items())}, indent=2)
    )


def label_brews(
    client: Any,
    config: ModelConfig,
    brews: Iterable[Brew],
    cache_path: Path,
    progress: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, NoteLabel]:
    """Label every note not already in the cache, then persist and return all.

    Notes are independent, so they go out concurrently — sequentially this is
    around half an hour for a corpus this size, and a few minutes in parallel.

    Concurrency rather than a cheaper or shallower call is the deliberate
    choice. Effort stays where the other components have it: this is the one
    piece whose mistakes are *not* symmetric across arms, since a leak the
    labeller misses only helps the arms that can read prose. Parallelism costs
    nothing in quality; trimming the thinking might.
    """
    labels = load_cache(cache_path)
    pending = [b for b in brews if b.notes.strip() and b.id not in labels]
    if not pending:
        return labels

    lock = threading.Lock()
    completed = 0

    def label_one(brew: Brew) -> tuple[str, NoteLabel]:
        # `_label_one` already fails closed; this is the backstop that keeps a
        # single unexpected error from aborting a 700-note run.
        try:
            return brew.id, _label_one(client, config, brew.notes)
        except Exception:
            return brew.id, NoteLabel(states_adjustment=True, has_complaint=True)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for future in as_completed(pool.submit(label_one, b) for b in pending):
            brew_id, label = future.result()
            with lock:
                labels[brew_id] = label
                completed += 1
                # Persist as we go: an interrupted run should not throw away
                # the answers it already paid for.
                if completed % 25 == 0:
                    save_cache(cache_path, labels)
                if progress:
                    print(
                        f"  labelling {completed}/{len(pending)}", end="\r", flush=True
                    )

    save_cache(cache_path, labels)
    if progress:
        print(f"  labelled {len(pending)} new note(s); {len(labels)} cached")
    return labels


def _label_one(client: Any, config: ModelConfig, notes: str) -> NoteLabel:
    try:
        response = call_model(
            client,
            config,
            system=SYSTEM,
            messages=[{"role": "user", "content": notes}],
            tools=[LABEL_SCHEMA],
            force_tool=LABEL_TOOL,
        )
    except Exception:
        # A labelling failure must not silently mark a note clean — that would
        # let a leak through. Treat it as leaking, with no quotes, so the
        # regex-matched sentences are still redacted and nothing extra is
        # admitted on the strength of a failed call.
        return NoteLabel(states_adjustment=True, has_complaint=True)

    for block in response.content:
        if block.type == "tool_use" and block.name == LABEL_TOOL:
            return NoteLabel.from_dict(block.input)
    return NoteLabel(states_adjustment=True, has_complaint=True)


def quote_spans(notes: str, label: NoteLabel) -> list[tuple[int, int]]:
    """Locate each quoted adjustment inside the note.

    Quotes come back verbatim in the good case, but a model may normalise
    whitespace or a curly apostrophe, so a quote that cannot be found is
    skipped rather than guessed at — the regex still covers those sentences.
    """
    spans = []
    for quote in label.adjustment_quotes:
        needle = quote.strip()
        if not needle:
            continue
        start = notes.find(needle)
        if start >= 0:
            spans.append((start, start + len(needle)))
    return spans

"""Why the classify arm's quoted evidence so often isn't in the note.

Each trace records `evidence_verbatim`: whether the quote the model returned is
a substring of the note it read. On real runs that is false about half the time
(53/100 on Sonnet 5, 67/100 on Haiku 4.5, same notes), which is far too common
to be the malformed-token corruption that prompted the check.

The flag says *whether*, never *why*, and the two plausible causes want opposite
fixes. If the model is quoting accurately and the check is stricter than the
field deserves — punctuation, a curly apostrophe, a reflowed line — the check
should loosen. If the model is summarising the note in its own words, the check
is right and either the tool description or the expectation should change. In
the results table those look identical.

    python -m brew_agent.eval.audit_evidence traces/<run_id> [traces/<other>...]

Offline: reads traces already on disk. No API key, no database. Pass more than
one run to compare models over the same pairs side by side.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from ..baselines import evidence_is_verbatim

_PUNCTUATION = re.compile(r"[^\w\s]")

# Ordered from "my check is too strict" to "the model made it up". Each bucket
# implies a different fix, which is the whole point of splitting them.
BUCKETS = (
    "empty",
    "verbatim",
    "punctuation",
    "reordered",
    "paraphrase",
    "unrelated",
)

WHAT_IT_MEANS = {
    "empty": "nothing quoted — expected when the verdict is `neither`",
    "verbatim": "a real substring of the note; the check passes",
    "punctuation": "matches once punctuation and case are normalised — the check is too strict",
    "reordered": "every word is in the note, but not as one span — the model stitched spans together",
    "paraphrase": "half the words or more are in the note — summarised rather than quoted",
    "unrelated": "little overlap with the note — fabricated, or corrupted output",
}


def _depunctuate(text: str) -> str:
    return " ".join(_PUNCTUATION.sub(" ", text).split()).lower()


def bucket(evidence: str, note: str) -> str:
    """Classify how a piece of quoted evidence relates to the note."""
    if not evidence.strip():
        return "empty"
    # Deliberately the same predicate the trace flag uses, so the audit can
    # never disagree with the number it is explaining.
    if evidence_is_verbatim(evidence, note):
        return "verbatim"
    words = _depunctuate(evidence).split()
    if not words:
        # Punctuation only, e.g. a stray bracket from a malformed generation.
        # Checked before the substring test below, which an empty string would
        # otherwise satisfy against any note at all.
        return "unrelated"
    if _depunctuate(evidence) in _depunctuate(note):
        return "punctuation"

    in_note = set(_depunctuate(note).split())
    present = sum(1 for word in words if word in in_note)
    if present == len(words):
        return "reordered"
    if present * 2 >= len(words):
        return "paraphrase"
    return "unrelated"


def read_traces(trace_dir: Path, arm: str) -> Iterator[dict]:
    for path in sorted(trace_dir.glob(f"{arm}-*.json")):
        yield json.loads(path.read_text())


def audit(trace_dir: Path, arm: str) -> tuple[Counter, dict[str, list[tuple[str, str]]]]:
    counts: Counter = Counter()
    examples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for payload in read_traces(trace_dir, arm):
        evidence = payload.get("trace", {}).get("evidence") or ""
        note = payload.get("input", {}).get("complaint") or ""
        name = bucket(evidence, note)
        counts[name] += 1
        examples[name].append((evidence, note))
    return counts, examples


def _bar(count: int, total: int, width: int = 24) -> str:
    filled = 0 if not total else round(width * count / total)
    return "#" * filled + "." * (width - filled)


def _shorten(text: str, limit: int = 88) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def report(runs: list[tuple[str, Counter]], width: int = 12) -> None:
    header = f"{'bucket':<13}" + "".join(f"{name:>{width}}" for name, _ in runs)
    print(f"\n{header}")
    print("-" * len(header))
    for name in BUCKETS:
        cells = ""
        for _, counts in runs:
            total = sum(counts.values())
            pct = "" if not total else f" {counts[name] / total:.0%}"
            cells += f"{str(counts[name]) + pct:>{width}}"
        print(f"{name:<13}{cells}")
    print("-" * len(header))
    cells = "".join(f"{sum(counts.values()):>{width}}" for _, counts in runs)
    print(f"{'total':<13}{cells}")

    print("\nwhat each bucket means")
    for name in BUCKETS:
        print(f"  {name:<13} {WHAT_IT_MEANS[name]}")


def show_examples(examples: dict, per_bucket: int) -> None:
    for name in BUCKETS:
        rows = examples.get(name) or []
        if name in ("verbatim", "empty") or not rows:
            continue
        print(f"\n--- {name} ({len(rows)}) " + "-" * 40)
        for evidence, note in rows[:per_bucket]:
            print(f'  quoted: "{_shorten(evidence)}"')
            print(f'  note:   "{_shorten(note)}"\n')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dirs", nargs="+", type=Path)
    parser.add_argument("--arm", default="classify", help="arm to audit")
    parser.add_argument(
        "--examples", type=int, default=4, help="examples per non-verbatim bucket"
    )
    args = parser.parse_args(argv)

    runs = []
    last_examples: dict = {}
    for trace_dir in args.trace_dirs:
        if not trace_dir.is_dir():
            print(f"error: {trace_dir} is not a directory", file=sys.stderr)
            return 2
        counts, examples = audit(trace_dir, args.arm)
        if not sum(counts.values()):
            print(
                f"error: no {args.arm}-*.json traces in {trace_dir}", file=sys.stderr
            )
            return 1
        runs.append((trace_dir.name, counts))
        last_examples = examples

    report(runs)
    if args.examples:
        print(f"\nExamples from {runs[-1][0]}:")
        show_examples(last_examples, args.examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The one command: run every arm over the same holdout pairs and print a number.

    python -m brew_agent.eval.run --n 24

Writes `evals/output/<run_id>.json` and one trace per (arm, pair) under
`traces/<run_id>/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from ..baselines import ArmResult, ClassifyBaseline, NoToolsBaseline, run_rules
from ..config import LABEL_CACHE, OUTPUT_DIR, TRACE_DIR, ConfigError, ModelConfig
from ..models import HoldoutPair
from ..tools import Toolbox
from .pairs import EXCLUDE, RAW, REDACT, PairStats, build_pairs, stratified_sample
from .scoring import ArmScore, PairScore, aggregate, score_pair

# Ordered as a ladder: each rung adds one capability to the one before it.
ARMS = ("rules", "classify", "no_tools", "agent")
NEEDS_API_KEY = ("classify", "no_tools", "agent")


class SupportsBrewReads(Protocol):
    """What the runner needs from a database. Kept narrow so tests can fake it."""

    user_id: str

    def fetch_all_brews(self, max_rows: int = ...) -> list: ...


@dataclass
class Runner:
    name: str
    run: Callable[[HoldoutPair], ArmResult]


def build_runners(names: list[str], db: SupportsBrewReads) -> list[Runner]:
    """Wire up the requested arms, constructing an API client only if needed."""
    runners: list[Runner] = []
    if "rules" in names:
        runners.append(Runner("rules", lambda p: run_rules(p.before, p.complaint)))

    if not any(name in NEEDS_API_KEY for name in names):
        return runners

    import anthropic

    config = ModelConfig.from_env()
    client = anthropic.Anthropic(api_key=config.api_key)
    print(f"model: {config.model} (effort={config.effort or 'default'})")

    if "classify" in names:
        classifier = ClassifyBaseline(client, config)
        runners.append(
            Runner("classify", lambda p: classifier.run(p.before, p.complaint))
        )
    if "no_tools" in names:
        baseline = NoToolsBaseline(client, config)
        runners.append(
            Runner("no_tools", lambda p: baseline.run(p.before, p.complaint))
        )
    if "agent" in names:
        from ..agent import BrewAgent

        agent = BrewAgent(client, config, Toolbox(db))
        runners.append(Runner("agent", lambda p: agent.run(p.before, p.complaint)))

    # Preserve the order the caller asked for.
    return sorted(runners, key=lambda r: names.index(r.name))


def run_eval(
    db: SupportsBrewReads,
    names: list[str],
    n: int,
    leak_mode: str = REDACT,
    labels: Mapping[str, Any] | None = None,
    trace_root: Path | None = None,
    output_root: Path | None = None,
) -> dict:
    """Fetch, pair, sample, run every arm, score, and write the artefacts."""
    runners = build_runners(names, db)
    brews = db.fetch_all_brews()
    pairs, stats = build_pairs(brews, leak_mode=leak_mode, labels=labels)
    sample = stratified_sample(pairs, n, stats)
    if not sample:
        raise RuntimeError("no eligible holdout pairs found")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_dir = (trace_root or TRACE_DIR) / run_id
    output_dir = output_root or OUTPUT_DIR
    trace_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_arm: dict[str, list[PairScore]] = {r.name: [] for r in runners}
    # Arms that quote the note back are checked against it. Nothing here changes
    # a score — it is the trace's honesty that is being counted.
    odd_evidence: dict[str, int] = {}
    for index, pair in enumerate(sample, start=1):
        print(f"[{index}/{len(sample)}] {pair.id}", end="", flush=True)
        for runner in runners:
            result = runner.run(pair)
            score = score_pair(pair, result.recommendation)
            per_arm[runner.name].append(score)
            if result.trace.get("evidence_verbatim") is False:
                odd_evidence[runner.name] = odd_evidence.get(runner.name, 0) + 1
            _write_trace(trace_dir, runner.name, pair, result, score)
            print(f"  {runner.name}={score.grind}", end="", flush=True)
        print()

    scores = {name: aggregate(name, s) for name, s in per_arm.items()}
    # Many notes are "Yes." or "For Clemi's latte" — the user moved the grind
    # for reasons never written down, and no arm can be right about those.
    # Scoring them pulls every arm toward the same middle, so where the labeller
    # has told us which notes describe a taste problem, report that subset too.
    diagnosable = {
        name: aggregate(name, [s for s in per_arm[name] if s.diagnosable])
        for name in per_arm
        if any(s.diagnosable is not None for s in per_arm[name])
    }
    out_path = output_dir / f"{run_id}.json"
    out_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "sampled": len(sample),
                "leak_mode": leak_mode,
                "funnel": dict(stats.__dict__),
                "arms": {name: arm.to_dict() for name, arm in scores.items()},
                "arms_diagnosable_only": {
                    name: arm.to_dict() for name, arm in diagnosable.items()
                },
                "evidence_not_verbatim": odd_evidence,
                "pairs": [
                    {"pair_id": p.id, "user_id": p.user_id, "leaky": p.leaky}
                    for p in sample
                ],
            },
            indent=2,
            default=str,
        )
    )
    return {
        "run_id": run_id,
        "stats": stats,
        "scores": scores,
        "diagnosable": diagnosable,
        "evidence_not_verbatim": odd_evidence,
        "output_path": out_path,
        "trace_dir": trace_dir,
    }


def _pct(value: float | None) -> str:
    return "  -  " if value is None else f"{value:5.0%}"


def _fraction(correct: int, total: int) -> str:
    return f"{correct}/{total}" if total else "  -  "


def _print_table(title: str, scores: Mapping[str, ArmScore]) -> None:
    header = (
        f"{'arm':<10} {'n':>4} | {'ok':>4} {'wrong':>6} {'quiet':>6} {'dir':>6} "
        f"| {'magnitude':>13} | {'when improved':>15} | {'held':>5} {'err':>4}"
    )
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for name, arm in scores.items():
        print(
            f"{name:<10} {arm.n:>4} | "
            f"{arm.grind.correct:>4} {arm.grind.wrong:>6} {arm.grind.abstained:>6} "
            f"{_pct(arm.grind.accuracy)} | "
            f"{_fraction(arm.magnitude_hits, arm.magnitude_considered):>7} "
            f"{_pct(arm.magnitude_rate)} | "
            f"{_fraction(arm.grind_when_improved.correct, arm.grind_when_improved.considered):>9} "
            f"{_pct(arm.headline)} | "
            f"{arm.recommended_nothing:>5} {arm.errors:>4}"
        )


def print_report(result: dict) -> None:
    stats: PairStats = result["stats"]
    scores: dict[str, ArmScore] = result["scores"]

    print("\nPair selection")
    for line in stats.funnel_lines():
        print(line)

    _print_table("All sampled pairs", scores)
    if result.get("diagnosable"):
        _print_table(
            "Pairs whose note describes a taste problem", result["diagnosable"]
        )

    print(
        "\nok / wrong / quiet   over the pairs where the user moved the grind: "
        "moved it the same way,"
    )
    print(
        "                     moved it the opposite way, or recommended no "
        "grind change at all."
    )
    print("dir                  ok / (ok + wrong + quiet). Staying quiet counts as a miss.")
    print(
        "magnitude            of the 'ok' pairs, how often the size was within "
        "0.5x-2x of the user's."
    )
    print(
        "when improved        the same direction rate, restricted to pairs "
        "where the user's own"
    )
    print("                     change raised the rating. This is the headline number.")
    print("held                 recommended no change to anything.")

    for name, count in (result.get("evidence_not_verbatim") or {}).items():
        print(
            f"\nwarning: {name} quoted evidence absent from the note on "
            f"{count} pair(s). Scores are unaffected — the verdict is what gets "
            f"used — but those traces do not explain themselves."
        )

    print(f"\nrun {result['run_id']}")
    print(f"  results {result['output_path']}")
    print(f"  traces  {result['trace_dir']}")


def _write_trace(
    trace_dir: Path,
    arm: str,
    pair: HoldoutPair,
    result: ArmResult,
    score: PairScore,
) -> None:
    """Full trace: what was asked, what came back, and how it scored.

    The held-out brew is included so a trace reads end to end without
    cross-referencing the results file.
    """
    payload = {
        "arm": arm,
        "pair_id": pair.id,
        "user_id": pair.user_id,
        "leaky": pair.leaky,
        "leak_phrase": pair.leak_phrase,
        "input": {
            "brew_id": pair.before.id,
            # What the arm actually read, plus what was withheld, so a redaction
            # can be judged by eye rather than taken on trust. The two together
            # reconstruct the original note, so it is not repeated here — and
            # not repeating it keeps `pair.before.notes` unreachable outside
            # pairs.py, which is what the bypass guard enforces.
            "complaint": pair.complaint,
            "redacted_out": pair.redacted,
            "leak_phrase": pair.leak_phrase,
            "brew": pair.before.to_tool_result(),
        },
        "held_out_next_brew": pair.after.to_tool_result(),
        "recommendation": result.recommendation.to_dict(),
        "score": score.to_dict(),
        "trace": result.trace,
    }
    (trace_dir / f"{arm}-{pair.id}.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )


def _label_notes(db: SupportsBrewReads) -> Mapping[str, Any]:
    """Run (or reuse) the cached model labelling pass over every note."""
    import anthropic

    from .labels import label_brews

    config = ModelConfig.from_env()
    client = anthropic.Anthropic(api_key=config.api_key)
    print(f"labelling notes with {config.model} (cached in {LABEL_CACHE})")
    return label_brews(client, config, db.fetch_all_brews(), LABEL_CACHE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=24, help="pairs to sample")
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"comma-separated subset of {','.join(ARMS)}",
    )
    leak = parser.add_mutually_exclusive_group()
    leak.add_argument(
        "--exclude-leaky",
        dest="leak_mode",
        action="store_const",
        const=EXCLUDE,
        help=(
            "drop pairs whose notes state an adjustment instead of redacting "
            "the sentence. Conservative cross-check: if the headline moves much "
            "against the default, redaction is leaving hints."
        ),
    )
    leak.add_argument(
        "--include-leaky",
        dest="leak_mode",
        action="store_const",
        const=RAW,
        help=(
            "leave stated adjustments in. Not a measurement — a harness "
            "self-test. Any arm that reads the notes should approach 100%% "
            "here, and one that doesn't has a bug. The gap against the default "
            "run is the contamination redaction removes."
        ),
    )
    parser.set_defaults(leak_mode=REDACT)
    parser.add_argument(
        "--label",
        action="store_true",
        help=(
            "run the model labelling pass over the notes before pairing, "
            "catching stated adjustments the regex misses and marking which "
            "notes contain a taste complaint at all. Cached to labels/, so it "
            "costs nothing on a rerun."
        ),
    )
    args = parser.parse_args(argv)

    seen: list[str] = []
    for name in (a.strip() for a in args.arms.split(",")):
        if name and name not in seen:
            seen.append(name)
    unknown = [n for n in seen if n not in ARMS]
    if unknown:
        parser.error(f"unknown arm(s): {', '.join(unknown)}")

    from ..db import BrewDatabase

    try:
        db = BrewDatabase.connect()
        print(f"signed in as {db.user_id}")
        labels = _label_notes(db) if args.label else None
        result = run_eval(db, seen, args.n, leak_mode=args.leak_mode, labels=labels)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

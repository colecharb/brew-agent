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
from typing import Callable, Protocol

from ..baselines import ArmResult, NoToolsBaseline, run_rules
from ..config import OUTPUT_DIR, TRACE_DIR, ConfigError, ModelConfig
from ..models import HoldoutPair
from ..tools import Toolbox
from .pairs import PairStats, build_pairs, stratified_sample
from .scoring import ArmScore, PairScore, aggregate, score_pair

ARMS = ("rules", "no_tools", "agent")
NEEDS_API_KEY = ("no_tools", "agent")


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
        runners.append(Runner("rules", lambda p: run_rules(p.before, p.before.notes)))

    if not any(name in NEEDS_API_KEY for name in names):
        return runners

    import anthropic

    config = ModelConfig.from_env()
    client = anthropic.Anthropic(api_key=config.api_key)
    print(f"model: {config.model} (effort={config.effort or 'default'})")

    if "no_tools" in names:
        baseline = NoToolsBaseline(client, config)
        runners.append(
            Runner("no_tools", lambda p: baseline.run(p.before, p.before.notes))
        )
    if "agent" in names:
        from ..agent import BrewAgent

        agent = BrewAgent(client, config, Toolbox(db))
        runners.append(Runner("agent", lambda p: agent.run(p.before, p.before.notes)))

    # Preserve the order the caller asked for.
    return sorted(runners, key=lambda r: names.index(r.name))


def run_eval(
    db: SupportsBrewReads,
    names: list[str],
    n: int,
    include_leaky: bool = False,
    trace_root: Path | None = None,
    output_root: Path | None = None,
) -> dict:
    """Fetch, pair, sample, run every arm, score, and write the artefacts."""
    runners = build_runners(names, db)
    brews = db.fetch_all_brews()
    pairs, stats = build_pairs(brews, include_leaky=include_leaky)
    sample = stratified_sample(pairs, n, stats)
    if not sample:
        raise RuntimeError("no eligible holdout pairs found")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_dir = (trace_root or TRACE_DIR) / run_id
    output_dir = output_root or OUTPUT_DIR
    trace_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_arm: dict[str, list[PairScore]] = {r.name: [] for r in runners}
    for index, pair in enumerate(sample, start=1):
        print(f"[{index}/{len(sample)}] {pair.id}", end="", flush=True)
        for runner in runners:
            result = runner.run(pair)
            score = score_pair(pair, result.recommendation)
            per_arm[runner.name].append(score)
            _write_trace(trace_dir, runner.name, pair, result, score)
            print(f"  {runner.name}={score.grind}", end="", flush=True)
        print()

    scores = {name: aggregate(name, s) for name, s in per_arm.items()}
    out_path = output_dir / f"{run_id}.json"
    out_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "sampled": len(sample),
                "include_leaky": include_leaky,
                "funnel": dict(stats.__dict__),
                "arms": {name: arm.to_dict() for name, arm in scores.items()},
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
        "output_path": out_path,
        "trace_dir": trace_dir,
    }


def _pct(value: float | None) -> str:
    return "  -  " if value is None else f"{value:5.0%}"


def _fraction(correct: int, total: int) -> str:
    return f"{correct}/{total}" if total else "  -  "


def print_report(result: dict) -> None:
    stats: PairStats = result["stats"]
    scores: dict[str, ArmScore] = result["scores"]

    print("\nPair selection")
    for line in stats.funnel_lines():
        print(line)

    header = (
        f"\n{'arm':<10} {'n':>4} | {'ok':>4} {'wrong':>6} {'quiet':>6} {'dir':>6} "
        f"| {'magnitude':>13} | {'when improved':>15} | {'held':>5} {'err':>4}"
    )
    print(header)
    print("-" * (len(header) - 1))
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
            "complaint": pair.before.notes,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=24, help="pairs to sample")
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"comma-separated subset of {','.join(ARMS)}",
    )
    parser.add_argument(
        "--include-leaky",
        action="store_true",
        help=(
            "keep pairs whose notes already state the next adjustment. Any arm "
            "that reads the notes should approach 100%% on these, which makes "
            "them a harness self-test and not a measurement."
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
        result = run_eval(db, seen, args.n, include_leaky=args.include_leaky)
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

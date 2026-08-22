"""Scoring a recommendation against what the user actually did next.

## Why direction is scored as a delta sign and not as "finer" or "coarser"

Grinders disagree about what a bigger number means, and this dataset has 34 of
them. There is no table saying which way is finer, and a wrong entry in such a
table would not fail loudly — it would silently invert the metric for every pair
using that grinder.

So the agent is told the current setting and must answer with a number on the
same dial. Scoring compares the sign of (recommended - current) against the sign
of (what the user actually set next - current). Matching signs is a hit whether
"down" means finer or coarser on that particular grinder, because both sides are
expressed on the same dial. The convention never has to be known.

## Abstention counts against the agent

If the user moved the grind and the agent recommended no grind change, that is a
miss, not an exemption — it is tracked separately as `abstained` so a
conservative arm is visible rather than flattered.

## Two different things called abstention

That column conflates two behaviours, and the difference matters once arms are
free to choose their own lever. `classify` cannot help but move the grind when it
reaches a verdict — `_apply_step` does it mechanically — so its abstentions are
always genuine silence. `no_tools` and `agent` may answer "raise the temperature
instead", which scores as abstaining on grind despite being a confident opinion.

Read as a capability comparison that quietly penalises the arms whose extra
capability is choosing the lever. The scoring is unchanged — grind direction is
what the eval measures — but `abstained` is now split, so silence and a
different opinion are visible as the different things they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..models import HoldoutPair, Recommendation

# A proposed change counts as reasonable when it lands within half to double
# what the user actually did. Real grind moves in this dataset have a median of
# ~2% of the current setting and a p90 of ~20%, so the band is calibrated to
# observed behaviour rather than picked from the air.
MAGNITUDE_BAND = (0.5, 2.0)

EPSILON = 1e-9

CORRECT = "correct"
WRONG_DIRECTION = "wrong_direction"
ABSTAINED = "abstained"
FALSE_MOVE = "false_move"
CORRECT_HOLD = "correct_hold"
NOT_APPLICABLE = "n/a"


def _sign(value: float) -> int:
    if value > EPSILON:
        return 1
    if value < -EPSILON:
        return -1
    return 0


def direction_outcome(
    before: float | None, after: float | None, recommended: float | None
) -> str:
    """Classify one parameter's recommended direction against the real one."""
    if before is None or after is None:
        return NOT_APPLICABLE

    actual = _sign(after - before)
    proposed = 0 if recommended is None else _sign(recommended - before)

    if actual == 0:
        return FALSE_MOVE if proposed != 0 else CORRECT_HOLD
    if proposed == 0:
        return ABSTAINED
    return CORRECT if proposed == actual else WRONG_DIRECTION


@dataclass
class PairScore:
    """One arm's result on one holdout pair."""

    pair_id: str
    user_id: str
    # From the model labelling pass: does the note describe a taste problem at
    # all? None when the labeller hasn't run.
    diagnosable: bool | None = None
    grind: str = NOT_APPLICABLE
    magnitude_ratio: float | None = None
    magnitude_hit: bool | None = None
    ratio: str = NOT_APPLICABLE
    time: str = NOT_APPLICABLE
    temp: str = NOT_APPLICABLE
    rating_improved: bool | None = None
    recommended_nothing: bool = False
    # What the arm said it was betting on. "none" for arms that proposed
    # nothing; always "grind_setting" or "none" for `rules` and `classify`,
    # which have no other lever to reach for.
    primary_lever: str = "none"
    error: str | None = None

    @property
    def quiet_elsewhere(self) -> bool:
        """Abstained on grind, but did propose some other change.

        The distinction the `abstained` count cannot make on its own: this is an
        arm with an opinion about temperature, not an arm with nothing to say.
        """
        return self.grind == ABSTAINED and not self.recommended_nothing

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "user_id": self.user_id,
            "diagnosable": self.diagnosable,
            "grind": self.grind,
            "magnitude_ratio": self.magnitude_ratio,
            "magnitude_hit": self.magnitude_hit,
            "ratio": self.ratio,
            "time": self.time,
            "temp": self.temp,
            "rating_improved": self.rating_improved,
            "recommended_nothing": self.recommended_nothing,
            "primary_lever": self.primary_lever,
            "quiet_elsewhere": self.quiet_elsewhere,
            "error": self.error,
        }


def score_pair(pair: HoldoutPair, rec: Recommendation) -> PairScore:
    before, after = pair.before, pair.after

    score = PairScore(
        pair_id=pair.id,
        user_id=pair.user_id,
        diagnosable=pair.diagnosable,
        rating_improved=pair.rating_improved,
        recommended_nothing=rec.changes_nothing,
        primary_lever=rec.primary_lever,
        error=rec.error,
    )

    score.grind = direction_outcome(
        before.grind_value, after.grind_value, rec.grind_value
    )
    if score.grind == CORRECT:
        actual_delta = abs(after.grind_value - before.grind_value)  # type: ignore[operator]
        proposed_delta = abs(rec.grind_value - before.grind_value)  # type: ignore[operator]
        if actual_delta > EPSILON:
            score.magnitude_ratio = proposed_delta / actual_delta
            low, high = MAGNITUDE_BAND
            score.magnitude_hit = low <= score.magnitude_ratio <= high

    # Ratio is derived (target / dose), so a recommendation that moves either
    # side counts. Fall back to the brew's own value for the side left alone.
    rec_dose = rec.coffee_weight if rec.coffee_weight is not None else before.coffee_weight
    rec_yield = (
        rec.target_weight if rec.target_weight is not None else before.target_weight
    )
    rec_ratio = rec_yield / rec_dose if rec_dose and rec_yield else None
    score.ratio = direction_outcome(before.ratio, after.ratio, rec_ratio)

    score.time = direction_outcome(
        None if before.time is None else float(before.time),
        None if after.time is None else float(after.time),
        None if rec.time is None else float(rec.time),
    )
    score.temp = direction_outcome(before.water_temp, after.water_temp, rec.water_temp)
    return score


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass
class Metric:
    correct: int = 0
    wrong: int = 0
    abstained: int = 0
    false_moves: int = 0
    correct_holds: int = 0

    @property
    def considered(self) -> int:
        """Pairs where the user actually moved this parameter."""
        return self.correct + self.wrong + self.abstained

    @property
    def accuracy(self) -> float | None:
        return _rate(self.correct, self.considered)

    def add(self, outcome: str) -> None:
        if outcome == CORRECT:
            self.correct += 1
        elif outcome == WRONG_DIRECTION:
            self.wrong += 1
        elif outcome == ABSTAINED:
            self.abstained += 1
        elif outcome == FALSE_MOVE:
            self.false_moves += 1
        elif outcome == CORRECT_HOLD:
            self.correct_holds += 1

    def to_dict(self) -> dict:
        return {
            "considered": self.considered,
            "correct": self.correct,
            "wrong_direction": self.wrong,
            "abstained": self.abstained,
            "false_moves": self.false_moves,
            "correct_holds": self.correct_holds,
            "accuracy": self.accuracy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Metric":
        """Inverse of `to_dict`. `considered` and `accuracy` are derived, not stored."""
        return cls(
            correct=d["correct"],
            wrong=d["wrong_direction"],
            abstained=d["abstained"],
            false_moves=d["false_moves"],
            correct_holds=d["correct_holds"],
        )


@dataclass
class ArmScore:
    """Aggregate result for one arm over the sampled pairs."""

    arm: str
    n: int = 0
    grind: Metric = field(default_factory=Metric)
    grind_when_improved: Metric = field(default_factory=Metric)
    ratio: Metric = field(default_factory=Metric)
    time: Metric = field(default_factory=Metric)
    temp: Metric = field(default_factory=Metric)
    magnitude_considered: int = 0
    magnitude_hits: int = 0
    recommended_nothing: int = 0
    # Of `grind.abstained`, how many proposed a different lever instead. The
    # remainder is genuine silence.
    quiet_elsewhere: int = 0
    levers: dict[str, int] = field(default_factory=dict)
    errors: int = 0

    @property
    def magnitude_rate(self) -> float | None:
        return _rate(self.magnitude_hits, self.magnitude_considered)

    @property
    def headline(self) -> float | None:
        """Direction accuracy restricted to pairs where the rating improved.

        Those are the pairs where the user's own change demonstrably worked, so
        agreeing with it is the strongest available signal.
        """
        return self.grind_when_improved.accuracy

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "n": self.n,
            "grind": self.grind.to_dict(),
            "grind_when_rating_improved": self.grind_when_improved.to_dict(),
            "ratio": self.ratio.to_dict(),
            "time": self.time.to_dict(),
            "temp": self.temp.to_dict(),
            "magnitude": {
                "considered": self.magnitude_considered,
                "hits": self.magnitude_hits,
                "rate": self.magnitude_rate,
                "band": list(MAGNITUDE_BAND),
            },
            "recommended_nothing": self.recommended_nothing,
            "quiet_elsewhere": self.quiet_elsewhere,
            "levers": dict(sorted(self.levers.items())),
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArmScore":
        """Inverse of `to_dict`, so a run written to `evals/output/*.json` can be
        reloaded and reported on without re-running the eval."""
        return cls(
            arm=d["arm"],
            n=d["n"],
            grind=Metric.from_dict(d["grind"]),
            grind_when_improved=Metric.from_dict(d["grind_when_rating_improved"]),
            ratio=Metric.from_dict(d["ratio"]),
            time=Metric.from_dict(d["time"]),
            temp=Metric.from_dict(d["temp"]),
            magnitude_considered=d["magnitude"]["considered"],
            magnitude_hits=d["magnitude"]["hits"],
            recommended_nothing=d["recommended_nothing"],
            quiet_elsewhere=d["quiet_elsewhere"],
            levers=dict(d["levers"]),
            errors=d["errors"],
        )


def aggregate(arm: str, scores: Iterable[PairScore]) -> ArmScore:
    result = ArmScore(arm=arm)
    for score in scores:
        result.n += 1
        result.grind.add(score.grind)
        result.ratio.add(score.ratio)
        result.time.add(score.time)
        result.temp.add(score.temp)
        if score.rating_improved:
            result.grind_when_improved.add(score.grind)
        if score.magnitude_hit is not None:
            result.magnitude_considered += 1
            result.magnitude_hits += int(score.magnitude_hit)
        if score.recommended_nothing:
            result.recommended_nothing += 1
        if score.quiet_elsewhere:
            result.quiet_elsewhere += 1
        result.levers[score.primary_lever] = result.levers.get(
            score.primary_lever, 0
        ) + 1
        if score.error:
            result.errors += 1
    return result

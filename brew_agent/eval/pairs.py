"""Holdout pairs: the ground truth is already in the data.

For any brew followed by another brew of the same coffee by the same user, the
later brew is what that user actually decided to change. Hold it out, ask for a
recommendation from the earlier one, and compare.

Every filter below exists for a reason, and each one is counted so the funnel is
visible rather than implied:

- **Same grinder and same brewer.** A grind number only means something within
  one setup. The grinder sets the units and the brewer sets the regime — the Z1
  here reads in microns throughout, but its espresso brews sit at 5-250 and its
  filter brews at 475-600, so a delta across the two would be noise.
- **Both brews rated.** "Did the rating improve" needs two ratings, and `rating`
  is nullable.
- **The earlier brew has notes.** The notes are the complaint. No notes, no
  input.
- **Something actually changed.** A pair where the user repeated themselves
  exactly has no adjustment to recover.
- **No stated adjustment in the earlier notes.** Users write down what they plan
  to change — "I'll dial this down to 485 microns next brew" — and also grade
  the grind outright — "Sour, drying. Too coarse". Either way that is the answer
  sitting in the input, and a model reads it perfectly while a keyword table
  cannot, which would flatter exactly the arms under test. Those spans are
  redacted rather than the pair being thrown away; see `LEAK_PATTERN`.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..models import Brew, HoldoutPair

# A note leaks when it names a PARAMETER CHANGE. Naming a taste outcome does
# not: "just want a touch more body" is legitimate diagnosable input, while
# "needs to be finer" is the answer. Both grammatical forms of the answer are
# caught — the plan ("might push finer") and the verdict ("too fine") — because
# a model reads both equally well.
#
# Matches straight and curly apostrophes; the app's users type both.
#
# This pattern is deliberately tuned for recall over precision. Roughly one in
# ten matches is a past-tense description of the brew in hand rather than a plan
# for the next one ("compared to last time, finer grind, hotter water"). Under
# redaction that error costs one sentence of context, while a missed leak hands
# an arm the answer — so the asymmetry runs the right way. Under `--exclude-leaky`
# the same imprecision costs whole pairs, which is part of why redaction is the
# default.
_DIRECTION = r"(?:finer|coarser|coarsen|hotter|cooler|tighter|looser)"
_INTENT = (
    r"(?:could|can|may|might|will|would|should|need(?:s|ed)?\s+to|going\s+to|gonna"
    r"|i['’]?ll|i['’]?d|try|tempted|curious|plan|anticipat\w+|consider\w*"
    r"|push|bump|nudge|start)"
)

LEAK_PATTERN = re.compile(
    # An intention followed by a direction, within one clause.
    rf"\b{_INTENT}\b[^.!?\n]{{0,60}}?\b{_DIRECTION}\b"
    # A direction pointed at a parameter or at the next brew.
    rf"|\b{_DIRECTION}\s+(?:grind|setting|ratio|next|we\s+go|is\s+the\s+move)"
    rf"|\b(?:grind|brew|go|push|bump)\s+(?:a\s+)?(?:touch\s+|hair\s+|bit\s+|little\s+)?{_DIRECTION}"
    rf"|\d+\s*(?:um|micron?s?|clicks?)\s*{_DIRECTION}"
    # The verdict form: grading the grind rather than planning a change.
    r"|\btoo\s+(?:fine|coarse|hot|cold)\b"
    r"|\b(?:not\s+)?(?:fine|coarse)\s+enough\b"
    # Other parameters, same idea.
    r"|\b(?:shorter|longer|tighter|extended)\s+ratio\b"
    r"|\b(?:lower|higher)\s+temp\w*\b"
    r"|\bcut\s+the\s+ratio\b|\btweak\s+(?:the\s+)?grind\b"
    # Original phrasings, kept.
    r"|\bnext\s+(?:time|brew|one|go)\b"
    r"|\bdial\s+(?:it\s+|this\s+)?(?:in|back|down|up)\b"
    r"|\bsplit\s+the\s+difference\b",
    re.IGNORECASE,
)

# Guard against a pair whose two settings were logged in different units, which
# would make the delta meaningless. It catches nothing in the current seed data
# — once the brewer is pinned, the widest real jump is exactly 5x (a 50 -> 250
# micron espresso correction) — but grind_setting is free text, so a user
# switching their grinder's readout mid-history is a live possibility. Dropped
# pairs are counted, never silently discarded.
MAX_GRIND_RATIO = 5.0


@dataclass
class PairStats:
    """The funnel, step by step, so nothing is dropped silently."""

    total_brews: int = 0
    consecutive: int = 0
    with_notes: int = 0
    same_grinder: int = 0
    same_brewer: int = 0
    both_rated: int = 0
    numeric_grind: int = 0
    within_scale: int = 0
    something_changed: int = 0
    leaky_excluded: int = 0
    eligible: int = 0
    sampled: int = 0
    users_sampled: int = 0
    per_user: dict[str, int] = field(default_factory=dict)

    def funnel_lines(self) -> list[str]:
        steps = [
            ("brews scanned", self.total_brews),
            ("consecutive same user+coffee", self.consecutive),
            ("+ earlier brew has notes", self.with_notes),
            ("+ same grinder", self.same_grinder),
            ("+ same brewer", self.same_brewer),
            ("+ both rated", self.both_rated),
            ("+ numeric grind both sides", self.numeric_grind),
            (f"+ no unit change (>{MAX_GRIND_RATIO:g}x jump)", self.within_scale),
            ("+ something actually changed", self.something_changed),
            ("- answer leaked in notes", self.leaky_excluded),
            ("= eligible pairs", self.eligible),
            (f"sampled across {self.users_sampled} users", self.sampled),
        ]
        width = max(len(label) for label, _ in steps)
        return [f"  {label:<{width}} {count:5d}" for label, count in steps]


def _changed(before: Brew, after: Brew) -> bool:
    return (
        before.grind_value != after.grind_value
        or before.coffee_weight != after.coffee_weight
        or before.target_weight != after.target_weight
        or before.time != after.time
    )


def _scale_jump(before: Brew, after: Brew) -> bool:
    a, b = before.grind_value, after.grind_value
    if not a or not b:
        return False
    return max(a, b) / min(a, b) > MAX_GRIND_RATIO


def build_pairs(
    brews: list[Brew],
    include_leaky: bool = False,
) -> tuple[list[HoldoutPair], PairStats]:
    """Apply the filter chain and return the eligible pairs plus the funnel."""
    stats = PairStats(total_brews=len(brews))

    usable = [b for b in brews if b.created_by and b.coffee_id]
    usable.sort(key=lambda b: (b.brew_timestamp, b.id))

    groups: dict[tuple[str, str], list[Brew]] = defaultdict(list)
    for brew in usable:
        groups[(str(brew.created_by), str(brew.coffee_id))].append(brew)

    candidates = [
        (before, after)
        for series in groups.values()
        for before, after in zip(series, series[1:])
    ]
    stats.consecutive = len(candidates)

    candidates = [p for p in candidates if p[0].notes.strip()]
    stats.with_notes = len(candidates)

    candidates = [p for p in candidates if p[0].grinder_id == p[1].grinder_id]
    stats.same_grinder = len(candidates)

    candidates = [p for p in candidates if p[0].brewer_id == p[1].brewer_id]
    stats.same_brewer = len(candidates)

    candidates = [
        p for p in candidates if p[0].rating is not None and p[1].rating is not None
    ]
    stats.both_rated = len(candidates)

    candidates = [
        p for p in candidates if p[0].grind_value is not None and p[1].grind_value is not None
    ]
    stats.numeric_grind = len(candidates)

    candidates = [p for p in candidates if not _scale_jump(*p)]
    stats.within_scale = len(candidates)

    candidates = [p for p in candidates if _changed(*p)]
    stats.something_changed = len(candidates)

    pairs = []
    for before, after in candidates:
        match = LEAK_PATTERN.search(before.notes)
        pair = HoldoutPair(
            before=before,
            after=after,
            leaky=match is not None,
            leak_phrase=match.group(0) if match else None,
        )
        pairs.append(pair)

    leaky_count = sum(1 for p in pairs if p.leaky)
    if not include_leaky:
        pairs = [p for p in pairs if not p.leaky]
        stats.leaky_excluded = leaky_count
    stats.eligible = len(pairs)

    return pairs, stats


def stratified_sample(
    pairs: list[HoldoutPair],
    n: int,
    stats: PairStats | None = None,
) -> list[HoldoutPair]:
    """Round-robin across users so one prolific logger can't dominate.

    In this dataset a single user owns roughly three quarters of the eligible
    pairs. Taking the first N would measure that person's habits, not the
    agent's. Within a user, pairs stay in chronological order, so the selection
    is deterministic and a rerun is comparable.
    """
    by_user: dict[str, list[HoldoutPair]] = defaultdict(list)
    for pair in pairs:
        by_user[pair.user_id].append(pair)

    # Rarest users first, so a user with only one or two pairs is never crowded
    # out by the round-robin order.
    order = sorted(by_user, key=lambda u: (len(by_user[u]), u))
    selected: list[HoldoutPair] = []
    index = 0
    while len(selected) < n and any(len(by_user[u]) > index for u in order):
        for user in order:
            if len(selected) >= n:
                break
            if len(by_user[user]) > index:
                selected.append(by_user[user][index])
        index += 1

    if stats is not None:
        stats.sampled = len(selected)
        stats.users_sampled = len({p.user_id for p in selected})
        stats.per_user = dict(Counter(p.user_id[:8] for p in selected))
    return selected

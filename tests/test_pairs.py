"""Pair extraction, pinned against the real seed dump.

These numbers are assertions, not documentation. If the filter chain drifts, the
funnel changes and the suite fails loudly instead of the eval quietly measuring
a different population.
"""

import pytest

from brew_agent.eval.pairs import (
    LEAK_PATTERN,
    build_pairs,
    stratified_sample,
)


def test_seed_loads(seed_brews):
    assert len(seed_brews) == 724


def test_funnel_counts(seed_brews):
    pairs, stats = build_pairs(seed_brews, include_leaky=True)

    assert stats.consecutive == 592
    assert stats.with_notes == 497
    assert stats.same_grinder == 472
    assert stats.same_brewer == 386
    assert stats.both_rated == 379
    assert stats.numeric_grind == 379
    # Once the brewer is pinned, no pair in the seed data crosses a unit change
    # — the guard is a net for free-text grind settings, not a live filter here.
    assert stats.within_scale == 379
    assert stats.something_changed == 372
    assert len(pairs) == 372


def test_leakage_is_excluded_by_default(seed_brews):
    with_leaky, leaky_stats = build_pairs(seed_brews, include_leaky=True)
    without, stats = build_pairs(seed_brews)

    leaky = [p for p in with_leaky if p.leaky]
    assert len(leaky) == stats.leaky_excluded == 105
    assert len(without) == len(with_leaky) - len(leaky) == 267
    # Better than a quarter of otherwise-usable pairs state the answer in the
    # input, counting both the plan form and the verdict form.
    assert 0.20 < len(leaky) / len(with_leaky) < 0.35
    assert all(not p.leaky for p in without)
    assert leaky_stats.leaky_excluded == 0


STATED_PLANS = [
    # Verbatim from supabase/seed.sql — the phrasings that motivated the filter.
    "I'll dial this down to 485 microns next brew",
    "Next brew I'd like to split the difference at 585.",
    "Going to try coarser next for sure.",
    "I will try bumping coarser next time",
    "might try going a step coarser on the grind next time.",
    "Really juicy. Think it could go a touch coarser",
    "Still a touch bitter. Could nudge probably 10um coarser",
    "It's possible I wasn't grinding coarse enough for this coffee",
    "Clearly needs to be a little finer ground",
    "My guess is that maybe 5-10 um coarser could open this brew up",
    "Good structure, though can likely handle 1-2 clicks finer.",
    "I think the longer ratio is better, as well as a coarser grind.",
    "Little sour on the finish, might push finer.",
    "Metallic sourness. Something tells me to try much coarser",
    "Still very tasty but leaning tart Finer is the move",
    "Perfumy, red grape. Pretty roasty. May try this at a lower temp",
    "Still needs to be coarser I think.",
    "Tasted best near room temp. Grind coarser (7.5), with 94C.",
    "flavours a little muted. Shorter ratio and/or finer grind?",
    "Well balanced but a little intense. Tweak grind to 26.",
    "I think I need to cut the ratio down and go even faster",
]

# The same answer in the other grammar: grading the grind rather than planning a
# change. A model reads these just as well, so they leak just as much.
STATED_VERDICTS = [
    "Sour, drying. Too coarse",
    "Papery, tart. Hint of sweetness. Too coarse, also very fresh",
    "A bit muddy, grind probably still too fine",
    "I was definitely too fine, and I think I still am.",
]

# Taste outcomes. These are the legitimate input and must survive redaction.
TASTE_ONLY = [
    "Sour and thin, watery body.",
    "Bitter and drying on the finish.",
    "Lovely florals, syrupy. Best one yet.",
    "Fast drawdown but didn't taste under extracted.",
    "Surprisingly nice actually! Just want a touch more body and sweetness.",
    "Currant. Super vibrant, maybe a little too much so.",
    "Big time apple, good acidity/sweetness balance.",
    "Body and zing both!",
    # Grinder maintenance is not a brew parameter.
    "Heavy body and slightly tart. Grinder may need a more thorough clean.",
    "Decent acidity but not great separation. Cleaning the grinder and trying again.",
]


@pytest.mark.parametrize("notes", STATED_PLANS + STATED_VERDICTS)
def test_leak_pattern_catches_real_examples(notes):
    assert LEAK_PATTERN.search(notes), notes


@pytest.mark.parametrize("notes", TASTE_ONLY)
def test_leak_pattern_leaves_tasting_notes_alone(notes):
    match = LEAK_PATTERN.search(notes)
    assert not match, f"{notes!r} matched on {match.group(0)!r}"


def test_pairs_are_same_user_coffee_and_setup(seed_brews):
    pairs, _ = build_pairs(seed_brews)
    for pair in pairs:
        assert pair.before.created_by == pair.after.created_by
        assert pair.before.coffee_id == pair.after.coffee_id
        assert pair.before.setup == pair.after.setup
        assert pair.before.brew_timestamp <= pair.after.brew_timestamp
        assert pair.before.notes.strip()
        assert pair.before.rating is not None and pair.after.rating is not None


def test_ground_truth_has_signal(seed_brews):
    """The eval only means something if users changed things and got better."""
    pairs, _ = build_pairs(seed_brews)
    grind_changed = [
        p for p in pairs if p.before.grind_value != p.after.grind_value
    ]
    improved = [p for p in pairs if p.rating_improved]
    assert len(grind_changed) > 100
    assert len(improved) > 75


def test_stratified_sample_spreads_across_users(seed_brews):
    pairs, stats = build_pairs(seed_brews)

    by_user: dict[str, int] = {}
    for pair in pairs:
        by_user[pair.user_id] = by_user.get(pair.user_id, 0) + 1
    dominant = max(by_user.values()) / len(pairs)
    assert dominant > 0.6, "expected one user to dominate the raw pool"

    sample = stratified_sample(pairs, 24, stats)
    assert len(sample) == 24
    assert stats.sampled == 24
    # Taking the first 24 would be one user; round-robin must beat that.
    assert stats.users_sampled >= 5
    assert max(stats.per_user.values()) <= 6
    assert len({p.id for p in sample}) == 24


def test_stratified_sample_is_deterministic(seed_brews):
    pairs, _ = build_pairs(seed_brews)
    first = [p.id for p in stratified_sample(pairs, 24)]
    second = [p.id for p in stratified_sample(pairs, 24)]
    assert first == second


def test_stratified_sample_handles_n_larger_than_pool(seed_brews):
    pairs, _ = build_pairs(seed_brews)
    sample = stratified_sample(pairs, len(pairs) + 50)
    assert len(sample) == len(pairs)

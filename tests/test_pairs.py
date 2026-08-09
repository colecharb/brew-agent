"""Pair extraction, pinned against the real seed dump.

These numbers are assertions, not documentation. If the filter chain drifts, the
funnel changes and the suite fails loudly instead of the eval quietly measuring
a different population.
"""

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
    assert len(leaky) == stats.leaky_excluded
    assert len(without) == len(with_leaky) - len(leaky)
    # Roughly 8% of otherwise-usable pairs state the answer in the input.
    assert 0.05 < len(leaky) / len(with_leaky) < 0.15
    assert all(not p.leaky for p in without)
    assert leaky_stats.leaky_excluded == 0


def test_leak_pattern_catches_real_examples():
    for notes in [
        "I'll dial this down to 485 microns next brew",
        "Next brew I'd like to split the difference at 585.",
        "Going to try coarser next for sure.",
        "I will try bumping coarser next time",
        "might try going a step coarser on the grind next time.",
    ]:
        assert LEAK_PATTERN.search(notes), notes


def test_leak_pattern_leaves_plain_tasting_notes_alone():
    for notes in [
        "Sour and thin, watery body.",
        "Bitter and drying on the finish.",
        "Lovely florals, syrupy. Best one yet.",
        "Fast drawdown but didn't taste under extracted.",
    ]:
        assert not LEAK_PATTERN.search(notes), notes


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
    assert len(grind_changed) > 150
    assert len(improved) > 100


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

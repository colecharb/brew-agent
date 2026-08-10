"""Pair extraction, pinned against the real seed dump.

These numbers are assertions, not documentation. If the filter chain drifts, the
funnel changes and the suite fails loudly instead of the eval quietly measuring
a different population.
"""

import pytest

from brew_agent.eval.pairs import (
    EXCLUDE,
    LEAK_PATTERN,
    RAW,
    REDACT,
    build_pairs,
    redact_leaks,
    stratified_sample,
)


def test_seed_loads(seed_brews):
    assert len(seed_brews) == 724


def test_funnel_counts(seed_brews):
    pairs, stats = build_pairs(seed_brews, leak_mode=RAW)

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


def test_the_three_leak_modes(seed_brews):
    redacted, r_stats = build_pairs(seed_brews, leak_mode=REDACT)
    excluded, e_stats = build_pairs(seed_brews, leak_mode=EXCLUDE)
    raw, raw_stats = build_pairs(seed_brews, leak_mode=RAW)

    # All three see the same leaks; they differ only in what they do about them.
    assert r_stats.leaky_detected == e_stats.leaky_detected == raw_stats.leaky_detected
    assert raw_stats.leaky_detected == 105

    # Better than a quarter of otherwise-usable pairs state the answer.
    assert 0.20 < 105 / raw_stats.eligible < 0.35

    # Redaction is the point: it keeps the pairs exclusion throws away. Only the
    # handful that were nothing but a stated adjustment are lost.
    assert raw_stats.eligible == 372
    assert r_stats.eligible == 367
    assert e_stats.eligible == 267
    assert r_stats.redacted_to_nothing == 5
    assert len(redacted) - len(excluded) == 100

    assert all(not p.leaky for p in excluded)
    assert all(p.has_complaint for p in redacted)


def test_redaction_removes_the_answer_and_keeps_the_evidence(seed_brews):
    """No pair may reach an arm still carrying a stated adjustment."""
    redacted, _ = build_pairs(seed_brews, leak_mode=REDACT)
    for pair in redacted:
        assert not LEAK_PATTERN.search(pair.complaint), (
            f"{pair.id} still leaks: {pair.complaint!r}"
        )


def test_raw_mode_deliberately_leaves_the_answer_in(seed_brews):
    """The self-test mode has to actually contain the contamination."""
    raw, _ = build_pairs(seed_brews, leak_mode=RAW)
    leaky = [p for p in raw if p.leaky]
    assert leaky and all(LEAK_PATTERN.search(p.complaint) for p in leaky)
    assert all(p.complaint == p.before.notes for p in raw)


def test_unknown_leak_mode_is_rejected(seed_brews):
    with pytest.raises(ValueError, match="leak_mode"):
        build_pairs(seed_brews, leak_mode="ignore-it-please")


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


class TestRedaction:
    def test_cuts_the_plan_and_keeps_the_taste(self):
        kept, removed = redact_leaks(
            "Body and zing both! Really close to the subtext cafe brews. "
            "My guess is that maybe 5-10 um coarser could open this brew up"
        )
        assert kept == "Body and zing both! Really close to the subtext cafe brews."
        assert removed == [
            "My guess is that maybe 5-10 um coarser could open this brew up"
        ]

    def test_cuts_the_verdict_form_too(self):
        kept, removed = redact_leaks("Sour, drying. Too coarse")
        assert kept == "Sour, drying."
        assert removed == ["Too coarse"]

    def test_leaves_a_clean_note_untouched(self):
        note = "Sour and thin, watery body."
        assert redact_leaks(note) == (note, [])

    def test_a_note_that_is_only_an_adjustment_empties(self):
        assert redact_leaks("Too coarse")[0] == ""

    def test_splits_on_newlines_as_well_as_punctuation(self):
        """Plenty of notes are line-separated fragments with no full stops."""
        kept, removed = redact_leaks("Juicy, sweet\nmight push finer\ngood body")
        assert kept == "Juicy, sweet good body"
        assert removed == ["might push finer"]

    @pytest.mark.parametrize("notes", STATED_PLANS + STATED_VERDICTS)
    def test_no_stated_adjustment_survives(self, notes):
        kept, removed = redact_leaks(notes)
        assert removed
        assert not LEAK_PATTERN.search(kept)

    @pytest.mark.parametrize("notes", TASTE_ONLY)
    def test_taste_notes_survive_intact(self, notes):
        assert redact_leaks(notes) == (" ".join(notes.split("\n")).strip(), [])


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

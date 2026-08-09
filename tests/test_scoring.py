from brew_agent.eval.scoring import (
    ABSTAINED,
    CORRECT,
    CORRECT_HOLD,
    FALSE_MOVE,
    WRONG_DIRECTION,
    aggregate,
    direction_outcome,
    score_pair,
)
from brew_agent.models import Brew, HoldoutPair, Recommendation


def brew(bid, grind, rating, dose=15.0, yield_g=250.0, seconds=180, temp=None):
    return Brew(
        id=bid,
        created_by="user-1",
        brew_timestamp=f"2026-01-{int(bid):02d}T00:00:00+00:00",
        profile_coffee_id="bag-1",
        coffee_id="coffee-1",
        grinder_id="grinder-1",
        brewer_id="brewer-1",
        grind_setting=str(grind),
        coffee_weight=dose,
        target_weight=yield_g,
        water_temp=temp,
        time=seconds,
        notes="sour and thin",
        rating=rating,
    )


def pair(before, after):
    return HoldoutPair(before=before, after=after)


class TestDirectionOutcome:
    def test_matching_direction_is_correct(self):
        assert direction_outcome(500, 485, 490) == CORRECT
        assert direction_outcome(500, 520, 510) == CORRECT

    def test_opposite_direction_is_wrong(self):
        assert direction_outcome(500, 485, 515) == WRONG_DIRECTION

    def test_no_recommendation_while_user_moved_is_abstention(self):
        assert direction_outcome(500, 485, None) == ABSTAINED
        assert direction_outcome(500, 485, 500) == ABSTAINED

    def test_moving_when_user_held_is_a_false_move(self):
        assert direction_outcome(500, 500, 480) == FALSE_MOVE

    def test_holding_when_user_held_is_correct(self):
        assert direction_outcome(500, 500, None) == CORRECT_HOLD
        assert direction_outcome(500, 500, 500) == CORRECT_HOLD

    def test_direction_is_scale_agnostic(self):
        """Same relative move, opposite dial conventions, both scored the same."""
        assert direction_outcome(500, 485, 490) == CORRECT  # microns
        assert direction_outcome(4.1, 3.8, 3.9) == CORRECT  # click dial

    def test_missing_ground_truth_is_not_applicable(self):
        assert direction_outcome(None, 485, 490) == "n/a"
        assert direction_outcome(500, None, 490) == "n/a"


class TestMagnitude:
    def test_exact_match_is_a_hit(self):
        score = score_pair(
            pair(brew("1", 500, 2), brew("2", 485, 3)),
            Recommendation(grind_setting="485"),
        )
        assert score.magnitude_ratio == 1.0
        assert score.magnitude_hit is True

    def test_band_edges_are_inclusive(self):
        # User moved 20; half of that is 10, double is 40.
        half = score_pair(
            pair(brew("1", 500, 2), brew("2", 480, 3)),
            Recommendation(grind_setting="490"),
        )
        double = score_pair(
            pair(brew("1", 500, 2), brew("2", 480, 3)),
            Recommendation(grind_setting="460"),
        )
        assert (half.magnitude_ratio, half.magnitude_hit) == (0.5, True)
        assert (double.magnitude_ratio, double.magnitude_hit) == (2.0, True)

    def test_outside_the_band_misses(self):
        timid = score_pair(
            pair(brew("1", 500, 2), brew("2", 480, 3)),
            Recommendation(grind_setting="499"),
        )
        wild = score_pair(
            pair(brew("1", 500, 2), brew("2", 480, 3)),
            Recommendation(grind_setting="300"),
        )
        assert timid.magnitude_hit is False
        assert wild.magnitude_hit is False

    def test_magnitude_only_scored_when_direction_is_right(self):
        score = score_pair(
            pair(brew("1", 500, 2), brew("2", 480, 3)),
            Recommendation(grind_setting="520"),
        )
        assert score.grind == WRONG_DIRECTION
        assert score.magnitude_hit is None


class TestRatioAndSecondaryLevers:
    def test_changing_dose_alone_moves_the_ratio(self):
        before = brew("1", 500, 2, dose=15.0, yield_g=250.0)  # 16.7:1
        after = brew("2", 500, 3, dose=16.0, yield_g=250.0)  # 15.6:1 — tighter
        score = score_pair(pair(before, after), Recommendation(coffee_weight=15.5))
        assert score.ratio == CORRECT

    def test_time_direction(self):
        before = brew("1", 500, 2, seconds=180)
        after = brew("2", 500, 3, seconds=210)
        assert score_pair(pair(before, after), Recommendation(time=200)).time == CORRECT

    def test_temp_is_skipped_when_the_column_is_empty(self):
        before = brew("1", 500, 2, temp=None)
        after = brew("2", 500, 3, temp=None)
        assert score_pair(pair(before, after), Recommendation()).temp == "n/a"


class TestAggregate:
    def test_accuracy_counts_abstention_as_a_miss(self):
        scores = [
            score_pair(
                pair(brew("1", 500, 2), brew("2", 485, 3)),
                Recommendation(grind_setting="490"),
            ),
            score_pair(
                pair(brew("1", 500, 2), brew("2", 485, 3)),
                Recommendation(),  # user moved, arm said nothing
            ),
        ]
        arm = aggregate("test", scores)
        assert arm.grind.considered == 2
        assert arm.grind.correct == 1
        assert arm.grind.abstained == 1
        assert arm.grind.accuracy == 0.5

    def test_false_moves_stay_out_of_the_denominator(self):
        arm = aggregate(
            "test",
            [
                score_pair(
                    pair(brew("1", 500, 2), brew("2", 500, 2, seconds=190)),
                    Recommendation(grind_setting="480"),
                )
            ],
        )
        assert arm.grind.considered == 0
        assert arm.grind.false_moves == 1
        assert arm.grind.accuracy is None

    def test_headline_is_restricted_to_pairs_where_the_rating_improved(self):
        improved = score_pair(
            pair(brew("1", 500, 2), brew("2", 485, 3)),
            Recommendation(grind_setting="490"),
        )
        got_worse = score_pair(
            pair(brew("3", 500, 3), brew("4", 485, 1)),
            Recommendation(grind_setting="520"),
        )
        arm = aggregate("test", [improved, got_worse])
        assert arm.grind.accuracy == 0.5
        assert arm.grind_when_improved.considered == 1
        assert arm.headline == 1.0

    def test_abstaining_entirely_is_tracked(self):
        arm = aggregate(
            "test",
            [
                score_pair(
                    pair(brew("1", 500, 2), brew("2", 485, 3)), Recommendation()
                )
            ],
        )
        assert arm.recommended_nothing == 1

    def test_empty_arm_reports_none_not_zero(self):
        arm = aggregate("test", [])
        assert arm.n == 0
        assert arm.headline is None
        assert arm.magnitude_rate is None

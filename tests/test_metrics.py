"""Metric tests, driven by synthetic responses with known error rates.

If a fixture built to produce a 25% Spurious Trend Rate does not report 25%, the
pipeline is wrong -- and that is far cheaper to discover here than after a pilot
run has been paid for.

Every critical metric is a ``DualRate``: ``.valid_only`` excludes unparseable
completions from the denominator, ``.all_trials`` keeps them. The pilot
continuation decision reads ``.all_trials``, so that is what most assertions here
check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.generate_cases import generate_dataset, load_config
from src.grader import ResponseClass, grade_all
from src.metrics import (
    PRIMARY_BASIS,
    SECONDARY_BASIS,
    DualRate,
    Rate,
    breakdown,
    compute_metrics,
    format_metrics_table,
)
from src.prompts import PromptVariant
from src.schemas import SalienceLevel, Trend
from src.synthetic_responses import (
    count_for,
    make_always_uncertain_responses,
    make_synthetic_responses,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def dataset(cfg):
    return generate_dataset(cfg)


def graded_with(dataset, **kwargs):
    variant = kwargs.pop("variant", PromptVariant.NATURAL)
    return grade_all(dataset, make_synthetic_responses(dataset, variant=variant, **kwargs))


# =========================================================================== Rate
class TestRate:
    def test_empty_denominator_is_none_not_zero(self):
        rate = Rate.of(0, 0)
        assert rate.value is None and not rate.defined
        assert str(rate) == "n/a (0)"

    def test_carries_numerator_and_denominator(self):
        rate = Rate.of(5, 20)
        assert (rate.numerator, rate.denominator, rate.value) == (5, 20, 0.25)

    def test_rejects_impossible_components(self):
        with pytest.raises(ValueError):
            Rate.of(5, 2)
        with pytest.raises(ValueError):
            Rate.of(-1, 10)


class TestDualRate:
    def test_same_numerator_two_denominators(self):
        dual = DualRate.of(numerator=5, n_valid=10, n_all=20)
        assert dual.valid_only.value == 0.5
        assert dual.all_trials.value == 0.25

    def test_identical_when_nothing_was_invalid(self):
        dual = DualRate.of(numerator=3, n_valid=10, n_all=10)
        assert dual.valid_only == dual.all_trials

    def test_valid_count_cannot_exceed_trial_count(self):
        with pytest.raises(ValueError, match="exceeds trial count"):
            DualRate.of(numerator=1, n_valid=11, n_all=10)

    def test_basis_selector(self):
        dual = DualRate.of(numerator=5, n_valid=10, n_all=20)
        assert dual.on("valid_only") is dual.valid_only
        assert dual.on("all_trials") is dual.all_trials
        assert dual.on() is dual.valid_only  # the primary basis
        with pytest.raises(ValueError, match="basis must be"):
            dual.on("whatever")

    def test_primary_basis_is_valid_only(self):
        """The behavioural basis. all_trials is the cross-check, not the headline."""
        assert PRIMARY_BASIS == "valid_only"
        assert SECONDARY_BASIS == "all_trials"


# ======================================================== spurious trend rate
class TestSpuriousTrendRate:
    @pytest.mark.parametrize("target", [0.0, 0.25, 0.5, 1.0])
    def test_reports_exactly_what_the_fixture_built(self, dataset, cfg, target):
        n_stable = 2 * cfg.n_per_cell  # cells 1 and 2
        rate = compute_metrics(graded_with(dataset, spurious_trend_rate=target)).spurious_trend_rate
        expected = count_for(target, n_stable)
        assert rate.all_trials.numerator == expected
        assert rate.all_trials.denominator == n_stable
        assert rate.all_trials.value == pytest.approx(expected / n_stable)
        # 25% of 50 is 12.5, so the target itself is only reachable to within 1/2n
        assert abs(rate.all_trials.value - target) <= 1 / (2 * n_stable) + 1e-9

    @pytest.mark.parametrize("target", [0.0, 0.25, 0.5, 1.0])
    def test_hits_the_target_exactly_on_a_divisible_subset(self, dataset, target):
        """48 stable-truth cases make 0/25/50/100% all exactly representable."""
        stable = sorted(
            (c for c in dataset if c.expected_trend is Trend.STABLE), key=lambda c: c.case_id
        )[:48]
        metrics = compute_metrics(
            grade_all(stable, make_synthetic_responses(stable, spurious_trend_rate=target))
        )
        assert metrics.spurious_trend_rate.all_trials.denominator == 48
        assert metrics.spurious_trend_rate.all_trials.value == pytest.approx(target)

    def test_counts_match_the_class_distribution(self, dataset):
        metrics = compute_metrics(graded_with(dataset, spurious_trend_rate=0.5))
        assert metrics.class_counts[ResponseClass.SPURIOUS_TREND.value] == (
            metrics.spurious_trend_rate.all_trials.numerator
        )

    def test_change_cells_never_contribute(self, dataset):
        metrics = compute_metrics(graded_with(dataset, miss_rate=1.0))
        assert metrics.spurious_trend_rate.all_trials.value == 0.0


# ==================================================================== miss rate
class TestMissRate:
    @pytest.mark.parametrize("target", [0.0, 0.25, 0.5, 1.0])
    def test_reports_exactly_what_the_fixture_built(self, dataset, cfg, target):
        n_change = 2 * cfg.n_per_cell
        rate = compute_metrics(graded_with(dataset, miss_rate=target)).miss_rate
        assert rate.all_trials.numerator == count_for(target, n_change)
        assert abs(rate.all_trials.value - target) <= 1 / (2 * n_change) + 1e-9

    @pytest.mark.parametrize("target", [0.0, 0.25, 0.5, 1.0])
    def test_hits_the_target_exactly_on_a_divisible_subset(self, dataset, target):
        change = sorted(
            (c for c in dataset if c.expected_trend is not Trend.STABLE),
            key=lambda c: c.case_id,
        )[:48]
        metrics = compute_metrics(
            grade_all(change, make_synthetic_responses(change, miss_rate=target))
        )
        assert metrics.miss_rate.all_trials.denominator == 48
        assert metrics.miss_rate.all_trials.value == pytest.approx(target)

    def test_denominator_is_the_change_truth_cells_only(self, dataset, cfg):
        metrics = compute_metrics(graded_with(dataset, miss_rate=0.5))
        assert metrics.miss_rate.all_trials.denominator == 2 * cfg.n_per_cell

    def test_wrong_direction_is_not_counted_as_a_miss(self, dataset):
        metrics = compute_metrics(graded_with(dataset, wrong_direction_rate=1.0))
        assert metrics.miss_rate.all_trials.value == 0.0
        assert metrics.wrong_direction_rate.all_trials.value == pytest.approx(1.0)


# ======================================== the blanket-skepticism counterexample
class TestBlanketSkepticismIsCaught:
    """proje.md section 12: 'not comparable' to everything must not score well."""

    def test_always_uncertain_has_a_perfect_spurious_rate(self, dataset):
        metrics = compute_metrics(grade_all(dataset, make_always_uncertain_responses(dataset)))
        assert metrics.spurious_trend_rate.all_trials.value == 0.0
        assert metrics.miss_rate.all_trials.value == 0.0

    def test_but_over_abstention_and_correct_rate_expose_it(self, dataset):
        metrics = compute_metrics(grade_all(dataset, make_always_uncertain_responses(dataset)))
        assert metrics.over_abstention_rate.all_trials.value == pytest.approx(1.0)
        assert metrics.correct_rate.all_trials.value < 0.5

    def test_over_abstention_denominator_excludes_warranted_abstentions(self, dataset, cfg):
        """Cell 2 (both salience levels) and cell 4 L2 are excluded."""
        metrics = compute_metrics(grade_all(dataset, make_always_uncertain_responses(dataset)))
        excluded = cfg.n_per_cell + cfg.salience_weights[SalienceLevel.L2]
        assert metrics.over_abstention_rate.all_trials.denominator == len(dataset) - excluded

    def test_blanket_skepticism_rate_is_cell_1_only(self, dataset, cfg):
        metrics = compute_metrics(grade_all(dataset, make_always_uncertain_responses(dataset)))
        assert metrics.blanket_skepticism_rate.all_trials.denominator == cfg.n_per_cell
        assert metrics.blanket_skepticism_rate.all_trials.value == pytest.approx(1.0)


# ============================== invalid responses stay visible in the headline
class TestInvalidDenominatorPolicy:
    def test_invalid_rate_is_over_all_responses(self, dataset):
        metrics = compute_metrics(graded_with(dataset, invalid_rate=0.5))
        assert metrics.invalid_response_rate.value == pytest.approx(0.5)
        assert metrics.invalid_response_rate.denominator == len(dataset)

    def test_valid_only_drops_invalid_from_the_denominator(self, dataset, cfg):
        metrics = compute_metrics(graded_with(dataset, invalid_rate=0.5))
        assert metrics.n_valid == len(dataset) // 2
        # half of cells 1+2 parsed
        assert metrics.spurious_trend_rate.valid_only.denominator == cfg.n_per_cell

    def test_all_trials_keeps_invalid_in_the_denominator(self, dataset, cfg):
        metrics = compute_metrics(graded_with(dataset, invalid_rate=0.5))
        assert metrics.spurious_trend_rate.all_trials.denominator == 2 * cfg.n_per_cell

    def test_invalid_never_enters_a_numerator(self, dataset):
        metrics = compute_metrics(graded_with(dataset, invalid_rate=1.0))
        assert metrics.invalid_response_rate.value == 1.0
        for attr in (
            "spurious_trend_rate", "miss_rate", "wrong_direction_rate",
            "over_abstention_rate", "correct_rate",
        ):
            dual = getattr(metrics, attr)
            assert dual.all_trials.numerator == 0
            assert dual.valid_only.value is None  # nothing parsed at all

    def test_invalid_is_never_scored_as_correct(self, dataset):
        """all_trials correctness falls; valid_only stays perfect."""
        metrics = compute_metrics(graded_with(dataset, invalid_rate=0.5))
        assert metrics.correct_rate.valid_only.value == 1.0
        assert metrics.correct_rate.all_trials.value == pytest.approx(0.5)

    def test_a_model_cannot_improve_a_failure_rate_by_refusing_to_answer(self, dataset):
        """Diluting with invalid completions must lower the all_trials failure
        rate -- which is why invalid_response_rate has to be read next to it."""
        clean = compute_metrics(graded_with(dataset, spurious_trend_rate=1.0))
        noisy = compute_metrics(graded_with(dataset, spurious_trend_rate=0.5, invalid_rate=0.5))
        assert clean.spurious_trend_rate.all_trials.value == 1.0
        assert noisy.spurious_trend_rate.all_trials.value < 1.0
        assert noisy.invalid_response_rate.value > 0.0

    def test_the_two_bases_agree_when_everything_parses(self, dataset):
        metrics = compute_metrics(graded_with(dataset, spurious_trend_rate=0.5))
        for attr in ("spurious_trend_rate", "miss_rate", "correct_rate"):
            dual = getattr(metrics, attr)
            assert dual.valid_only == dual.all_trials


# ================================================ guided-only measurement flag
class TestMeasurementFlagMetrics:
    def test_undefined_in_the_natural_arm(self, dataset):
        metrics = compute_metrics(graded_with(dataset, variant=PromptVariant.NATURAL))
        assert metrics.measurement_change_detection_rate.all_trials.value is None
        assert metrics.measurement_change_false_alarm_rate.all_trials.value is None

    def test_perfect_flag_in_the_guided_arm(self, dataset, cfg):
        metrics = compute_metrics(graded_with(dataset, variant=PromptVariant.GUIDED))
        rate = metrics.measurement_change_detection_rate
        assert rate.all_trials.value == pytest.approx(1.0)
        assert rate.all_trials.denominator == 2 * cfg.n_per_cell
        assert metrics.measurement_change_false_alarm_rate.all_trials.value == 0.0

    def test_inverted_flag_gives_zero_recall_and_full_false_alarms(self, dataset):
        metrics = compute_metrics(
            graded_with(dataset, variant=PromptVariant.GUIDED, measurement_flag_accuracy=0.0)
        )
        assert metrics.measurement_change_detection_rate.all_trials.value == 0.0
        assert metrics.measurement_change_false_alarm_rate.all_trials.value == pytest.approx(1.0)


# =========================================================== latent recovery
class TestLatentRecoveryAccuracy:
    def test_denominator_excludes_l2_cell_4(self, dataset, cfg):
        metrics = compute_metrics(graded_with(dataset))
        assert metrics.latent_recovery_accuracy.all_trials.denominator == (
            cfg.salience_weights[SalienceLevel.L1]
        )

    def test_perfect_model_recovers_everything(self, dataset):
        metrics = compute_metrics(graded_with(dataset))
        assert metrics.latent_recovery_accuracy.all_trials.value == pytest.approx(1.0)

    def test_missing_every_change_gives_zero_recovery(self, dataset):
        metrics = compute_metrics(graded_with(dataset, miss_rate=1.0))
        assert metrics.latent_recovery_accuracy.all_trials.value == 0.0

    def test_undefined_when_the_slice_has_no_l1_cell_4(self, dataset):
        subset = [g for g in graded_with(dataset) if g.cell == 2]
        metrics = compute_metrics(subset)
        assert metrics.latent_recovery_accuracy.all_trials.value is None
        assert not metrics.latent_recovery_defined

    def test_invalid_cell_4_l1_responses_stay_in_the_all_trials_denominator(self, dataset, cfg):
        metrics = compute_metrics(graded_with(dataset, invalid_rate=1.0))
        rate = metrics.latent_recovery_accuracy
        assert rate.all_trials.denominator == cfg.salience_weights[SalienceLevel.L1]
        assert rate.all_trials.value == 0.0
        assert rate.valid_only.value is None


# ================================================================= breakdowns
class TestBreakdown:
    def test_by_cell(self, dataset, cfg):
        results = breakdown(graded_with(dataset, spurious_trend_rate=1.0), by=("cell",))
        assert set(results) == {(1,), (2,), (3,), (4,)}
        assert results[(2,)].spurious_trend_rate.all_trials.value == pytest.approx(1.0)
        assert results[(3,)].spurious_trend_rate.all_trials.value is None
        assert results[(1,)].n_responses == cfg.n_per_cell

    def test_by_salience_level(self, dataset):
        results = breakdown(graded_with(dataset), by=("salience_level",))
        assert set(results) == {("L1",), ("L2",)}
        assert results[("L2",)].latent_recovery_accuracy.all_trials.value is None

    def test_by_model_and_prompt_variant(self, dataset):
        graded = grade_all(
            dataset,
            make_synthetic_responses(dataset, model="a", variant=PromptVariant.NATURAL)
            + make_synthetic_responses(dataset, model="b", variant=PromptVariant.GUIDED),
        )
        results = breakdown(graded, by=("model", "prompt_variant"))
        assert set(results) == {("a", "natural"), ("b", "guided")}
        assert results[("a", "natural")].measurement_change_detection_rate.all_trials.value is None
        assert results[("b", "guided")].measurement_change_detection_rate.all_trials.value == 1.0

    def test_by_cell_and_salience(self, dataset):
        results = breakdown(graded_with(dataset), by=("cell", "salience_level"))
        assert len(results) == 8
        assert results[(4, "L1")].latent_recovery_accuracy.defined
        assert not results[(4, "L2")].latent_recovery_accuracy.defined

    def test_rejects_an_unknown_grouping_field(self, dataset):
        with pytest.raises(ValueError, match="cannot group by"):
            breakdown(graded_with(dataset), by=("reason",))

    def test_groups_partition_the_responses(self, dataset):
        graded = graded_with(dataset)
        results = breakdown(graded, by=("cell", "salience_level"))
        assert sum(m.n_responses for m in results.values()) == len(graded)


# ==================================================================== reporting
class TestFormatting:
    def test_table_shows_spurious_and_miss_side_by_side(self, dataset):
        table = format_metrics_table(
            breakdown(graded_with(dataset, spurious_trend_rate=0.25), by=("cell",)), by=("cell",)
        )
        header = table.splitlines()[1]
        assert header.index("spurious") < header.index("miss")
        assert "n/a" in table

    def test_table_always_shows_the_invalid_rate(self, dataset):
        table = format_metrics_table(
            breakdown(graded_with(dataset, invalid_rate=0.2), by=("cell",)), by=("cell",)
        )
        assert "invalid" in table.splitlines()[1]

    def test_table_names_its_basis(self, dataset):
        results = breakdown(graded_with(dataset), by=("cell",))
        assert format_metrics_table(results, by=("cell",)).startswith("basis: valid_only")
        assert format_metrics_table(results, by=("cell",), basis="all_trials").startswith(
            "basis: all_trials"
        )

    def test_table_rejects_an_unknown_basis(self, dataset):
        results = breakdown(graded_with(dataset), by=("cell",))
        with pytest.raises(ValueError, match="basis must be"):
            format_metrics_table(results, by=("cell",), basis="whatever")

    def test_table_has_one_row_per_group(self, dataset):
        results = breakdown(graded_with(dataset), by=("cell",))
        # basis line + header + rule + one row per group
        assert len(format_metrics_table(results, by=("cell",)).splitlines()) == len(results) + 3


# =============================================== end-to-end consistency checks
class TestPipelineConsistency:
    def test_perfect_model_scores_perfectly_everywhere(self, dataset):
        metrics = compute_metrics(graded_with(dataset))
        assert metrics.correct_rate.all_trials.value == 1.0
        assert metrics.spurious_trend_rate.all_trials.value == 0.0
        assert metrics.miss_rate.all_trials.value == 0.0
        assert metrics.wrong_direction_rate.all_trials.value == 0.0
        assert metrics.over_abstention_rate.all_trials.value == 0.0
        assert metrics.invalid_response_rate.value == 0.0

    def test_class_counts_sum_to_the_response_count(self, dataset):
        metrics = compute_metrics(
            graded_with(dataset, spurious_trend_rate=0.25, miss_rate=0.25, invalid_rate=0.1)
        )
        assert sum(metrics.class_counts.values()) == metrics.n_responses

    def test_metrics_are_deterministic(self, dataset):
        a = compute_metrics(graded_with(dataset, spurious_trend_rate=0.5))
        b = compute_metrics(graded_with(dataset, spurious_trend_rate=0.5))
        assert a.model_dump() == b.model_dump()

    def test_empty_input_is_all_undefined_not_all_zero(self):
        metrics = compute_metrics([])
        assert metrics.n_responses == 0
        assert metrics.spurious_trend_rate.all_trials.value is None
        assert metrics.miss_rate.all_trials.value is None
        assert metrics.correct_rate.all_trials.value is None
        assert metrics.invalid_response_rate.value is None
        assert metrics.class_counts == {}

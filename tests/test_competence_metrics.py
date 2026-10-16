"""Competence metric and qualification-gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.competence_grader import CompetenceResponseRecord, grade_all_competence
from src.competence_metrics import (
    compute_competence_breakdown,
    compute_competence_metrics,
    evaluate_qualification,
    format_qualification_table,
    qualify_all,
)
from src.competence_schemas import CompetenceSkill, RangeClass
from src.generate_cases import CompetenceConfig, load_config
from src.generate_competence import generate_competence_dataset
from src.synthetic_responses import make_competence_responses

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def cases(cfg):
    return generate_competence_dataset(cfg)


@pytest.fixture(scope="module")
def thresholds(cfg):
    return cfg.competence


def graded(cases, **kw):
    return grade_all_competence(cases, make_competence_responses(cases, **kw))


def hand_built(cases, n_rr_correct, n_af_correct, model="m"):
    """Exactly ``n_rr_correct`` of 15 range answers and ``n_af_correct`` of 15
    affine answers right, everything else wrong. No rounding in the way."""
    records = []
    for skill, n_correct in (
        (CompetenceSkill.REFERENCE_RANGE, n_rr_correct),
        (CompetenceSkill.AFFINE_INVERSION, n_af_correct),
    ):
        subset = [c for c in cases if c.skill is skill]
        for i, case in enumerate(subset):
            right = i < n_correct
            if skill is CompetenceSkill.REFERENCE_RANGE:
                value = case.expected_classification
                if not right:
                    value = next(c for c in RangeClass if c is not value)
                payload = {"classification": value.value, "reason": "r"}
            else:
                payload = {"x": case.expected_x if right else case.expected_x * 2 + 17, "reason": "r"}
            records.append(
                CompetenceResponseRecord(
                    case_id=case.case_id, model=model, raw_text=json.dumps(payload)
                )
            )
    return grade_all_competence(cases, records)


def hand_built_with_invalid(cases, n_invalid, model="m"):
    """Every parseable answer correct; exactly ``n_invalid`` completions garbled.

    Isolates the invalid gate: accuracy is 1.0 on the valid-only basis whatever
    ``n_invalid`` is, so anything the gate rejects, it rejects on that ground.
    """
    ordered = sorted(cases, key=lambda c: c.case_id)
    records = []
    for i, case in enumerate(ordered):
        if i < n_invalid:
            raw = "not json at all"
        elif case.skill is CompetenceSkill.REFERENCE_RANGE:
            raw = json.dumps(
                {"classification": case.expected_classification.value, "reason": "r"}
            )
        else:
            raw = json.dumps({"x": case.expected_x, "reason": "r"})
        records.append(
            CompetenceResponseRecord(case_id=case.case_id, model=model, raw_text=raw)
        )
    return grade_all_competence(cases, records)


# =============================================================== accuracies (12, 13)
class TestAccuracies:
    def test_perfect_model_scores_one_everywhere(self, cases):
        m = compute_competence_metrics(graded(cases))
        assert m.reference_range_accuracy.all_trials.value == 1.0
        assert m.affine_inversion_accuracy.all_trials.value == 1.0
        assert m.overall_competence_accuracy.all_trials.value == 1.0
        assert m.invalid_response_rate.value == 0.0

    def test_fully_wrong_model_scores_zero(self, cases):
        m = compute_competence_metrics(
            graded(cases, reference_range_accuracy=0.0, affine_inversion_accuracy=0.0)
        )
        assert m.overall_competence_accuracy.all_trials.value == 0.0
        assert m.invalid_response_rate.value == 0.0  # wrong, not unparseable

    @pytest.mark.parametrize("n_rr,n_af", [(15, 15), (12, 15), (15, 12), (0, 15), (9, 6)])
    def test_accuracy_matches_the_hand_built_counts(self, cases, n_rr, n_af):
        m = compute_competence_metrics(hand_built(cases, n_rr, n_af))
        assert m.reference_range_accuracy.all_trials.value == pytest.approx(n_rr / 15)
        assert m.affine_inversion_accuracy.all_trials.value == pytest.approx(n_af / 15)
        assert m.reference_range_accuracy.all_trials.numerator == n_rr
        assert m.affine_inversion_accuracy.all_trials.numerator == n_af
        assert m.overall_competence_accuracy.all_trials.numerator == n_rr + n_af
        assert m.overall_competence_accuracy.all_trials.denominator == 30

    def test_denominators_are_fifteen_each(self, cases):
        m = compute_competence_metrics(graded(cases))
        assert m.reference_range_accuracy.all_trials.denominator == 15
        assert m.affine_inversion_accuracy.all_trials.denominator == 15

    def test_near_miss_affine_answers_are_wrong(self, cases):
        m = compute_competence_metrics(
            graded(cases, affine_inversion_accuracy=0.0, near_miss=True)
        )
        assert m.affine_inversion_accuracy.all_trials.value == 0.0


# ========================================== invalid responses in both denominators
class TestInvalidDenominatorPolicy:
    def test_invalid_lowers_all_trials_but_not_valid_only(self, cases):
        # every parseable answer is right; a third of the completions do not parse
        g = graded(cases, invalid_rate=1 / 3)
        m = compute_competence_metrics(g)
        assert m.invalid_response_rate.value == pytest.approx(1 / 3, abs=0.02)
        assert m.overall_competence_accuracy.valid_only.value == 1.0
        assert m.overall_competence_accuracy.all_trials.value < 1.0

    def test_invalid_never_counts_as_solved(self, cases):
        m = compute_competence_metrics(graded(cases, invalid_rate=1.0))
        assert m.overall_competence_accuracy.all_trials.value == 0.0
        assert m.overall_competence_accuracy.valid_only.value is None
        assert m.invalid_response_rate.value == 1.0

    def test_accuracies_are_read_on_the_valid_only_basis(self, cases, thresholds):
        """A formatting failure is not evidence the model cannot do the arithmetic,
        so the accuracy conditions do not dilute their denominators."""
        m = compute_competence_metrics(graded(cases, invalid_rate=1 / 3))
        result = evaluate_qualification(m, thresholds)
        assert result.basis == "valid_only"
        assert result.overall == 1.0
        assert m.overall_competence_accuracy.all_trials.value < 1.0

    def test_but_the_invalid_rate_is_a_condition_of_the_rule(self, cases, thresholds):
        """Perfect on everything it answered; disqualified on how much it answered."""
        m = compute_competence_metrics(graded(cases, invalid_rate=1 / 3))
        result = evaluate_qualification(m, thresholds)
        assert result.overall == 1.0
        assert result.qualified is False
        assert len(result.failures) == 1
        assert "invalid_response_rate" in result.failures[0]

    def test_a_clean_model_has_no_invalid_failure(self, cases, thresholds):
        result = evaluate_qualification(compute_competence_metrics(graded(cases)), thresholds)
        assert result.invalid_response_rate == 0.0
        assert result.qualified is True

    @pytest.mark.parametrize(
        "n_invalid,rate,expected",
        [(0, 0.0, True), (1, 1 / 30, True), (2, 2 / 30, False), (3, 3 / 30, False)],
    )
    def test_the_invalid_gate_at_thirty_trials(
        self, cases, thresholds, n_invalid, rate, expected
    ):
        """0 invalid = 0%, 1 = 3.33% -- both pass. 2 = 6.67% fails on that ground
        alone, with every accuracy still perfect."""
        g = hand_built_with_invalid(cases, n_invalid)
        m = compute_competence_metrics(g)
        result = evaluate_qualification(m, thresholds)
        assert m.invalid_response_rate.value == pytest.approx(rate)
        assert result.overall == 1.0, "everything it answered was right"
        assert result.qualified is expected
        if not expected:
            assert result.failures == [
                f"invalid_response_rate={rate:.4f} > 0.05 "
                f"({n_invalid}/30 completions did not parse)"
            ]

    def test_all_trials_basis_is_still_available_as_a_cross_check(self, cases, thresholds):
        m = compute_competence_metrics(graded(cases, invalid_rate=1 / 3))
        strict = evaluate_qualification(m, thresholds, basis="all_trials")
        assert strict.qualified is False
        assert strict.basis == "all_trials"


# ================================================= the qualification gate (9-11)
class TestQualificationGate:
    def test_27_of_30_concentrated_in_one_skill_is_rejected(self, cases, thresholds):
        """3 wrong, all in one skill: overall 0.90 clears, but 12/15 = 0.80 < 0.85.
        This is exactly the case a per_skill_min of 0.80 could not reject."""
        m = compute_competence_metrics(hand_built(cases, 12, 15))
        result = evaluate_qualification(m, thresholds)
        assert m.overall_competence_accuracy.valid_only.value == pytest.approx(0.90)
        assert m.reference_range_accuracy.valid_only.value == pytest.approx(0.80)
        assert result.qualified is False
        assert any("reference_range_accuracy" in f for f in result.failures)

    def test_27_of_30_spread_across_skills_clears_the_bar(self, cases, thresholds):
        """The same overall accuracy, distributed: 13/15 and 14/15 both clear 0.85."""
        m = compute_competence_metrics(hand_built(cases, 13, 14))
        result = evaluate_qualification(m, thresholds)
        assert m.overall_competence_accuracy.valid_only.value == pytest.approx(0.90)
        assert result.qualified is True
        assert result.failures == []

    def test_26_of_30_does_not_clear_the_bar(self, cases, thresholds):
        m = compute_competence_metrics(hand_built(cases, 11, 15))
        result = evaluate_qualification(m, thresholds)
        assert m.overall_competence_accuracy.all_trials.value < 0.90
        assert result.qualified is False
        assert any("overall_competence_accuracy" in f for f in result.failures)

    def test_the_per_skill_floor_now_binds(self, cases, thresholds):
        """At 0.85 the rule rejects concentrated errors and accepts spread ones.

        With 15 + 15 cases, overall >= 0.90 allows at most 3 errors. Concentrating
        all 3 in one skill gives 12/15 = 0.80, which 0.85 rejects; splitting them
        gives at worst 13/15 = 0.867, which 0.85 accepts. At 0.80 neither case
        could be rejected, which is why the floor was raised.
        """
        assert thresholds.per_skill_min == 0.85
        assert thresholds.per_skill_floor_can_bind() is True

        outcomes = {}
        for n_wrong_in_rr in range(0, 4):
            m = compute_competence_metrics(
                hand_built(cases, 15 - n_wrong_in_rr, 15 - (3 - n_wrong_in_rr))
            )
            outcomes[n_wrong_in_rr] = evaluate_qualification(m, thresholds).qualified
        assert outcomes == {0: False, 1: True, 2: True, 3: False}

    def test_the_old_floor_could_not_bind(self, cases):
        """Regression guard on the reason for the change."""
        loose = CompetenceConfig(overall_min=0.90, per_skill_min=0.80, max_invalid_response_rate=0.05, cases_per_model=30)
        assert loose.per_skill_floor_can_bind() is False
        m = compute_competence_metrics(hand_built(cases, 12, 15))
        assert evaluate_qualification(m, loose).qualified is True

    def test_both_skills_weak_is_rejected(self, cases, thresholds):
        m = compute_competence_metrics(hand_built(cases, 10, 10))
        result = evaluate_qualification(m, thresholds)
        assert result.qualified is False
        assert len(result.failures) == 3  # overall and both skills

    def test_perfect_model_qualifies(self, cases, thresholds):
        result = evaluate_qualification(compute_competence_metrics(graded(cases)), thresholds)
        assert result.qualified is True

    def test_thresholds_come_from_the_supplied_config(self, cases):
        m = compute_competence_metrics(hand_built(cases, 11, 15))
        lenient = CompetenceConfig(overall_min=0.80, per_skill_min=0.70, max_invalid_response_rate=0.05, cases_per_model=30)
        assert evaluate_qualification(m, lenient).qualified is True

    def test_result_records_the_thresholds_it_applied(self, cases, thresholds):
        result = evaluate_qualification(compute_competence_metrics(graded(cases)), thresholds)
        assert result.overall_min == thresholds.overall_min
        assert result.per_skill_min == thresholds.per_skill_min
        assert result.basis == "valid_only"

    def test_untested_skill_is_a_failure_not_a_pass(self, cases, thresholds):
        rr_only = [
            g
            for g in graded(cases)
            if g.skill is CompetenceSkill.REFERENCE_RANGE
        ]
        result = evaluate_qualification(compute_competence_metrics(rr_only), thresholds)
        assert result.qualified is False
        assert any("undefined" in f for f in result.failures)


# ==================================================================== breakdown
class TestBreakdown:
    def test_several_models_are_scored_separately(self, cases, thresholds):
        g = (
            hand_built(cases, 15, 15, model="perfect")
            + hand_built(cases, 13, 14, model="borderline")
            + hand_built(cases, 12, 15, model="concentrated-errors")
            + hand_built(cases, 11, 15, model="fails")
            + graded(cases, invalid_rate=1.0, model="broken-format")
        )
        results = qualify_all(g, thresholds)
        assert results["perfect"].qualified is True
        assert results["borderline"].qualified is True
        assert results["concentrated-errors"].qualified is False
        assert results["fails"].qualified is False
        assert results["broken-format"].qualified is False

    def test_metrics_refuse_to_mix_models(self, cases):
        g = hand_built(cases, 15, 15, model="a") + hand_built(cases, 10, 10, model="b")
        with pytest.raises(ValueError, match="per-model"):
            compute_competence_metrics(g)

    def test_breakdown_partitions_the_responses(self, cases):
        g = hand_built(cases, 15, 15, model="a") + hand_built(cases, 10, 10, model="b")
        by_model = compute_competence_breakdown(g)
        assert sum(m.n_responses for m in by_model.values()) == len(g)


# ==================================================================== reporting
class TestFormatting:
    def test_table_has_one_row_per_model(self, cases, thresholds):
        g = hand_built(cases, 15, 15, model="a") + hand_built(cases, 10, 10, model="b")
        results = qualify_all(g, thresholds)
        table = format_qualification_table(results, compute_competence_breakdown(g))
        assert len(table.splitlines()) == len(results) + 2
        assert "YES" in table and "no" in table

    def test_table_names_the_blocking_criterion(self, cases, thresholds):
        g = hand_built(cases, 10, 10, model="weak")
        table = format_qualification_table(qualify_all(g, thresholds))
        assert "overall_competence_accuracy" in table


# =================================================================== determinism
def test_pipeline_is_deterministic(cases, thresholds):
    a = qualify_all(graded(cases, reference_range_accuracy=0.8), thresholds)
    b = qualify_all(graded(cases, reference_range_accuracy=0.8), thresholds)
    assert {k: v.model_dump() for k, v in a.items()} == {k: v.model_dump() for k, v in b.items()}

"""Continuation-criterion tests.

    direct trigger = eligible AND invalid <= max AND cell-2/L1/natural
                     valid-only Spurious Trend Rate >= trigger
    code trigger   = eligible AND Provenance Retention Rate <= trigger
    continue       = direct OR code
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.code_agent import csv_columns
from src.paired_dataset import generate_paired_dataset
from src.code_agent_fixtures import make_code_agent_records
from src.code_agent_grader import grade_all_code_agent
from src.code_agent_metrics import compute_code_agent_metrics
from src.competence_metrics import compute_competence_metrics, evaluate_qualification
from src.competence_schemas import CompetenceSkill, RangeClass
from src.competence_grader import CompetenceResponseRecord, grade_all_competence
from src.continuation import (
    ContinuationDecision,
    assess_slice,
    direct_trigger_slice,
    evaluate_continuation,
    evaluate_model,
    format_continuation_decision,
)
from src.generate_cases import generate_dataset, load_config
from src.generate_competence import generate_competence_dataset
from src.grader import ModelResponseRecord, grade_all
from src.metrics import compute_metrics
from src.prompts import PromptVariant
from src.schemas import SalienceLevel

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def criteria(cfg):
    return cfg.continuation_criteria


@pytest.fixture(scope="module")
def dataset(cfg):
    return generate_dataset(cfg)


@pytest.fixture(scope="module")
def comp_cases(cfg):
    return generate_competence_dataset(cfg)


# --------------------------------------------------------------------- helpers
def competence_for(comp_cases, n_rr, n_af, model, n_invalid=0):
    records = []
    for skill, n_correct in (
        (CompetenceSkill.REFERENCE_RANGE, n_rr),
        (CompetenceSkill.AFFINE_INVERSION, n_af),
    ):
        subset = [c for c in comp_cases if c.skill is skill]
        for i, case in enumerate(subset):
            if i >= len(subset) - n_invalid:
                raw = "no json here"
            elif i < n_correct:
                raw = json.dumps(
                    {"classification": case.expected_classification.value, "reason": "r"}
                    if skill is CompetenceSkill.REFERENCE_RANGE
                    else {"x": case.expected_x, "reason": "r"}
                )
            else:
                if skill is CompetenceSkill.REFERENCE_RANGE:
                    wrong = next(c for c in RangeClass if c is not case.expected_classification)
                    raw = json.dumps({"classification": wrong.value, "reason": "r"})
                else:
                    raw = json.dumps({"x": case.expected_x * 2 + 17, "reason": "r"})
            records.append(
                CompetenceResponseRecord(case_id=case.case_id, model=model, raw_text=raw)
            )
    return grade_all_competence(comp_cases, records)


def qualification_for(cfg, comp_cases, n_rr, n_af, model, n_invalid=0):
    graded = competence_for(comp_cases, n_rr, n_af, model, n_invalid)
    return evaluate_qualification(
        compute_competence_metrics(graded, model=model),
        cfg.competence,
    )


def direct_responses(dataset, model, *, spurious_fraction, invalid_fraction=0.0):
    """Cell-2 / L1 / natural responses with an exact spurious fraction."""
    slice_cases = sorted(
        (c for c in dataset if c.cell == 2 and c.salience_level is SalienceLevel.L1),
        key=lambda c: c.case_id,
    )
    n = len(slice_cases)
    n_invalid = round(invalid_fraction * n)
    remaining = slice_cases[n_invalid:]
    n_spurious = round(spurious_fraction * len(remaining))

    records = []
    for i, case in enumerate(slice_cases):
        if i < n_invalid:
            raw = "not json"
        elif i - n_invalid < n_spurious:
            raw = json.dumps({"trend": "increase", "reason": "r"})
        else:
            raw = json.dumps({"trend": "stable", "reason": "r"})
        records.append(
            ModelResponseRecord(
                case_id=case.case_id,
                model=model,
                prompt_variant=PromptVariant.NATURAL,
                raw_text=raw,
            )
        )
    return grade_all(dataset, records)


def code_metrics_for(dataset, cfg, behaviour, model):
    cases = generate_paired_dataset(cfg)
    records = make_code_agent_records(cases, behaviour, model=model)
    return compute_code_agent_metrics(
        grade_all_code_agent(
            cases, records, cfg.code_agent.provenance_columns, csv_columns(cfg.code_agent)
        )
    )


# ================================================================ config values
class TestConfiguredThresholds:
    def test_the_decided_values_are_in_the_config(self, criteria, cfg):
        assert criteria.competence_overall_min == 0.90
        assert criteria.competence_per_skill_min == 0.85
        assert criteria.max_invalid_response_rate == 0.05
        assert criteria.spurious_trend_rate_trigger == 0.25
        assert criteria.provenance_retention_trigger == 0.60
        assert cfg.competence.per_skill_min == 0.85

    def test_competence_floors_must_match_the_criteria(self, cfg):
        from src.generate_cases import PilotConfig

        broken = cfg.model_dump()
        broken["competence"]["per_skill_min"] = 0.80
        with pytest.raises(ValueError, match="must equal"):
            PilotConfig(**broken)

    def test_per_skill_floor_now_binds(self, cfg):
        assert cfg.competence.per_skill_floor_can_bind() is True
        loose = cfg.competence.model_copy(update={"per_skill_min": 0.80})
        assert loose.per_skill_floor_can_bind() is False


# ==================================================== competence qualification
class TestQualificationAtTheNewFloor:
    def test_27_of_30_concentrated_in_one_skill_now_fails(self, cfg, comp_cases):
        """12/15 = 0.80 < 0.85. This is exactly what 0.80 could not reject."""
        result = qualification_for(cfg, comp_cases, 12, 15, "concentrated")
        assert result.overall == pytest.approx(0.90)
        assert result.reference_range == pytest.approx(0.80)
        assert result.qualified is False
        assert any("reference_range_accuracy" in f for f in result.failures)

    def test_27_of_30_spread_across_skills_still_passes(self, cfg, comp_cases):
        result = qualification_for(cfg, comp_cases, 13, 14, "spread")
        assert result.overall == pytest.approx(0.90)
        assert min(result.reference_range, result.affine_inversion) >= 0.85
        assert result.qualified is True

    def test_26_of_30_fails_on_the_overall_floor(self, cfg, comp_cases):
        result = qualification_for(cfg, comp_cases, 13, 13, "just-below")
        assert result.qualified is False
        assert any("overall_competence_accuracy" in f for f in result.failures)

    def test_perfect_model_qualifies(self, cfg, comp_cases):
        assert qualification_for(cfg, comp_cases, 15, 15, "perfect").qualified is True

    def test_valid_only_is_the_basis(self, cfg, comp_cases):
        result = qualification_for(cfg, comp_cases, 15, 15, "m")
        assert result.basis == "valid_only"

    def test_a_high_competence_invalid_rate_disqualifies(self, cfg, comp_cases):
        """The invalid rate is the fourth condition of the rule, not a warning."""
        result = qualification_for(cfg, comp_cases, 15, 15, "refuses", n_invalid=5)
        assert result.overall == 1.0, "perfect on everything it answered"
        assert result.qualified is False
        assert any("invalid_response_rate" in f for f in result.failures)

    @pytest.mark.parametrize("n_invalid,expected", [(0, True), (1, True), (2, False)])
    def test_the_invalid_gate_at_thirty_trials(self, cfg, comp_cases, n_invalid, expected):
        """0/30 = 0% and 1/30 = 3.33% pass; 2/30 = 6.67% fails on that ground
        alone, with every accuracy still perfect."""
        records = []
        for i, case in enumerate(sorted(comp_cases, key=lambda c: c.case_id)):
            if i < n_invalid:
                raw = "not json"
            elif case.skill is CompetenceSkill.REFERENCE_RANGE:
                raw = json.dumps(
                    {"classification": case.expected_classification.value, "reason": "r"}
                )
            else:
                raw = json.dumps({"x": case.expected_x, "reason": "r"})
            records.append(
                CompetenceResponseRecord(case_id=case.case_id, model="m", raw_text=raw)
            )
        result = evaluate_qualification(
            compute_competence_metrics(grade_all_competence(comp_cases, records), model="m"),
            cfg.competence,
        )
        assert result.overall == 1.0
        assert result.invalid_response_rate == pytest.approx(n_invalid / 30)
        assert result.qualified is expected


# ================================================================ slice gating
class TestSliceEligibility:
    def test_clean_slice_is_eligible(self, dataset, criteria):
        graded = direct_responses(dataset, "m", spurious_fraction=0.5)
        metrics = compute_metrics(direct_trigger_slice(graded, "m"))
        assessment = assess_slice(metrics, criteria)
        assert assessment.eligible is True
        assert assessment.needs_structured_output_rerun is False

    def test_slice_over_five_percent_invalid_is_not_eligible(self, dataset, criteria):
        graded = direct_responses(dataset, "m", spurious_fraction=1.0, invalid_fraction=0.2)
        metrics = compute_metrics(direct_trigger_slice(graded, "m"))
        assessment = assess_slice(metrics, criteria)
        assert assessment.invalid_response_rate > criteria.max_invalid_response_rate
        assert assessment.eligible is False
        assert assessment.needs_structured_output_rerun is True
        assert "invalid_response_rate" in assessment.reason

    def test_a_rerun_slice_is_not_asked_to_rerun_again(self, dataset, criteria):
        graded = direct_responses(dataset, "m", spurious_fraction=1.0, invalid_fraction=0.2)
        metrics = compute_metrics(direct_trigger_slice(graded, "m"))
        assessment = assess_slice(metrics, criteria, already_rerun=True)
        assert assessment.eligible is False
        assert assessment.needs_structured_output_rerun is False

    def test_empty_slice_is_not_eligible(self, criteria):
        assessment = assess_slice(compute_metrics([]), criteria)
        assert assessment.eligible is False
        assert "no responses" in assessment.reason


# ============================================================== direct trigger
class TestDirectTrigger:
    def test_slice_is_cell_2_l1_natural_only(self, dataset, cfg):
        graded = direct_responses(dataset, "m", spurious_fraction=0.5)
        rows = direct_trigger_slice(graded, "m")
        assert rows
        assert all(r.cell == 2 for r in rows)
        assert all(r.salience_level is SalienceLevel.L1 for r in rows)
        assert all(r.prompt_variant is PromptVariant.NATURAL for r in rows)
        assert len(rows) == cfg.salience_weights[SalienceLevel.L1]

    def test_fires_above_the_threshold(self, dataset, cfg, comp_cases, criteria):
        decision = evaluate_model(
            "m",
            criteria,
            qualification=qualification_for(cfg, comp_cases, 15, 15, "m"),
            direct_graded=direct_responses(dataset, "m", spurious_fraction=0.6),
        )
        assert decision.spurious_trend_rate >= criteria.spurious_trend_rate_trigger
        assert decision.direct_trigger is True

    def test_does_not_fire_below_the_threshold(self, dataset, cfg, comp_cases, criteria):
        decision = evaluate_model(
            "m",
            criteria,
            qualification=qualification_for(cfg, comp_cases, 15, 15, "m"),
            direct_graded=direct_responses(dataset, "m", spurious_fraction=0.0),
        )
        assert decision.spurious_trend_rate == 0.0
        assert decision.direct_trigger is False

    def test_does_not_fire_for_an_unqualified_model(self, dataset, cfg, comp_cases, criteria):
        decision = evaluate_model(
            "m",
            criteria,
            qualification=qualification_for(cfg, comp_cases, 12, 15, "m"),
            direct_graded=direct_responses(dataset, "m", spurious_fraction=1.0),
        )
        assert decision.eligible is False
        assert decision.direct_trigger is False

    def test_does_not_fire_when_the_slice_is_too_invalid(self, dataset, cfg, comp_cases, criteria):
        decision = evaluate_model(
            "m",
            criteria,
            qualification=qualification_for(cfg, comp_cases, 15, 15, "m"),
            direct_graded=direct_responses(
                dataset, "m", spurious_fraction=1.0, invalid_fraction=0.2
            ),
        )
        assert decision.spurious_trend_rate == 1.0, "valid-only rate is still high"
        assert decision.direct_slice.eligible is False
        assert decision.direct_trigger is False
        assert decision.direct_slice.needs_structured_output_rerun is True

    def test_the_reported_rate_is_the_valid_only_one(self, dataset, cfg, comp_cases, criteria):
        """Even on a slice the gate rejects, the rate recorded is the behavioural
        one -- so the report says what the model did, and separately says the
        slice may not be used."""
        graded = direct_responses(dataset, "m", spurious_fraction=1.0, invalid_fraction=0.2)
        metrics = compute_metrics(direct_trigger_slice(graded, "m"))
        assert metrics.spurious_trend_rate.valid_only.value == 1.0
        assert metrics.spurious_trend_rate.all_trials.value < 1.0
        decision = evaluate_model(
            "m",
            criteria,
            qualification=qualification_for(cfg, comp_cases, 15, 15, "m"),
            direct_graded=graded,
        )
        assert decision.spurious_trend_rate == metrics.spurious_trend_rate.valid_only.value
        assert decision.basis == "valid_only"

    def test_one_invalid_completion_disqualifies_a_fifteen_case_slice(
        self, dataset, cfg, comp_cases, criteria
    ):
        """A consequence of the configured threshold worth knowing before the run:
        the cell-2/L1/natural slice has 15 cases, so a single unparseable
        completion is 6.7% and exceeds max_invalid_response_rate=0.05. At this n
        the gate is effectively 'zero parse failures or re-run'."""
        graded = direct_responses(dataset, "m", spurious_fraction=1.0, invalid_fraction=1 / 15)
        metrics = compute_metrics(direct_trigger_slice(graded, "m"))
        assert metrics.invalid_response_rate.numerator == 1
        assert metrics.invalid_response_rate.value > criteria.max_invalid_response_rate
        decision = evaluate_model(
            "m",
            criteria,
            qualification=qualification_for(cfg, comp_cases, 15, 15, "m"),
            direct_graded=graded,
        )
        assert decision.direct_trigger is False
        assert decision.direct_slice.needs_structured_output_rerun is True

    def test_a_clean_slice_fires_normally(self, dataset, cfg, comp_cases, criteria):
        graded = direct_responses(dataset, "m", spurious_fraction=1.0)
        decision = evaluate_model(
            "m",
            criteria,
            qualification=qualification_for(cfg, comp_cases, 15, 15, "m"),
            direct_graded=graded,
        )
        assert decision.direct_slice.invalid_response_rate == 0.0
        assert decision.direct_trigger is True


# ================================================================ code trigger
class TestCodeTrigger:
    def test_fires_when_provenance_is_ignored(self, dataset, cfg, comp_cases, criteria):
        decision = evaluate_model(
            "m",
            criteria,
            qualification=qualification_for(cfg, comp_cases, 15, 15, "m"),
            code_metrics=code_metrics_for(dataset, cfg, "ignores_provenance", "m"),
        )
        assert decision.provenance_retention_rate == 0.0
        assert decision.code_trigger is True

    def test_does_not_fire_when_provenance_is_retained(self, dataset, cfg, comp_cases, criteria):
        decision = evaluate_model(
            "m",
            criteria,
            qualification=qualification_for(cfg, comp_cases, 15, 15, "m"),
            code_metrics=code_metrics_for(dataset, cfg, "inverts_calibration", "m"),
        )
        assert decision.provenance_retention_rate == 1.0
        assert decision.code_trigger is False

    def test_retention_counts_level_one_so_reading_alone_defeats_the_trigger(
        self, dataset, cfg, comp_cases, criteria
    ):
        """Reading the column and doing nothing scores PRR 1.0 -- the trigger
        does not fire, which is why Effective Provenance Use is also reported."""
        metrics = code_metrics_for(dataset, cfg, "reads_but_ignores", "m")
        decision = evaluate_model(
            "m",
            criteria,
            qualification=qualification_for(cfg, comp_cases, 15, 15, "m"),
            code_metrics=metrics,
        )
        assert metrics.provenance_retention_rate.valid_only.value == 1.0
        assert metrics.effective_provenance_use_rate.valid_only.value == 0.0
        assert decision.code_trigger is False

    def test_does_not_fire_for_an_unqualified_model(self, dataset, cfg, comp_cases, criteria):
        decision = evaluate_model(
            "m",
            criteria,
            qualification=qualification_for(cfg, comp_cases, 10, 10, "m"),
            code_metrics=code_metrics_for(dataset, cfg, "ignores_provenance", "m"),
        )
        assert decision.code_trigger is False

    def test_absent_code_results_cannot_fire_the_trigger(self, cfg, comp_cases, criteria):
        decision = evaluate_model(
            "m", criteria, qualification=qualification_for(cfg, comp_cases, 15, 15, "m")
        )
        assert decision.code_trigger is False
        assert any("no code-agent results" in n for n in decision.notes)


# ==================================================================== OR logic
class TestOrLogic:
    def _decide(self, dataset, cfg, comp_cases, criteria, *, spurious, behaviour, model="m"):
        return evaluate_continuation(
            criteria,
            qualifications={model: qualification_for(cfg, comp_cases, 15, 15, model)},
            direct_graded=direct_responses(dataset, model, spurious_fraction=spurious),
            code_metrics_by_model={model: code_metrics_for(dataset, cfg, behaviour, model)},
        )

    def test_direct_only(self, dataset, cfg, comp_cases, criteria):
        d = self._decide(
            dataset, cfg, comp_cases, criteria, spurious=0.6, behaviour="inverts_calibration"
        )
        assert d.per_model["m"].direct_trigger is True
        assert d.per_model["m"].code_trigger is False
        assert d.project_continue is True

    def test_code_only(self, dataset, cfg, comp_cases, criteria):
        d = self._decide(
            dataset, cfg, comp_cases, criteria, spurious=0.0, behaviour="ignores_provenance"
        )
        assert d.per_model["m"].direct_trigger is False
        assert d.per_model["m"].code_trigger is True
        assert d.project_continue is True

    def test_both(self, dataset, cfg, comp_cases, criteria):
        d = self._decide(
            dataset, cfg, comp_cases, criteria, spurious=0.6, behaviour="ignores_provenance"
        )
        assert d.project_continue is True

    def test_neither(self, dataset, cfg, comp_cases, criteria):
        d = self._decide(
            dataset, cfg, comp_cases, criteria, spurious=0.0, behaviour="inverts_calibration"
        )
        assert d.per_model["m"].triggers is False
        assert d.project_continue is False

    def test_one_triggering_model_carries_the_project(self, dataset, cfg, comp_cases, criteria):
        decision = evaluate_continuation(
            criteria,
            qualifications={
                "quiet": qualification_for(cfg, comp_cases, 15, 15, "quiet"),
                "loud": qualification_for(cfg, comp_cases, 15, 15, "loud"),
            },
            direct_graded=(
                direct_responses(dataset, "quiet", spurious_fraction=0.0)
                + direct_responses(dataset, "loud", spurious_fraction=0.8)
            ),
        )
        assert decision.per_model["quiet"].triggers is False
        assert decision.per_model["loud"].triggers is True
        assert decision.project_continue is True
        assert decision.triggering_models == ["loud"]

    def test_an_unqualified_loud_model_does_not_carry_it(self, dataset, cfg, comp_cases, criteria):
        decision = evaluate_continuation(
            criteria,
            qualifications={"loud": qualification_for(cfg, comp_cases, 10, 10, "loud")},
            direct_graded=direct_responses(dataset, "loud", spurious_fraction=1.0),
        )
        assert decision.project_continue is False

    def test_decision_records_the_criteria_it_applied(self, dataset, cfg, comp_cases, criteria):
        d = self._decide(
            dataset, cfg, comp_cases, criteria, spurious=0.6, behaviour="ignores_provenance"
        )
        assert d.criteria.spurious_trend_rate_trigger == criteria.spurious_trend_rate_trigger
        assert d.basis == "valid_only"

    def test_rerun_list_names_the_blocked_slices(self, dataset, cfg, comp_cases, criteria):
        decision = evaluate_continuation(
            criteria,
            qualifications={"m": qualification_for(cfg, comp_cases, 15, 15, "m")},
            direct_graded=direct_responses(
                dataset, "m", spurious_fraction=1.0, invalid_fraction=0.2
            ),
        )
        assert decision.slices_needing_rerun == ["m/direct"]

    def test_report_renders(self, dataset, cfg, comp_cases, criteria):
        d = self._decide(
            dataset, cfg, comp_cases, criteria, spurious=0.6, behaviour="ignores_provenance"
        )
        text = format_continuation_decision(d)
        assert "project_continue = True" in text
        assert "basis: valid_only" in text

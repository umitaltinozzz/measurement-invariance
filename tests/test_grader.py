"""Grader tests: every (cell, salience, answer) combination lands in one class.

The classification table is the contract between the dataset and every number in
the paper, so it is tested exhaustively rather than by example.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.generate_cases import generate_dataset, load_config
from src.grader import (
    CORRECT_CLASSES,
    ModelResponseRecord,
    ResponseClass,
    abstention_is_warranted,
    grade_all,
    grade_response,
    latent_recovery_is_defined,
)
from src.prompts import PromptVariant
from src.response_schemas import TrendResponse, parse_response
from src.schemas import SalienceLevel, Trend

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(load_config(CONFIG_PATH))


def pick(dataset, cell, salience=SalienceLevel.L1):
    return next(c for c in dataset if c.cell == cell and c.salience_level is salience)


def answer(case, trend, variant=PromptVariant.NATURAL, flag=None, model="m"):
    payload = {"trend": trend}
    if variant is PromptVariant.GUIDED:
        payload["measurement_change_detected"] = (
            case.measurement_function_change if flag is None else flag
        )
    payload["reason"] = "because"
    return ModelResponseRecord(
        case_id=case.case_id, model=model, prompt_variant=variant, raw_text=json.dumps(payload)
    )


def cls_of(case, trend, **kw):
    return grade_response(case, answer(case, trend, **kw)).response_class


# =============================================================== strict parsing
class TestStrictParsing:
    def test_valid_natural_response_parses(self):
        r = parse_response('{"trend": "stable", "reason": "flat"}', PromptVariant.NATURAL)
        assert r.ok and r.trend is TrendResponse.STABLE
        assert r.measurement_change_detected is None

    def test_valid_guided_response_parses(self):
        r = parse_response(
            '{"trend": "increase", "measurement_change_detected": true, "reason": "x"}',
            PromptVariant.GUIDED,
        )
        assert r.ok and r.measurement_change_detected is True

    def test_surrounding_whitespace_is_tolerated(self):
        assert parse_response('\n  {"trend": "stable", "reason": "x"}  \n', PromptVariant.NATURAL).ok

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "not json at all",
            '```json\n{"trend": "stable", "reason": "x"}\n```',
            'Here you go: {"trend": "stable", "reason": "x"}',
            '{"trend": "stable", "reason": "x"} and that is my analysis',
            '[{"trend": "stable", "reason": "x"}]',
            '"just a string"',
            '{"trend": "stable"}',
            '{"reason": "x"}',
            '{"trend": "rising", "reason": "x"}',
            '{"trend": "stable", "reason": "x", "confidence": 0.9}',
        ],
    )
    def test_malformed_natural_responses_are_rejected(self, raw):
        result = parse_response(raw, PromptVariant.NATURAL)
        assert not result.ok and result.error

    def test_natural_schema_rejects_the_guided_field(self):
        raw = '{"trend": "stable", "measurement_change_detected": true, "reason": "x"}'
        assert not parse_response(raw, PromptVariant.NATURAL).ok

    def test_guided_schema_requires_the_flag(self):
        assert not parse_response('{"trend": "stable", "reason": "x"}', PromptVariant.GUIDED).ok

    def test_flag_is_not_coerced_from_a_string(self):
        raw = '{"trend": "stable", "measurement_change_detected": "true", "reason": "x"}'
        assert not parse_response(raw, PromptVariant.GUIDED).ok


# ============================================================ invalid responses
class TestInvalidResponses:
    def test_unparseable_becomes_invalid_not_a_wrong_trend(self, dataset):
        case = pick(dataset, 2)
        graded = grade_response(
            case,
            ModelResponseRecord(
                case_id=case.case_id,
                model="m",
                prompt_variant=PromptVariant.NATURAL,
                raw_text="I think the temperature went up.",
            ),
        )
        assert graded.response_class is ResponseClass.INVALID_RESPONSE
        assert graded.response_class is not ResponseClass.SPURIOUS_TREND
        assert graded.is_correct is False
        assert graded.reported_trend is None
        assert graded.parse_error

    def test_invalid_carries_no_recovery_judgement(self, dataset):
        case = pick(dataset, 4, SalienceLevel.L1)
        graded = grade_response(
            case,
            ModelResponseRecord(
                case_id=case.case_id,
                model="m",
                prompt_variant=PromptVariant.NATURAL,
                raw_text="{}",
            ),
        )
        assert graded.latent_recovery_defined is True
        assert graded.latent_recovery_correct is None


# ================================================================ cell-by-cell
class TestCell1:
    """Nothing changed and nothing is hidden."""

    def test_stable_is_correct(self, dataset):
        assert cls_of(pick(dataset, 1), "stable") is ResponseClass.CORRECT_STABLE

    @pytest.mark.parametrize("trend", ["increase", "decrease"])
    def test_any_direction_is_spurious(self, dataset, trend):
        assert cls_of(pick(dataset, 1), trend) is ResponseClass.SPURIOUS_TREND

    def test_uncertain_is_over_abstention_and_flagged_as_blanket_skepticism(self, dataset):
        graded = grade_response(pick(dataset, 1), answer(pick(dataset, 1), "uncertain"))
        assert graded.response_class is ResponseClass.OVER_ABSTENTION
        assert graded.blanket_skepticism is True


class TestCell2:
    """The main condition: latent fixed, measurement function intervened upon."""

    @pytest.mark.parametrize("salience", list(SalienceLevel))
    def test_ground_truth_is_stable(self, dataset, salience):
        assert pick(dataset, 2, salience).expected_trend is Trend.STABLE

    @pytest.mark.parametrize("salience", list(SalienceLevel))
    @pytest.mark.parametrize("trend", ["increase", "decrease"])
    def test_any_direction_is_spurious_trend(self, dataset, salience, trend):
        assert cls_of(pick(dataset, 2, salience), trend) is ResponseClass.SPURIOUS_TREND

    def test_spurious_holds_even_when_the_series_screams_otherwise(self, dataset):
        """The observed level shifts by at least 10%; truth is still stable."""
        for case in [c for c in dataset if c.cell == 2]:
            direction = "increase" if case.observed_shift_ratio > 0 else "decrease"
            assert cls_of(case, direction) is ResponseClass.SPURIOUS_TREND

    @pytest.mark.parametrize("salience", list(SalienceLevel))
    def test_stable_is_correct(self, dataset, salience):
        assert cls_of(pick(dataset, 2, salience), "stable") is ResponseClass.CORRECT_STABLE

    @pytest.mark.parametrize("salience", list(SalienceLevel))
    def test_uncertain_is_a_warranted_abstention(self, dataset, salience):
        assert cls_of(pick(dataset, 2, salience), "uncertain") is ResponseClass.CORRECT_ABSTENTION

    def test_abstention_and_stable_are_distinguishable(self, dataset):
        case = pick(dataset, 2)
        assert cls_of(case, "stable") is not cls_of(case, "uncertain")
        assert cls_of(case, "stable") in CORRECT_CLASSES
        assert cls_of(case, "uncertain") in CORRECT_CLASSES

    def test_uncertain_is_not_blanket_skepticism_here(self, dataset):
        graded = grade_response(pick(dataset, 2), answer(pick(dataset, 2), "uncertain"))
        assert graded.blanket_skepticism is False


class TestCell3:
    """Over-abstention control, not a reasoning-capability measure.

    With the measurement function held fixed the real change is plainly visible
    in the raw series, so what cell 3 detects is a model applying skepticism it
    has no reason to apply.
    """

    def test_right_direction_is_correct_change(self, dataset):
        case = pick(dataset, 3)
        assert cls_of(case, case.expected_trend.value) is ResponseClass.CORRECT_CHANGE

    def test_stable_is_missed_change(self, dataset):
        assert cls_of(pick(dataset, 3), "stable") is ResponseClass.MISSED_CHANGE

    def test_opposite_direction_is_wrong_direction(self, dataset):
        case = pick(dataset, 3)
        opposite = "decrease" if case.expected_trend is Trend.INCREASE else "increase"
        assert cls_of(case, opposite) is ResponseClass.WRONG_DIRECTION

    def test_uncertain_is_over_abstention(self, dataset):
        assert cls_of(pick(dataset, 3), "uncertain") is ResponseClass.OVER_ABSTENTION

    @pytest.mark.parametrize("salience", list(SalienceLevel))
    def test_abstention_is_never_warranted(self, dataset, salience):
        assert abstention_is_warranted(pick(dataset, 3, salience)) is False


class TestCell4:
    """Graded against latent ground truth even though the readings were transformed."""

    @pytest.mark.parametrize("salience", list(SalienceLevel))
    def test_graded_on_latent_truth_not_observed_direction(self, dataset, salience):
        for case in [c for c in dataset if c.cell == 4 and c.salience_level is salience]:
            assert cls_of(case, case.expected_trend.value) is ResponseClass.CORRECT_CHANGE
            naive = case.naive_observed_trend()
            if naive is not case.expected_trend and naive is not Trend.STABLE:
                assert cls_of(case, naive.value) is ResponseClass.WRONG_DIRECTION

    def test_stable_is_missed_change(self, dataset):
        assert cls_of(pick(dataset, 4), "stable") is ResponseClass.MISSED_CHANGE

    def test_l1_uncertain_is_over_abstention(self, dataset):
        assert cls_of(pick(dataset, 4, SalienceLevel.L1), "uncertain") is (
            ResponseClass.OVER_ABSTENTION
        )

    def test_l2_uncertain_is_a_warranted_abstention(self, dataset):
        """At L2 the device id says the instrument changed but not by how much,
        so the latent change cannot be recovered and abstaining is correct."""
        assert cls_of(pick(dataset, 4, SalienceLevel.L2), "uncertain") is (
            ResponseClass.CORRECT_ABSTENTION
        )


# ======================================================== latent recoverability
class TestLatentRecoverability:
    def test_defined_only_for_cell_4_at_l1(self, dataset):
        for case in dataset:
            expected = case.cell == 4 and case.salience_level is SalienceLevel.L1
            assert latent_recovery_is_defined(case) is expected

    def test_l2_cell_4_never_carries_a_recovery_judgement(self, dataset):
        for case in [c for c in dataset if c.cell == 4 and c.salience_level is SalienceLevel.L2]:
            graded = grade_response(case, answer(case, case.expected_trend.value))
            assert graded.latent_recovery_defined is False
            assert graded.latent_recovery_correct is None

    def test_l1_cell_4_recovery_tracks_correct_change(self, dataset):
        case = pick(dataset, 4, SalienceLevel.L1)
        assert grade_response(case, answer(case, case.expected_trend.value)).latent_recovery_correct
        assert not grade_response(case, answer(case, "stable")).latent_recovery_correct
        assert not grade_response(case, answer(case, "uncertain")).latent_recovery_correct


# ============================================================ the guided flag
class TestMeasurementFlag:
    def test_natural_arm_carries_no_flag(self, dataset):
        graded = grade_response(pick(dataset, 2), answer(pick(dataset, 2), "stable"))
        assert graded.measurement_change_detected is None
        assert graded.measurement_change_correct is None

    def test_guided_flag_is_scored_against_the_case(self, dataset):
        case = pick(dataset, 2)
        right = grade_response(
            case, answer(case, "stable", variant=PromptVariant.GUIDED, flag=True)
        )
        wrong = grade_response(
            case, answer(case, "stable", variant=PromptVariant.GUIDED, flag=False)
        )
        assert right.measurement_change_correct is True
        assert wrong.measurement_change_correct is False

    def test_flag_does_not_change_the_trend_class(self, dataset):
        case = pick(dataset, 2)
        for flag in (True, False):
            graded = grade_response(
                case, answer(case, "increase", variant=PromptVariant.GUIDED, flag=flag)
            )
            assert graded.response_class is ResponseClass.SPURIOUS_TREND


# ===================================================================== batching
class TestGradeAll:
    def test_grades_every_record(self, dataset):
        records = [answer(c, "stable") for c in dataset]
        assert len(grade_all(dataset, records)) == len(dataset)

    def test_unknown_case_id_raises_rather_than_being_dropped(self, dataset):
        record = answer(dataset[0], "stable").model_copy(update={"case_id": "ghost"})
        with pytest.raises(KeyError, match="ghost"):
            grade_all(dataset, [record])

    def test_mismatched_case_and_record_raises(self, dataset):
        with pytest.raises(ValueError, match="is for"):
            grade_response(dataset[1], answer(dataset[0], "stable"))

    def test_classification_is_total(self, dataset):
        """No (cell, salience, answer) combination falls through unclassified."""
        for case in dataset:
            for trend in TrendResponse:
                graded = grade_response(case, answer(case, trend.value))
                assert isinstance(graded.response_class, ResponseClass)
                assert graded.response_class is not ResponseClass.INVALID_RESPONSE

    def test_grading_is_deterministic(self, dataset):
        records = [answer(c, "uncertain") for c in dataset]
        assert [g.model_dump(mode="json") for g in grade_all(dataset, records)] == [
            g.model_dump(mode="json") for g in grade_all(dataset, records)
        ]

"""Competence grader tests: strict parsing, numeric tolerance, no silent passes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.competence_grader import (
    CompetenceClass,
    CompetenceResponseRecord,
    grade_all_competence,
    grade_competence_response,
    parse_competence_response,
)
from src.competence_schemas import (
    CompetenceSkill,
    RangeClass,
    numeric_tolerance,
)
from src.generate_cases import load_config
from src.generate_competence import generate_competence_dataset

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"


@pytest.fixture(scope="module")
def cases():
    return generate_competence_dataset(load_config(CONFIG_PATH))


@pytest.fixture(scope="module")
def rr_case(cases):
    return next(c for c in cases if c.skill is CompetenceSkill.REFERENCE_RANGE)


@pytest.fixture(scope="module")
def af_case(cases):
    return next(c for c in cases if c.skill is CompetenceSkill.AFFINE_INVERSION)


def answer(case, payload, model="m"):
    return CompetenceResponseRecord(
        case_id=case.case_id, model=model, raw_text=json.dumps(payload)
    )


def raw(case, text, model="m"):
    return CompetenceResponseRecord(case_id=case.case_id, model=model, raw_text=text)


# ============================================================== strict parsing (8)
class TestStrictParsing:
    def test_valid_reference_range_response(self):
        r = parse_competence_response(
            '{"classification": "within", "reason": "x"}', CompetenceSkill.REFERENCE_RANGE
        )
        assert r.ok and r.response.classification is RangeClass.WITHIN

    def test_valid_affine_response_accepts_float_and_int(self):
        for payload in ('{"x": 50.0, "reason": "r"}', '{"x": 50, "reason": "r"}'):
            r = parse_competence_response(payload, CompetenceSkill.AFFINE_INVERSION)
            assert r.ok and float(r.response.x) == 50.0

    def test_surrounding_whitespace_is_tolerated(self):
        assert parse_competence_response(
            '\n {"x": 1, "reason": "r"} \n', CompetenceSkill.AFFINE_INVERSION
        ).ok

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "within",
            "The result is within the range.",
            '```json\n{"classification": "within", "reason": "x"}\n```',
            'Answer: {"classification": "within", "reason": "x"}',
            '{"classification": "within"}',
            '{"classification": "inside", "reason": "x"}',
            '{"classification": "within", "reason": "x", "confidence": 1}',
            '[{"classification": "within", "reason": "x"}]',
        ],
    )
    def test_malformed_reference_range_is_rejected(self, text):
        result = parse_competence_response(text, CompetenceSkill.REFERENCE_RANGE)
        assert not result.ok and result.error

    @pytest.mark.parametrize(
        "text",
        [
            '{"x": "50", "reason": "r"}',
            '{"x": true, "reason": "r"}',
            '{"x": null, "reason": "r"}',
            '{"x": "(65-5)/1.2", "reason": "r"}',
            '{"reason": "r"}',
            '{"x": 50}',
            '{"x": 50, "reason": 5}',
            '{"x": 50, "reason": "r", "steps": "..."}',
        ],
    )
    def test_malformed_affine_is_rejected(self, text):
        result = parse_competence_response(text, CompetenceSkill.AFFINE_INVERSION)
        assert not result.ok and result.error

    def test_non_finite_numbers_are_rejected(self):
        for text in ('{"x": NaN, "reason": "r"}', '{"x": Infinity, "reason": "r"}'):
            assert not parse_competence_response(text, CompetenceSkill.AFFINE_INVERSION).ok

    def test_skills_do_not_accept_each_others_schema(self):
        assert not parse_competence_response(
            '{"x": 50, "reason": "r"}', CompetenceSkill.REFERENCE_RANGE
        ).ok
        assert not parse_competence_response(
            '{"classification": "within", "reason": "r"}', CompetenceSkill.AFFINE_INVERSION
        ).ok


# =========================================================== invalid handling (8)
class TestInvalidResponses:
    def test_unparseable_gets_its_own_class(self, rr_case):
        graded = grade_competence_response(rr_case, raw(rr_case, "within, obviously"))
        assert graded.outcome is CompetenceClass.INVALID_RESPONSE
        assert graded.outcome is not CompetenceClass.INCORRECT
        assert graded.is_correct is False
        assert graded.parse_error

    def test_invalid_is_never_scored_correct_even_if_the_text_is_right(self, rr_case):
        graded = grade_competence_response(
            rr_case, raw(rr_case, rr_case.expected_classification.value)
        )
        assert graded.is_correct is False

    def test_invalid_carries_no_reported_answer(self, af_case):
        graded = grade_competence_response(af_case, raw(af_case, "x is 50"))
        assert graded.reported_answer is None
        assert graded.absolute_error is None


# ============================================================ reference range
class TestReferenceRangeGrading:
    def test_correct_classification(self, rr_case):
        graded = grade_competence_response(
            rr_case,
            answer(rr_case, {"classification": rr_case.expected_classification.value, "reason": "r"}),
        )
        assert graded.outcome is CompetenceClass.CORRECT and graded.is_correct

    def test_wrong_classification(self, rr_case):
        wrong = next(c for c in RangeClass if c is not rr_case.expected_classification)
        graded = grade_competence_response(
            rr_case, answer(rr_case, {"classification": wrong.value, "reason": "r"})
        )
        assert graded.outcome is CompetenceClass.INCORRECT and not graded.is_correct

    def test_every_generated_case_grades_correctly_on_its_own_truth(self, cases):
        for case in cases:
            if case.skill is not CompetenceSkill.REFERENCE_RANGE:
                continue
            graded = grade_competence_response(
                case,
                answer(case, {"classification": case.expected_classification.value, "reason": "r"}),
            )
            assert graded.is_correct, case.case_id

    def test_endpoint_answered_within_is_correct(self, cases):
        from src.competence_schemas import BoundaryKind

        for case in cases:
            if getattr(case, "boundary_kind", None) in (
                BoundaryKind.AT_LOWER,
                BoundaryKind.AT_UPPER,
            ):
                graded = grade_competence_response(
                    case, answer(case, {"classification": "within", "reason": "r"})
                )
                assert graded.is_correct
                for wrong in ("below", "above"):
                    assert not grade_competence_response(
                        case, answer(case, {"classification": wrong, "reason": "r"})
                    ).is_correct


# ================================================ affine inversion tolerance (6, 7)
class TestAffineGrading:
    def test_exact_answer_is_correct(self, af_case):
        graded = grade_competence_response(
            af_case, answer(af_case, {"x": af_case.expected_x, "reason": "r"})
        )
        assert graded.is_correct
        assert graded.absolute_error == pytest.approx(0.0)

    def test_integer_form_of_an_integer_answer_is_correct(self, cases):
        case = next(
            c
            for c in cases
            if c.skill is CompetenceSkill.AFFINE_INVERSION and float(c.expected_x).is_integer()
        )
        graded = grade_competence_response(
            case, answer(case, {"x": int(case.expected_x), "reason": "r"})
        )
        assert graded.is_correct

    def test_value_just_inside_the_band_is_correct(self, af_case):
        tol = numeric_tolerance(af_case.expected_x)
        graded = grade_competence_response(
            af_case, answer(af_case, {"x": af_case.expected_x + tol * 0.99, "reason": "r"})
        )
        assert graded.is_correct
        assert graded.tolerance == pytest.approx(tol)

    def test_value_just_outside_the_band_is_incorrect(self, af_case):
        tol = numeric_tolerance(af_case.expected_x)
        graded = grade_competence_response(
            af_case, answer(af_case, {"x": af_case.expected_x + tol * 1.01, "reason": "r"})
        )
        assert graded.outcome is CompetenceClass.INCORRECT
        assert graded.absolute_error > graded.tolerance

    def test_plainly_wrong_answer_is_incorrect(self, af_case):
        graded = grade_competence_response(
            af_case, answer(af_case, {"x": af_case.expected_x * 2 + 17, "reason": "r"})
        )
        assert not graded.is_correct

    def test_the_uninverted_output_is_not_accepted(self, cases):
        """Answering y instead of x is the characteristic failure; it must fail."""
        for case in cases:
            if case.skill is not CompetenceSkill.AFFINE_INVERSION:
                continue
            if abs(case.y - case.expected_x) <= numeric_tolerance(case.expected_x):
                continue  # degenerate case where y == x anyway
            graded = grade_competence_response(case, answer(case, {"x": case.y, "reason": "r"}))
            assert not graded.is_correct, case.case_id

    def test_every_generated_case_grades_correctly_on_its_own_truth(self, cases):
        for case in cases:
            if case.skill is not CompetenceSkill.AFFINE_INVERSION:
                continue
            graded = grade_competence_response(
                case, answer(case, {"x": case.expected_x, "reason": "r"})
            )
            assert graded.is_correct, case.case_id

    def test_comparison_is_numeric_not_string(self, cases):
        case = next(
            c
            for c in cases
            if c.skill is CompetenceSkill.AFFINE_INVERSION and float(c.expected_x).is_integer()
        )
        as_int = grade_competence_response(case, answer(case, {"x": int(case.expected_x), "reason": "r"}))
        as_float = grade_competence_response(case, answer(case, {"x": float(case.expected_x), "reason": "r"}))
        assert as_int.is_correct and as_float.is_correct
        assert as_int.reported_answer != case.case_id  # sanity: it stored a number


# ===================================================================== batching
class TestGradeAll:
    def test_grades_every_record(self, cases):
        records = [
            answer(c, {"classification": "within", "reason": "r"})
            if c.skill is CompetenceSkill.REFERENCE_RANGE
            else answer(c, {"x": 0, "reason": "r"})
            for c in cases
        ]
        assert len(grade_all_competence(cases, records)) == len(cases)

    def test_unknown_case_id_raises(self, cases):
        record = CompetenceResponseRecord(case_id="ghost", model="m", raw_text="{}")
        with pytest.raises(KeyError, match="ghost"):
            grade_all_competence(cases, [record])

    def test_mismatched_case_and_record_raises(self, cases):
        with pytest.raises(ValueError, match="is for"):
            grade_competence_response(cases[1], raw(cases[0], "{}"))

    def test_grading_is_deterministic(self, cases):
        records = [raw(c, "{}") for c in cases]
        first = grade_all_competence(cases, records)
        second = grade_all_competence(cases, records)
        assert [g.model_dump() for g in first] == [g.model_dump() for g in second]

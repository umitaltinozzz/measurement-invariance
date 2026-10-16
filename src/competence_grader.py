"""Strict parsing and deterministic grading of competence pre-test responses.

Same two commitments as the main grader:

* parsing is strict -- exact JSON, no fences, no extra keys, no coercion -- and
  an unparseable completion becomes its own class rather than being counted as a
  wrong answer;
* nothing here assumes the runner used constrained decoding. If a runner later
  adds structured outputs the invalid rate falls, and that shows up in the
  metrics; it changes nothing about how an answer is scored.

Numeric answers are compared numerically, never as strings: ``50``, ``50.0`` and
``49.99999`` are all the same answer to "what is x", and a string comparison
would score two of them wrong for reasons that have nothing to do with the skill
being tested.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, ValidationError

from .competence_schemas import (
    CompetenceCase,
    CompetenceSkill,
    RangeClass,
    numeric_tolerance,
    within_tolerance,
)


class CompetenceClass(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    INVALID_RESPONSE = "invalid_response"


class ReferenceRangeResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    classification: RangeClass
    reason: str = Field(strict=True)


class AffineInversionResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # StrictInt | StrictFloat accepts 50 and 50.0 -- both are valid JSON numbers
    # for the same answer -- while rejecting "50", true, and null. A plain
    # strict float would reject the integer form, and a lax float would accept
    # the string form; neither is what "give x as a number" means.
    x: Union[StrictInt, StrictFloat]
    reason: str = Field(strict=True)


CompetenceParsedModel = Union[ReferenceRangeResponse, AffineInversionResponse]

_MODEL_BY_SKILL = {
    CompetenceSkill.REFERENCE_RANGE: ReferenceRangeResponse,
    CompetenceSkill.AFFINE_INVERSION: AffineInversionResponse,
}


class CompetenceParseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    skill: CompetenceSkill
    response: Optional[CompetenceParsedModel] = None
    error: Optional[str] = None


def parse_competence_response(raw_text: str, skill: CompetenceSkill) -> CompetenceParseResult:
    """Parse one completion under the schema its skill asked for."""
    skill = CompetenceSkill(skill)
    model_cls = _MODEL_BY_SKILL[skill]

    if raw_text is None or not raw_text.strip():
        return CompetenceParseResult(ok=False, skill=skill, error="empty response")

    try:
        payload: Any = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        return CompetenceParseResult(ok=False, skill=skill, error=f"not valid JSON: {exc.msg}")

    if not isinstance(payload, dict):
        return CompetenceParseResult(
            ok=False,
            skill=skill,
            error=f"top level is {type(payload).__name__}, not an object",
        )

    try:
        parsed = model_cls(**payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        return CompetenceParseResult(ok=False, skill=skill, error=f"schema mismatch: {problems}")

    if isinstance(parsed, AffineInversionResponse) and not math.isfinite(float(parsed.x)):
        return CompetenceParseResult(
            ok=False, skill=skill, error="x is not a finite number"
        )

    return CompetenceParseResult(ok=True, skill=skill, response=parsed)


class CompetenceResponseRecord(BaseModel):
    """One raw completion, as it comes back from a runner."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    model: str
    raw_text: str


class GradedCompetenceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    model: str
    skill: CompetenceSkill

    outcome: CompetenceClass
    is_correct: bool
    expected_answer: str
    reported_answer: Optional[str] = None
    parse_error: Optional[str] = None

    #: affine only: how far the number was from the truth, and the tolerance it
    #: was judged against. Kept so a near miss can be told from a wild one.
    absolute_error: Optional[float] = None
    tolerance: Optional[float] = None


def grade_competence_response(
    case: CompetenceCase,
    record: CompetenceResponseRecord,
    parsed: Optional[CompetenceParseResult] = None,
) -> GradedCompetenceResponse:
    if record.case_id != case.case_id:
        raise ValueError(f"record is for {record.case_id}, not {case.case_id}")

    parsed = parsed or parse_competence_response(record.raw_text, case.skill)
    common = dict(case_id=case.case_id, model=record.model, skill=case.skill)

    if not parsed.ok:
        return GradedCompetenceResponse(
            **common,
            outcome=CompetenceClass.INVALID_RESPONSE,
            is_correct=False,
            expected_answer=str(case.expected_answer),
            parse_error=parsed.error,
        )

    if case.skill is CompetenceSkill.REFERENCE_RANGE:
        reported = parsed.response.classification
        correct = reported is case.expected_classification
        return GradedCompetenceResponse(
            **common,
            outcome=CompetenceClass.CORRECT if correct else CompetenceClass.INCORRECT,
            is_correct=correct,
            expected_answer=case.expected_classification.value,
            reported_answer=reported.value,
        )

    reported_x = float(parsed.response.x)
    correct = within_tolerance(reported_x, case.expected_x)
    return GradedCompetenceResponse(
        **common,
        outcome=CompetenceClass.CORRECT if correct else CompetenceClass.INCORRECT,
        is_correct=correct,
        expected_answer=repr(case.expected_x),
        reported_answer=repr(reported_x),
        absolute_error=abs(reported_x - case.expected_x),
        tolerance=numeric_tolerance(case.expected_x),
    )


def grade_all_competence(
    cases: Iterable[CompetenceCase], records: Sequence[CompetenceResponseRecord]
) -> List[GradedCompetenceResponse]:
    """Grade a batch, joining on ``case_id``. Unknown ids raise rather than drop."""
    by_id: Dict[str, CompetenceCase] = {c.case_id: c for c in cases}
    graded: List[GradedCompetenceResponse] = []
    for record in records:
        case = by_id.get(record.case_id)
        if case is None:
            raise KeyError(f"no competence case named {record.case_id!r}")
        graded.append(grade_competence_response(case, record))
    return graded

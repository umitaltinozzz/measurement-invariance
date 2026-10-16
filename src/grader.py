"""Deterministic grading of model responses against latent ground truth.

No LLM judge, no fuzzy matching: the classification is a total function of
(case, parsed response).  ``reason`` text is stored but never graded -- rating
free-text justifications would put a second, unvalidated measurement function
between the model and the reported number, which is the exact failure mode this
project studies.

Response classes
----------------
correct_stable      truth is stable, model said stable
correct_change      truth is a real change, model named the right direction
correct_abstention  model abstained where abstention is epistemically warranted
spurious_trend      truth is stable, model claimed a direction   <- primary failure
missed_change       truth is a real change, model said stable
wrong_direction     truth is a real change, model named the opposite direction
over_abstention     model abstained where the log did support an answer
invalid_response    the completion did not parse under its schema

When abstention is warranted
----------------------------
``uncertain`` is not uniformly good or bad; it depends on what the log actually
supports.  The rule used here:

    abstention_warranted = measurement_function_change
                           and (not true_world_change or salience is L2)

* cell 1 -- nothing changed and nothing is hidden.  ``uncertain`` is blanket
  skepticism, flagged as such on top of ``over_abstention``.
* cell 2 (L1 and L2) -- the instrument changed and the truth is "no change".
  Declining to call a trend is a defensible reading, weaker than normalising to
  ``stable`` but not a failure. Reported separately from ``correct_stable`` so
  the two can be told apart.
* cell 3 -- the instrument is constant and the change is plainly in the series.
  ``uncertain`` is over-abstention (proje.md section 12's blanket skepticism).
* cell 4 L1 -- reference ranges expose the transformation, so the latent change
  is recoverable. ``uncertain`` is over-abstention.
* cell 4 L2 -- the embedded id says THAT the device changed but not BY HOW MUCH,
  so no coefficients are available and the latent change genuinely cannot be
  recovered. ``uncertain`` is the correct answer there, and that same fact is
  why L2 x cell 4 is excluded from Latent Recovery Accuracy.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from .prompts import PromptVariant
from .response_schemas import ParseResult, TrendResponse, parse_response
from .schemas import BenchmarkCase, SalienceLevel, Trend


class ResponseClass(str, Enum):
    CORRECT_STABLE = "correct_stable"
    CORRECT_CHANGE = "correct_change"
    CORRECT_ABSTENTION = "correct_abstention"
    SPURIOUS_TREND = "spurious_trend"
    MISSED_CHANGE = "missed_change"
    WRONG_DIRECTION = "wrong_direction"
    OVER_ABSTENTION = "over_abstention"
    INVALID_RESPONSE = "invalid_response"


CORRECT_CLASSES = frozenset(
    {
        ResponseClass.CORRECT_STABLE,
        ResponseClass.CORRECT_CHANGE,
        ResponseClass.CORRECT_ABSTENTION,
    }
)


class ModelResponseRecord(BaseModel):
    """One raw completion, as it comes back from a runner."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    model: str
    prompt_variant: PromptVariant
    raw_text: str


class GradedResponse(BaseModel):
    """One graded completion, carrying every field the metrics group by."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    model: str
    prompt_variant: PromptVariant
    cell: int
    salience_level: SalienceLevel
    expected_trend: Trend

    # Carried so the metrics layer never has to re-open the dataset to work out
    # a denominator.
    true_world_change: bool
    measurement_function_change: bool
    abstention_warranted: bool

    response_class: ResponseClass
    is_correct: bool
    reported_trend: Optional[TrendResponse] = None
    parse_error: Optional[str] = None

    # Blanket skepticism: abstaining in cell 1, where nothing changed at all and
    # nothing was concealed. Tracked on top of the class, not instead of it.
    blanket_skepticism: bool = False

    # Guided arm only; None in the natural arm, where the flag is not requested.
    measurement_change_detected: Optional[bool] = None
    measurement_change_correct: Optional[bool] = None

    # Latent Recovery Accuracy is only defined where recovery is possible at all.
    latent_recovery_defined: bool = False
    latent_recovery_correct: Optional[bool] = None


def abstention_is_warranted(case: BenchmarkCase) -> bool:
    """See the module docstring for the rationale of each branch."""
    if not case.measurement_function_change:
        return False
    if not case.true_world_change:
        return True
    return case.salience_level is SalienceLevel.L2


def latent_recovery_is_defined(case: BenchmarkCase) -> bool:
    """Recovering the latent change requires all three of:

    1. a real latent change to recover (cells 3 and 4),
    2. a measurement function standing in the way of reading it off (cell 4),
    3. enough exposed provenance to invert that function (L1's reference range).

    That is exactly cell 4 at L1. L2 x cell 4 is epistemically undefined and is
    excluded from the metric's denominator rather than scored as a failure.
    """
    return (
        case.true_world_change
        and case.measurement_function_change
        and case.salience_level is SalienceLevel.L1
    )


def _classify_trend(case: BenchmarkCase, reported: TrendResponse) -> ResponseClass:
    if reported is TrendResponse.UNCERTAIN:
        return (
            ResponseClass.CORRECT_ABSTENTION
            if abstention_is_warranted(case)
            else ResponseClass.OVER_ABSTENTION
        )

    if case.expected_trend is Trend.STABLE:
        if reported is TrendResponse.STABLE:
            return ResponseClass.CORRECT_STABLE
        # increase or decrease where the latent state never moved
        return ResponseClass.SPURIOUS_TREND

    if reported is TrendResponse.STABLE:
        return ResponseClass.MISSED_CHANGE
    if reported.value == case.expected_trend.value:
        return ResponseClass.CORRECT_CHANGE
    return ResponseClass.WRONG_DIRECTION


def grade_response(
    case: BenchmarkCase, record: ModelResponseRecord, parsed: Optional[ParseResult] = None
) -> GradedResponse:
    if record.case_id != case.case_id:
        raise ValueError(f"record is for {record.case_id}, not {case.case_id}")

    parsed = parsed or parse_response(record.raw_text, record.prompt_variant)

    common = dict(
        case_id=case.case_id,
        model=record.model,
        prompt_variant=record.prompt_variant,
        cell=case.cell,
        salience_level=case.salience_level,
        expected_trend=case.expected_trend,
        true_world_change=case.true_world_change,
        measurement_function_change=case.measurement_function_change,
        abstention_warranted=abstention_is_warranted(case),
    )

    if not parsed.ok:
        # An unparseable completion is a distinct outcome. It is never folded
        # into a trend class, in either direction.
        return GradedResponse(
            **common,
            response_class=ResponseClass.INVALID_RESPONSE,
            is_correct=False,
            parse_error=parsed.error,
            latent_recovery_defined=latent_recovery_is_defined(case),
            latent_recovery_correct=None,
        )

    reported = parsed.trend
    assert reported is not None  # guaranteed by parsed.ok
    response_class = _classify_trend(case, reported)

    flag = parsed.measurement_change_detected
    flag_correct = None if flag is None else flag == case.measurement_function_change

    recovery_defined = latent_recovery_is_defined(case)
    recovery_correct = (
        response_class is ResponseClass.CORRECT_CHANGE if recovery_defined else None
    )

    return GradedResponse(
        **common,
        response_class=response_class,
        is_correct=response_class in CORRECT_CLASSES,
        reported_trend=reported,
        blanket_skepticism=(
            reported is TrendResponse.UNCERTAIN
            and not case.true_world_change
            and not case.measurement_function_change
        ),
        measurement_change_detected=flag,
        measurement_change_correct=flag_correct,
        latent_recovery_defined=recovery_defined,
        latent_recovery_correct=recovery_correct,
    )


def grade_all(
    cases: Iterable[BenchmarkCase], records: Sequence[ModelResponseRecord]
) -> List[GradedResponse]:
    """Grade a batch, joining on ``case_id``.

    Raises on an unknown ``case_id`` rather than dropping the row: a response
    that silently vanishes would shrink a denominator without anyone noticing.
    """
    by_id: Dict[str, BenchmarkCase] = {c.case_id: c for c in cases}
    graded: List[GradedResponse] = []
    for record in records:
        case = by_id.get(record.case_id)
        if case is None:
            raise KeyError(f"no case named {record.case_id!r} in the supplied dataset")
        graded.append(grade_response(case, record))
    return graded

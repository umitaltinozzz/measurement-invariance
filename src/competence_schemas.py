"""Schemas for the mandatory competence pre-test (proje.md section 7).

The pre-test exists to close a confound, not to be interesting on its own.  If a
model fails at L1 in the main benchmark, that failure has two possible readings:

1. it can do the arithmetic but does not deploy it when provenance is implicit
   -- a measurement-invariance failure, which is the paper's claim; or
2. it cannot do the arithmetic at all -- a plain capability failure, which says
   nothing about invariance.

Reading (1) is only available for models that have been shown to do (2)'s work.
So the pre-test asks the two prerequisite skills directly and in isolation:

* **reference-range competence** -- given a closed interval and a result, say
  whether the result is below, within, or above it;
* **affine-inversion competence** -- given ``y = a*x + b`` and ``y``, recover ``x``.

Ground truth is computed here, programmatically, and enforced by validators, the
same way the main benchmark does it.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Literal, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Below this the affine construction stops being numerically well conditioned:
#: recovering x from y divides by a, so a small a amplifies any float error.
MIN_AFFINE_GAIN = 0.2


def numeric_tolerance(expected: float) -> float:
    """Tolerance for a recovered numeric answer.

    Relative for anything of appreciable size, with an absolute floor so that an
    expected value at or near zero stays checkable.
    """
    return max(1e-6, abs(expected) * 1e-4)


def within_tolerance(actual: float, expected: float) -> bool:
    """Never compare a model's number as a string; compare it numerically.

    NaN and infinity are rejected outright rather than propagating through the
    comparison, where ``abs(nan - x) <= tol`` is silently False anyway but for
    the wrong reason.
    """
    if actual is None or not math.isfinite(actual):
        return False
    return abs(actual - expected) <= numeric_tolerance(expected)


class CompetenceSkill(str, Enum):
    REFERENCE_RANGE = "reference_range"
    AFFINE_INVERSION = "affine_inversion"


class RangeClass(str, Enum):
    BELOW = "below"
    WITHIN = "within"
    ABOVE = "above"


class BoundaryKind(str, Enum):
    """Where a reference-range case sits relative to the interval endpoints.

    Tracked so the boundary cases can be placed deliberately rather than landing
    wherever sampling puts them, and so accuracy can be broken out by them later.
    """

    NONE = "none"
    AT_LOWER = "at_lower"  # result == low, which is WITHIN under a closed interval
    AT_UPPER = "at_upper"  # result == high, likewise WITHIN
    JUST_BELOW = "just_below"
    JUST_ABOVE = "just_above"


def classify_against_range(result: float, reference_range: Tuple[float, float]) -> RangeClass:
    """The ground truth function. Closed interval: both endpoints are WITHIN."""
    low, high = reference_range
    if result < low:
        return RangeClass.BELOW
    if result > high:
        return RangeClass.ABOVE
    return RangeClass.WITHIN


class ReferenceRangeCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    random_seed: int
    skill: Literal[CompetenceSkill.REFERENCE_RANGE] = CompetenceSkill.REFERENCE_RANGE

    analyte: str
    result: float
    reference_range: Tuple[float, float]
    expected_classification: RangeClass
    boundary_kind: BoundaryKind = BoundaryKind.NONE

    @field_validator("reference_range")
    @classmethod
    def _range_ordered(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError(f"reference_range must be increasing, got {v}")
        return v

    @field_validator("analyte")
    @classmethod
    def _analyte_is_fictional(cls, v: str) -> str:
        # Real analyte names would let a model answer from memorised clinical
        # knowledge instead of from the interval it was given.
        if not v.startswith(("Analyte-", "Marker-", "Index-", "Compound-")):
            raise ValueError(f"analyte {v!r} must be a fictional placeholder name")
        return v

    @model_validator(mode="after")
    def _label_matches_the_arithmetic(self) -> "ReferenceRangeCase":
        expected = classify_against_range(self.result, self.reference_range)
        if self.expected_classification is not expected:
            raise ValueError(
                f"expected_classification={self.expected_classification.value} contradicts "
                f"result={self.result} against closed interval {self.reference_range} "
                f"(should be {expected.value})"
            )
        return self

    @model_validator(mode="after")
    def _boundary_kind_matches_the_numbers(self) -> "ReferenceRangeCase":
        low, high = self.reference_range
        actual = BoundaryKind.NONE
        if self.result == low:
            actual = BoundaryKind.AT_LOWER
        elif self.result == high:
            actual = BoundaryKind.AT_UPPER
        if self.boundary_kind in (BoundaryKind.AT_LOWER, BoundaryKind.AT_UPPER):
            if self.boundary_kind is not actual:
                raise ValueError(
                    f"boundary_kind={self.boundary_kind.value} but result={self.result} "
                    f"is not that endpoint of {self.reference_range}"
                )
        elif actual is not BoundaryKind.NONE:
            raise ValueError(
                f"result sits exactly on an endpoint of {self.reference_range} but "
                f"boundary_kind is {self.boundary_kind.value}"
            )
        return self

    @property
    def expected_answer(self) -> str:
        return self.expected_classification.value


class AffineInversionCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    random_seed: int
    skill: Literal[CompetenceSkill.AFFINE_INVERSION] = CompetenceSkill.AFFINE_INVERSION

    a: float
    b: float
    y: float
    expected_x: float

    @field_validator("a")
    @classmethod
    def _gain_positive_and_conditioned(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"gain a must be positive, got {v}")
        if v < MIN_AFFINE_GAIN:
            raise ValueError(
                f"gain a={v} is below {MIN_AFFINE_GAIN}; inverting it amplifies float "
                "error enough to make the tolerance check meaningless"
            )
        return v

    @model_validator(mode="after")
    def _answer_satisfies_the_equation(self) -> "AffineInversionCase":
        reconstructed = self.a * self.expected_x + self.b
        if not within_tolerance(reconstructed, self.y):
            raise ValueError(
                f"expected_x={self.expected_x} does not satisfy y={self.a}*x+{self.b}: "
                f"gives {reconstructed}, not {self.y}"
            )
        return self

    @property
    def expected_answer(self) -> float:
        return self.expected_x


#: Discriminated on ``skill`` so a JSONL line round-trips back to the right type.
CompetenceCase = Annotated[
    Union[ReferenceRangeCase, AffineInversionCase], Field(discriminator="skill")
]


class CompetenceCaseAdapter(BaseModel):
    """Wrapper used only to validate a bare dict into the right case type."""

    model_config = ConfigDict(frozen=True)

    case: CompetenceCase

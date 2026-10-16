"""Prompts for the competence pre-test.

These deliberately share no vocabulary with the main benchmark.  Nothing here
mentions instruments, calibration, sensors, provenance, trends, world change or
comparability, and the affine questions are posed as plain arithmetic rather than
as anything to do with a measuring device.

The reason is that the pre-test's whole job is to establish a capability floor
*independently*.  If it primed the model to think about instrument changes, a
model could pass it for the wrong reason, and the confound it exists to close
would still be open.  ``tests/test_competence_generator.py`` asserts the absence
of that vocabulary.

The closed-interval convention is stated explicitly in every reference-range
prompt. Leaving it implicit would test whether the model guesses the same
convention as the grader, which is a different question from whether it can
apply one it was given.
"""

from __future__ import annotations

from typing import NamedTuple

from .competence_schemas import (
    AffineInversionCase,
    CompetenceCase,
    CompetenceSkill,
    ReferenceRangeCase,
)

SYSTEM_PROMPT = (
    "You answer with a single JSON object and no other text: no preamble, no markdown "
    "fences, no commentary."
)

REFERENCE_RANGE_SCHEMA = """{
  "classification": "below" | "within" | "above",
  "reason": "<at most two sentences>"
}"""

AFFINE_SCHEMA = """{
  "x": <number>,
  "reason": "<at most two sentences>"
}"""

_REFERENCE_RANGE_TEMPLATE = """A laboratory reports a result for {analyte}.

Reference range: [{low:g}, {high:g}]
Result: {result:g}

The reference range is a CLOSED interval: a result equal to {low:g} or equal to
{high:g} counts as within the range. A result less than {low:g} is below the range;
a result greater than {high:g} is above the range.

Classify the result.

Answer with exactly this JSON object:

{schema}

Respond with the JSON object only."""

_AFFINE_TEMPLATE = """A linear relationship converts an input number x into an output number y:

    y = a * x + b

For this problem, a = {a:g} and b = {b:g}.

The output is y = {y:g}. Find x.

Answer with exactly this JSON object:

{schema}

Give "x" as a plain JSON number, not as a string and not as an expression.

Respond with the JSON object only."""


class CompetencePrompt(NamedTuple):
    system: str
    user: str
    skill: CompetenceSkill


def build_reference_range_prompt(case: ReferenceRangeCase) -> CompetencePrompt:
    low, high = case.reference_range
    return CompetencePrompt(
        system=SYSTEM_PROMPT,
        user=_REFERENCE_RANGE_TEMPLATE.format(
            analyte=case.analyte,
            low=low,
            high=high,
            result=case.result,
            schema=REFERENCE_RANGE_SCHEMA,
        ),
        skill=CompetenceSkill.REFERENCE_RANGE,
    )


def build_affine_prompt(case: AffineInversionCase) -> CompetencePrompt:
    return CompetencePrompt(
        system=SYSTEM_PROMPT,
        user=_AFFINE_TEMPLATE.format(a=case.a, b=case.b, y=case.y, schema=AFFINE_SCHEMA),
        skill=CompetenceSkill.AFFINE_INVERSION,
    )


def build_competence_prompt(case: CompetenceCase) -> CompetencePrompt:
    if case.skill is CompetenceSkill.REFERENCE_RANGE:
        return build_reference_range_prompt(case)
    return build_affine_prompt(case)

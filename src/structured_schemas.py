"""Minimal JSON Schemas used for constrained decoding on compatible providers.

These describe the existing strict parsers exactly; they do not change how an
answer is graded. Keeping the schemas inline avoids provider-specific behavior
for ``$ref`` expansion.
"""

from __future__ import annotations

from .prompts import PromptVariant
from .competence_schemas import CompetenceSkill

_TREND = {"type": "string", "enum": ["increase", "decrease", "stable", "uncertain"]}
_REASON = {"type": "string"}

NATURAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"trend": _TREND, "reason": _REASON},
    "required": ["trend", "reason"],
    "additionalProperties": False,
}

GUIDED_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "trend": _TREND,
        "measurement_change_detected": {"type": "boolean"},
        "reason": _REASON,
    },
    "required": ["trend", "measurement_change_detected", "reason"],
    "additionalProperties": False,
}

REFERENCE_RANGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": ["below", "within", "above"]},
        "reason": _REASON,
    },
    "required": ["classification", "reason"],
    "additionalProperties": False,
}

AFFINE_INVERSION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "number"}, "reason": _REASON},
    "required": ["x", "reason"],
    "additionalProperties": False,
}

CODE_AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}, "answer": _TREND, "reason": _REASON},
    "required": ["code", "answer", "reason"],
    "additionalProperties": False,
}

CODE_AGENT_RESPONSE_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "code_lines": {"type": "array", "items": {"type": "string"}},
        "answer": _TREND, "reason": _REASON,
    },
    "required": ["code_lines", "answer", "reason"],
    "additionalProperties": False,
}


def direct_response_schema(variant: PromptVariant | str) -> dict:
    return (
        GUIDED_RESPONSE_SCHEMA
        if PromptVariant(variant) is PromptVariant.GUIDED
        else NATURAL_RESPONSE_SCHEMA
    )


def competence_response_schema(skill: CompetenceSkill | str) -> dict:
    return (
        REFERENCE_RANGE_RESPONSE_SCHEMA
        if CompetenceSkill(skill) is CompetenceSkill.REFERENCE_RANGE
        else AFFINE_INVERSION_RESPONSE_SCHEMA
    )

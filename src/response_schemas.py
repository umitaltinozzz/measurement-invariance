"""Strict parsing of model responses.

Parsing is deliberately unforgiving: exact JSON, no markdown fences, no prose
around it, no extra keys, no missing keys, no coercion of ``"true"`` to ``True``.

The reason is a measurement-validity one, and it is the same argument the paper
itself makes.  A lenient parser is a *measurement function* applied to the
model's output, and a lenient one changes what is being measured: recovering
``{"trend": "increase"}`` out of a paragraph means the number reported as
"spurious trend rate" is partly a number about the parser.  A formatting failure
is a real, separately reportable event, so it gets its own class
(``invalid_response``) and its own rate, and is never silently reclassified as a
wrong trend.

If invalid-response rates come out high in the pilot, the fix belongs in the
runner (constrained decoding / structured outputs), not here.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .prompts import PromptVariant


class TrendResponse(str, Enum):
    """The answer space offered to the model. Wider than ground truth: models may
    abstain with ``uncertain``, which no case is ever labelled with."""

    INCREASE = "increase"
    DECREASE = "decrease"
    STABLE = "stable"
    UNCERTAIN = "uncertain"


# Strictness is applied per field rather than model-wide. Model-wide strict mode
# would also require `trend` to arrive as a TrendResponse instance, which JSON
# cannot express; the enum must accept its own string value. Everything else is
# strict, so `"true"` is not silently read as `True` and a number is not read as
# a reason string.


class NaturalResponse(BaseModel):
    """Main arm. No measurement-related field is requested or accepted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trend: TrendResponse
    reason: str = Field(strict=True)


class GuidedResponse(BaseModel):
    """Ablation arm. Adds the provenance flag."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trend: TrendResponse
    measurement_change_detected: bool = Field(strict=True)
    reason: str = Field(strict=True)


ParsedModel = Union[NaturalResponse, GuidedResponse]

_MODEL_BY_VARIANT = {
    PromptVariant.NATURAL: NaturalResponse,
    PromptVariant.NATURAL_SUSTAINED: NaturalResponse,
    PromptVariant.GUIDED: GuidedResponse,
}


class ParseResult(BaseModel):
    """Outcome of parsing one raw completion.

    Exactly one of ``response`` / ``error`` is set.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    ok: bool
    variant: PromptVariant
    response: Optional[ParsedModel] = None
    error: Optional[str] = None

    @property
    def trend(self) -> Optional[TrendResponse]:
        return self.response.trend if self.response is not None else None

    @property
    def measurement_change_detected(self) -> Optional[bool]:
        return getattr(self.response, "measurement_change_detected", None)


def parse_response(raw_text: str, variant: PromptVariant) -> ParseResult:
    """Parse one completion under the schema its prompt variant asked for.

    Only surrounding whitespace is tolerated. Everything else -- fences, prose,
    trailing commentary, extra or missing keys, wrong types -- is a parse
    failure, reported as such rather than guessed at.
    """
    variant = PromptVariant(variant)
    model_cls = _MODEL_BY_VARIANT[variant]

    if raw_text is None:
        return ParseResult(ok=False, variant=variant, error="empty response")
    # Reasoning models may still wrap an answer in an explicit think block or
    # emit a Harmony channel transcript. Only the final channel is a response.
    text = raw_text
    if "<|channel|>" in text:
        final = re.search(r"<\|channel\|>final\s*(.*)", text, flags=re.DOTALL)
        if final is None:
            return ParseResult(ok=False, variant=variant, error="missing final channel")
        text = final.group(1)
        text = re.split(r"<\|(?!channel\|)[^|]+\|>", text, maxsplit=1)[0]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return ParseResult(ok=False, variant=variant, error="empty response")

    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        return ParseResult(ok=False, variant=variant, error=f"not valid JSON: {exc.msg}")

    if not isinstance(payload, dict):
        return ParseResult(
            ok=False, variant=variant, error=f"top level is {type(payload).__name__}, not an object"
        )

    try:
        parsed = model_cls(**payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        return ParseResult(ok=False, variant=variant, error=f"schema mismatch: {problems}")

    return ParseResult(ok=True, variant=variant, response=parsed)

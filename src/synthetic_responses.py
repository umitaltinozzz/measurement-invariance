"""Synthetic model responses with known error rates.

Not a model and not a baseline: a way to exercise the grader and the metrics
pipeline against answers whose classification is known in advance, so that the
numbers can be validated before any API budget is spent.  If a fixture built to
produce a 25% Spurious Trend Rate does not report 25%, the bug is in the
pipeline, and finding that out costs nothing here and a full pilot run later.

Corruptions are applied to deterministic prefixes of each subset (ordered by
``case_id``), never sampled, so a fixture is exactly reproducible and its target
rates are exact rather than approximate.

Note on ``invalid_rate``: invalid responses are excluded from the denominators of
the trend metrics, so with ``invalid_rate > 0`` the realised Spurious Trend Rate
is ``n_spurious / (n_stable - n_invalid)``, not the requested fraction. Pass
``invalid_rate=0`` when a fixture needs to hit an exact target rate.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List

from .grader import ModelResponseRecord
from .prompts import PromptVariant
from .response_schemas import TrendResponse
from .schemas import BenchmarkCase, Trend

MALFORMED_COMPLETION = "Sure! Here is my analysis:\n```json\n{'trend': 'stable'}\n```"


def count_for(fraction: float, n: int) -> int:
    """How many of ``n`` cases a requested fraction corrupts.

    Round-half-up, not Python's banker's rounding: ``round(12.5)`` is 12, which
    would turn a requested 25% of 50 into 24%. Exported so tests can assert
    against the same arithmetic instead of reimplementing it.

    A fraction is only exactly representable when ``fraction * n`` is an integer.
    With 50 stable-truth cases, 25% is 12.5 and therefore is not; the realised
    rate is 13/50. Slice to a divisible subset when an exact rate is needed.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return math.floor(fraction * n + 0.5)


def _counts_for(n: int, fractions: Dict[str, float]) -> List[str]:
    """Behaviour label per index, corruptions first, then ``correct``."""
    labels: List[str] = []
    for behaviour, fraction in fractions.items():
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{behaviour} fraction must be in [0, 1], got {fraction}")
        labels.extend([behaviour] * count_for(fraction, n))
    if len(labels) > n:
        raise ValueError(
            f"requested behaviours cover {len(labels)} of only {n} cases; fractions must sum to <= 1"
        )
    labels.extend(["correct"] * (n - len(labels)))
    return labels


def _naive_direction(case: BenchmarkCase) -> TrendResponse:
    """What a provenance-blind reader would claim. Used for spurious answers so
    the fixture resembles the failure mode being measured."""
    naive = case.naive_observed_trend()
    if naive is Trend.INCREASE:
        return TrendResponse.INCREASE
    if naive is Trend.DECREASE:
        return TrendResponse.DECREASE
    return TrendResponse.INCREASE  # cell 1: no shift at all, pick a side


def _opposite(trend: Trend) -> TrendResponse:
    return (
        TrendResponse.DECREASE if trend is Trend.INCREASE else TrendResponse.INCREASE
    )


def _render(
    trend: TrendResponse, variant: PromptVariant, flag: bool, reason: str
) -> str:
    payload: Dict[str, object] = {"trend": trend.value}
    if PromptVariant(variant) is PromptVariant.GUIDED:
        payload["measurement_change_detected"] = flag
    payload["reason"] = reason
    return json.dumps(payload)


def make_synthetic_responses(
    cases: Iterable[BenchmarkCase],
    model: str = "synthetic",
    variant: PromptVariant = PromptVariant.NATURAL,
    *,
    spurious_trend_rate: float = 0.0,
    over_abstention_rate: float = 0.0,
    miss_rate: float = 0.0,
    wrong_direction_rate: float = 0.0,
    invalid_rate: float = 0.0,
    measurement_flag_accuracy: float = 1.0,
) -> List[ModelResponseRecord]:
    """Build one response per case with the requested error profile.

    ``spurious_trend_rate`` and ``over_abstention_rate`` apply to the
    stable-truth subset (cells 1 and 2); ``miss_rate``, ``wrong_direction_rate``
    and ``over_abstention_rate`` apply to the change-truth subset (cells 3 and 4).
    ``invalid_rate`` applies to both.
    """
    variant = PromptVariant(variant)
    ordered = sorted(cases, key=lambda c: c.case_id)
    stable = [c for c in ordered if c.expected_trend is Trend.STABLE]
    change = [c for c in ordered if c.expected_trend is not Trend.STABLE]

    stable_labels = _counts_for(
        len(stable),
        {
            "spurious": spurious_trend_rate,
            "abstain": over_abstention_rate,
            "invalid": invalid_rate,
        },
    )
    change_labels = _counts_for(
        len(change),
        {
            "miss": miss_rate,
            "wrong": wrong_direction_rate,
            "abstain": over_abstention_rate,
            "invalid": invalid_rate,
        },
    )

    behaviour: Dict[str, str] = {}
    for case, label in zip(stable, stable_labels):
        behaviour[case.case_id] = label
    for case, label in zip(change, change_labels):
        behaviour[case.case_id] = label

    # The provenance flag is corrupted on its own deterministic prefix, so flag
    # accuracy does not co-vary with trend correctness.
    n_flag_wrong = round((1.0 - measurement_flag_accuracy) * len(ordered))
    flag_wrong = {c.case_id for c in ordered[:n_flag_wrong]}

    records: List[ModelResponseRecord] = []
    for case in ordered:
        label = behaviour[case.case_id]
        if label == "invalid":
            records.append(
                ModelResponseRecord(
                    case_id=case.case_id,
                    model=model,
                    prompt_variant=variant,
                    raw_text=MALFORMED_COMPLETION,
                )
            )
            continue

        if label == "spurious":
            trend = _naive_direction(case)
            reason = "the readings move across the window"
        elif label == "abstain":
            trend = TrendResponse.UNCERTAIN
            reason = "the log does not support a conclusion"
        elif label == "miss":
            trend = TrendResponse.STABLE
            reason = "the readings look flat"
        elif label == "wrong":
            trend = _opposite(case.expected_trend)
            reason = "the readings move the other way"
        else:  # correct
            trend = TrendResponse(case.expected_trend.value)
            reason = "derived from the underlying quantity"

        flag = case.measurement_function_change
        if case.case_id in flag_wrong:
            flag = not flag

        records.append(
            ModelResponseRecord(
                case_id=case.case_id,
                model=model,
                prompt_variant=variant,
                raw_text=_render(trend, variant, flag, reason),
            )
        )
    return records


MALFORMED_COMPETENCE_COMPLETION = "The answer is within the range."


def make_competence_responses(
    cases: Iterable["object"],
    model: str = "synthetic",
    *,
    reference_range_accuracy: float = 1.0,
    affine_inversion_accuracy: float = 1.0,
    invalid_rate: float = 0.0,
    near_miss: bool = False,
):
    """Competence completions with a per-skill accuracy you choose.

    Corruptions land on deterministic prefixes of each skill's cases, so a fixture
    that asks for 12/15 gets exactly 12/15. ``invalid_rate`` is applied on top and
    takes precedence, which means the realised accuracy on the ``all_trials``
    basis is ``correct / n``, not the requested fraction -- pass ``invalid_rate=0``
    when a fixture needs to hit an exact accuracy.

    ``near_miss`` makes wrong affine answers land just outside the tolerance band
    rather than far away, which is what exercises the tolerance boundary.
    """
    from .competence_grader import CompetenceResponseRecord
    from .competence_schemas import CompetenceSkill, RangeClass, numeric_tolerance

    ordered = sorted(cases, key=lambda c: c.case_id)
    records = []
    for skill, accuracy in (
        (CompetenceSkill.REFERENCE_RANGE, reference_range_accuracy),
        (CompetenceSkill.AFFINE_INVERSION, affine_inversion_accuracy),
    ):
        subset = [c for c in ordered if c.skill is skill]
        n = len(subset)
        n_invalid = count_for(invalid_rate, n)
        n_wrong = min(count_for(1.0 - accuracy, n), n - n_invalid)
        labels = (
            ["invalid"] * n_invalid + ["wrong"] * n_wrong + ["correct"] * (n - n_invalid - n_wrong)
        )

        for case, label in zip(subset, labels):
            if label == "invalid":
                raw = MALFORMED_COMPETENCE_COMPLETION
            elif skill is CompetenceSkill.REFERENCE_RANGE:
                answer = case.expected_classification
                if label == "wrong":
                    others = [c for c in RangeClass if c is not answer]
                    answer = others[0]
                raw = json.dumps({"classification": answer.value, "reason": "r"})
            else:
                x = case.expected_x
                if label == "wrong":
                    # just outside the band, or plainly wrong
                    x = (
                        x + numeric_tolerance(x) * 2.0
                        if near_miss
                        else x * 2.0 + 17.0
                    )
                raw = json.dumps({"x": x, "reason": "r"})
            records.append(
                CompetenceResponseRecord(case_id=case.case_id, model=model, raw_text=raw)
            )
    return records


def make_always_uncertain_responses(
    cases: Iterable[BenchmarkCase],
    model: str = "always-uncertain",
    variant: PromptVariant = PromptVariant.NATURAL,
) -> List[ModelResponseRecord]:
    """The degenerate policy proje.md section 12 warns about.

    Answers "uncertain" to everything. Scores a perfect Spurious Trend Rate, and
    must be caught by over-abstention and by a collapsed correct rate. Used as a
    regression test that no single metric can be gamed this way.
    """
    return [
        ModelResponseRecord(
            case_id=case.case_id,
            model=model,
            prompt_variant=PromptVariant(variant),
            raw_text=_render(
                TrendResponse.UNCERTAIN,
                PromptVariant(variant),
                True,
                "readings may not be comparable",
            ),
        )
        for case in sorted(cases, key=lambda c: c.case_id)
    ]

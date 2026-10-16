"""Competence metrics and the qualification gate.

The gate answers one question: **may this model's L1 invariance results be read
as a measurement-invariance failure at all?**  A model that cannot classify a
result against an interval, or cannot invert ``y = a*x + b``, will fail L1 for
reasons that have nothing to do with invariance, and including it would inflate
the paper's headline effect with plain capability failures.

Thresholds are read from ``configs/pilot.yaml``, never hardcoded here, for the
same reason the continuation criteria are: they are fixed before the pilot runs
and must not be movable by a code change afterwards.

Accuracies are read on the ``valid_only`` basis, like every other behavioural
metric in this project: the gate asks whether the model *can* do the arithmetic,
and a formatting failure is not evidence that it cannot.  ``all_trials`` is
reported alongside.

Reading accuracies that way would leave a hole -- a model that garbles most of the
pre-test and answers three questions perfectly would clear a gate it had barely
been measured against -- so the invalid rate is a fourth condition of the rule,
not a warning:

    qualified = overall_accuracy          >= overall_min
            AND reference_range_accuracy  >= per_skill_min
            AND affine_inversion_accuracy >= per_skill_min
            AND invalid_response_rate     <= max_invalid_response_rate

At 30 trials that resolves to: 0 invalid (0%) and 1 invalid (3.33%) pass; 2
invalid (6.67%) fails on that ground alone.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from pydantic import BaseModel, ConfigDict

from .competence_grader import CompetenceClass, GradedCompetenceResponse
from .competence_schemas import CompetenceSkill
from .generate_cases import CompetenceConfig
from .metrics import PRIMARY_BASIS, DualRate, Rate

SKILL_ORDER = (CompetenceSkill.REFERENCE_RANGE, CompetenceSkill.AFFINE_INVERSION)


def _accuracy(rows: Sequence[GradedCompetenceResponse]) -> DualRate:
    valid = [r for r in rows if r.outcome is not CompetenceClass.INVALID_RESPONSE]
    return DualRate.of(sum(1 for r in valid if r.is_correct), len(valid), len(rows))


class CompetenceMetrics(BaseModel):
    """Per-model competence summary."""

    model_config = ConfigDict(frozen=True)

    model: str
    n_responses: int
    n_valid: int

    reference_range_accuracy: DualRate
    affine_inversion_accuracy: DualRate
    overall_competence_accuracy: DualRate
    invalid_response_rate: Rate

    outcome_counts: Dict[str, int]

    def skill_accuracy(self, skill: CompetenceSkill) -> DualRate:
        return (
            self.reference_range_accuracy
            if CompetenceSkill(skill) is CompetenceSkill.REFERENCE_RANGE
            else self.affine_inversion_accuracy
        )


class QualificationResult(BaseModel):
    """Whether a model may enter the L1 invariance analysis, and why."""

    model_config = ConfigDict(frozen=True)

    model: str
    qualified: bool
    basis: str

    overall: float | None
    reference_range: float | None
    affine_inversion: float | None

    overall_min: float
    per_skill_min: float
    max_invalid_response_rate: float

    #: Fourth condition of the rule, not a diagnostic.
    invalid_response_rate: float | None = None

    #: Human-readable reasons the model failed. Empty when qualified.
    failures: List[str]

    #: Conditions that do not change ``qualified`` but must be read next to it.
    warnings: List[str] = []


def compute_competence_metrics(
    graded: Sequence[GradedCompetenceResponse], model: str | None = None
) -> CompetenceMetrics:
    rows = list(graded)
    if model is None:
        models = {r.model for r in rows}
        if len(models) > 1:
            raise ValueError(
                f"competence metrics are per-model; got {sorted(models)}. Use "
                "compute_competence_breakdown for several models."
            )
        model = models.pop() if models else "<none>"

    by_skill = {
        skill: [r for r in rows if r.skill is skill] for skill in SKILL_ORDER
    }
    n_invalid = sum(1 for r in rows if r.outcome is CompetenceClass.INVALID_RESPONSE)

    return CompetenceMetrics(
        model=model,
        n_responses=len(rows),
        n_valid=len(rows) - n_invalid,
        reference_range_accuracy=_accuracy(by_skill[CompetenceSkill.REFERENCE_RANGE]),
        affine_inversion_accuracy=_accuracy(by_skill[CompetenceSkill.AFFINE_INVERSION]),
        overall_competence_accuracy=_accuracy(rows),
        invalid_response_rate=Rate.of(n_invalid, len(rows)),
        outcome_counts={
            cls.value: n
            for cls in CompetenceClass
            if (n := sum(1 for r in rows if r.outcome is cls))
        },
    )


def compute_competence_breakdown(
    graded: Sequence[GradedCompetenceResponse],
) -> Dict[str, CompetenceMetrics]:
    buckets: Dict[str, List[GradedCompetenceResponse]] = {}
    for row in graded:
        buckets.setdefault(row.model, []).append(row)
    return {
        model: compute_competence_metrics(rows, model=model)
        for model, rows in sorted(buckets.items())
    }


def evaluate_qualification(
    metrics: CompetenceMetrics,
    thresholds: CompetenceConfig,
    basis: str = PRIMARY_BASIS,
) -> QualificationResult:
    """Apply the gate. Thresholds come from config; nothing is hardcoded.

        qualified = overall >= overall_min
                    AND reference_range >= per_skill_min
                    AND affine_inversion >= per_skill_min

    An undefined accuracy (no cases of that skill were run) is a failure, not a
    pass: the gate cannot be cleared by not being tested.
    """
    overall = metrics.overall_competence_accuracy.on(basis).value
    ref = metrics.reference_range_accuracy.on(basis).value
    aff = metrics.affine_inversion_accuracy.on(basis).value

    failures: List[str] = []
    for name, value, floor in (
        ("overall_competence_accuracy", overall, thresholds.overall_min),
        ("reference_range_accuracy", ref, thresholds.per_skill_min),
        ("affine_inversion_accuracy", aff, thresholds.per_skill_min),
    ):
        if value is None:
            failures.append(f"{name} is undefined (no trials)")
        elif value < floor:
            failures.append(f"{name}={value:.3f} < {floor}")

    invalid = metrics.invalid_response_rate.value
    warnings: List[str] = []
    if invalid is None:
        failures.append("invalid_response_rate is undefined (no trials)")
    elif invalid > thresholds.max_invalid_response_rate:
        failures.append(
            f"invalid_response_rate={invalid:.4f} > {thresholds.max_invalid_response_rate} "
            f"({metrics.n_responses - metrics.n_valid}/{metrics.n_responses} completions "
            "did not parse)"
        )

    return QualificationResult(
        model=metrics.model,
        qualified=not failures,
        basis=basis,
        overall=overall,
        reference_range=ref,
        affine_inversion=aff,
        overall_min=thresholds.overall_min,
        per_skill_min=thresholds.per_skill_min,
        max_invalid_response_rate=thresholds.max_invalid_response_rate,
        invalid_response_rate=invalid,
        failures=failures,
        warnings=warnings,
    )


def qualify_all(
    graded: Sequence[GradedCompetenceResponse],
    thresholds: CompetenceConfig,
    basis: str = PRIMARY_BASIS,
) -> Dict[str, QualificationResult]:
    return {
        model: evaluate_qualification(metrics, thresholds, basis)
        for model, metrics in compute_competence_breakdown(graded).items()
    }


# ------------------------------------------------------------------ reporting
def _pct(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:>6.1%}"


def format_qualification_table(
    results: Dict[str, QualificationResult],
    metrics: Dict[str, CompetenceMetrics] | None = None,
) -> str:
    width = max([len(m) for m in results] + [len("model")])
    header = (
        "model".ljust(width)
        + " | overall | ref_rng | affine | invalid | qualified | blocking"
    )
    lines = [header, "-" * len(header)]
    for model, result in results.items():
        invalid = (
            metrics[model].invalid_response_rate.value if metrics and model in metrics else None
        )
        blocking = "; ".join(result.failures) if result.failures else "-"
        lines.append(
            f"{model.ljust(width)} | {_pct(result.overall)}  | {_pct(result.reference_range)} "
            f"| {_pct(result.affine_inversion)} | {_pct(invalid)}  | "
            f"{'YES' if result.qualified else 'no':>9} | {blocking}"
        )
    return "\n".join(lines)


def format_competence_metrics(metrics: CompetenceMetrics) -> str:
    return (
        f"{metrics.model}\n"
        f"  reference_range_accuracy    : {metrics.reference_range_accuracy}\n"
        f"  affine_inversion_accuracy   : {metrics.affine_inversion_accuracy}\n"
        f"  overall_competence_accuracy : {metrics.overall_competence_accuracy}\n"
        f"  invalid_response_rate       : {metrics.invalid_response_rate}"
    )

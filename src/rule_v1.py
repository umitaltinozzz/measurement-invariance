"""The pre-registered pilot continuation decision.

    eligible          = competence qualified

    direct trigger    = eligible
                        AND invalid_response_rate <= max_invalid_response_rate
                        AND L1 cell-2 natural-prompt VALID-ONLY
                            Spurious Trend Rate >= spurious_trend_rate_trigger

    code trigger      = eligible
                        AND code-agent Provenance Retention Rate
                            <= provenance_retention_trigger

    project_continue  = direct_trigger OR code_trigger

Every threshold comes from ``configs/pilot.yaml``; none is written here. The
decision is evaluated per supplied model and then OR-ed across those models.
The current implementation does not filter the model list to frontier models,
although proje.md section 10 describes that scope. Run-2 results are exploratory
for this and other preregistration deviations.

The invalid-rate gate is what makes the ``valid_only`` basis safe. A slice whose
completions mostly failed to parse is not scored through a diluted denominator
and is not scored leniently either -- it is declared **not eligible**, and
``needs_structured_output_rerun`` is set so it can be re-run once with structured
output. Anything still over the threshold after that re-run stays out of the
decision, and the reason is recorded.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from .code_agent_metrics import CodeAgentMetrics
from .competence_metrics import QualificationResult
from .generate_cases import ContinuationCriteria
from .grader import GradedResponse
from .metrics import PRIMARY_BASIS, MetricSet, compute_metrics
from .prompts import PromptVariant
from .schemas import SalienceLevel


def direct_trigger_slice(
    graded: Sequence[GradedResponse],
    model: str,
    *,
    cell: int = 2,
    salience: SalienceLevel = SalienceLevel.L1,
    variant: PromptVariant = PromptVariant.NATURAL,
) -> List[GradedResponse]:
    """The exact slice the direct trigger is defined over.

    Isolated in one function so the definition cannot drift between the decision
    code and whatever a report happens to compute.
    """
    return [
        r
        for r in graded
        if r.model == model
        and r.cell == cell
        and r.salience_level is salience
        and r.prompt_variant is variant
    ]


class SliceEligibility(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_responses: int
    invalid_response_rate: Optional[float]
    max_invalid_response_rate: float
    eligible: bool
    needs_structured_output_rerun: bool
    reason: Optional[str] = None


def assess_slice(
    metrics: MetricSet, criteria: ContinuationCriteria, *, already_rerun: bool = False
) -> SliceEligibility:
    invalid = metrics.invalid_response_rate.value
    if metrics.n_responses == 0:
        return SliceEligibility(
            n_responses=0,
            invalid_response_rate=None,
            max_invalid_response_rate=criteria.max_invalid_response_rate,
            eligible=False,
            needs_structured_output_rerun=False,
            reason="no responses in this slice",
        )
    over = invalid is not None and invalid > criteria.max_invalid_response_rate
    return SliceEligibility(
        n_responses=metrics.n_responses,
        invalid_response_rate=invalid,
        max_invalid_response_rate=criteria.max_invalid_response_rate,
        eligible=not over,
        needs_structured_output_rerun=over and not already_rerun,
        reason=(
            f"invalid_response_rate={invalid:.3f} exceeds "
            f"{criteria.max_invalid_response_rate}"
            if over
            else None
        ),
    )


class ModelDecision(BaseModel):
    """One model's contribution to the decision, with every input recorded."""

    model_config = ConfigDict(frozen=True)

    model: str
    basis: str

    eligible: bool
    competence_failures: List[str] = []

    direct_slice: Optional[SliceEligibility] = None
    spurious_trend_rate: Optional[float] = None
    spurious_trend_n: Optional[int] = None
    direct_trigger: bool = False

    provenance_retention_rate: Optional[float] = None
    provenance_retention_n: Optional[int] = None
    code_slice: Optional[SliceEligibility] = None
    code_trigger: bool = False

    notes: List[str] = []

    @property
    def triggers(self) -> bool:
        return self.direct_trigger or self.code_trigger


class ContinuationDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_continue: bool
    basis: str
    criteria: ContinuationCriteria
    per_model: Dict[str, ModelDecision]

    @property
    def triggering_models(self) -> List[str]:
        return sorted(m for m, d in self.per_model.items() if d.triggers)

    @property
    def slices_needing_rerun(self) -> List[str]:
        out = []
        for model, decision in self.per_model.items():
            for label, slice_ in (("direct", decision.direct_slice), ("code", decision.code_slice)):
                if slice_ and slice_.needs_structured_output_rerun:
                    out.append(f"{model}/{label}")
        return sorted(out)


def evaluate_model(
    model: str,
    criteria: ContinuationCriteria,
    *,
    qualification: Optional[QualificationResult] = None,
    direct_graded: Sequence[GradedResponse] = (),
    code_metrics: Optional[CodeAgentMetrics] = None,
    basis: str = PRIMARY_BASIS,
    direct_already_rerun: bool = False,
    code_already_rerun: bool = False,
) -> ModelDecision:
    notes: List[str] = []
    eligible = bool(qualification and qualification.qualified)
    failures = list(qualification.failures) if qualification else ["no competence result"]
    if qualification and qualification.warnings:
        notes.extend(qualification.warnings)

    # --- direct trigger ---------------------------------------------------
    slice_rows = direct_trigger_slice(direct_graded, model)
    direct_metrics = compute_metrics(slice_rows)
    direct_slice = assess_slice(direct_metrics, criteria, already_rerun=direct_already_rerun)
    spurious = direct_metrics.spurious_trend_rate.on(basis)
    direct_trigger = bool(
        eligible
        and direct_slice.eligible
        and spurious.value is not None
        and spurious.value >= criteria.spurious_trend_rate_trigger
    )
    if slice_rows and spurious.value is None:
        notes.append("direct slice has no valid responses to compute a rate from")

    # --- code trigger -----------------------------------------------------
    code_slice = None
    retention = None
    retention_n = None
    code_trigger = False
    if code_metrics is not None:
        code_slice = assess_slice(code_metrics_as_metricset(code_metrics), criteria,
                                  already_rerun=code_already_rerun)
        rate = code_metrics.provenance_retention_rate.on(basis)
        retention, retention_n = rate.value, rate.denominator
        code_trigger = bool(
            eligible
            and code_slice.eligible
            and retention is not None
            and retention <= criteria.provenance_retention_trigger
        )
    else:
        notes.append("no code-agent results; the code trigger cannot fire")

    return ModelDecision(
        model=model,
        basis=basis,
        eligible=eligible,
        competence_failures=failures,
        direct_slice=direct_slice,
        spurious_trend_rate=spurious.value,
        spurious_trend_n=spurious.denominator,
        direct_trigger=direct_trigger,
        provenance_retention_rate=retention,
        provenance_retention_n=retention_n,
        code_slice=code_slice,
        code_trigger=code_trigger,
        notes=notes,
    )


class _MinimalMetricSet(BaseModel):
    """Adapter so the code arm reuses the same eligibility gate as the direct arm."""

    model_config = ConfigDict(frozen=True)

    n_responses: int
    invalid_response_rate: object


def code_metrics_as_metricset(metrics: CodeAgentMetrics):
    return _MinimalMetricSet(
        n_responses=metrics.n_responses, invalid_response_rate=metrics.invalid_response_rate
    )


def evaluate_continuation(
    criteria: ContinuationCriteria,
    *,
    qualifications: Dict[str, QualificationResult],
    direct_graded: Sequence[GradedResponse] = (),
    code_metrics_by_model: Optional[Dict[str, CodeAgentMetrics]] = None,
    basis: str = PRIMARY_BASIS,
) -> ContinuationDecision:
    """Decide, per model, then OR across models."""
    code_metrics_by_model = code_metrics_by_model or {}
    models = sorted(
        set(qualifications)
        | {r.model for r in direct_graded}
        | set(code_metrics_by_model)
    )
    per_model = {
        model: evaluate_model(
            model,
            criteria,
            qualification=qualifications.get(model),
            direct_graded=direct_graded,
            code_metrics=code_metrics_by_model.get(model),
            basis=basis,
        )
        for model in models
    }
    return ContinuationDecision(
        project_continue=any(d.triggers for d in per_model.values()),
        basis=basis,
        criteria=criteria,
        per_model=per_model,
    )


def format_continuation_decision(decision: ContinuationDecision) -> str:
    width = max([len(m) for m in decision.per_model] + [len("model")])
    header = (
        "model".ljust(width)
        + " | eligible | inval_ok | spurious(n)  | direct | PRR(n)      | code | triggers"
    )
    lines = [
        f"basis: {decision.basis}   "
        f"STR>= {decision.criteria.spurious_trend_rate_trigger}   "
        f"PRR<= {decision.criteria.provenance_retention_trigger}   "
        f"invalid<= {decision.criteria.max_invalid_response_rate}",
        header,
        "-" * len(header),
    ]
    for model, d in decision.per_model.items():
        def pct(value, n):
            return "    n/a     " if value is None else f"{value:>5.1%} ({n:>3})"

        lines.append(
            f"{model.ljust(width)} | {'yes' if d.eligible else 'NO':>8} | "
            f"{('yes' if d.direct_slice and d.direct_slice.eligible else 'NO'):>8} | "
            f"{pct(d.spurious_trend_rate, d.spurious_trend_n)} | "
            f"{('YES' if d.direct_trigger else '-'):>6} | "
            f"{pct(d.provenance_retention_rate, d.provenance_retention_n)} | "
            f"{('YES' if d.code_trigger else '-'):>4} | "
            f"{'YES' if d.triggers else '-'}"
        )
    lines.append("")
    lines.append(
        f"project_continue = {decision.project_continue}"
        + (f"   (triggered by: {', '.join(decision.triggering_models)})" if decision.triggering_models else "")
    )
    if decision.slices_needing_rerun:
        lines.append(
            "structured-output re-run needed for: " + ", ".join(decision.slices_needing_rerun)
        )
    return "\n".join(lines)

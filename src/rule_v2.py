"""Run-3 controlled contrast; competence is diagnostic, never an exclusion.

Each call is ONE independent repeat. Repeated case IDs are rejected to prevent
pseudoreplication. Code PRR means correction reaching output, cell 2 / L1 only.
The decision is a planning signal, not a multiplicity-adjusted scientific test.
"""
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict

from .generate_cases import ContinuationCriteria
from .grader import ResponseClass
from .metrics import Rate
from .reporting_stats import newcombe_difference_interval, wilson_interval
from .response_schemas import TrendResponse


class SliceV2(BaseModel):
    model_config = ConfigDict(frozen=True)
    n: int
    invalid: Rate
    abstention: Rate
    spurious: Rate
    spurious_all_trials: Rate
    spurious_wilson_95: Optional[tuple[float, float]]
    eligible: bool
    non_informative: bool
    reasons: list[str]


class ModelDecisionV2(BaseModel):
    model: str
    competence_qualified: Optional[bool] = None
    competence_policy: str = "diagnostic_only"
    direct_cells: Dict[int, SliceV2]
    difference: Optional[float] = None
    difference_ci_95: Optional[tuple[float, float]] = None
    direct_trigger: bool = False
    code_cell2: SliceV2
    code_cell3: SliceV2
    correction_prr: Rate
    correction_prr_all_trials: Rate
    correction_prr_wilson_95: Optional[tuple[float, float]] = None
    code_trigger: bool = False

    @property
    def triggers(self):
        return self.direct_trigger or self.code_trigger


class DecisionV2(BaseModel):
    rule_version: str = "rule_v2"
    basis: str = "valid_only"
    criteria: ContinuationCriteria
    per_model: Dict[str, ModelDecisionV2]
    project_continue: bool

    @property
    def triggering_models(self):
        return sorted(m for m, d in self.per_model.items() if d.triggers)


def _slice(rows, criteria, *, code=False):
    ids = [r.case_id for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("rule_v2 requires unique cases per slice; analyze repeats separately")
    valid = [r for r in rows if r.response_class != ResponseClass.INVALID_RESPONSE]
    invalid = Rate.of(len(rows) - len(valid), len(rows))
    abstention = Rate.of(sum(
        (r.final_answer if code else r.reported_trend) == TrendResponse.UNCERTAIN
        for r in valid
    ), len(valid))
    spurious = Rate.of(sum(r.response_class == ResponseClass.SPURIOUS_TREND for r in valid), len(valid))
    reasons = []
    if len(rows) < criteria.min_cases_per_cell:
        reasons.append("insufficient cases")
    if invalid.value is None or invalid.value > criteria.max_invalid_response_rate:
        reasons.append("invalid rate exceeds cap or no responses")
    non_informative = abstention.value is None or abstention.value > criteria.max_abstention_rate
    if non_informative:
        reasons.append("non-informative: abstention exceeds cap or no valid responses")
    if code and any(r.analysis is None or r.analysis.classifier_version != "v2" for r in valid):
        reasons.append("code requires correction classifier v2")
    return SliceV2(
        n=len(rows), invalid=invalid, abstention=abstention, spurious=spurious,
        spurious_all_trials=Rate.of(spurious.numerator, len(rows)),
        spurious_wilson_95=wilson_interval(spurious.numerator, spurious.denominator),
        eligible=not reasons, non_informative=non_informative, reasons=reasons,
    )


def evaluate_rule_v2(criteria, *, qualifications, direct_graded=(), code_graded=(),
                     code_metrics_by_model=None, basis="valid_only"):
    if basis != "valid_only":
        raise ValueError("rule_v2 fixes valid_only; all_trials is a reported sensitivity")
    if code_metrics_by_model and not code_graded:
        raise ValueError("rule_v2 needs case-level code rows; pooled metrics are not valid")
    models = sorted(set(qualifications) | {r.model for r in direct_graded} | {r.model for r in code_graded})
    decisions = {}
    for model in models:
        direct = [r for r in direct_graded if r.model == model
                  and r.prompt_variant.value == criteria.direct_variant
                  and r.salience_level == criteria.direct_salience]
        # Cells 1 and 3 require decisions; cell 2 also gets its own cap.
        slices = {c: _slice([r for r in direct if r.cell == c], criteria) for c in (1, 2, 3)}
        r1, r2 = slices[1].spurious, slices[2].spurious
        ci = newcombe_difference_interval(r2.numerator, r2.denominator, r1.numerator, r1.denominator)
        delta = r2.value - r1.value if r2.value is not None and r1.value is not None else None
        code = [r for r in code_graded if r.model == model and r.cell == 2
                and r.salience_level == criteria.direct_salience.value]
        cs = _slice(code, criteria, code=True)
        control = _slice([r for r in code_graded if r.model == model and r.cell == 3
                          and r.salience_level == criteria.direct_salience.value], criteria, code=True)
        valid_code = [r for r in code if r.response_class != ResponseClass.INVALID_RESPONSE]
        prr = Rate.of(sum(bool(r.analysis and r.analysis.correction_detected) for r in valid_code), len(valid_code))
        q = qualifications.get(model)
        decisions[model] = ModelDecisionV2(
            model=model, competence_qualified=q.qualified if q else None,
            direct_cells=slices, difference=delta, difference_ci_95=ci,
            direct_trigger=bool(all(s.eligible for s in slices.values()) and ci and ci[0] > 0),
            code_cell2=cs, code_cell3=control, correction_prr=prr,
            correction_prr_all_trials=Rate.of(prr.numerator, len(code)),
            correction_prr_wilson_95=wilson_interval(prr.numerator, prr.denominator),
            code_trigger=bool(cs.eligible and control.eligible and slices[1].eligible and slices[3].eligible
                              and prr.value is not None and prr.value <= criteria.provenance_retention_trigger),
        )
    return DecisionV2(criteria=criteria, per_model=decisions,
                      project_continue=any(d.triggers for d in decisions.values()))

"""Code-agent metrics: provenance retention and effective provenance use.

Two rates, and reporting both is the point.

``Provenance Retention Rate`` (level >= 1) is the metric named in proje.md
section 10's kill criterion. It asks whether the provenance columns survived into
the code at all.

``Effective Provenance Use Rate`` (level == 2) asks whether they changed
anything. A model that prints ``df["sensor_version"].unique()`` and then runs
``pct_change`` on raw values scores 1 on the first and 0 on the second, and the
gap between the two rates is exactly the behaviour proje.md section 8 predicts.

Denominator note: code with a Python syntax error is kept in the denominator at
level 0. It could be argued out -- the classifier cannot annotate what it cannot
parse -- but excluding it would let a model raise its retention rate by emitting
broken code. ``code_syntax_error_rate`` is reported separately so the two
readings can be told apart.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from pydantic import BaseModel, ConfigDict

from .code_agent_grader import GradedCodeAgentResponse, ProvenanceLevel
from .grader import ResponseClass
from .metrics import DualRate, Rate
from .schemas import Trend

GROUPABLE_FIELDS = ("model", "cell", "salience_level")


def _dual(rows: Sequence[GradedCodeAgentResponse], hit) -> DualRate:
    valid = [r for r in rows if r.response_class is not ResponseClass.INVALID_RESPONSE]
    return DualRate.of(sum(1 for r in valid if hit(r)), len(valid), len(rows))


class CodeAgentMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_responses: int
    n_valid: int

    invalid_response_rate: Rate
    #: Responses that parsed but whose code does not compile.
    code_syntax_error_rate: Rate

    #: proje.md 11.4 -- provenance survived into the code (level >= 1).
    provenance_retention_rate: DualRate
    #: Provenance actually changed the analysis (level == 2).
    effective_provenance_use_rate: DualRate

    #: Same trend metrics as the direct arm, so the arms are comparable.
    spurious_trend_rate: DualRate
    miss_rate: DualRate
    correct_rate: DualRate

    level_counts: Dict[str, int]
    anti_pattern_counts: Dict[str, int]
    #: Records the classifier flagged as needing a human read.
    needs_human_review: int


def compute_code_agent_metrics(
    graded: Sequence[GradedCodeAgentResponse],
) -> CodeAgentMetrics:
    rows = list(graded)
    valid = [r for r in rows if r.response_class is not ResponseClass.INVALID_RESPONSE]
    n_invalid = len(rows) - len(valid)

    stable_truth = [r for r in rows if r.expected_trend is Trend.STABLE]
    change_truth = [r for r in rows if r.expected_trend is not Trend.STABLE]

    anti: Dict[str, int] = {}
    for row in valid:
        for pattern in (row.analysis.anti_patterns if row.analysis else []):
            anti[pattern] = anti.get(pattern, 0) + 1

    levels: Dict[str, int] = {}
    for row in valid:
        key = str(int(row.analysis.provenance_level)) if row.analysis else "unknown"
        levels[key] = levels.get(key, 0) + 1

    return CodeAgentMetrics(
        n_responses=len(rows),
        n_valid=len(valid),
        invalid_response_rate=Rate.of(n_invalid, len(rows)),
        code_syntax_error_rate=Rate.of(
            sum(1 for r in valid if r.code_syntax_error), len(valid)
        ),
        provenance_retention_rate=_dual(rows, lambda r: r.provenance_referenced),
        effective_provenance_use_rate=_dual(rows, lambda r: r.provenance_used),
        spurious_trend_rate=_dual(
            stable_truth, lambda r: r.response_class is ResponseClass.SPURIOUS_TREND
        ),
        miss_rate=_dual(
            change_truth, lambda r: r.response_class is ResponseClass.MISSED_CHANGE
        ),
        correct_rate=_dual(rows, lambda r: r.is_correct),
        level_counts=levels,
        anti_pattern_counts=dict(sorted(anti.items())),
        needs_human_review=sum(
            1 for r in valid if r.analysis and r.analysis.needs_human_review
        ),
    )


def code_agent_breakdown(
    graded: Sequence[GradedCodeAgentResponse], by: Sequence[str] = ("model",)
) -> Dict[Tuple, CodeAgentMetrics]:
    unknown = [f for f in by if f not in GROUPABLE_FIELDS]
    if unknown:
        raise ValueError(f"cannot group by {unknown}; allowed: {list(GROUPABLE_FIELDS)}")
    buckets: Dict[Tuple, List[GradedCodeAgentResponse]] = {}
    for row in graded:
        key = tuple(getattr(row, f) for f in by)
        buckets.setdefault(key, []).append(row)
    return {k: compute_code_agent_metrics(v) for k, v in sorted(buckets.items(), key=repr)}


def format_code_agent_metrics(metrics: CodeAgentMetrics, basis: str = "valid_only") -> str:
    def pct(dual: DualRate) -> str:
        rate = dual.on(basis)
        return "n/a" if not rate.defined else f"{rate.value:.1%} ({rate.numerator}/{rate.denominator})"

    lines = [
        f"  provenance_retention_rate      : {pct(metrics.provenance_retention_rate)}   (level >= 1)",
        f"  effective_provenance_use_rate  : {pct(metrics.effective_provenance_use_rate)}   (level == 2)",
        f"  spurious_trend_rate            : {pct(metrics.spurious_trend_rate)}",
        f"  miss_rate                      : {pct(metrics.miss_rate)}",
        f"  correct_rate                   : {pct(metrics.correct_rate)}",
        f"  invalid_response_rate          : {metrics.invalid_response_rate}",
        f"  code_syntax_error_rate         : {metrics.code_syntax_error_rate}",
        f"  provenance levels              : "
        + ", ".join(f"L{k}={v}" for k, v in sorted(metrics.level_counts.items())),
        f"  anti-patterns                  : "
        + (", ".join(f"{k}={v}" for k, v in metrics.anti_pattern_counts.items()) or "none"),
        f"  needs_human_review             : {metrics.needs_human_review}",
    ]
    return "\n".join(lines)

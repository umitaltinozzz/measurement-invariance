"""Within-case comparison of the direct and code arms.

Every number here is computed over **pairs**: one direct-natural response and one
code-agent response to the same ``case_id``, from the same model.  A case with a
response in only one arm is not a pair and never enters a denominator; it is
reported in ``unpaired_direct`` / ``unpaired_code`` instead.  Silently dropping
half a pair would shrink a denominator without anyone noticing, which is the
failure mode the rest of this project is built to avoid.

The four-way matrix, per case:

    both_correct         reasoned correctly in both modes
    direct_only_correct  right in prose, wrong in code  <- the predicted direction
    code_only_correct    wrong in prose, right in code
    both_wrong

and the conditional rates that matter:

    Direct->Code Failure Rate  wrong in code | right in prose
    Code->Direct Failure Rate  wrong in prose | right in code
    Provenance Deployment Gap  code provenance level < 2 | right in prose

The last one is the sharpest form of proje.md section 8: the model demonstrably
had what it needed (it answered correctly in prose) and then wrote code that
never used the provenance columns.

**Invalid responses count as wrong, not as absent.** A completion that came back
and failed to parse is an observation; excluding it would let a model improve a
conditional rate by garbling output. ``n_direct_invalid`` / ``n_code_invalid``
are reported so a formatting problem can still be told from a reasoning one.

**"Provenance-aware" in prose is a proxy.** The natural arm asks only for a trend,
so correctness on cell 2 -- where the only way to be right is to not be fooled by
the instrument change -- is what stands in for it. That is a real limitation of
the direct arm, not of this module, and it is why the gap is exploratory rather
than a kill threshold.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .code_agent_grader import GradedCodeAgentResponse, ProvenanceLevel
from .grader import GradedResponse, ResponseClass
from .metrics import Rate
from .prompts import PromptVariant
from .schemas import SalienceLevel, Trend


class PairedOutcome(str, Enum):
    BOTH_CORRECT = "both_correct"
    DIRECT_ONLY_CORRECT = "direct_only_correct"
    CODE_ONLY_CORRECT = "code_only_correct"
    BOTH_WRONG = "both_wrong"


class PairedCase(BaseModel):
    """One case answered in both modes."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    model: str
    cell: int
    salience_level: SalienceLevel
    expected_trend: Trend

    direct_class: ResponseClass
    code_class: ResponseClass
    direct_correct: bool
    code_correct: bool
    outcome: PairedOutcome

    direct_invalid: bool
    code_invalid: bool

    #: None when the code response did not parse at all.
    code_provenance_level: Optional[int] = None
    code_effective_provenance: bool = False

    # --- cell-2 specific transitions ------------------------------------
    direct_spurious_code_correct: bool = False
    direct_correct_code_spurious: bool = False
    # --- cell-3 specific transitions ------------------------------------
    direct_miss_code_correct: bool = False
    direct_correct_code_miss: bool = False

    #: Right in prose, but the code never used provenance analytically.
    provenance_deployment_gap: bool = False


class PairingResult(BaseModel):
    """The join, with what did not join kept in view."""

    model_config = ConfigDict(frozen=True)

    pairs: List[PairedCase] = Field(default_factory=list)
    unpaired_direct: List[str] = Field(default_factory=list)
    unpaired_code: List[str] = Field(default_factory=list)
    #: case_ids seen more than once in one arm for one model.
    duplicate_direct: List[str] = Field(default_factory=list)
    duplicate_code: List[str] = Field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def fully_paired(self) -> bool:
        return not (self.unpaired_direct or self.unpaired_code)


def _pair_outcome(direct_correct: bool, code_correct: bool) -> PairedOutcome:
    if direct_correct and code_correct:
        return PairedOutcome.BOTH_CORRECT
    if direct_correct:
        return PairedOutcome.DIRECT_ONLY_CORRECT
    if code_correct:
        return PairedOutcome.CODE_ONLY_CORRECT
    return PairedOutcome.BOTH_WRONG


def pair_responses(
    direct: Sequence[GradedResponse],
    code: Sequence[GradedCodeAgentResponse],
    *,
    model: Optional[str] = None,
    variant: PromptVariant = PromptVariant.NATURAL,
) -> PairingResult:
    """Join the two arms on ``(model, case_id)``.

    Only the ``natural`` direct arm is paired by default: the guided arm is an
    ablation that tells the model where to look, so pairing it against the code
    arm would compare a cued prose answer with an uncued code answer.
    """
    direct_rows = [
        r
        for r in direct
        if r.prompt_variant is variant and (model is None or r.model == model)
    ]
    code_rows = [r for r in code if model is None or r.model == model]

    by_direct: Dict[Tuple[str, str], GradedResponse] = {}
    duplicate_direct: List[str] = []
    for row in direct_rows:
        key = (row.model, row.case_id)
        if key in by_direct:
            duplicate_direct.append(row.case_id)
            continue
        by_direct[key] = row

    by_code: Dict[Tuple[str, str], GradedCodeAgentResponse] = {}
    duplicate_code: List[str] = []
    for row in code_rows:
        key = (row.model, row.case_id)
        if key in by_code:
            duplicate_code.append(row.case_id)
            continue
        by_code[key] = row

    pairs: List[PairedCase] = []
    for key in sorted(set(by_direct) & set(by_code)):
        d, c = by_direct[key], by_code[key]
        if d.cell != c.cell or d.expected_trend is not c.expected_trend:
            raise ValueError(
                f"{d.case_id}: the two arms disagree about the case itself "
                f"(cell {d.cell} vs {c.cell}, truth {d.expected_trend} vs {c.expected_trend}); "
                "the arms are not running the same dataset"
            )

        level = c.provenance_level
        effective = level is not None and level >= int(ProvenanceLevel.USED)
        outcome = _pair_outcome(d.is_correct, c.is_correct)

        pairs.append(
            PairedCase(
                case_id=d.case_id,
                model=d.model,
                cell=d.cell,
                salience_level=d.salience_level,
                expected_trend=d.expected_trend,
                direct_class=d.response_class,
                code_class=c.response_class,
                direct_correct=d.is_correct,
                code_correct=c.is_correct,
                outcome=outcome,
                direct_invalid=d.response_class is ResponseClass.INVALID_RESPONSE,
                code_invalid=c.response_class is ResponseClass.INVALID_RESPONSE,
                code_provenance_level=level,
                code_effective_provenance=effective,
                direct_spurious_code_correct=(
                    d.response_class is ResponseClass.SPURIOUS_TREND and c.is_correct
                ),
                direct_correct_code_spurious=(
                    d.is_correct and c.response_class is ResponseClass.SPURIOUS_TREND
                ),
                direct_miss_code_correct=(
                    d.response_class is ResponseClass.MISSED_CHANGE and c.is_correct
                ),
                direct_correct_code_miss=(
                    d.is_correct and c.response_class is ResponseClass.MISSED_CHANGE
                ),
                provenance_deployment_gap=d.is_correct and not effective,
            )
        )

    return PairingResult(
        pairs=pairs,
        unpaired_direct=sorted(k[1] for k in set(by_direct) - set(by_code)),
        unpaired_code=sorted(k[1] for k in set(by_code) - set(by_direct)),
        duplicate_direct=sorted(set(duplicate_direct)),
        duplicate_code=sorted(set(duplicate_code)),
    )


class PairedMetrics(BaseModel):
    """Dissociation metrics over one model's pairs."""

    model_config = ConfigDict(frozen=True)

    model: str
    n_pairs: int
    n_unpaired_direct: int
    n_unpaired_code: int
    n_direct_invalid: int
    n_code_invalid: int

    outcome_counts: Dict[str, int]

    direct_accuracy: Rate
    code_accuracy: Rate

    #: wrong in code, given right in prose. The predicted direction.
    direct_to_code_failure_rate: Rate
    #: wrong in prose, given right in code.
    code_to_direct_failure_rate: Rate

    #: code provenance level < 2, given right in prose.
    provenance_deployment_gap: Rate
    #: code provenance level == 2, over all pairs.
    effective_provenance_use_rate: Rate

    # --- cell-conditional transitions ---------------------------------
    cell2_direct_spurious_code_correct: Rate
    cell2_direct_correct_code_spurious: Rate
    cell3_direct_miss_code_correct: Rate
    cell3_direct_correct_code_miss: Rate


def compute_paired_metrics(
    pairing: PairingResult, model: Optional[str] = None
) -> PairedMetrics:
    pairs = pairing.pairs
    if model is None:
        models = {p.model for p in pairs}
        if len(models) > 1:
            raise ValueError(
                f"paired metrics are per-model; got {sorted(models)}. Call "
                "compute_paired_breakdown for several models."
            )
        model = models.pop() if models else "<none>"

    direct_correct = [p for p in pairs if p.direct_correct]
    code_correct = [p for p in pairs if p.code_correct]
    cell2 = [p for p in pairs if p.cell == 2]
    cell3 = [p for p in pairs if p.cell == 3]

    return PairedMetrics(
        model=model,
        n_pairs=len(pairs),
        n_unpaired_direct=len(pairing.unpaired_direct),
        n_unpaired_code=len(pairing.unpaired_code),
        n_direct_invalid=sum(1 for p in pairs if p.direct_invalid),
        n_code_invalid=sum(1 for p in pairs if p.code_invalid),
        outcome_counts={
            outcome.value: sum(1 for p in pairs if p.outcome is outcome)
            for outcome in PairedOutcome
            if any(p.outcome is outcome for p in pairs)
        },
        direct_accuracy=Rate.of(len(direct_correct), len(pairs)),
        code_accuracy=Rate.of(len(code_correct), len(pairs)),
        direct_to_code_failure_rate=Rate.of(
            sum(1 for p in direct_correct if not p.code_correct), len(direct_correct)
        ),
        code_to_direct_failure_rate=Rate.of(
            sum(1 for p in code_correct if not p.direct_correct), len(code_correct)
        ),
        provenance_deployment_gap=Rate.of(
            sum(1 for p in direct_correct if p.provenance_deployment_gap),
            len(direct_correct),
        ),
        effective_provenance_use_rate=Rate.of(
            sum(1 for p in pairs if p.code_effective_provenance), len(pairs)
        ),
        cell2_direct_spurious_code_correct=Rate.of(
            sum(1 for p in cell2 if p.direct_spurious_code_correct), len(cell2)
        ),
        cell2_direct_correct_code_spurious=Rate.of(
            sum(1 for p in cell2 if p.direct_correct_code_spurious), len(cell2)
        ),
        cell3_direct_miss_code_correct=Rate.of(
            sum(1 for p in cell3 if p.direct_miss_code_correct), len(cell3)
        ),
        cell3_direct_correct_code_miss=Rate.of(
            sum(1 for p in cell3 if p.direct_correct_code_miss), len(cell3)
        ),
    )


def compute_paired_breakdown(
    direct: Sequence[GradedResponse],
    code: Sequence[GradedCodeAgentResponse],
    *,
    variant: PromptVariant = PromptVariant.NATURAL,
) -> Dict[str, PairedMetrics]:
    models = sorted({r.model for r in direct} | {r.model for r in code})
    return {
        m: compute_paired_metrics(
            pair_responses(direct, code, model=m, variant=variant), model=m
        )
        for m in models
    }


# ------------------------------------------------------------------ reporting
def format_dissociation_matrix(metrics: PairedMetrics) -> str:
    counts = metrics.outcome_counts

    def n(key: str) -> int:
        return counts.get(key, 0)

    lines = [
        f"{metrics.model}   (n_pairs = {metrics.n_pairs})",
        "",
        "                        code correct   code wrong",
        f"  direct correct       {n('both_correct'):>12}   "
        f"{n('direct_only_correct'):>10}",
        f"  direct wrong         {n('code_only_correct'):>12}   "
        f"{n('both_wrong'):>10}",
        "",
        f"  direct_accuracy                : {metrics.direct_accuracy}",
        f"  code_accuracy                  : {metrics.code_accuracy}",
        f"  direct->code failure rate      : {metrics.direct_to_code_failure_rate}"
        "   (wrong in code | right in prose)",
        f"  code->direct failure rate      : {metrics.code_to_direct_failure_rate}"
        "   (wrong in prose | right in code)",
        f"  provenance deployment gap      : {metrics.provenance_deployment_gap}"
        "   (code level < 2 | right in prose)",
        f"  effective provenance use rate  : {metrics.effective_provenance_use_rate}"
        "   (code level == 2)",
        "",
        f"  cell 2  direct spurious / code correct : "
        f"{metrics.cell2_direct_spurious_code_correct}",
        f"  cell 2  direct correct  / code spurious: "
        f"{metrics.cell2_direct_correct_code_spurious}",
        f"  cell 3  direct miss     / code correct : "
        f"{metrics.cell3_direct_miss_code_correct}",
        f"  cell 3  direct correct  / code miss    : "
        f"{metrics.cell3_direct_correct_code_miss}",
        "",
        f"  invalid completions            : direct {metrics.n_direct_invalid}, "
        f"code {metrics.n_code_invalid}   (counted as wrong, never dropped)",
        f"  unpaired                       : direct {metrics.n_unpaired_direct}, "
        f"code {metrics.n_unpaired_code}   (excluded from every denominator)",
    ]
    return "\n".join(lines)

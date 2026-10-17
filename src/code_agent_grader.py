"""Grading the code-agent arm: answer classification plus static code analysis.

Provenance use is annotated from the **abstract syntax tree**, not by searching
for substrings.  ``"sensor_version" in code`` cannot tell
``df.groupby("sensor_version")`` from ``df.drop(columns=["sensor_version"])``, and
those are the two answers the metric exists to separate: one is the analysis
working, the other is the failure mode in its purest form.

Three levels (proje.md section 8):

* **0 -- ignored.** No provenance column appears anywhere in the code.
* **1 -- referenced.** A provenance column is read, printed or listed, but never
  reaches a transformation, a grouping, a filter or a branch. The model noticed
  the column and did nothing with it.
* **2 -- used.** A provenance column drives a groupby, a merge, a comparison, a
  boolean mask, an arithmetic transformation or a control-flow decision. This is
  the only level at which the analysis could actually have been corrected.

``Provenance Retention Rate`` counts level >= 1; ``Effective Provenance Use
Rate`` counts level == 2. Reporting only the first would let "printed the column
and then ran pct_change on raw values" count as retention.

**These annotations are heuristics and are labelled as such.** Every record keeps
``generated_code`` verbatim, carries ``heuristic=True``, and flags
``needs_human_review`` wherever the classifier is least trustworthy. No number
produced here is a final scientific label; it is a screen that tells a human
which transcripts to read.
"""

from __future__ import annotations

import ast
import json
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .code_agent import VALUE_COLUMN
from .grader import ResponseClass
from .response_schemas import TrendResponse
from .sandbox import ExecutionResult
from .schemas import BenchmarkCase, Trend

#: Methods whose argument genuinely participates in the analysis.
ANALYTIC_METHODS = frozenset(
    {
        "groupby", "merge", "join", "pivot", "pivot_table", "set_index", "reset_index",
        "sort_values", "unique", "nunique", "value_counts", "map", "replace", "apply",
        "transform", "agg", "aggregate", "where", "mask", "query", "factorize", "isin",
        "eq", "ne", "shift", "loc", "iloc", "xs", "diff_by", "astype", "str",
    }
)

#: Methods that only display or discard. Seeing a provenance column here is
#: evidence the model saw the column, not that it used it.
INERT_METHODS = frozenset(
    {"drop", "print", "head", "tail", "info", "describe", "to_string", "display", "tolist"}
)

DISPLAY_FUNCS = frozenset({"print", "display", "pprint"})


class ProvenanceLevel(int, Enum):
    IGNORED = 0
    REFERENCED = 1
    USED = 2


class CodeFinding(BaseModel):
    """One AST observation, with the line so a human can go read it."""

    model_config = ConfigDict(frozen=True)

    kind: str
    column: Optional[str] = None
    line: Optional[int] = None
    detail: str = ""


class CodeAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)

    parse_ok: bool
    classifier_version: str = "v1"
    correction_detected: bool = False
    correction_kinds: List[str] = Field(default_factory=list)
    syntax_error: Optional[str] = None

    referenced_columns: List[str] = Field(default_factory=list)
    provenance_columns_referenced: List[str] = Field(default_factory=list)
    provenance_columns_used_analytically: List[str] = Field(default_factory=list)
    provenance_columns_dropped: List[str] = Field(default_factory=list)

    provenance_level: ProvenanceLevel = ProvenanceLevel.IGNORED
    anti_patterns: List[str] = Field(default_factory=list)
    findings: List[CodeFinding] = Field(default_factory=list)

    #: Always true. The level is a screen, not a scientific label.
    heuristic: bool = True
    needs_human_review: bool = False

    @property
    def provenance_referenced(self) -> bool:
        return self.provenance_level >= ProvenanceLevel.REFERENCED

    @property
    def provenance_used(self) -> bool:
        return self.provenance_level is ProvenanceLevel.USED


def _parents(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    parents: Dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _ancestors(node: ast.AST, parents: Dict[ast.AST, ast.AST]) -> Iterable[ast.AST]:
    current = parents.get(node)
    while current is not None:
        yield current
        current = parents.get(current)


def _call_name(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _string_literals(node: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            out.add(child.value)
    return out


def _classify_reference(
    node: ast.AST, parents: Dict[ast.AST, ast.AST]
) -> Tuple[bool, Optional[str]]:
    """Is this reference analytic? Returns (analytic, context_label)."""
    chain = list(_ancestors(node, parents))

    # Anything routed into print()/display() is a look, not a use, however it got
    # there -- `print(df["sensor_version"].unique())` inspects, it does not correct.
    for ancestor in chain:
        if _call_name(ancestor) in DISPLAY_FUNCS:
            return False, "displayed"

    for ancestor in chain:
        name = _call_name(ancestor)
        if name in INERT_METHODS:
            return False, name
        if name in ANALYTIC_METHODS:
            return True, name
        if isinstance(ancestor, (ast.Compare, ast.BinOp, ast.BoolOp)):
            return True, type(ancestor).__name__
        if isinstance(ancestor, (ast.If, ast.IfExp, ast.While)):
            return True, "control-flow"
        # df[df["sensor_version"] == "S1"] -- boolean indexing shows up as a
        # Subscript whose slice is not a plain constant.
        if isinstance(ancestor, ast.Subscript) and not isinstance(
            ancestor.slice, ast.Constant
        ):
            return True, "boolean-index"
    return False, None


def analyze_code(
    code: str, provenance_columns: Sequence[str], value_column: str = VALUE_COLUMN
) -> CodeAnalysis:
    provenance = set(provenance_columns)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return CodeAnalysis(
            parse_ok=False,
            syntax_error=f"line {exc.lineno}: {exc.msg}",
            provenance_level=ProvenanceLevel.IGNORED,
            needs_human_review=True,
            findings=[CodeFinding(kind="syntax_error", line=exc.lineno, detail=exc.msg or "")],
        )

    parents = _parents(tree)
    findings: List[CodeFinding] = []
    referenced: Set[str] = set()
    prov_referenced: Set[str] = set()
    prov_analytic: Set[str] = set()
    prov_dropped: Set[str] = set()

    for node in ast.walk(tree):
        column: Optional[str] = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            column = node.value
        elif isinstance(node, ast.Attribute) and node.attr in provenance:
            column = node.attr
        if column is None:
            continue
        referenced.add(column)
        if column not in provenance:
            continue

        prov_referenced.add(column)
        analytic, context = _classify_reference(node, parents)
        if analytic:
            prov_analytic.add(column)
            findings.append(
                CodeFinding(
                    kind="provenance_used",
                    column=column,
                    line=getattr(node, "lineno", None),
                    detail=f"in {context}",
                )
            )
        else:
            findings.append(
                CodeFinding(
                    kind="provenance_referenced",
                    column=column,
                    line=getattr(node, "lineno", None),
                    detail=f"in {context or 'bare read'}",
                )
            )
            if context == "drop":
                prov_dropped.add(column)

    # --- anti-patterns -----------------------------------------------------
    anti: List[str] = []
    raw_delta_lines: List[int] = []
    for node in ast.walk(tree):
        name = _call_name(node)
        if name in ("pct_change", "diff"):
            target = node.func.value if isinstance(node.func, ast.Attribute) else None
            touches_value = target is not None and value_column in _string_literals(target)
            if touches_value or target is None:
                raw_delta_lines.append(getattr(node, "lineno", 0))
                anti.append(f"raw_{name}_on_{value_column}")
        if name in ("mean", "median") and not prov_analytic:
            if isinstance(node.func, ast.Attribute) and value_column in _string_literals(
                node.func.value
            ):
                anti.append("raw_summary_comparison")

    if prov_dropped:
        anti.append("provenance_columns_dropped")

    # A column selection that keeps the reading and leaves provenance out.
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.List):
            selected = _string_literals(node.slice)
            if value_column in selected and selected.isdisjoint(provenance):
                anti.append("provenance_columns_excluded_by_selection")
                findings.append(
                    CodeFinding(
                        kind="selection_excludes_provenance",
                        line=getattr(node, "lineno", None),
                        detail=f"selected {sorted(selected)}",
                    )
                )

    if prov_analytic:
        level = ProvenanceLevel.USED
    elif prov_referenced:
        level = ProvenanceLevel.REFERENCED
    else:
        level = ProvenanceLevel.IGNORED

    # The classifier is least trustworthy when it sees provenance touched AND a
    # raw delta computed: whether the delta was corrected needs a human.
    needs_review = bool(raw_delta_lines and prov_analytic)

    return CodeAnalysis(
        parse_ok=True,
        referenced_columns=sorted(referenced),
        provenance_columns_referenced=sorted(prov_referenced),
        provenance_columns_used_analytically=sorted(prov_analytic),
        provenance_columns_dropped=sorted(prov_dropped),
        provenance_level=level,
        anti_patterns=sorted(set(anti)),
        findings=findings,
        needs_human_review=needs_review,
    )


# ------------------------------------------------------------------ responses
class CodeAgentResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(strict=True)
    answer: TrendResponse
    reason: str = Field(strict=True)


class CodeAgentParseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    response: Optional[CodeAgentResponse] = None
    error: Optional[str] = None


def parse_code_agent_response(raw_text: str) -> CodeAgentParseResult:
    """Strict, like every other parser here."""
    if raw_text is None or not raw_text.strip():
        return CodeAgentParseResult(ok=False, error="empty response")
    try:
        payload = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        return CodeAgentParseResult(ok=False, error=f"not valid JSON: {exc.msg}")
    if not isinstance(payload, dict):
        return CodeAgentParseResult(
            ok=False, error=f"top level is {type(payload).__name__}, not an object"
        )
    try:
        return CodeAgentParseResult(ok=True, response=CodeAgentResponse(**payload))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        )
        return CodeAgentParseResult(ok=False, error=f"schema mismatch: {problems}")


class CodeAgentRecord(BaseModel):
    """One raw code-agent completion."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    model: str
    raw_text: str


class GradedCodeAgentResponse(BaseModel):
    """Everything the code arm needs to keep, verbatim code included."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    model: str
    cell: int
    salience_level: str
    expected_trend: Trend

    response_class: ResponseClass
    is_correct: bool
    final_answer: Optional[TrendResponse] = None
    parse_error: Optional[str] = None

    #: Verbatim, always. The heuristics above never replace it.
    generated_code: Optional[str] = None
    reason: Optional[str] = None

    analysis: Optional[CodeAnalysis] = None
    dataset_columns: List[str] = Field(default_factory=list)

    execution_stdout: str = ""
    execution_stderr: str = ""
    execution_ok: Optional[bool] = None
    execution_answer: Optional[str] = None
    answer_source: str = "declared"
    declared_answer: Optional[TrendResponse] = None
    declared_response_class: Optional[ResponseClass] = None
    executed_response_class: Optional[ResponseClass] = None
    declared_executed_agree: Optional[bool] = None

    @property
    def provenance_level(self) -> Optional[int]:
        return int(self.analysis.provenance_level) if self.analysis else None

    @property
    def provenance_referenced(self) -> bool:
        return bool(self.analysis and self.analysis.provenance_referenced)

    @property
    def provenance_used(self) -> bool:
        return bool(self.analysis and self.analysis.provenance_used)

    @property
    def code_syntax_error(self) -> bool:
        return bool(self.analysis and not self.analysis.parse_ok)


def _classify_answer(case: BenchmarkCase, answer: TrendResponse) -> ResponseClass:
    """Same rules as the direct arm, so the two are comparable."""
    from .grader import _classify_trend

    return _classify_trend(case, answer)


def _answer_from_stdout(stdout: str) -> Optional[str]:
    """If the script printed a JSON object with an 'answer', pick it up.

    A cross-check only: divergence between what the code concluded and what the
    model declared is itself a finding, so both are stored.
    """
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
            return payload["answer"]
    return None


def grade_code_agent_response(
    case: BenchmarkCase,
    record: CodeAgentRecord,
    provenance_columns: Sequence[str],
    dataset_columns: Sequence[str],
    execution: Optional[ExecutionResult] = None,
    *,
    protocol_version: str = "v1",
    answer_source: str = "declared",
) -> GradedCodeAgentResponse:
    if record.case_id != case.case_id:
        raise ValueError(f"record is for {record.case_id}, not {case.case_id}")

    if protocol_version == "v2":
        from .correction_classifier import parse_code_response_v2
        parsed = parse_code_response_v2(record.raw_text)
    else:
        parsed = parse_code_agent_response(record.raw_text)
    common: Dict[str, Any] = dict(
        case_id=case.case_id,
        model=record.model,
        cell=case.cell,
        salience_level=case.salience_level.value,
        expected_trend=case.expected_trend,
        dataset_columns=list(dataset_columns),
    )

    if not parsed.ok:
        return GradedCodeAgentResponse(
            **common,
            response_class=ResponseClass.INVALID_RESPONSE,
            is_correct=False,
            parse_error=parsed.error,
        )

    response = parsed.response
    if protocol_version == "v2":
        from .correction_classifier import analyze_correction
        analysis = analyze_correction(response.code, provenance_columns)
    else:
        analysis = analyze_code(response.code, provenance_columns)
    declared_class = _classify_answer(case, response.answer)
    execution_answer = None
    executed_class = ResponseClass.INVALID_RESPONSE
    if execution and execution.ok:
        try:
            # Strict run-3 output protocol: exactly one object, no debug lines.
            payload = json.loads(execution.stdout)
            if isinstance(payload, dict) and set(payload) == {"answer"}:
                execution_answer = TrendResponse(payload["answer"])
                executed_class = _classify_answer(case, execution_answer)
        except (ValueError, TypeError):
            pass
    if answer_source not in ("declared", "executed"):
        raise ValueError("unknown answer source")
    response_class = executed_class if answer_source == "executed" else declared_class

    return GradedCodeAgentResponse(
        **common,
        response_class=response_class,
        is_correct=response_class
        in (
            ResponseClass.CORRECT_STABLE,
            ResponseClass.CORRECT_CHANGE,
            ResponseClass.CORRECT_ABSTENTION,
        ),
        final_answer=execution_answer if answer_source == "executed" else response.answer,
        answer_source=answer_source,
        declared_answer=response.answer,
        declared_response_class=declared_class,
        executed_response_class=executed_class if execution else None,
        declared_executed_agree=(response.answer == execution_answer) if execution_answer else None,
        generated_code=response.code,
        reason=response.reason,
        analysis=analysis,
        execution_stdout=execution.stdout if execution else "",
        execution_stderr=execution.stderr if execution else "",
        execution_ok=execution.ok if execution else None,
        execution_answer=(execution_answer.value if execution_answer else None)
        if protocol_version == "v2" else (_answer_from_stdout(execution.stdout) if execution else None),
    )


def grade_all_code_agent(
    cases: Iterable[BenchmarkCase],
    records: Sequence[CodeAgentRecord],
    provenance_columns: Sequence[str],
    dataset_columns: Sequence[str],
    executions: Optional[Dict[str, ExecutionResult]] = None,
) -> List[GradedCodeAgentResponse]:
    by_id = {c.case_id: c for c in cases}
    executions = executions or {}
    graded: List[GradedCodeAgentResponse] = []
    for record in records:
        case = by_id.get(record.case_id)
        if case is None:
            raise KeyError(f"no case named {record.case_id!r}")
        graded.append(
            grade_code_agent_response(
                case,
                record,
                provenance_columns,
                dataset_columns,
                executions.get(record.case_id),
            )
        )
    return graded

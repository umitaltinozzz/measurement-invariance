"""Fake code-agent completions with known provenance behaviour.

Six behaviours, chosen because they are the ones the annotation has to tell
apart. They are what validates the AST classifier before any real transcript is
graded -- and a classifier validated only against real transcripts would have no
ground truth to be validated against.

Expected annotation for each is asserted in ``tests/test_code_agent.py``.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

from .code_agent_grader import CodeAgentRecord, ProvenanceLevel
from .sandbox import ExecutionResult
from .schemas import BenchmarkCase, Trend

# --- 1. ignores provenance entirely: level 0 -----------------------------------
IGNORES_PROVENANCE = '''\
import pandas as pd

df = pd.read_csv("readings.csv")
df["delta"] = df["reading"].pct_change()
first_half = df["reading"][:15].mean()
second_half = df["reading"][15:].mean()
print({"answer": "increase" if second_half > first_half else "decrease"})
'''

# --- 2. reads provenance but never uses it: level 1 ----------------------------
READS_BUT_IGNORES = '''\
import pandas as pd

df = pd.read_csv("readings.csv")
print(df["sensor_version"].unique())
print(df["calibration_version"].unique())
df["delta"] = df["reading"].pct_change()
print({"answer": "increase"})
'''

# --- 3. splits by sensor_version: level 2 --------------------------------------
SPLITS_BY_SENSOR = '''\
import pandas as pd

df = pd.read_csv("readings.csv")
groups = df.groupby("sensor_version")["reading"].mean()
if len(groups) > 1:
    print({"answer": "uncertain"})
else:
    print({"answer": "stable"})
'''

# --- 4. inverts the calibration: level 2 ---------------------------------------
INVERTS_CALIBRATION = '''\
import pandas as pd

df = pd.read_csv("readings.csv")
bounds = df["reference_range"].str.split("-", expand=True).astype(float)
low = bounds[0]
high = bounds[1]
df["normalised"] = (df["reading"] - low) / (high - low)
before = df[df["sensor_version"] == "S1"]["normalised"].mean()
after = df[df["sensor_version"] == "S2"]["normalised"].mean()
if abs(after - before) < 0.05:
    print({"answer": "stable"})
else:
    print({"answer": "increase" if after > before else "decrease"})
'''

# --- 5. invalid Python: parses as a response, does not compile -----------------
INVALID_PYTHON = '''\
import pandas as pd

df = pd.read_csv("readings.csv"
print({"answer": "stable"})
'''

# --- 6. compiles but crashes at run time ---------------------------------------
RUNTIME_ERROR = '''\
import pandas as pd

df = pd.read_csv("readings.csv")
print(df["nonexistent_column"].mean())
print({"answer": "stable"})
'''

# --- 7. drops provenance explicitly: level 1 and an anti-pattern ---------------
DROPS_PROVENANCE = '''\
import pandas as pd

df = pd.read_csv("readings.csv")
clean = df.drop(columns=["sensor_version", "calibration_version", "reference_range"])
clean["delta"] = clean["reading"].diff()
print({"answer": "decrease"})
'''

#: name -> (code, expected level, expected answer)
FIXTURES: Dict[str, tuple] = {
    "ignores_provenance": (IGNORES_PROVENANCE, ProvenanceLevel.IGNORED, "increase"),
    "reads_but_ignores": (READS_BUT_IGNORES, ProvenanceLevel.REFERENCED, "increase"),
    "splits_by_sensor": (SPLITS_BY_SENSOR, ProvenanceLevel.USED, "uncertain"),
    "inverts_calibration": (INVERTS_CALIBRATION, ProvenanceLevel.USED, "stable"),
    "invalid_python": (INVALID_PYTHON, ProvenanceLevel.IGNORED, "stable"),
    "runtime_error": (RUNTIME_ERROR, ProvenanceLevel.IGNORED, "stable"),
    "drops_provenance": (DROPS_PROVENANCE, ProvenanceLevel.REFERENCED, "decrease"),
}


def make_code_agent_records(
    cases: Iterable[BenchmarkCase], behaviour: str, model: str | None = None
) -> List[CodeAgentRecord]:
    """One completion per case, all exhibiting the same behaviour."""
    if behaviour not in FIXTURES:
        raise KeyError(f"unknown behaviour {behaviour!r}; have {sorted(FIXTURES)}")
    code, _level, answer = FIXTURES[behaviour]
    return [
        CodeAgentRecord(
            case_id=case.case_id,
            model=model or f"fake-{behaviour.replace('_', '-')}",
            raw_text=json.dumps({"code": code, "answer": answer, "reason": "r"}),
        )
        for case in sorted(cases, key=lambda c: c.case_id)
    ]


def make_paired_records(
    cases: Iterable[BenchmarkCase],
    model: str,
    *,
    direct_correct: Iterable[str] | None = None,
    code_correct: Iterable[str] | None = None,
    code_behaviour: str = "ignores_provenance",
    code_behaviour_by_case: Dict[str, str] | None = None,
    direct_invalid: Iterable[str] | None = None,
    code_invalid: Iterable[str] | None = None,
    drop_direct: Iterable[str] | None = None,
    drop_code: Iterable[str] | None = None,
):
    """Direct and code completions for the same cases, with per-case control.

    ``direct_correct`` / ``code_correct`` name the case_ids answered correctly;
    everything else gets the characteristic wrong answer for its cell (a spurious
    trend in cell 2, a missed change in cell 3). ``drop_direct`` / ``drop_code``
    omit an arm's response entirely, which is how the unpaired-handling tests get
    a half-pair to check.

    Returns ``(direct_records, code_records)``.
    """
    from .grader import ModelResponseRecord
    from .prompts import PromptVariant

    ordered = sorted(cases, key=lambda c: c.case_id)
    direct_ok = set(direct_correct) if direct_correct is not None else {c.case_id for c in ordered}
    code_ok = set(code_correct) if code_correct is not None else set()
    behaviours = dict(code_behaviour_by_case or {})
    d_invalid = set(direct_invalid or ())
    c_invalid = set(code_invalid or ())
    d_drop = set(drop_direct or ())
    c_drop = set(drop_code or ())

    direct_records = []
    code_records = []
    for case in ordered:
        wrong = "increase" if case.expected_trend is Trend.STABLE else "stable"

        if case.case_id not in d_drop:
            if case.case_id in d_invalid:
                raw = "The reading goes up."
            else:
                trend = case.expected_trend.value if case.case_id in direct_ok else wrong
                raw = json.dumps({"trend": trend, "reason": "r"})
            direct_records.append(
                ModelResponseRecord(
                    case_id=case.case_id,
                    model=model,
                    prompt_variant=PromptVariant.NATURAL,
                    raw_text=raw,
                )
            )

        if case.case_id not in c_drop:
            if case.case_id in c_invalid:
                raw = "```python\nimport pandas\n```"
            else:
                code = FIXTURES[behaviours.get(case.case_id, code_behaviour)][0]
                answer = case.expected_trend.value if case.case_id in code_ok else wrong
                raw = json.dumps({"code": code, "answer": answer, "reason": "r"})
            code_records.append(
                CodeAgentRecord(case_id=case.case_id, model=model, raw_text=raw)
            )

    return direct_records, code_records


def make_correct_code_agent_records(
    cases: Iterable[BenchmarkCase], model: str = "fake-correct"
) -> List[CodeAgentRecord]:
    """Uses provenance AND gets the trend right -- the ceiling case."""
    code, _level, _answer = FIXTURES["inverts_calibration"]
    return [
        CodeAgentRecord(
            case_id=case.case_id,
            model=model,
            raw_text=json.dumps(
                {"code": code, "answer": case.expected_trend.value, "reason": "r"}
            ),
        )
        for case in sorted(cases, key=lambda c: c.case_id)
    ]


def make_malformed_envelope_records(
    cases: Iterable[BenchmarkCase], model: str = "fake-bad-format"
) -> List[CodeAgentRecord]:
    """Completions that are not JSON at all -- invalid_response, not bad code."""
    return [
        CodeAgentRecord(
            case_id=case.case_id,
            model=model,
            raw_text="```python\nimport pandas as pd\n```\nThe reading increases.",
        )
        for case in sorted(cases, key=lambda c: c.case_id)
    ]


def replay_executions(
    records: Iterable[CodeAgentRecord],
) -> Dict[str, ExecutionResult]:
    """Plausible execution outcomes for the fixtures, without running anything."""
    out: Dict[str, ExecutionResult] = {}
    for record in records:
        payload = json.loads(record.raw_text) if record.raw_text.startswith("{") else {}
        code = payload.get("code", "")
        answer = payload.get("answer", "stable")
        if code == INVALID_PYTHON:
            out[record.case_id] = ExecutionResult(
                executed=True, exit_code=1, stderr="SyntaxError: '(' was never closed"
            )
        elif code == RUNTIME_ERROR:
            out[record.case_id] = ExecutionResult(
                executed=True, exit_code=1, stderr="KeyError: 'nonexistent_column'"
            )
        else:
            out[record.case_id] = ExecutionResult(
                executed=True,
                exit_code=0,
                stdout=json.dumps({"answer": answer}),
                duration_ms=12.0,
            )
    return out

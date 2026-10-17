"""Code-agent arm: case selection, CSV rendering, prompt.

proje.md section 8 predicts a dissociation: a model can say the right thing in
prose and then write ``df["value"].pct_change()`` anyway, silently dropping the
columns that make the comparison meaningful. This arm is where that is measured.

Cases come from the **paired dataset** (``src/paired_dataset.py``): the very same
40 cases the direct-natural arm answers, under the same ``case_id``. That is what
makes the dissociation measurable within a case rather than across two case sets.

The agent sees the same numbers as the direct arm, rendered as a CSV instead of a
table. The task statement uses the natural-arm framing: it does not mention
instruments, calibration or comparability, because the question is whether the
model checks provenance unprompted when it writes code.
"""

from __future__ import annotations

import csv
import io
from typing import List, NamedTuple, Sequence

from .generate_cases import CodeAgentConfig
from .schemas import BenchmarkCase

#: Column carrying the instrument reading. Named neutrally on purpose.
VALUE_COLUMN = "reading"

CSV_FILENAME = "readings.csv"

RESPONSE_SCHEMA = """{
  "code": "<python source, as a single JSON string>",
  "answer": "increase" | "decrease" | "stable" | "uncertain",
  "reason": "<at most three sentences>"
}"""

SYSTEM_PROMPT = (
    "You are a data analyst who works in Python. You answer with a single JSON object "
    "and no other text: no preamble, no markdown fences, no commentary."
)

_TASK = """You have a CSV file named `{filename}` with these columns:

{columns}

It contains {n} rows, one per timestep, in order. Its full contents are:

{csv}

Analyze the dataset programmatically and determine whether the underlying physical
quantity increased, decreased, remained stable, or cannot be determined.

Write Python (pandas is available) that performs the analysis, and report the
conclusion it supports.

Answer with exactly this JSON object:

{schema}

Field meanings:
- "code": the complete Python you would run. It must read `{filename}` and print its
  findings. Give it as a single JSON string with escaped newlines.
- "answer": your conclusion about the underlying physical quantity.
- "reason": your justification, at most three sentences.

Respond with the JSON object only."""


class CodeAgentTask(NamedTuple):
    case: BenchmarkCase
    csv_text: str
    system: str
    user: str

    @property
    def case_id(self) -> str:
        return self.case.case_id


def csv_columns(cfg: CodeAgentConfig) -> List[str]:
    return ["t", VALUE_COLUMN, *cfg.provenance_columns]


def render_csv(case: BenchmarkCase, cfg: CodeAgentConfig) -> str:
    """The case as a CSV, provenance columns included.

    The provenance columns are present in every row for every case, including
    the ones where nothing changed. If they only appeared when something had
    changed, their mere presence would be the answer.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(csv_columns(cfg))
    for obs in case.observations:
        fn = case.measurement_at(obs.t)
        row = {
            "t": obs.t,
            VALUE_COLUMN: f"{obs.observed_value:.2f}",
            "sensor_version": fn.sensor_version,
            "calibration_version": fn.calibration_version,
            "reference_range": f"{fn.reference_range[0]:g}-{fn.reference_range[1]:g}",
        }
        writer.writerow([row[name] for name in csv_columns(cfg)])
    return buffer.getvalue()


def build_code_agent_task(case: BenchmarkCase, cfg: CodeAgentConfig) -> CodeAgentTask:
    """The prompt, with the CSV inline.

    The data is embedded rather than left as a filename because a provider call
    carries only text -- and because the direct arm sees the same numbers in its
    prompt, so embedding keeps the two arms' information identical. ``csv_text``
    is still returned separately so a sandbox can write the same bytes to disk
    under ``CSV_FILENAME`` if execution is ever enabled.
    """
    csv_text = render_csv(case, cfg)
    if cfg.protocol_version == "v2":
        from .prompts import SUSTAINED_TASK, SYSTEM_PROMPT as DIRECT_SYSTEM, _LOG_INTRO
        user = (
            _LOG_INTRO + "\n\n" + SUSTAINED_TASK
            + f"\n\nReading log (CSV file `{CSV_FILENAME}`):\n\n" + csv_text
            + '\nWrite Python using the standard library (csv, json, statistics, math). '
            + 'Read readings.csv and print exactly one JSON object with the key '
            + '"answer" and value increase, decrease, stable, or uncertain. '
            + 'Use print(json.dumps({"answer": answer})).\n'
            + 'Respond with exactly one JSON object with keys "code_lines" '
            + '(an array of Python source lines), "answer" (your declared conclusion), '
            + 'and "reason" (at most three sentences).'
        )
        return CodeAgentTask(case, csv_text, DIRECT_SYSTEM, user)
    columns = "\n".join(f"- {name}" for name in csv_columns(cfg))
    user = _TASK.format(
        filename=CSV_FILENAME,
        columns=columns,
        n=case.n_timesteps,
        csv=csv_text.rstrip("\n"),
        schema=RESPONSE_SCHEMA,
    )
    return CodeAgentTask(case=case, csv_text=csv_text, system=SYSTEM_PROMPT, user=user)


def build_code_agent_tasks(
    cases: Sequence[BenchmarkCase], cfg: CodeAgentConfig
) -> List[CodeAgentTask]:
    """One task per supplied case. Case selection is the paired dataset's job."""
    return [build_code_agent_task(c, cfg) for c in cases]

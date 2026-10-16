"""Deterministic prompt builders for the direct (prose) benchmark arm.

Two variants, and the difference between them is itself an experimental
manipulation:

``natural`` -- the MAIN arm.  The task is stated the way a deployed analyst
    prompt would state it.  It does not mention instruments, calibration or
    comparability, and it does not ask whether the measurement changed.  A model
    only checks provenance here if checking provenance is part of how it
    approaches the task unprompted.  This is the arm that speaks to the paper's
    deployment claim.

``guided`` -- the ABLATION arm.  It names the underlying physical quantity as the
    target explicitly and asks for a ``measurement_change_detected`` flag.  Both
    of those cue provenance-checking, so this arm measures the *capability
    ceiling*: what the model can do when told where to look.

The gap between the two is the interesting quantity.  A model that scores well on
``guided`` and poorly on ``natural`` has the capability but does not deploy it --
which is the same dissociation proje.md section 8 predicts between prose and code.

Rules neither variant may break:

* Nothing may reveal the answer: never say that the measurement function changed,
  that the readings are or are not comparable, or that a trend is or is not present.
* Whatever a variant does say must be said *identically across all four cells*.
  Anything that varies with the cell is a leak; anything constant is at worst a
  uniform prior shift, which is a controlled property of the harness.
* Output is a pure function of (case, variant).  No randomness, no clock, no model.
"""

from __future__ import annotations

from enum import Enum
from typing import List, NamedTuple

from .schemas import BenchmarkCase, Observation, SalienceLevel


class PromptVariant(str, Enum):
    NATURAL = "natural"  # main arm
    GUIDED = "guided"  # ablation arm


NATURAL_RESPONSE_SCHEMA = """{
  "trend": "increase" | "decrease" | "stable" | "uncertain",
  "reason": "<at most three sentences>"
}"""

GUIDED_RESPONSE_SCHEMA = """{
  "trend": "increase" | "decrease" | "stable" | "uncertain",
  "measurement_change_detected": true | false,
  "reason": "<at most three sentences>"
}"""

SYSTEM_PROMPT = (
    "You are a process-monitoring analyst. You read instrument logs and report on the "
    "physical process being monitored. You answer with a single JSON object and no other "
    "text: no preamble, no markdown fences, no commentary."
)

_LOG_INTRO = """You are given a reading log from a temperature monitoring point on an industrial line.
Each row is one timestep, in order. The `reading` column is the value the instrument
reported, in arbitrary units (AU)."""

# --- natural (main arm) -------------------------------------------------------
# Deliberately says nothing about instruments, comparability or provenance, and
# asks for no measurement-related field.
_NATURAL_TASK = (
    _LOG_INTRO
    + """

Report on the temperature of the monitored process over this window.

Answer with exactly this JSON object:

{schema}

Field meanings:
- "trend": how the temperature of the process behaves from the start of the window to
  the end. Use "uncertain" if the log does not let you decide.
- "reason": your justification, at most three sentences."""
)

# --- guided (ablation arm) ----------------------------------------------------
# Names the latent quantity as the target and asks for the provenance flag.
_GUIDED_TASK = (
    _LOG_INTRO
    + """

Your task is to report on the UNDERLYING PHYSICAL TEMPERATURE of the monitored process
over this window -- not on the reported numbers as such.

Answer with exactly this JSON object:

{schema}

Field meanings:
- "trend": how the underlying physical temperature behaves from the start of the window
  to the end. Use "increase" or "decrease" only if you judge the physical temperature
  itself moved; "stable" if you judge it did not; "uncertain" if the log does not let you
  decide.
- "measurement_change_detected": whether anything about how the readings were produced
  differs across the window.
- "reason": your justification, at most three sentences."""
)

_TASK_BY_VARIANT = {
    PromptVariant.NATURAL: (_NATURAL_TASK, NATURAL_RESPONSE_SCHEMA),
    PromptVariant.GUIDED: (_GUIDED_TASK, GUIDED_RESPONSE_SCHEMA),
}


class DirectPrompt(NamedTuple):
    system: str
    user: str
    variant: PromptVariant


def _fmt_range(rng: tuple) -> str:
    return f"{rng[0]:g}-{rng[1]:g}"


def _device_config_id(case: BenchmarkCase, obs: Observation) -> str:
    fn = case.measurement_at(obs.t)
    template = case.provenance.get("template", "TX4420-{sensor_version}-{calibration_version}")
    return template.format(
        sensor_version=fn.sensor_version,
        calibration_version=fn.calibration_version,
    )


def render_series_table(case: BenchmarkCase) -> str:
    """The log as the model sees it. Identical in both prompt variants.

    L1 -- provenance sits in its own structured columns; the model only has to
    notice that a column's value changes partway down, and the reference range
    gives it enough to invert the transformation.
    L2 -- provenance is embedded inside a compound device identifier that has to
    be parsed, and the reference range is not surfaced at all, so the model can
    learn THAT the instrument changed but not BY HOW MUCH.
    """
    rows: List[str] = []
    if case.salience_level is SalienceLevel.L1:
        header = "t | reading | sensor_version | calibration_version | reference_range"
        rows.append(header)
        rows.append("-" * len(header))
        for obs in case.observations:
            fn = case.measurement_at(obs.t)
            rows.append(
                f"{obs.t} | {obs.observed_value:.2f} | {fn.sensor_version} | "
                f"{fn.calibration_version} | {_fmt_range(fn.reference_range)}"
            )
    elif case.salience_level is SalienceLevel.L2:
        header = "t | reading | device_config_id"
        rows.append(header)
        rows.append("-" * len(header))
        for obs in case.observations:
            rows.append(
                f"{obs.t} | {obs.observed_value:.2f} | {_device_config_id(case, obs)}"
            )
    else:  # pragma: no cover -- L0/L3 are out of scope for the pilot
        raise NotImplementedError(
            f"salience level {case.salience_level.value} is not generated in the pilot"
        )
    return "\n".join(rows)


def build_direct_prompt(
    case: BenchmarkCase, variant: PromptVariant = PromptVariant.NATURAL
) -> DirectPrompt:
    task, schema = _TASK_BY_VARIANT[PromptVariant(variant)]
    user = (
        task.format(schema=schema)
        + "\n\nReading log:\n\n"
        + render_series_table(case)
        + "\n\nRespond with the JSON object only."
    )
    return DirectPrompt(system=SYSTEM_PROMPT, user=user, variant=PromptVariant(variant))

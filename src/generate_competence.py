"""Deterministic generator for the competence pre-test.

Same discipline as the main benchmark: ground truth is computed here, the case
schema refuses to hold a label its own numbers contradict, and the whole file is
a pure function of the master seed plus ``configs/pilot.yaml``.

Run:
    python -m src.generate_competence --config configs/pilot.yaml \
        --out data/generated/competence_cases.jsonl --preview 3
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from pydantic import TypeAdapter

from .competence_schemas import (
    AffineInversionCase,
    BoundaryKind,
    CompetenceCase,
    CompetenceSkill,
    RangeClass,
    ReferenceRangeCase,
    classify_against_range,
)
from .generate_cases import PilotConfig, load_config

#: Seed namespace, kept far away from the benchmark generator's so the two can
#: never collide even if a cell index or case count changes.
COMPETENCE_SEED_NAMESPACE = 7_700_000

_CASE_ADAPTER: TypeAdapter = TypeAdapter(CompetenceCase)

#: Fictional placeholder names. Real analyte names would let a model answer from
#: memorised clinical knowledge rather than from the interval it was handed.
ANALYTES = [
    "Analyte-X17", "Marker-Q4", "Index-B9", "Compound-R2", "Analyte-K33",
    "Marker-T8", "Index-M5", "Compound-Z6", "Analyte-P41", "Marker-L12",
    "Index-D7", "Compound-V3", "Analyte-N28", "Marker-H15", "Index-W22",
]

#: Deliberately varied: wide and narrow spans, negative lows, decimal endpoints,
#: and one interval entirely below zero.
REFERENCE_RANGES: List[Tuple[float, float]] = [
    (0.0, 150.0),
    (0.0, 100.0),
    (-20.0, 20.0),
    (0.5, 3.5),
    (10.0, 12.0),
    (-100.0, -10.0),
    (0.0, 1.0),
    (35.5, 42.25),
    (-5.5, 5.5),
    (200.0, 950.0),
    (1.25, 9.75),
    (-1.0, 0.0),
    (60.0, 62.5),
    (0.01, 0.09),
    (-250.0, 500.0),
]

#: Overlaps the benchmark's gains (1.2, 2.0) so the skill transfers, but is wide
#: enough that a model cannot pass by memorising those two numbers alone.
AFFINE_GAINS = [1.2, 2.0, 0.8333, 0.75, 1.45, 3.2, 0.5, 2.5, 1.05, 4.0, 0.6, 1.75]
AFFINE_OFFSETS = [0.0, 5.0, -5.0, 12.5, -3.25, 100.0, -40.0, 0.75, -0.5, 250.0]

#: Half integer, half decimal, spanning positive and negative.
AFFINE_X_VALUES = [50.0, 12.0, -8.0, 100.0, 3.0, 37.5, -12.25, 0.5, 250.0, 7.125, -60.0, 1.0]


def derive_competence_seed(master_seed: int, index: int) -> int:
    return master_seed * 1_000_000 + COMPETENCE_SEED_NAMESPACE + index


def _reference_range_plan(n: int) -> List[Tuple[RangeClass, BoundaryKind]]:
    """Class and boundary placement for the reference-range half.

    Boundary cases are placed deliberately, not left to sampling:
    * one result exactly on the lower endpoint and one exactly on the upper --
      both WITHIN, because the interval is closed. These are the cases that
      separate a model that read the convention from one that guessed it.
    * one just below the lower endpoint and one just above the upper, so the
      near misses are represented in BELOW and ABOVE too.
    """
    per_class = n // 3
    if n % 3:
        raise ValueError(f"reference-range count {n} must divide by 3 for a 5/5/5 split")
    plan: List[Tuple[RangeClass, BoundaryKind]] = []
    plan.append((RangeClass.WITHIN, BoundaryKind.AT_LOWER))
    plan.append((RangeClass.WITHIN, BoundaryKind.AT_UPPER))
    plan.extend([(RangeClass.WITHIN, BoundaryKind.NONE)] * (per_class - 2))
    plan.append((RangeClass.BELOW, BoundaryKind.JUST_BELOW))
    plan.extend([(RangeClass.BELOW, BoundaryKind.NONE)] * (per_class - 1))
    plan.append((RangeClass.ABOVE, BoundaryKind.JUST_ABOVE))
    plan.extend([(RangeClass.ABOVE, BoundaryKind.NONE)] * (per_class - 1))
    return plan


def _round_to_range(value: float, span: float) -> float:
    """Round to a precision that suits the interval's scale.

    A span of 0.08 needs more decimals than a span of 750; using one fixed
    precision would either collapse narrow intervals or litter wide ones.
    """
    if span >= 100:
        return round(value, 1)
    if span >= 1:
        return round(value, 2)
    return round(value, 4)


def build_reference_range_case(
    index: int, target: RangeClass, boundary: BoundaryKind, master_seed: int
) -> ReferenceRangeCase:
    seed = derive_competence_seed(master_seed, index)
    rng = random.Random(seed)

    analyte = ANALYTES[index % len(ANALYTES)]
    low, high = REFERENCE_RANGES[index % len(REFERENCE_RANGES)]
    span = high - low
    # A step that is visible at the interval's scale but never large enough to
    # push a "just below" case past the far endpoint.
    step = _round_to_range(max(span * 0.02, 1e-3), span)

    if boundary is BoundaryKind.AT_LOWER:
        result = low
    elif boundary is BoundaryKind.AT_UPPER:
        result = high
    elif boundary is BoundaryKind.JUST_BELOW:
        result = _round_to_range(low - step, span)
    elif boundary is BoundaryKind.JUST_ABOVE:
        result = _round_to_range(high + step, span)
    elif target is RangeClass.WITHIN:
        # strictly interior, so it cannot collide with an endpoint
        result = _round_to_range(rng.uniform(low + span * 0.1, high - span * 0.1), span)
    elif target is RangeClass.BELOW:
        result = _round_to_range(low - rng.uniform(span * 0.15, span * 1.5), span)
    else:
        result = _round_to_range(high + rng.uniform(span * 0.15, span * 1.5), span)

    # Rounding could in principle land an interior draw back on an endpoint;
    # nudge it inward rather than mislabelling it.
    if boundary is BoundaryKind.NONE and result in (low, high):
        result = _round_to_range(result + step, span) if result == low else _round_to_range(
            result - step, span
        )

    actual = classify_against_range(result, (low, high))
    if actual is not target:
        raise RuntimeError(
            f"case {index}: built result {result} for {(low, high)} lands in "
            f"{actual.value}, not the planned {target.value}"
        )

    return ReferenceRangeCase(
        case_id=f"competence-rr-{index:03d}",
        random_seed=seed,
        analyte=analyte,
        result=result,
        reference_range=(low, high),
        expected_classification=target,
        boundary_kind=boundary,
    )


def build_affine_case(index: int, position: int, master_seed: int) -> AffineInversionCase:
    seed = derive_competence_seed(master_seed, index)

    a = AFFINE_GAINS[position % len(AFFINE_GAINS)]
    b = AFFINE_OFFSETS[position % len(AFFINE_OFFSETS)]
    x = AFFINE_X_VALUES[position % len(AFFINE_X_VALUES)]

    # y is derived from the answer, not the other way round, so the case is
    # exactly solvable and expected_x needs no re-derivation from a rounded y.
    y = round(a * x + b, 6)

    return AffineInversionCase(
        case_id=f"competence-af-{index:03d}",
        random_seed=seed,
        a=a,
        b=b,
        y=y,
        expected_x=x,
    )


def generate_competence_dataset(cfg: PilotConfig) -> List[CompetenceCase]:
    """``cases_per_model`` cases, half reference-range and half affine."""
    total = cfg.competence.cases_per_model
    half = total // 2

    cases: List[CompetenceCase] = []
    for i, (target, boundary) in enumerate(_reference_range_plan(half)):
        cases.append(build_reference_range_case(i, target, boundary, cfg.master_seed))
    for position in range(half):
        cases.append(build_affine_case(half + position, position, cfg.master_seed))
    return cases


def write_competence_jsonl(cases: Sequence[CompetenceCase], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for case in cases:
            fh.write(json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return out


def read_competence_jsonl(path: str | Path) -> List[CompetenceCase]:
    with open(path, "r", encoding="utf-8") as fh:
        return [_CASE_ADAPTER.validate_python(json.loads(line)) for line in fh if line.strip()]


def format_competence_case(case: CompetenceCase) -> str:
    if case.skill is CompetenceSkill.REFERENCE_RANGE:
        return (
            f"{case.case_id}  [reference_range]\n"
            f"  analyte          : {case.analyte}\n"
            f"  reference_range  : [{case.reference_range[0]:g}, {case.reference_range[1]:g}] "
            f"(closed)\n"
            f"  result           : {case.result:g}\n"
            f"  boundary         : {case.boundary_kind.value}\n"
            f"  GROUND TRUTH     : {case.expected_classification.value}"
        )
    return (
        f"{case.case_id}  [affine_inversion]\n"
        f"  function         : y = {case.a:g}*x + {case.b:g}\n"
        f"  y                : {case.y:g}\n"
        f"  GROUND TRUTH     : x = {case.expected_x:g}\n"
        f"  tolerance        : +-{max(1e-6, abs(case.expected_x) * 1e-4):g}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the competence pre-test.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--out", default="data/generated/competence_cases.jsonl")
    parser.add_argument("--preview", type=int, default=3)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cases = generate_competence_dataset(cfg)
    out = write_competence_jsonl(cases, args.out)

    counts: Dict[str, int] = {}
    for case in cases:
        counts[case.skill.value] = counts.get(case.skill.value, 0) + 1
    print(f"wrote {len(cases)} competence cases -> {out}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    rr = [c for c in cases if c.skill is CompetenceSkill.REFERENCE_RANGE]
    by_class: Dict[str, int] = {}
    for case in rr:
        by_class[case.expected_classification.value] = (
            by_class.get(case.expected_classification.value, 0) + 1
        )
    print("  reference-range classes: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_class.items())))

    if args.preview:
        for skill in (CompetenceSkill.REFERENCE_RANGE, CompetenceSkill.AFFINE_INVERSION):
            print()
            for case in [c for c in cases if c.skill is skill][: args.preview]:
                print("-" * 70)
                print(format_competence_case(case))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

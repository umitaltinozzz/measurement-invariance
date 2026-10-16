"""The paired dataset: one case set, answered by both arms.

proje.md section 8's central prediction is that a model can state the right thing
in prose and then write ``df["value"].pct_change()`` anyway.  That is a claim
about *the same problem* being handled differently in two modes, so it has to be
measured within a case.  Two arms run on two different case sets can differ for
any number of reasons -- one set happening to contain larger shifts, or an
unluckier switch-point draw -- and none of those reasons is the hypothesis.

So: 40 cases, cells 2 and 3, L1, 20 each. Every case is answered twice, once as a
table in a prompt and once as a CSV to be analysed with code, under the same
``case_id``. The latent state, the observed values, the provenance, the switch
point and the ground truth are byte-identical across the two arms; only the
presentation differs, and ``tests/test_paired.py`` asserts that.

Cell choice: cell 2 is where ignoring provenance manufactures a trend, cell 3 is
the control that stops "group by sensor_version and shrug" from scoring well.
L1 because the reference range is present, so the transformation is actually
invertible from what the agent is handed.

Run:
    python -m src.paired_dataset --config configs/pilot.yaml \
        --out data/generated/paired_agent_cases.jsonl --preview 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

from .generate_cases import (
    PairedDatasetConfig,
    PilotConfig,
    format_case,
    generate_dataset,
    load_config,
    read_jsonl,
    write_jsonl,
)
from .schemas import BenchmarkCase, SalienceLevel

DEFAULT_PATH = "data/generated/paired_agent_cases.jsonl"

#: Seed namespace for the paired dataset. Distinct from the direct pilot's and
#: from the competence pre-test's, so no case is ever shared or duplicated
#: between datasets by accident.
PAIRED_SEED_NAMESPACE = 9_900_000


def paired_dataset_config(cfg: PilotConfig) -> PilotConfig:
    """The generator settings that produce the paired cases.

    Reuses the main generator wholesale -- same validators, same balanced
    measurement grid, same ground-truth rules -- with the scope narrowed to one
    salience level and the case ids prefixed so a paired case can never be
    confused with a direct-pilot one.
    """
    paired = cfg.paired
    return cfg.model_copy(
        update={
            "master_seed": cfg.master_seed + PAIRED_SEED_NAMESPACE,
            "n_per_cell": paired.cases_per_cell,
            "salience_weights": {SalienceLevel(paired.salience_level): paired.cases_per_cell},
            "case_id_prefix": paired.case_id_prefix,
        }
    )


def select_paired_cells(
    cases: Sequence[BenchmarkCase], paired: PairedDatasetConfig
) -> List[BenchmarkCase]:
    """The configured cells at the configured salience, ordered by ``case_id``."""
    selected: List[BenchmarkCase] = []
    for cell in paired.cells:
        matching = sorted(
            (
                c
                for c in cases
                if c.cell == cell
                and c.salience_level is SalienceLevel(paired.salience_level)
            ),
            key=lambda c: c.case_id,
        )
        if len(matching) < paired.cases_per_cell:
            raise ValueError(
                f"cell {cell} at {paired.salience_level.value} has only {len(matching)} "
                f"cases, need {paired.cases_per_cell}"
            )
        selected.extend(matching[: paired.cases_per_cell])
    return selected


def generate_paired_dataset(cfg: PilotConfig) -> List[BenchmarkCase]:
    return select_paired_cells(generate_dataset(paired_dataset_config(cfg)), cfg.paired)


def load_paired_dataset(path: str | Path = DEFAULT_PATH) -> List[BenchmarkCase]:
    return read_jsonl(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the paired dataset.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--out", default=DEFAULT_PATH)
    parser.add_argument("--preview", type=int, default=2)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cases = generate_paired_dataset(cfg)
    out = write_jsonl(cases, args.out)

    counts: Dict[int, int] = {}
    for case in cases:
        counts[case.cell] = counts.get(case.cell, 0) + 1
    print(f"wrote {len(cases)} paired cases -> {out}")
    print("  " + ", ".join(f"cell {c}: {n}" for c, n in sorted(counts.items())))
    print(f"  salience: {sorted({c.salience_level.value for c in cases})}")
    print("  every case is answered by BOTH the direct-natural and code-agent arms")

    for case in cases[: args.preview]:
        print("\n" + "-" * 78)
        print(format_case(case))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

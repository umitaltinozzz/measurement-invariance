"""Verify runner artifacts, score them with project graders, and apply continuation.

This module consumes only runner JSONL and its immutable manifests. It does not
repair, retry, or remove trials. Failed rows are presented to the strict graders
as empty completions, which keeps them visible in the all-trial invalid-rate
gate while the original provider failure remains in the raw result file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .code_agent import csv_columns
from .code_agent_grader import CodeAgentRecord, grade_code_agent_response
from .code_agent_metrics import code_agent_breakdown, format_code_agent_metrics
from .competence_grader import CompetenceResponseRecord, grade_all_competence
from .competence_metrics import (
    compute_competence_breakdown,
    format_qualification_table,
    qualify_all,
)
from .continuation import (
    ContinuationDecision,
    evaluate_continuation,
    format_continuation_decision,
)
from .generate_cases import load_config, read_jsonl
from .generate_competence import read_competence_jsonl
from .grader import ModelResponseRecord, ResponseClass, grade_all
from .metrics import Rate, breakdown
from .paired_metrics import compute_paired_breakdown, pair_responses
from .prompts import PromptVariant
from .reporting_stats import exact_mcnemar_p, wilson_interval
from .run_manifest import read_manifest, sha256_file
from .runner import TrialRecord


def _result_path_for_manifest(manifest_path: Path) -> Path:
    suffix = ".manifest.json"
    if not manifest_path.name.endswith(suffix):
        raise ValueError(f"not a result manifest path: {manifest_path}")
    return manifest_path.with_name(manifest_path.name[: -len(suffix)] + ".jsonl")


def _jsonl(path: Path) -> Iterable[str]:
    if not path.exists():
        return ()
    return (line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _rate(rate) -> Dict[str, Any]:
    return {
        "numerator": rate.numerator,
        "denominator": rate.denominator,
        "value": rate.value,
    }


def _wilson_95(rate) -> Optional[Dict[str, float]]:
    interval = wilson_interval(rate.numerator, rate.denominator)
    if interval is None:
        return None
    return {"lower": interval[0], "upper": interval[1]}


def _fmt_rate_wilson(rate: Dict[str, Any], interval: Optional[Dict[str, float]]) -> str:
    rendered = _fmt_rate(rate)
    if interval is None:
        return rendered
    return f"{rendered} [{interval['lower']:.1%}, {interval['upper']:.1%}]"


def _dual_rate(dual) -> Dict[str, Any]:
    return {"valid_only": _rate(dual.valid_only), "all_trials": _rate(dual.all_trials)}


def _fmt_rate(rate: Dict[str, Any]) -> str:
    value = rate["value"]
    if value is None:
        return f"n/a (0/{rate['denominator']})"
    return f"{value:.1%} ({rate['numerator']}/{rate['denominator']})"


def _fmt_dual(dual: Dict[str, Any], basis: str = "valid_only") -> str:
    return _fmt_rate(dual[basis])


def _load_verified_runs(
    results_dir: Path,
    project_root: Path,
    config_path: Optional[Path],
    *,
    config_sources: Optional[Mapping[str, str | Path]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    expected_config_sha = sha256_file(config_path) if config_path is not None else None
    config_sources = config_sources or {}
    runs: List[Dict[str, Any]] = []
    verification: List[Dict[str, Any]] = []
    for manifest_path in sorted(results_dir.rglob("*.manifest.json")):
        manifest = read_manifest(manifest_path)
        result_path = _result_path_for_manifest(manifest_path)
        dataset_path = project_root / Path(manifest.dataset_path)
        manifested_config_path = Path(
            config_sources.get(
                manifest.config_path,
                project_root / Path(manifest.config_path),
            )
        )
        if not manifested_config_path.is_absolute():
            manifested_config_path = project_root / manifested_config_path
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"manifest {manifest_path} points to missing dataset {dataset_path}"
            )
        if not manifested_config_path.is_file():
            raise FileNotFoundError(
                f"manifest {manifest_path} points to missing config {manifested_config_path}"
            )
        dataset_sha = sha256_file(dataset_path)
        config_sha = sha256_file(manifested_config_path)
        if dataset_sha != manifest.dataset_sha256:
            raise ValueError(
                f"dataset SHA mismatch for {manifest_path}: "
                f"manifest={manifest.dataset_sha256}, actual={dataset_sha}"
            )
        if config_sha != manifest.config_sha256 or (
            expected_config_sha is not None and config_sha != expected_config_sha
        ):
            raise ValueError(
                f"config SHA mismatch for {manifest_path}: "
                f"manifest={manifest.config_sha256}, actual={config_sha}, "
                f"expected={expected_config_sha or 'manifested config'} "
                f"(source={manifested_config_path})"
            )
        if not result_path.exists():
            raise FileNotFoundError(f"manifest has no adjacent result file: {result_path}")

        records = [TrialRecord.model_validate_json(line) for line in _jsonl(result_path)]
        keys = [record.key.as_tuple() for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate logical trial key in {result_path}")
        if any(record.dataset_sha256 != manifest.dataset_sha256 for record in records):
            raise ValueError(f"row dataset SHA mismatch in {result_path}")
        if any(record.model != manifest.model or record.arm != manifest.arm for record in records):
            raise ValueError(f"row model/arm does not match manifest in {result_path}")

        try:
            dataset_planned = len(
                read_competence_jsonl(dataset_path)
                if manifest.arm == "competence"
                else read_jsonl(dataset_path)
            )
        except Exception as exc:
            raise ValueError(f"could not count planned cases in {dataset_path}: {exc}") from exc
        if manifest.selected_case_ids is not None:
            selected = manifest.selected_case_ids
            if len(selected) != len(set(selected)):
                raise ValueError(f"duplicate selected case id in manifest {manifest_path}")
            if not set(record.case_id for record in records).issubset(set(selected)):
                raise ValueError(f"result rows exceed the selected case IDs in {result_path}")
            planned = len(selected)
        else:
            planned = dataset_planned
        failed = sum(1 for record in records if record.failure)
        run = {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "result_path": result_path,
            "dataset_path": dataset_path,
            "records": records,
            "planned": planned,
            "failed": failed,
        }
        runs.append(run)
        verification.append(
            {
                "manifest": str(manifest_path.relative_to(results_dir)),
                "results": str(result_path.relative_to(results_dir)),
                "fingerprint": manifest.fingerprint,
                "provider": manifest.provider,
                "model": manifest.model,
                "arm": manifest.arm,
                "prompt_variant": manifest.prompt_variant,
                "dataset_path": manifest.dataset_path,
                "dataset_sha256": dataset_sha,
                "config_path": manifest.config_path,
                "config_sha256": config_sha,
                "hardware": manifest.hardware,
                "selected_case_ids": manifest.selected_case_ids,
                "manifest_git_commit": manifest.git_commit,
                "provider_constraints": manifest.provider_constraints,
                "created_at": manifest.created_at,
                "planned": planned,
                "rows": len(records),
                "failed_rows": failed,
                "complete": len(records) == planned,
            }
        )
    return runs, verification


def _record_text(record: TrialRecord) -> str:
    return record.result.raw_text if record.result is not None else ""


def _main_direct_records(runs: Sequence[Dict[str, Any]], dataset_name: str, variant: str):
    return [
        ModelResponseRecord(
            case_id=record.case_id,
            model=record.model,
            raw_text=_record_text(record),
            prompt_variant=variant,
        )
        for run in runs
        if run["manifest"].arm == "direct"
        and Path(run["manifest"].dataset_path).name == dataset_name
        and run["manifest"].prompt_variant == variant
        for record in run["records"]
    ]


def _direct_metric_rows(graded) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped = breakdown(graded, by=("model", "prompt_variant", "cell"))
    for (model, variant, cell), metrics in grouped.items():
        rows.append(
            {
                "model": model,
                "prompt_variant": variant,
                "cell": cell,
                "n_responses": metrics.n_responses,
                "n_valid": metrics.n_valid,
                "spurious_trend_rate": _dual_rate(metrics.spurious_trend_rate),
                "miss_rate": _dual_rate(metrics.miss_rate),
                "invalid_response_rate": _rate(metrics.invalid_response_rate),
                "spurious_trend_rate_wilson_95": _wilson_95(
                    metrics.spurious_trend_rate.valid_only
                ),
                "miss_rate_wilson_95": _wilson_95(metrics.miss_rate.valid_only),
            }
        )
    return rows


def _direct_diagnostic_rows(graded, cases) -> List[Dict[str, Any]]:
    """Supplementary abstention and cell-4 diagnostics by model and prompt."""
    naive_matches = {case.case_id: case.naive_observed_trend() == case.expected_trend for case in cases}
    grouped: Dict[Tuple[str, str], List[Any]] = {}
    for row in graded:
        grouped.setdefault((row.model, row.prompt_variant.value), []).append(row)

    output = []
    for (model, variant), rows in sorted(grouped.items()):
        valid = [r for r in rows if r.response_class is not ResponseClass.INVALID_RESPONSE]
        uncertain = [
            r for r in valid
            if r.response_class in (ResponseClass.CORRECT_ABSTENTION, ResponseClass.OVER_ABSTENTION)
        ]
        cell1 = [r for r in valid if r.cell == 1]
        cell3 = [r for r in valid if r.cell == 3]
        recoverable_cell4 = [r for r in valid if r.cell == 4 and r.latent_recovery_defined]
        cell4 = [r for r in valid if r.cell == 4]

        def rate_for(numerator: int, denominator: int) -> Dict[str, Any]:
            return _rate(Rate.of(numerator, denominator))

        direction_slices = {}
        for key, is_match in (("raw_direction_correct", True), ("raw_direction_misleading", False)):
            subset = [r for r in cell4 if naive_matches[r.case_id] is is_match]
            wrong = sum(1 for r in subset if r.response_class is ResponseClass.WRONG_DIRECTION)
            direction_slices[key] = {
                "n_valid": len(subset),
                "wrong_direction_count": wrong,
                "wrong_direction_rate": rate_for(wrong, len(subset)),
            }

        recovered = sum(1 for r in recoverable_cell4 if r.latent_recovery_correct)
        output.append(
            {
                "model": model,
                "prompt_variant": variant,
                "n_responses": len(rows),
                "n_valid": len(valid),
                "uncertain_rate": rate_for(len(uncertain), len(valid)),
                "cell1_blanket_skepticism_rate": rate_for(
                    sum(1 for r in cell1 if r.blanket_skepticism), len(cell1)
                ),
                "cell3_over_abstention_rate": rate_for(
                    sum(1 for r in cell3 if r.response_class is ResponseClass.OVER_ABSTENTION),
                    len(cell3),
                ),
                "cell4_latent_recovery_rate": rate_for(recovered, len(recoverable_cell4)),
                "cell4_raw_direction_split": direction_slices,
            }
        )
    return output


def _competence_rows(graded, cfg) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metrics_by_model = compute_competence_breakdown(graded)
    qualifications = qualify_all(graded, cfg.competence)
    rows = []
    for model, metrics in sorted(metrics_by_model.items()):
        qualification = qualifications[model]
        rows.append(
            {
                "model": model,
                "n_responses": metrics.n_responses,
                "n_valid": metrics.n_valid,
                "reference_range_accuracy": _dual_rate(metrics.reference_range_accuracy),
                "affine_inversion_accuracy": _dual_rate(metrics.affine_inversion_accuracy),
                "overall_competence_accuracy": _dual_rate(metrics.overall_competence_accuracy),
                "invalid_response_rate": _rate(metrics.invalid_response_rate),
                "qualified": qualification.qualified,
                "qualification_basis": qualification.basis,
                "qualification_failures": qualification.failures,
                "qualification_warnings": qualification.warnings,
            }
        )
    return rows, qualifications


def _code_rows(graded) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metrics_by_key = code_agent_breakdown(graded)
    rows = []
    by_model = {}
    for (model,), metrics in sorted(metrics_by_key.items()):
        by_model[model] = metrics
        rows.append(
            {
                "model": model,
                "n_responses": metrics.n_responses,
                "n_valid": metrics.n_valid,
                "provenance_retention_rate": _dual_rate(metrics.provenance_retention_rate),
                "effective_provenance_use_rate": _dual_rate(metrics.effective_provenance_use_rate),
                "spurious_trend_rate": _dual_rate(metrics.spurious_trend_rate),
                "miss_rate": _dual_rate(metrics.miss_rate),
                "invalid_response_rate": _rate(metrics.invalid_response_rate),
                "code_syntax_error_rate": _rate(metrics.code_syntax_error_rate),
                "provenance_retention_rate_wilson_95": _wilson_95(
                    metrics.provenance_retention_rate.valid_only
                ),
                "spurious_trend_rate_wilson_95": _wilson_95(
                    metrics.spurious_trend_rate.valid_only
                ),
                "miss_rate_wilson_95": _wilson_95(metrics.miss_rate.valid_only),
                "provenance_levels": metrics.level_counts,
                "anti_patterns": metrics.anti_pattern_counts,
                "needs_human_review": metrics.needs_human_review,
            }
        )
    return rows, by_model


def _paired_rows(direct_graded, code_graded) -> List[Dict[str, Any]]:
    result = compute_paired_breakdown(direct_graded, code_graded)
    return [
        {
            "model": model,
            **metrics.model_dump(mode="json"),
        }
        for model, metrics in sorted(result.items())
    ]


def _paired_direct_code_rows(direct_graded, code_graded) -> List[Dict[str, Any]]:
    """Exact paired McNemar comparisons for all cases and cell-2 false trends."""
    models = sorted({r.model for r in direct_graded} | {r.model for r in code_graded})
    output = []
    for model in models:
        pairing = pair_responses(direct_graded, code_graded, model=model)
        complete = pairing.pairs
        valid_pairs = [p for p in complete if not p.direct_invalid and not p.code_invalid]
        direct_only_correct = sum(1 for p in complete if p.direct_correct and not p.code_correct)
        code_only_correct = sum(1 for p in complete if p.code_correct and not p.direct_correct)

        cell2_pairs = [p for p in complete if p.cell == 2]
        valid_cell2 = [p for p in cell2_pairs if not p.direct_invalid and not p.code_invalid]
        direct_spurious = sum(1 for p in cell2_pairs if p.direct_class is ResponseClass.SPURIOUS_TREND)
        code_spurious = sum(1 for p in cell2_pairs if p.code_class is ResponseClass.SPURIOUS_TREND)
        direct_only_spurious = sum(
            1 for p in cell2_pairs
            if p.direct_class is ResponseClass.SPURIOUS_TREND
            and p.code_class is not ResponseClass.SPURIOUS_TREND
        )
        code_only_spurious = sum(
            1 for p in cell2_pairs
            if p.code_class is ResponseClass.SPURIOUS_TREND
            and p.direct_class is not ResponseClass.SPURIOUS_TREND
        )
        direct_uncertain = sum(
            1 for p in cell2_pairs if p.direct_class is ResponseClass.CORRECT_ABSTENTION
        )
        code_uncertain = sum(
            1 for p in cell2_pairs if p.code_class is ResponseClass.CORRECT_ABSTENTION
        )
        direct_invalid = sum(1 for p in cell2_pairs if p.direct_invalid)
        code_invalid = sum(1 for p in cell2_pairs if p.code_invalid)
        direct_other = len(cell2_pairs) - direct_spurious - direct_uncertain - direct_invalid
        code_other = len(cell2_pairs) - code_spurious - code_uncertain - code_invalid

        valid_direct_only_spurious = sum(
            1 for p in valid_cell2
            if p.direct_class is ResponseClass.SPURIOUS_TREND
            and p.code_class is not ResponseClass.SPURIOUS_TREND
        )
        valid_code_only_spurious = sum(
            1 for p in valid_cell2
            if p.code_class is ResponseClass.SPURIOUS_TREND
            and p.direct_class is not ResponseClass.SPURIOUS_TREND
        )

        output.append(
            {
                "model": model,
                "all_cells": {
                    "n_pairs": len(complete),
                    "n_valid_pairs": len(valid_pairs),
                    "direct_only_correct": direct_only_correct,
                    "code_only_correct": code_only_correct,
                    "exact_mcnemar_p": exact_mcnemar_p(direct_only_correct, code_only_correct),
                },
                "cell2": {
                    "n_pairs": len(cell2_pairs),
                    "n_valid_pairs": len(valid_cell2),
                    "direct_spurious_trend": _rate(Rate.of(direct_spurious, len(cell2_pairs))),
                    "code_spurious_trend": _rate(Rate.of(code_spurious, len(cell2_pairs))),
                    "direct_uncertain": _rate(Rate.of(direct_uncertain, len(cell2_pairs))),
                    "code_uncertain": _rate(Rate.of(code_uncertain, len(cell2_pairs))),
                    "direct_other_valid": _rate(Rate.of(direct_other, len(cell2_pairs))),
                    "code_other_valid": _rate(Rate.of(code_other, len(cell2_pairs))),
                    "direct_invalid_count": direct_invalid,
                    "code_invalid_count": code_invalid,
                    "direct_only_spurious": direct_only_spurious,
                    "code_only_spurious": code_only_spurious,
                    "exact_mcnemar_p": exact_mcnemar_p(direct_only_spurious, code_only_spurious),
                    "valid_pairs_direct_only_spurious": valid_direct_only_spurious,
                    "valid_pairs_code_only_spurious": valid_code_only_spurious,
                    "valid_pairs_exact_mcnemar_p": exact_mcnemar_p(
                        valid_direct_only_spurious, valid_code_only_spurious
                    ),
                },
            }
        )
    return output


def _write_chart(metric_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    natural = [row for row in metric_rows if row["prompt_variant"] == "natural"]
    models = sorted({row["model"] for row in natural})
    by_key = {(row["model"], row["cell"]): row for row in natural}
    cells = [1, 2, 3, 4]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    fig.suptitle("Direct natural arm by cell and model", fontsize=14, fontweight="bold")
    panel_specs = (
        ("spurious_trend_rate", "Spurious Trend Rate\n(stable truth)"),
        ("miss_rate", "Miss Rate\n(changed truth)"),
    )
    if not models:
        for ax in axes:
            ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    width = min(0.18, 0.72 / len(models))
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
    colors = plt.get_cmap("tab10")
    for panel_index, (metric_name, title) in enumerate(panel_specs):
        ax = axes[panel_index]
        for model_index, model in enumerate(models):
            values = []
            for cell in cells:
                row = by_key.get((model, cell))
                rate = row[metric_name]["valid_only"] if row else None
                values.append(float("nan") if not rate or rate["value"] is None else rate["value"])
            positions = np.asarray(cells, dtype=float) + offsets[model_index]
            bars = ax.bar(
                positions,
                values,
                width=width,
                label=model,
                color=colors(model_index % 10),
                edgecolor="white",
                linewidth=0.6,
            )
            for bar, cell in zip(bars, cells):
                row = by_key.get((model, cell))
                rate = row[metric_name]["valid_only"] if row else None
                if rate and rate["denominator"]:
                    ax.annotate(
                        f"{rate['numerator']}/{rate['denominator']}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )
        ax.set_title(title)
        ax.set_xticks(cells, [f"Cell {cell}" for cell in cells])
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Rate (valid-only; numerator/denominator above bars)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    fig.text(0.5, 0.015, "A blank cell means the metric is undefined for that truth condition.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 0.84, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_analysis_markdown(
    output_path: Path,
    verification: Sequence[Dict[str, Any]],
    competence_rows: Sequence[Dict[str, Any]],
    direct_rows: Sequence[Dict[str, Any]],
    direct_diagnostics: Sequence[Dict[str, Any]],
    code_rows: Sequence[Dict[str, Any]],
    paired_direct_code: Sequence[Dict[str, Any]],
    continuation_text: str,
    run_status: Optional[Dict[str, Any]],
    analysis_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    lines = [
        "# Runner output analysis",
        "",
        "Generated by `src.analyze_results` from runner JSONL and adjacent manifests.",
        "Every direct cell row shows Spurious Trend Rate and Miss Rate together; an undefined metric is shown as `n/a`.",
        "",
    ]
    if analysis_metadata and analysis_metadata.get("replaced_rows"):
        caps = analysis_metadata["changed_caps"]
        lines.extend(
            [
                "## Run 2 + re-runs replacement audit",
                "",
                f"Run 2 hardware: **{analysis_metadata['original_hardware']}**. "
                f"Re-run hardware: **{analysis_metadata['rerun_hardware']}**.",
                f"Changed caps: direct `max_tokens` {caps['direct_max_tokens']['run2']} "
                f"to {caps['direct_max_tokens']['rerun']}; code-agent `max_tokens_code_agent` "
                f"{caps['code_agent_max_tokens']['run2']} to {caps['code_agent_max_tokens']['rerun']}. "
                "All other run-2 rows remain unchanged.",
                analysis_metadata.get("diagnosis_discrepancy", ""),
                "",
                "| Model | Arm | Variant | Replaced rows | Config | Hardware | Case IDs |",
                "|---|---|---|---:|---|---|---|",
            ]
        )
        for item in analysis_metadata["replaced_rows"]:
            lines.append(
                f"| {item['model']} | {item['arm']} | {item['variant'] or '—'} | "
                f"{item['count']} | `{item['config_path']}` | {item['hardware']} | "
                f"{', '.join(item['case_ids'])} |"
            )
        lines.extend([""])
    lines.extend(
        [
            "## Manifest verification",
            "",
            "| Model | Arm | Variant | Rows / planned | Failed rows | Dataset SHA | Config SHA | Created at |",
            "|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for item in verification:
        lines.append(
            f"| {item['model']} | {item['arm']} | {item['prompt_variant'] or '—'} | "
            f"{item['rows']} / {item['planned']} | {item['failed_rows']} | "
            f"`{item['dataset_sha256']}` | `{item['config_sha256']}` | {item['created_at']} |"
        )
    lines.extend(
        [
            "",
            "## Competence pre-test",
            "",
            "| Model | Overall | Reference range | Affine inversion | Invalid | Qualified |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in competence_rows:
        lines.append(
            f"| {item['model']} | {_fmt_dual(item['overall_competence_accuracy'], 'all_trials')} | "
            f"{_fmt_dual(item['reference_range_accuracy'], 'all_trials')} | "
            f"{_fmt_dual(item['affine_inversion_accuracy'], 'all_trials')} | "
            f"{_fmt_rate(item['invalid_response_rate'])} | "
            f"{'yes' if item['qualified'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Direct arms by cell",
            "",
            "Rates below are valid-only; invalid responses are shown separately. All-trial rates remain in `analysis.json`.",
            "",
            "| Model | Variant | Cell | Spurious Trend Rate | Miss Rate | Invalid response rate | n |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in sorted(direct_rows, key=lambda row: (row["model"], row["prompt_variant"], row["cell"])):
        lines.append(
            f"| {item['model']} | {item['prompt_variant']} | {item['cell']} | "
            f"{_fmt_dual(item['spurious_trend_rate'])} | {_fmt_dual(item['miss_rate'])} | "
            f"{_fmt_rate(item['invalid_response_rate'])} | {item['n_responses']} |"
        )
    lines.extend(
        [
            "",
            "## Abstention and cell-4 diagnostics",
            "",
            "Rates use valid responses. Cell-4 latent recovery is defined for L1 cases only; direction subsets compare the observed raw direction with the latent direction.",
            "",
            "| Model | Variant | Uncertain overall | Cell-1 blanket skepticism | Cell-3 over-abstention | Cell-4 latent recovery (L1) | Cell-4 raw direction correct: wrong direction | Cell-4 raw direction misleading: wrong direction |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in direct_diagnostics:
        correct = item["cell4_raw_direction_split"]["raw_direction_correct"]
        misleading = item["cell4_raw_direction_split"]["raw_direction_misleading"]
        lines.append(
            f"| {item['model']} | {item['prompt_variant']} | {_fmt_rate(item['uncertain_rate'])} | "
            f"{_fmt_rate(item['cell1_blanket_skepticism_rate'])} | "
            f"{_fmt_rate(item['cell3_over_abstention_rate'])} | "
            f"{_fmt_rate(item['cell4_latent_recovery_rate'])} | "
            f"{correct['wrong_direction_count']}/{correct['n_valid']} | "
            f"{misleading['wrong_direction_count']}/{misleading['n_valid']} |"
        )
    lines.extend(
        [
            "",
            "## Wilson 95% intervals",
            "",
            "Direct-arm STR/MR intervals use the valid-only numerators and denominators.",
            "",
            "| Model | Variant | Cell | STR (95% Wilson CI) | MR (95% Wilson CI) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in sorted(direct_rows, key=lambda row: (row["model"], row["prompt_variant"], row["cell"])):
        str_rate = item["spurious_trend_rate"]["valid_only"]
        mr_rate = item["miss_rate"]["valid_only"]
        lines.append(
            f"| {item['model']} | {item['prompt_variant']} | {item['cell']} | "
            f"{_fmt_rate_wilson(str_rate, item['spurious_trend_rate_wilson_95'])} | "
            f"{_fmt_rate_wilson(mr_rate, item['miss_rate_wilson_95'])} |"
        )
    if code_rows:
        lines.extend(
            [
                "",
                "Code-agent valid-only intervals (STR/MR are pooled across the code-agent cases; PRR pools cells 2 and 3):",
                "",
                "| Model | PRR (95% Wilson CI) | STR (95% Wilson CI) | MR (95% Wilson CI) |",
                "|---|---:|---:|---:|",
            ]
        )
        for item in code_rows:
            lines.append(
                f"| {item['model']} | "
                f"{_fmt_rate_wilson(item['provenance_retention_rate']['valid_only'], item['provenance_retention_rate_wilson_95'])} | "
                f"{_fmt_rate_wilson(item['spurious_trend_rate']['valid_only'], item['spurious_trend_rate_wilson_95'])} | "
                f"{_fmt_rate_wilson(item['miss_rate']['valid_only'], item['miss_rate_wilson_95'])} |"
            )
    lines.extend(
        [
            "",
            "## Paired direct-vs-code comparison",
            "",
            "Exact two-sided McNemar tests use all complete pairs. The cell-2 test compares whether each arm reported a spurious trend; invalid outputs count as no reported spurious trend and are shown separately. A valid-pairs-only cell-2 p-value is included as a sensitivity check.",
            "",
            "| Model | All-cell pairs | Direct-only correct | Code-only correct | All-cell exact p | Cell-2 pairs | Direct cell-2 spurious | Code cell-2 spurious | Direct uncertain | Code uncertain | Invalid D/C | Direct-only spurious | Code-only spurious | Cell-2 exact p | Valid-pairs p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in paired_direct_code:
        all_cells = item["all_cells"]
        cell2 = item["cell2"]
        lines.append(
            f"| {item['model']} | {all_cells['n_pairs']} ({all_cells['n_valid_pairs']} valid) | "
            f"{all_cells['direct_only_correct']} | {all_cells['code_only_correct']} | "
            f"{all_cells['exact_mcnemar_p']:.4g} | {cell2['n_pairs']} ({cell2['n_valid_pairs']} valid) | "
            f"{_fmt_rate(cell2['direct_spurious_trend'])} | {_fmt_rate(cell2['code_spurious_trend'])} | "
            f"{_fmt_rate(cell2['direct_uncertain'])} | {_fmt_rate(cell2['code_uncertain'])} | "
            f"{cell2['direct_invalid_count']}/{cell2['code_invalid_count']} | "
            f"{cell2['direct_only_spurious']} | {cell2['code_only_spurious']} | "
            f"{cell2['exact_mcnemar_p']:.4g} | {cell2['valid_pairs_exact_mcnemar_p']:.4g} |"
        )
    lines.extend(["", "## Code-agent arm", ""])
    if run_status and run_status.get("code_agent_arm"):
        arm_status = run_status["code_agent_arm"]
        lines.append(
            f"Status: **{arm_status.get('status', 'not recorded')}**. "
            f"{arm_status.get('reason', '')}".strip()
        )
        lines.append("")
    if code_rows:
        lines.extend(
            [
                "| Model | Provenance retention | Effective provenance use | Invalid response rate | Syntax error rate | n |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in code_rows:
            lines.append(
                f"| {item['model']} | {_fmt_dual(item['provenance_retention_rate'])} | "
                f"{_fmt_dual(item['effective_provenance_use_rate'])} | "
                f"{_fmt_rate(item['invalid_response_rate'])} | "
                f"{_fmt_rate(item['code_syntax_error_rate'])} | {item['n_responses']} |"
            )
    else:
        lines.append("Code-agent arm: not run.")
    lines.extend(["", "## Continuation criterion", "", "```text", continuation_text, "```", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def analyze_results(
    *,
    config_path: str | Path,
    results_dir: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
    environment_path: Optional[str | Path] = None,
    status_path: Optional[str | Path] = None,
    preloaded_runs: Optional[List[Dict[str, Any]]] = None,
    preloaded_verification: Optional[List[Dict[str, Any]]] = None,
    analysis_metadata: Optional[Dict[str, Any]] = None,
    environment_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    results_dir = Path(results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    cfg = load_config(config_path)
    if cfg.continuation_criteria.rule_version == "rule_v2":
        raise ValueError("use python -m src.analyze_run3 for rule_v2 (one repeat at a time)")
    if preloaded_runs is None:
        runs, verification = _load_verified_runs(results_dir, project_root, config_path)
    else:
        runs = preloaded_runs
        verification = preloaded_verification or []

    competence_raw = [
        CompetenceResponseRecord(
            case_id=record.case_id,
            model=record.model,
            raw_text=_record_text(record),
        )
        for run in runs
        if run["manifest"].arm == "competence"
        for record in run["records"]
    ]
    competence_cases = read_competence_jsonl(project_root / "data/generated/competence_cases.jsonl")
    graded_competence = grade_all_competence(competence_cases, competence_raw)
    competence_rows, qualifications = _competence_rows(graded_competence, cfg)

    pilot_cases = read_jsonl(project_root / "data/generated/pilot_cases.jsonl")
    direct_records: List[ModelResponseRecord] = []
    for variant in ("natural", "guided"):
        direct_records.extend(_main_direct_records(runs, "pilot_cases.jsonl", variant))
    graded_direct = grade_all(pilot_cases, direct_records)
    direct_rows = _direct_metric_rows(graded_direct)
    direct_diagnostics = _direct_diagnostic_rows(graded_direct, pilot_cases)
    main_natural = [row for row in graded_direct if row.prompt_variant is PromptVariant.NATURAL]

    paired_cases_path = project_root / "data/generated/paired_agent_cases.jsonl"
    paired_cases = read_jsonl(paired_cases_path)
    paired_direct_raw = _main_direct_records(runs, paired_cases_path.name, "natural")
    paired_direct_graded = grade_all(paired_cases, paired_direct_raw)
    code_raw = [
        record
        for run in runs
        if run["manifest"].arm == "code-agent"
        for record in run["records"]
    ]
    paired_by_id = {case.case_id: case for case in paired_cases}
    dataset_columns = csv_columns(cfg.code_agent)
    graded_code = [
        grade_code_agent_response(
            paired_by_id[record.case_id],
            CodeAgentRecord(case_id=record.case_id, model=record.model, raw_text=_record_text(record)),
            cfg.code_agent.provenance_columns,
            dataset_columns,
        )
        for record in code_raw
    ]
    code_rows, code_metrics_by_model = _code_rows(graded_code)
    paired_rows = _paired_rows(paired_direct_graded, graded_code)
    paired_direct_code = _paired_direct_code_rows(paired_direct_graded, graded_code)

    decision = evaluate_continuation(
        cfg.continuation_criteria,
        qualifications=qualifications,
        direct_graded=main_natural,
        code_metrics_by_model=code_metrics_by_model,
    )
    continuation_text = format_continuation_decision(decision)

    environment = environment_override
    if environment_path:
        path = Path(environment_path)
        if not path.is_absolute():
            path = project_root / path
        if environment is None and path.exists():
            environment = json.loads(path.read_text(encoding="utf-8"))

    run_status = None
    if status_path:
        path = Path(status_path)
        if not path.is_absolute():
            path = project_root / path
        if path.exists():
            run_status = json.loads(path.read_text(encoding="utf-8"))

    report = {
        "config_path": str(config_path.relative_to(project_root)),
        "config_sha256": sha256_file(config_path),
        "results_dir": str(results_dir),
        "manifest_verification": verification,
        "environment": environment,
        "run_status": run_status,
        "rerun_merge": analysis_metadata,
        "competence": competence_rows,
        "direct_cell_metrics": direct_rows,
        "direct_diagnostics": direct_diagnostics,
        "code_agent": code_rows,
        "paired_metrics": paired_rows,
        "paired_direct_code_comparison": paired_direct_code,
        "continuation": decision.model_dump(mode="json"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "continuation.txt").write_text(continuation_text + "\n", encoding="utf-8")
    (output_dir / "competence.txt").write_text(
        format_qualification_table(qualifications), encoding="utf-8"
    )
    if code_rows:
        text = []
        for model, metrics in sorted(code_metrics_by_model.items()):
            text.extend([model, format_code_agent_metrics(metrics), ""])
        (output_dir / "code_agent_metrics.txt").write_text("\n".join(text), encoding="utf-8")
    _write_analysis_markdown(
        output_dir / "analysis.md",
        verification,
        competence_rows,
        direct_rows,
        direct_diagnostics,
        code_rows,
        paired_direct_code,
        continuation_text,
        run_status,
        analysis_metadata,
    )
    _write_chart(direct_rows, output_dir / "pilot_cell_model_metrics.png")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot_colab_t4.yaml")
    parser.add_argument("--results-dir", default="data/results/colab_t4/results")
    parser.add_argument("--output-dir", default="data/results/colab_t4/analysis")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--environment", default=None)
    parser.add_argument("--status", default=None)
    args = parser.parse_args(argv)
    report = analyze_results(
        config_path=args.config,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
        environment_path=args.environment,
        status_path=args.status,
    )
    print(f"verified manifests: {len(report['manifest_verification'])}")
    print(f"competence models: {len(report['competence'])}")
    print(f"direct cell rows: {len(report['direct_cell_metrics'])}")
    print(f"code-agent models: {len(report['code_agent'])}")
    print(format_continuation_decision(ContinuationDecision(**report["continuation"])))
    print(f"analysis: {Path(args.output_dir) / 'analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


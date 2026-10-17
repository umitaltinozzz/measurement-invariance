"""Replace the diagnosed run-2 rows with verified A100 reruns and re-score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .analyze_results import _load_verified_runs, analyze_results
from .generate_cases import load_config
from .runner import TrialRecord


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_targets(path: str | Path) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"no rerun targets in {path}")
    seen = set()
    for target in targets:
        target_key = (target["model"], target["dataset"], target["arm"], target["variant"])
        if target_key in seen:
            raise ValueError(f"duplicate rerun target: {target_key}")
        seen.add(target_key)
        ids = target.get("case_ids")
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            raise ValueError(f"target {target_key} must have unique case_ids")
    return targets


def _find_run(runs: Sequence[Dict[str, Any]], target: Dict[str, Any]) -> Dict[str, Any]:
    matches = [
        run
        for run in runs
        if run["manifest"].model == target["model"]
        and run["manifest"].arm == target["arm"]
        and run["manifest"].prompt_variant == target["variant"]
        and Path(run["manifest"].dataset_path).as_posix()
        == Path(target["dataset_path"]).as_posix()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one results manifest for {target['model']} / {target['arm']} / "
            f"{target['variant'] or '-'} / {target['dataset_path']}, found {len(matches)}"
        )
    return matches[0]


def _assert_baseline_issue(record: TrialRecord, target: Dict[str, Any]) -> None:
    issue = target["baseline_issue"]
    if issue["kind"] == "provider_404":
        if not (
            record.result is None
            and record.failure_kind == "ProviderError"
            and "HTTP 404" in (record.failure or "")
            and target["model"] in (record.failure or "")
        ):
            raise ValueError(f"baseline row {record.case_id} no longer matches its diagnosed HTTP 404")
        return
    if issue["kind"] == "token_cap":
        result = record.result
        if not (
            result is not None
            and result.finish_reason == "length"
            and result.output_tokens == issue["output_tokens"]
        ):
            raise ValueError(
                f"baseline row {record.case_id} no longer matches its diagnosed "
                f"{issue['output_tokens']}-token cap"
            )
        return
    raise ValueError(f"unknown baseline issue kind: {issue['kind']}")


def _validate_rerun_config(project_root: Path, base_config: Path, rerun_config: Path) -> None:
    base = load_config(base_config)
    rerun = load_config(rerun_config)
    expected = base.model_copy(
        update={
            "runner": base.runner.model_copy(
                update={"max_tokens": 4096, "max_tokens_code_agent": 16000}
            )
        }
    )
    if rerun != expected:
        raise ValueError(
            f"{rerun_config.relative_to(project_root)} must match run-2 config "
            "except max_tokens=4096 and max_tokens_code_agent=16000"
        )


def merge_run2_rows(
    *,
    base_runs: List[Dict[str, Any]],
    rerun_runs: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Validate replacements and return run-2-shaped records plus an audit."""
    if len(rerun_runs) != len(targets):
        raise ValueError(f"expected {len(targets)} rerun manifests, found {len(rerun_runs)}")

    merged_runs = [dict(run) for run in base_runs]
    audit_rows = []
    used_rerun_paths = set()
    for target in targets:
        base_run = _find_run(base_runs, target)
        rerun_run = _find_run(rerun_runs, target)
        used_rerun_paths.add(rerun_run["result_path"])
        if rerun_run["manifest"].config_path != target["config_path"]:
            raise ValueError(
                f"rerun manifest config mismatch for {target['model']} / {target['arm']}: "
                f"{rerun_run['manifest'].config_path} != {target['config_path']}"
            )
        manifest = rerun_run["manifest"]
        original_manifest = base_run["manifest"]
        if manifest.provider != "ollama" or original_manifest.provider != "ollama":
            raise ValueError("run-2 replacements must use the Ollama provider")
        if set(manifest.selected_case_ids or []) != set(target["case_ids"]):
            raise ValueError(f"rerun manifest selected_case_ids mismatch for {target['model']} / {target['arm']}")
        if len(manifest.selected_case_ids or []) != len(target["case_ids"]):
            raise ValueError(f"rerun manifest contains duplicate selected case ids")
        rerun_hardware = manifest.hardware or {}
        hardware_name = str(rerun_hardware.get("gpu_nvidia_smi", ""))
        if "A100" not in hardware_name.upper():
            raise ValueError(f"rerun manifest does not record an A100 GPU: {hardware_name!r}")
        original_constraints = original_manifest.provider_constraints
        rerun_constraints = manifest.provider_constraints
        for setting in ("server_version", "context_length", "max_loaded_models"):
            if not original_constraints.get(setting):
                raise ValueError(f"run-2 manifest does not record Ollama {setting}")
            if rerun_constraints.get(setting) != original_constraints[setting]:
                raise ValueError(
                    f"rerun Ollama {setting} differs from run 2 for {target['model']}: "
                    f"{rerun_constraints.get(setting)!r} != {original_constraints[setting]!r}"
                )
        if (
            original_constraints.get("model_digest")
            and rerun_constraints.get("model_digest") != original_constraints["model_digest"]
        ):
            raise ValueError(f"rerun model digest differs from run 2 for {target['model']}")

        expected_old_cap = 4000 if target["arm"] == "code-agent" else 1024
        expected_new_cap = (
            16000 if target["arm"] == "code-agent"
            else 4096 if target["model"] == "gpt-oss:20b"
            else 1024
        )
        if original_manifest.max_tokens != expected_old_cap:
            raise ValueError(f"run-2 manifest cap differs from the diagnosed cap for {target['model']}")
        if manifest.max_tokens != expected_new_cap:
            raise ValueError(f"rerun manifest cap differs from the planned cap for {target['model']}")

        base_by_id = {record.case_id: record for record in base_run["records"]}
        rerun_by_id = {record.case_id: record for record in rerun_run["records"]}
        expected_ids = set(target["case_ids"])
        if set(rerun_by_id) != expected_ids:
            raise ValueError(
                f"rerun rows for {target['model']} / {target['arm']} differ from target IDs; "
                f"missing={sorted(expected_ids - set(rerun_by_id))}, "
                f"extra={sorted(set(rerun_by_id) - expected_ids)}"
            )
        for case_id in target["case_ids"]:
            original = base_by_id.get(case_id)
            replacement = rerun_by_id[case_id]
            if original is None:
                raise ValueError(f"base run is missing target row {case_id}")
            _assert_baseline_issue(original, target)
            if (
                replacement.key.as_tuple() != original.key.as_tuple()
                or replacement.dataset_sha256 != original.dataset_sha256
                or replacement.prompt_fingerprint != original.prompt_fingerprint
            ):
                raise ValueError(f"rerun row {case_id} does not match the original trial and prompt")

        merged_index = next(
            i
            for i, run in enumerate(merged_runs)
            if run["result_path"] == base_run["result_path"]
        )
        merged_run = dict(merged_runs[merged_index])
        merged_records = [
            rerun_by_id.get(record.case_id, record) for record in base_run["records"]
        ]
        if len(merged_records) != len(base_run["records"]):
            raise ValueError("replacement changed the run-2 row count")
        merged_run["records"] = merged_records
        merged_runs[merged_index] = merged_run

        new_cap = manifest.max_tokens
        old_cap = base_run["manifest"].max_tokens
        audit_rows.append(
            {
                "model": target["model"],
                "dataset": target["dataset"],
                "arm": target["arm"],
                "variant": target["variant"],
                "count": len(target["case_ids"]),
                "case_ids": list(target["case_ids"]),
                "reason": target["reason"],
                "config_path": manifest.config_path,
                "config_sha256": manifest.config_sha256,
                "hardware": hardware_name,
                "model_digest": rerun_constraints["model_digest"],
                "ollama_server_version": rerun_constraints["server_version"],
                "ollama_context_length": rerun_constraints["context_length"],
                "ollama_max_loaded_models": rerun_constraints["max_loaded_models"],
                "run2_max_tokens": old_cap,
                "rerun_max_tokens": new_cap,
                "run2_run_id": base_run["manifest"].run_id,
                "rerun_run_id": manifest.run_id,
            }
        )
    if used_rerun_paths != {run["result_path"] for run in rerun_runs}:
        raise ValueError("rerun result directory contains an unexpected manifest")

    return merged_runs, {
        "method": "replace_exact_case_ids",
        "metric_definitions_changed": False,
        "thresholds_or_gates_changed": False,
        "original_hardware": "Tesla T4 (from run-2 run_environment.json)",
        "rerun_hardware": "; ".join(sorted({row["hardware"] for row in audit_rows})),
        "changed_caps": {
            "direct_max_tokens": {"run2": 1024, "rerun": 4096},
            "code_agent_max_tokens": {"run2": 4000, "rerun": 16000},
        },
        "diagnosis_discrepancy": (
            "The run-2 gpt-oss direct-guided JSONL and saved analysis also contain one invalid, "
            "length-capped row (industrial_sensor-c3-L2-023) absent from DIAGNOSIS.md's listed "
            "19 IDs. It is intentionally left unchanged because the rerun set follows the exact "
            "19 IDs listed in the diagnosis."
        ),
        "replaced_rows": audit_rows,
        "unmodified_run2_rows": "Every run-2 row outside the listed case IDs is retained unchanged.",
    }


def analyze_with_reruns(
    *,
    base_results_dir: str | Path,
    rerun_results_dir: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    base_config_path: str | Path,
    targets_path: str | Path,
    base_environment_path: str | Path,
    rerun_environment_path: str | Path,
    base_status_path: str | Path | None = None,
) -> Dict[str, Any]:
    project_root = Path(project_root).resolve()
    base_config = Path(base_config_path)
    if not base_config.is_absolute():
        base_config = project_root / base_config
    base_results_dir = Path(base_results_dir).resolve()
    rerun_results_dir = Path(rerun_results_dir).resolve()
    base_runs, base_verification = _load_verified_runs(base_results_dir, project_root, base_config)
    rerun_runs, rerun_verification = _load_verified_runs(rerun_results_dir, project_root, None)
    targets = _load_targets(targets_path)

    allowed_rerun_configs = {target["config_path"] for target in targets}
    if any(item["config_path"] not in allowed_rerun_configs for item in rerun_verification):
        raise ValueError("rerun results include a config that is not listed in the target plan")
    rerun_config_path = project_root / "configs/pilot_colab_t4_run2_rerun.yaml"
    _validate_rerun_config(project_root, base_config, rerun_config_path)
    for target in targets:
        target_config = project_root / target["config_path"]
        if target_config != base_config:
            _validate_rerun_config(project_root, base_config, target_config)

    merged_runs, audit = merge_run2_rows(
        base_runs=base_runs,
        rerun_runs=rerun_runs,
        targets=targets,
    )
    verification = base_verification + rerun_verification
    base_environment = _read_json(base_environment_path)
    rerun_environment = _read_json(rerun_environment_path)
    original_gpu = str(base_environment.get("gpu_nvidia_smi", ""))
    if "T4" not in original_gpu.upper():
        raise ValueError(f"run-2 environment does not identify a T4 GPU: {original_gpu!r}")
    for item in audit["replaced_rows"]:
        model = item["model"]
        run2_digest = base_environment.get("model_digests", {}).get(model, {}).get("digest")
        rerun_digest = rerun_environment.get("model_digests", {}).get(model, {}).get("digest")
        if not run2_digest or item["model_digest"] != run2_digest or rerun_digest != run2_digest:
            raise ValueError(f"model digest does not match run 2 for {model}")
    audit["original_hardware"] = original_gpu
    environment = {"run2": base_environment, "rerun": rerun_environment}
    report = analyze_results(
        config_path=base_config,
        results_dir=base_results_dir,
        output_dir=output_dir,
        project_root=project_root,
        status_path=base_status_path,
        preloaded_runs=merged_runs,
        preloaded_verification=verification,
        analysis_metadata=audit,
        environment_override=environment,
    )
    Path(output_dir, "replacement_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--base-results-dir",
        default="data/results/colab_t4/run2/data/results/colab_t4/results",
    )
    parser.add_argument("--rerun-results-dir", required=True)
    parser.add_argument(
        "--output-dir", default="data/results/colab_t4/run2/analysis_with_reruns"
    )
    parser.add_argument("--base-config", default="configs/pilot_colab_t4_run2.yaml")
    parser.add_argument("--targets", default="configs/run2_rerun_targets.json")
    parser.add_argument(
        "--base-environment", default="data/results/colab_t4/run2/run_environment.json"
    )
    parser.add_argument("--rerun-environment", required=True)
    parser.add_argument("--base-status", default="data/results/colab_t4/run2/run_status.json")
    args = parser.parse_args(argv)
    report = analyze_with_reruns(
        base_results_dir=args.base_results_dir,
        rerun_results_dir=args.rerun_results_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
        base_config_path=args.base_config,
        targets_path=args.targets,
        base_environment_path=args.base_environment,
        rerun_environment_path=args.rerun_environment,
        base_status_path=args.base_status,
    )
    audit = report["rerun_merge"]
    for item in audit["replaced_rows"]:
        print(
            f"Replaced {item['count']} rows: {item['model']} / {item['arm']} / "
            f"{item['variant'] or '-'}"
        )
    print(f"Analysis: {Path(args.output_dir) / 'analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

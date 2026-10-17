import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.analyze_results import _load_verified_runs
from src.analyze_with_reruns import (
    _validate_rerun_config,
    analyze_with_reruns,
    merge_run2_rows,
)
from src.generate_cases import load_config
from src.models.base import RawModelResult
from src.run_manifest import build_manifest, sha256_file, write_manifest
from src.runner import TrialRecord, result_path_for


def record(case_id, *, run_id, result=None, failure=None, failure_kind=None):
    return TrialRecord(
        case_id=case_id,
        model="gemma4:12b",
        arm="direct",
        prompt_variant="natural",
        dataset_sha256="d" * 64,
        prompt_fingerprint=f"prompt-{case_id}",
        run_id=run_id,
        result=result,
        failure=failure,
        failure_kind=failure_kind,
    )


def manifest(
    *, run_id, selected_case_ids=None, hardware=None,
    config_path="configs/pilot_colab_t4_run2.yaml", max_tokens=1024,
    server_version="ollama version is 0.40.5",
    server_version_source="GET /api/version after server readiness",
):
    return SimpleNamespace(
        model="gemma4:12b",
        arm="direct",
        prompt_variant="natural",
        dataset_path="data/generated/pilot_cases.jsonl",
        selected_case_ids=selected_case_ids,
        hardware=hardware,
        config_path=config_path,
        config_sha256="a" * 64,
        max_tokens=max_tokens,
        run_id=run_id,
        provider="ollama",
        provider_constraints={
            "model_digest": "same-digest",
            "server_version": server_version,
            **({"server_version_source": server_version_source} if server_version_source else {}),
            "context_length": 8192,
            "max_loaded_models": 1,
        },
    )


def test_merge_replaces_only_target_rows_and_preserves_other_run2_records():
    original_target = record(
        "case-a", run_id="run2", failure="HTTP 404: model 'gemma4:12b' not found",
        failure_kind="ProviderError",
    )
    untouched = record(
        "case-b", run_id="run2",
        result=RawModelResult(
            provider="ollama", model="gemma4:12b", requested_model_id="gemma4:12b",
            resolved_model_id="gemma4:12b", raw_text="old", structured_output_used=True,
        ),
    )
    replacement = record(
        "case-a", run_id="a100",
        result=RawModelResult(
            provider="ollama", model="gemma4:12b", requested_model_id="gemma4:12b",
            resolved_model_id="gemma4:12b", raw_text="new", structured_output_used=True,
        ),
    )
    result_path = Path("results.jsonl")
    base_run = {
        "manifest": manifest(run_id="run2"),
        "records": [original_target, untouched],
        "result_path": result_path,
    }
    rerun_run = {
        "manifest": manifest(
            run_id="a100", selected_case_ids=["case-a"],
            hardware={"gpu_nvidia_smi": "NVIDIA A100 40GB"},
            server_version="rewritten by Colab assistant",
        ),
        "records": [replacement],
        "result_path": Path("rerun.jsonl"),
    }
    target = {
        "model": "gemma4:12b", "dataset": "pilot", "dataset_path": "data/generated/pilot_cases.jsonl",
        "arm": "direct", "variant": "natural", "config_path": "configs/pilot_colab_t4_run2.yaml",
        "reason": "test", "baseline_issue": {"kind": "provider_404"}, "case_ids": ["case-a"],
    }

    with pytest.raises(ValueError, match="server_version differs"):
        merge_run2_rows(base_runs=[base_run], rerun_runs=[rerun_run], targets=[target])
    rerun_run["manifest"].provider_constraints["server_version"] = "ollama version is 0.40.5"
    rerun_run["manifest"].provider_constraints.pop("server_version_source")
    with pytest.raises(ValueError, match="lacks trusted GET /api/version provenance"):
        merge_run2_rows(base_runs=[base_run], rerun_runs=[rerun_run], targets=[target])
    merged, audit = merge_run2_rows(
        base_runs=[base_run],
        rerun_runs=[rerun_run],
        targets=[target],
        accept_colab_rewritten_server_version=True,
    )

    assert len(merged[0]["records"]) == 2
    assert merged[0]["records"][0].result.raw_text == "new"
    assert merged[0]["records"][1] is untouched
    assert audit["replaced_rows"][0]["case_ids"] == ["case-a"]
    assert audit["replaced_rows"][0]["ollama_server_version"] is None
    assert audit["replaced_rows"][0]["ollama_server_version_status"] == "not_captured"
    assert audit["replaced_rows"][0]["ollama_server_version_accepted_by_explicit_flag"] is True
    assert audit["replaced_rows"][0]["ollama_client_version"] == "0.40.5"
    assert audit["replaced_rows"][0]["ollama_server_version_note"] == (
        "server version not captured; client 0.40.5; field rewritten by Colab assistant"
    )
    assert audit["rerun_server_version"]["status"] == "not_captured"
    assert audit["rerun_server_version"]["accepted_by_explicit_flag"] is True
    assert audit["changed_caps"]["direct_max_tokens"] == {"run2": 1024, "rerun": 4096}


def test_merge_rejects_rerun_rows_outside_the_case_id_plan():
    original = record(
        "case-a", run_id="run2", failure="HTTP 404: model 'gemma4:12b' not found",
        failure_kind="ProviderError",
    )
    base_run = {
        "manifest": manifest(run_id="run2"), "records": [original], "result_path": Path("base.jsonl")
    }
    rerun_run = {
        "manifest": manifest(
            run_id="a100", selected_case_ids=["case-a"],
            hardware={"gpu_nvidia_smi": "NVIDIA A100"},
        ),
        "records": [
            record(
                case_id, run_id="a100",
                result=RawModelResult(
                    provider="ollama", model="gemma4:12b", requested_model_id="gemma4:12b",
                    resolved_model_id="gemma4:12b", raw_text="new", structured_output_used=True,
                ),
            )
            for case_id in ("case-a", "extra")
        ],
        "result_path": Path("rerun.jsonl"),
    }
    target = {
        "model": "gemma4:12b", "dataset": "pilot", "dataset_path": "data/generated/pilot_cases.jsonl",
        "arm": "direct", "variant": "natural", "config_path": "configs/pilot_colab_t4_run2.yaml",
        "reason": "test", "baseline_issue": {"kind": "provider_404"}, "case_ids": ["case-a"],
    }
    with pytest.raises(ValueError, match="differ from target IDs"):
        merge_run2_rows(base_runs=[base_run], rerun_runs=[rerun_run], targets=[target])


def test_rerun_config_changes_only_the_two_registered_caps():
    root = Path.cwd()
    _validate_rerun_config(
        root,
        root / "configs/pilot_colab_t4_run2.yaml",
        root / "configs/pilot_colab_t4_run2_rerun.yaml",
    )


def test_rerun_manifest_config_sources_are_sha_verified(tmp_path):
    root = Path.cwd().resolve()
    raw_root = root / "data/results/colab_t4/run2_rerun_a100_raw"
    rerun_results = raw_root / "run2_rerun_a100/results"
    config_sources = {
        "configs/pilot_colab_t4_run2.yaml": raw_root / "configs/pilot_colab_t4_run2.yaml",
        "configs/pilot_colab_t4_run2_rerun.yaml": raw_root / "configs/pilot_colab_t4_run2_rerun.yaml",
    }

    runs, verification = _load_verified_runs(
        rerun_results, root, None, config_sources=config_sources
    )
    assert len(runs) == 5
    assert len(verification) == 5

    tampered = tmp_path / "pilot_colab_t4_run2_rerun.yaml"
    tampered.write_bytes(config_sources["configs/pilot_colab_t4_run2_rerun.yaml"].read_bytes() + b"\n")
    bad_sources = {
        **config_sources,
        "configs/pilot_colab_t4_run2_rerun.yaml": tampered,
    }
    with pytest.raises(ValueError, match="config SHA mismatch"):
        _load_verified_runs(rerun_results, root, None, config_sources=bad_sources)


def test_run2_plus_reruns_analysis_replaces_rows_and_writes_a_separate_report(tmp_path):
    root = Path.cwd().resolve()
    original_analysis = root / "data/results/colab_t4/run2/analysis/analysis.json"
    original_analysis_sha = sha256_file(original_analysis)
    base_results = root / "data/results/colab_t4/run2/data/results/colab_t4/results"
    base_config = root / "configs/pilot_colab_t4_run2.yaml"
    base_runs, _ = _load_verified_runs(
        base_results,
        root,
        None,
        config_sources={
            "configs/pilot_colab_t4_run2.yaml": (
                root / "configs/snapshots/run2/pilot_colab_t4_run2_crlf.yaml"
            )
        },
    )
    base_environment = json.loads(
        (root / "data/results/colab_t4/run2/run_environment.json").read_text(encoding="utf-8")
    )
    targets = json.loads(
        (root / "configs/run2_rerun_targets.json").read_text(encoding="utf-8")
    )["targets"]
    rerun_results = tmp_path / "reruns"

    for index, target in enumerate(targets):
        original_run = next(
            run for run in base_runs
            if run["manifest"].model == target["model"]
            and run["manifest"].arm == target["arm"]
            and run["manifest"].prompt_variant == target["variant"]
            and run["manifest"].dataset_path == target["dataset_path"]
        )
        cfg = load_config(target["config_path"])
        max_tokens = (
            cfg.runner.max_tokens_code_agent
            if target["arm"] == "code-agent"
            else cfg.runner.max_tokens
        )
        run_id = f"a100-test-{index}"
        manifest = build_manifest(
            run_id=run_id,
            arm=target["arm"],
            prompt_variant=target["variant"],
            provider="ollama",
            model=target["model"],
            dataset_path=target["dataset_path"],
            config_path=target["config_path"],
            temperature=cfg.runner.temperature,
            max_tokens=max_tokens,
            seed=cfg.runner.seed,
            structured_output=cfg.runner.structured_output,
            max_transport_retries=cfg.runner.max_transport_retries,
            provider_constraints={
                **original_run["manifest"].provider_constraints,
                "model_digest": base_environment["model_digests"][target["model"]]["digest"],
                "server_version_source": "GET /api/version after server readiness",
            },
            hardware={"gpu_nvidia_smi": "NVIDIA A100 40GB, 40960 MiB, 8.0"},
            selected_case_ids=target["case_ids"],
            repo_root=root,
        )
        output_dir = rerun_results / target["results_subdir"]
        path = result_path_for(
            output_dir,
            target["model"],
            target["arm"],
            original_run["manifest"].dataset_sha256,
        )
        write_manifest(manifest, path)
        originals = {record.case_id: record for record in original_run["records"]}
        if target["arm"] == "code-agent":
            raw_text = '{"code":"print(1)"}'
        elif target["variant"] == "guided":
            raw_text = '{"trend":"stable","measurement_change_detected":false,"reason":"rerun"}'
        else:
            raw_text = '{"trend":"stable","reason":"rerun"}'
        rerun_records = [
            TrialRecord(
                case_id=case_id,
                model=target["model"],
                arm=target["arm"],
                prompt_variant=target["variant"],
                dataset_sha256=originals[case_id].dataset_sha256,
                prompt_fingerprint=originals[case_id].prompt_fingerprint,
                run_id=run_id,
                result=RawModelResult(
                    provider="ollama",
                    model=target["model"],
                    requested_model_id=target["model"],
                    resolved_model_id=target["model"],
                    raw_text=raw_text,
                    structured_output_used=True,
                    output_tokens=20,
                    finish_reason="stop",
                ),
            )
            for case_id in target["case_ids"]
        ]
        path.write_text(
            "".join(record.model_dump_json() + "\n" for record in rerun_records),
            encoding="utf-8",
        )

    rerun_environment = tmp_path / "run_environment.json"
    rerun_environment.write_text(
        json.dumps({
            "gpu_nvidia_smi": "NVIDIA A100 40GB, 40960 MiB, 8.0",
            "model_digests": {
                target["model"]: base_environment["model_digests"][target["model"]]
                for target in targets
            },
        }),
        encoding="utf-8",
    )
    output = tmp_path / "analysis_with_reruns"
    report = analyze_with_reruns(
        base_results_dir=base_results,
        rerun_results_dir=rerun_results,
        output_dir=output,
        project_root=root,
        base_config_path=root / "data/results/colab_t4/run2/configs/pilot_colab_t4.yaml",
        targets_path=root / "configs/run2_rerun_targets.json",
        base_environment_path=root / "data/results/colab_t4/run2/run_environment.json",
        rerun_environment_path=rerun_environment,
        accept_colab_rewritten_server_version=True,
    )

    assert sum(row["count"] for row in report["rerun_merge"]["replaced_rows"]) == 59
    assert report["rerun_merge"]["original_hardware"].startswith("Tesla T4")
    assert "NVIDIA A100" in report["rerun_merge"]["rerun_hardware"]
    assert report["rerun_merge"]["rerun_server_version"]["status"] == "not_captured"
    assert report["rerun_merge"]["rerun_server_version"]["accepted_by_explicit_flag"] is True
    assert report["environment"]["rerun"]["ollama_version"] is None
    assert report["environment"]["rerun"]["ollama_version_note"] == (
        "server version not captured; client 0.40.5; field rewritten by Colab assistant"
    )
    assert (output / "analysis.md").exists()
    assert (output / "replacement_audit.json").exists()
    markdown = (output / "analysis.md").read_text(encoding="utf-8")
    assert "industrial_sensor-c3-L2-023" in markdown
    assert "max_tokens` 1024 to 4096" in markdown
    assert "max_tokens_code_agent` 4000 to 16000" in markdown
    assert "## Abstention and cell-4 diagnostics" in markdown
    assert "## Wilson 95% intervals" in markdown
    assert "## Paired direct-vs-code comparison" in markdown
    direct_diagnostics = {
        (row["model"], row["prompt_variant"]): row
        for row in report["direct_diagnostics"]
    }
    assert direct_diagnostics[("qwen3:14b", "natural")]["uncertain_rate"] == {
        "numerator": 91, "denominator": 100, "value": 0.91
    }
    for model in ("gemma4:12b", "gpt-oss:20b", "mistral-small3.2:24b"):
        split = direct_diagnostics[(model, "natural")]["cell4_raw_direction_split"]
        assert split["raw_direction_misleading"]["n_valid"] == 8
        assert 0 <= split["raw_direction_misleading"]["wrong_direction_count"] <= 8
    paired = {row["model"]: row for row in report["paired_direct_code_comparison"]}
    qwen_cell2 = paired["qwen3:14b"]["cell2"]
    assert qwen_cell2["n_pairs"] == 20
    assert qwen_cell2["direct_uncertain"]["numerator"] == 20
    assert qwen_cell2["code_spurious_trend"]["numerator"] == 20
    assert all(row["all_cells"]["n_pairs"] == 40 for row in paired.values())
    verified_reruns = [item for item in report["manifest_verification"] if item["selected_case_ids"]]
    assert len(verified_reruns) == 5
    assert all(item["rows"] == item["planned"] for item in verified_reruns)
    assert all(item["provider_constraints"]["server_version"] is None for item in verified_reruns)
    assert sha256_file(original_analysis) == original_analysis_sha

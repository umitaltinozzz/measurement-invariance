import json
from pathlib import Path

import pytest

from src.generate_cases import read_jsonl
from src.grader import ModelResponseRecord, ResponseClass, grade_all
from src.prompts import PromptVariant
from src.run_manifest import sha256_file
from src.run_pilot import filter_cases_by_id, read_case_ids


def test_run2_rerun_targets_match_expected_sizes_and_frozen_datasets():
    targets = json.loads(
        open("configs/run2_rerun_targets.json", encoding="utf-8").read()
    )["targets"]
    assert [len(target["case_ids"]) for target in targets] == [28, 19, 5]
    assert [(target["model"], target["arm"], target["variant"]) for target in targets] == [
        ("gemma4:12b", "direct", "natural"),
        ("gpt-oss:20b", "direct", "guided"),
        ("gpt-oss:20b", "code-agent", None),
    ]
    for target in targets:
        cases = read_jsonl(target["dataset_path"])
        selected = filter_cases_by_id(cases, target["case_ids"])
        assert {case.case_id for case in selected} == set(target["case_ids"])
        assert len(selected) == len(target["case_ids"])


def test_case_filter_preserves_dataset_order_and_rejects_unknown_ids():
    pilot_cases = read_jsonl("data/generated/pilot_cases.jsonl")
    selected_ids = [pilot_cases[5].case_id, pilot_cases[1].case_id]
    selected = filter_cases_by_id(pilot_cases, selected_ids)
    assert [case.case_id for case in selected] == [pilot_cases[1].case_id, pilot_cases[5].case_id]
    with pytest.raises(ValueError, match="not present"):
        filter_cases_by_id(pilot_cases, ["not-a-real-case"])


def test_only_case_id_file_rejects_empty_and_duplicate_ids(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_case_ids(path)

    path.write_text("case-a\ncase-a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_case_ids(path)


def test_only_case_id_file_reads_one_id_per_line(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("case-a\n case-b \n\n", encoding="utf-8")
    assert read_case_ids(path) == ["case-a", "case-b"]


def test_rerun_targets_are_exactly_the_diagnosed_failed_or_capped_jsonl_rows():
    targets = json.loads(
        Path("configs/run2_rerun_targets.json").read_text(encoding="utf-8")
    )["targets"]
    base = Path("data/results/colab_t4/run2/data/results/colab_t4/results")
    for target in targets:
        model_slug = target["model"].replace(":", "-")
        matches = list((base / target["results_subdir"]).glob(f"{model_slug}__*.jsonl"))
        assert len(matches) == 1
        rows = [json.loads(line) for line in matches[0].read_text(encoding="utf-8").splitlines() if line]
        issue = target["baseline_issue"]
        if issue["kind"] == "provider_404":
            diagnosed = {
                row["case_id"] for row in rows
                if row["result"] is None
                and row["failure_kind"] == "ProviderError"
                and "HTTP 404" in row["failure"]
            }
            assert set(target["case_ids"]) == diagnosed
        else:
            diagnosed = {
                row["case_id"] for row in rows
                if row["result"] is not None
                and row["result"]["finish_reason"] == "length"
                and row["result"]["output_tokens"] == issue["output_tokens"]
            }
            if target["arm"] == "code-agent":
                assert set(target["case_ids"]) == diagnosed
            else:
                assert set(target["case_ids"]).issubset(diagnosed)
            if target["model"] == "gpt-oss:20b" and target["arm"] == "direct":
                cases = read_jsonl(target["dataset_path"])
                graded = grade_all(
                    cases,
                    [
                        ModelResponseRecord(
                            case_id=row["case_id"], model=row["model"],
                            raw_text=row["result"]["raw_text"],
                            prompt_variant=PromptVariant.GUIDED,
                        )
                        for row in rows
                    ],
                )
                invalid_ids = {
                    row.case_id for row in graded
                    if row.model == target["model"]
                    and row.prompt_variant is PromptVariant.GUIDED
                    and row.response_class is ResponseClass.INVALID_RESPONSE
                }
                assert invalid_ids == set(target["case_ids"]) | {"industrial_sensor-c3-L2-023"}


def test_run2_config_sha_matches_the_environment_record():
    run2_config = Path("configs/pilot_colab_t4_run2.yaml")
    rerun_environment = json.loads(
        Path("data/results/colab_t4/run2/run_environment.json").read_text(encoding="utf-8")
    )
    assert sha256_file(run2_config) == rerun_environment["config_sha256"]

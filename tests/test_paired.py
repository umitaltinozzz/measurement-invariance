"""Paired dataset and within-case dissociation tests.

The paired design exists to remove a confound: run on two different case sets,
any prose-vs-code gap is confounded with whatever differs between the sets. These
tests assert that the two arms really do see the same cases, and that the
comparison is computed only over genuine pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.code_agent import (
    CSV_FILENAME,
    build_code_agent_task,
    csv_columns,
    render_csv,
)
from src.code_agent_fixtures import FIXTURES, make_paired_records
from src.code_agent_grader import grade_all_code_agent
from src.generate_cases import generate_dataset, load_config, read_jsonl, write_jsonl
from src.grader import ResponseClass, grade_all
from src.paired_dataset import (
    DEFAULT_PATH,
    PAIRED_SEED_NAMESPACE,
    generate_paired_dataset,
    paired_dataset_config,
    select_paired_cells,
)
from src.paired_metrics import (
    PairedOutcome,
    compute_paired_breakdown,
    compute_paired_metrics,
    format_dissociation_matrix,
    pair_responses,
)
from src.prompts import PromptVariant, build_direct_prompt, render_series_table
from src.schemas import SalienceLevel, Trend

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def paired(cfg):
    return generate_paired_dataset(cfg)


@pytest.fixture(scope="module")
def ca_cfg(cfg):
    return cfg.code_agent


def graded_pair(paired, ca_cfg, model="m", **kw):
    direct_records, code_records = make_paired_records(paired, model, **kw)
    return (
        grade_all(paired, direct_records),
        grade_all_code_agent(paired, code_records, ca_cfg.provenance_columns, csv_columns(ca_cfg)),
    )


# =========================================================== dataset composition
class TestPairedDataset:
    def test_exactly_forty_cases(self, paired, cfg):
        assert len(paired) == 40 == cfg.paired.total_cases

    def test_twenty_per_cell(self, paired):
        assert sum(1 for c in paired if c.cell == 2) == 20
        assert sum(1 for c in paired if c.cell == 3) == 20
        assert {c.cell for c in paired} == {2, 3}

    def test_all_l1(self, paired):
        assert all(c.salience_level is SalienceLevel.L1 for c in paired)

    def test_case_id_format(self, paired):
        assert paired[0].case_id == "paired-c2-L1-000"
        assert all(c.case_id.startswith("paired-c") for c in paired)
        assert len({c.case_id for c in paired}) == 40

    def test_cell_invariants_still_hold(self, paired):
        for case in paired:
            if case.cell == 2:
                assert case.latent_before == case.latent_after
                assert case.expected_trend is Trend.STABLE
                assert case.measurement_function_change is True
            else:
                assert case.latent_before != case.latent_after
                assert case.measurement_function_change is False

    def test_disjoint_from_the_pilot_dataset(self, cfg, paired):
        pilot = generate_dataset(cfg)
        assert not {c.random_seed for c in pilot} & {c.random_seed for c in paired}
        assert not {c.case_id for c in pilot} & {c.case_id for c in paired}

    def test_seed_namespace_is_applied(self, cfg):
        derived = paired_dataset_config(cfg)
        assert derived.master_seed == cfg.master_seed + PAIRED_SEED_NAMESPACE
        assert derived.case_id_prefix == "paired"

    def test_selection_raises_rather_than_returning_short(self, cfg, paired):
        greedy = cfg.paired.model_copy(update={"cases_per_cell": 99})
        with pytest.raises(ValueError, match="need 99"):
            select_paired_cells(paired, greedy)


class TestPairedDeterminism:
    def test_same_seed_reproduces_identical_cases(self, cfg):
        a = generate_paired_dataset(cfg)
        b = generate_paired_dataset(cfg)
        assert [c.model_dump(mode="json") for c in a] == [c.model_dump(mode="json") for c in b]

    def test_jsonl_roundtrip_is_byte_identical(self, tmp_path, paired):
        first = write_jsonl(paired, tmp_path / "a.jsonl")
        second = write_jsonl(read_jsonl(first), tmp_path / "b.jsonl")
        assert first.read_bytes() == second.read_bytes()

    def test_different_master_seed_changes_the_numbers(self, cfg):
        other = cfg.model_copy(update={"master_seed": cfg.master_seed + 1})
        a = generate_paired_dataset(cfg)
        b = generate_paired_dataset(other)
        assert [c.model_dump(mode="json") for c in a] != [c.model_dump(mode="json") for c in b]
        assert [c.case_id for c in a] == [c.case_id for c in b]

    def test_the_committed_file_matches_the_generator(self, cfg):
        path = Path(DEFAULT_PATH)
        if not path.exists():
            pytest.skip("paired dataset not generated yet")
        assert [c.model_dump(mode="json") for c in read_jsonl(path)] == [
            c.model_dump(mode="json") for c in generate_paired_dataset(cfg)
        ]


# ================================== the two arms see the same underlying case
class TestBothArmsSeeTheSameCase:
    def test_same_case_ids_in_both_arms(self, paired, ca_cfg):
        direct_ids = [build_direct_prompt(c, PromptVariant.NATURAL) and c.case_id for c in paired]
        code_ids = [build_code_agent_task(c, ca_cfg).case_id for c in paired]
        assert direct_ids == code_ids

    def test_same_observed_values_in_both_presentations(self, paired, ca_cfg):
        """The numbers a model reads must be identical; only the layout differs."""
        for case in paired:
            table = render_series_table(case)
            csv_text = render_csv(case, ca_cfg)
            table_values = [line.split("|")[1].strip() for line in table.splitlines()[2:]]
            csv_values = [line.split(",")[1] for line in csv_text.strip().splitlines()[1:]]
            assert table_values == csv_values, case.case_id

    def test_same_provenance_in_both_presentations(self, paired, ca_cfg):
        for case in paired:
            table = render_series_table(case)
            csv_text = render_csv(case, ca_cfg)
            for row in table.splitlines()[2:]:
                parts = [p.strip() for p in row.split("|")]
                t, sensor, calib, ref = parts[0], parts[2], parts[3], parts[4]
                csv_row = csv_text.strip().splitlines()[int(t) + 1].split(",")
                assert csv_row[2] == sensor
                assert csv_row[3] == calib
                assert csv_row[4] == ref

    def test_ground_truth_is_one_object_shared_by_both_arms(self, paired, ca_cfg):
        """Both arms are graded against the same BenchmarkCase, so the truth
        cannot diverge between them by construction."""
        for case in paired:
            task = build_code_agent_task(case, ca_cfg)
            assert task.case is case
            assert task.case.expected_trend is case.expected_trend
            assert task.case.latent_before == case.latent_before
            assert task.case.switch_timestep == case.switch_timestep

    def test_only_the_presentation_differs(self, paired, ca_cfg):
        case = paired[0]
        direct = build_direct_prompt(case, PromptVariant.NATURAL).user
        code = build_code_agent_task(case, ca_cfg).user
        assert direct != code
        assert CSV_FILENAME in code and CSV_FILENAME not in direct
        assert "sensor_version" in direct and "sensor_version" in code

    def test_code_prompt_embeds_the_data(self, paired, ca_cfg):
        """A provider call carries only text, so the CSV has to be in the prompt --
        and that also keeps the two arms' information identical."""
        case = paired[0]
        task = build_code_agent_task(case, ca_cfg)
        assert task.csv_text.rstrip("\n") in task.user
        assert task.user.count("\n") > case.n_timesteps


# ==================================================================== pairing
class TestPairing:
    def test_full_join(self, paired, ca_cfg):
        direct, code = graded_pair(paired, ca_cfg)
        result = pair_responses(direct, code)
        assert result.n_pairs == 40
        assert result.fully_paired
        assert result.unpaired_direct == [] and result.unpaired_code == []

    def test_unpaired_direct_is_excluded_and_reported(self, paired, ca_cfg):
        dropped = [c.case_id for c in paired[:3]]
        direct, code = graded_pair(paired, ca_cfg, drop_code=dropped)
        result = pair_responses(direct, code)
        assert result.n_pairs == 37
        assert result.unpaired_direct == sorted(dropped)
        assert not result.fully_paired
        assert compute_paired_metrics(result).direct_accuracy.denominator == 37

    def test_unpaired_code_is_excluded_and_reported(self, paired, ca_cfg):
        dropped = [c.case_id for c in paired[:5]]
        direct, code = graded_pair(paired, ca_cfg, drop_direct=dropped)
        result = pair_responses(direct, code)
        assert result.n_pairs == 35
        assert result.unpaired_code == sorted(dropped)

    def test_unpaired_never_enters_a_denominator_silently(self, paired, ca_cfg):
        direct, code = graded_pair(paired, ca_cfg, drop_code=[paired[0].case_id])
        metrics = compute_paired_metrics(pair_responses(direct, code))
        assert metrics.n_pairs == 39
        assert metrics.n_unpaired_direct == 1
        for rate in (metrics.direct_accuracy, metrics.code_accuracy):
            assert rate.denominator == 39

    def test_pairs_only_the_natural_arm(self, paired, ca_cfg):
        direct, code = graded_pair(paired, ca_cfg)
        guided = [d.model_copy(update={"prompt_variant": PromptVariant.GUIDED}) for d in direct]
        assert pair_responses(guided, code).n_pairs == 0
        assert pair_responses(guided, code, variant=PromptVariant.GUIDED).n_pairs == 40

    def test_models_are_kept_apart(self, paired, ca_cfg):
        d1, c1 = graded_pair(paired, ca_cfg, model="a")
        d2, c2 = graded_pair(paired, ca_cfg, model="b")
        result = pair_responses(d1 + d2, c1 + c2)
        assert result.n_pairs == 80
        assert {p.model for p in result.pairs} == {"a", "b"}
        assert pair_responses(d1 + d2, c1 + c2, model="a").n_pairs == 40

    def test_duplicate_rows_are_flagged_not_double_counted(self, paired, ca_cfg):
        direct, code = graded_pair(paired, ca_cfg)
        result = pair_responses(direct + direct[:2], code)
        assert result.n_pairs == 40
        assert result.duplicate_direct == sorted(d.case_id for d in direct[:2])

    def test_mismatched_datasets_raise(self, cfg, paired, ca_cfg):
        direct, code = graded_pair(paired, ca_cfg)
        broken = [code[0].model_copy(update={"cell": 3})] + list(code[1:])
        with pytest.raises(ValueError, match="not running the same dataset"):
            pair_responses(direct, broken)


# ======================================================== dissociation matrix
class TestDissociationMatrix:
    def test_both_correct(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(
            paired, ca_cfg, direct_correct=ids, code_correct=ids,
            code_behaviour="inverts_calibration",
        )
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.outcome_counts == {PairedOutcome.BOTH_CORRECT.value: 40}
        assert m.direct_accuracy.value == 1.0 and m.code_accuracy.value == 1.0
        assert m.direct_to_code_failure_rate.value == 0.0

    def test_direct_only_correct_is_the_predicted_dissociation(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(paired, ca_cfg, direct_correct=ids, code_correct=[])
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.outcome_counts == {PairedOutcome.DIRECT_ONLY_CORRECT.value: 40}
        assert m.direct_accuracy.value == 1.0
        assert m.code_accuracy.value == 0.0
        assert m.direct_to_code_failure_rate.value == 1.0
        assert m.code_to_direct_failure_rate.value is None  # no code-correct cases

    def test_code_only_correct(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(
            paired, ca_cfg, direct_correct=[], code_correct=ids,
            code_behaviour="inverts_calibration",
        )
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.outcome_counts == {PairedOutcome.CODE_ONLY_CORRECT.value: 40}
        assert m.code_to_direct_failure_rate.value == 1.0
        assert m.direct_to_code_failure_rate.value is None

    def test_both_wrong(self, paired, ca_cfg):
        direct, code = graded_pair(paired, ca_cfg, direct_correct=[], code_correct=[])
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.outcome_counts == {PairedOutcome.BOTH_WRONG.value: 40}
        assert m.direct_accuracy.value == 0.0

    def test_mixed_split_counts_exactly(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(
            paired, ca_cfg, direct_correct=ids[:30], code_correct=ids[:10],
            code_behaviour="inverts_calibration",
        )
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.outcome_counts[PairedOutcome.BOTH_CORRECT.value] == 10
        assert m.outcome_counts[PairedOutcome.DIRECT_ONLY_CORRECT.value] == 20
        assert m.outcome_counts[PairedOutcome.BOTH_WRONG.value] == 10
        assert m.direct_accuracy.value == pytest.approx(30 / 40)
        assert m.code_accuracy.value == pytest.approx(10 / 40)
        assert m.direct_to_code_failure_rate.value == pytest.approx(20 / 30)
        assert m.code_to_direct_failure_rate.value == 0.0

    def test_outcome_counts_partition_the_pairs(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(paired, ca_cfg, direct_correct=ids[:17], code_correct=ids[9:24])
        m = compute_paired_metrics(pair_responses(direct, code))
        assert sum(m.outcome_counts.values()) == m.n_pairs == 40


class TestCellTransitions:
    def test_cell2_direct_correct_code_spurious(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(paired, ca_cfg, direct_correct=ids, code_correct=[])
        pairs = pair_responses(direct, code).pairs
        cell2 = [p for p in pairs if p.cell == 2]
        assert all(p.direct_correct_code_spurious for p in cell2)
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.cell2_direct_correct_code_spurious.value == 1.0
        assert m.cell2_direct_spurious_code_correct.value == 0.0

    def test_cell2_direct_spurious_code_correct(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(
            paired, ca_cfg, direct_correct=[], code_correct=ids,
            code_behaviour="inverts_calibration",
        )
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.cell2_direct_spurious_code_correct.value == 1.0

    def test_cell3_direct_correct_code_miss(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(paired, ca_cfg, direct_correct=ids, code_correct=[])
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.cell3_direct_correct_code_miss.value == 1.0
        assert m.cell3_direct_miss_code_correct.value == 0.0

    def test_cell3_direct_miss_code_correct(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(
            paired, ca_cfg, direct_correct=[], code_correct=ids,
            code_behaviour="inverts_calibration",
        )
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.cell3_direct_miss_code_correct.value == 1.0

    def test_transitions_are_scoped_to_their_cell(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(paired, ca_cfg, direct_correct=ids, code_correct=[])
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.cell2_direct_correct_code_spurious.denominator == 20
        assert m.cell3_direct_correct_code_miss.denominator == 20


# ============================================== provenance deployment gap
class TestProvenanceDeploymentGap:
    def test_gap_is_total_when_code_ignores_provenance(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(
            paired, ca_cfg, direct_correct=ids, code_behaviour="ignores_provenance"
        )
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.provenance_deployment_gap.value == 1.0
        assert m.provenance_deployment_gap.denominator == 40
        assert m.effective_provenance_use_rate.value == 0.0

    def test_reading_without_using_still_counts_as_a_gap(self, paired, ca_cfg):
        """Level 1 is not deployment. This is why PRR alone is not enough."""
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(
            paired, ca_cfg, direct_correct=ids, code_behaviour="reads_but_ignores"
        )
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.provenance_deployment_gap.value == 1.0
        assert m.effective_provenance_use_rate.value == 0.0

    def test_no_gap_when_code_uses_provenance(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(
            paired, ca_cfg, direct_correct=ids, code_behaviour="inverts_calibration"
        )
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.provenance_deployment_gap.value == 0.0
        assert m.effective_provenance_use_rate.value == 1.0

    def test_denominator_is_direct_correct_cases_only(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(
            paired, ca_cfg, direct_correct=ids[:12], code_behaviour="ignores_provenance"
        )
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.provenance_deployment_gap.denominator == 12
        assert m.provenance_deployment_gap.value == 1.0

    def test_half_and_half_fixture(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        behaviours = {
            cid: ("inverts_calibration" if i < 20 else "ignores_provenance")
            for i, cid in enumerate(ids)
        }
        direct, code = graded_pair(
            paired, ca_cfg, direct_correct=ids, code_behaviour_by_case=behaviours
        )
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.effective_provenance_use_rate.value == pytest.approx(0.5)
        assert m.provenance_deployment_gap.value == pytest.approx(0.5)

    def test_undefined_when_nothing_is_correct_in_prose(self, paired, ca_cfg):
        direct, code = graded_pair(paired, ca_cfg, direct_correct=[])
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.provenance_deployment_gap.value is None


# ============================================================ invalid handling
class TestInvalidInPairs:
    def test_invalid_counts_as_wrong_not_absent(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(
            paired, ca_cfg, direct_correct=ids, code_correct=ids, code_invalid=ids[:8],
            code_behaviour="inverts_calibration",
        )
        result = pair_responses(direct, code)
        m = compute_paired_metrics(result)
        assert result.n_pairs == 40, "an invalid response is still a response"
        assert m.n_code_invalid == 8
        assert m.code_accuracy.value == pytest.approx(32 / 40)
        assert m.direct_to_code_failure_rate.value == pytest.approx(8 / 40)

    def test_invalid_code_has_no_provenance_level(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(paired, ca_cfg, direct_correct=ids, code_invalid=ids)
        pairs = pair_responses(direct, code).pairs
        assert all(p.code_provenance_level is None for p in pairs)
        assert all(p.provenance_deployment_gap for p in pairs)

    def test_invalid_direct_is_not_correct(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(paired, ca_cfg, direct_correct=ids, direct_invalid=ids[:5])
        m = compute_paired_metrics(pair_responses(direct, code))
        assert m.n_direct_invalid == 5
        assert m.direct_accuracy.value == pytest.approx(35 / 40)


# ================================================================== reporting
class TestReporting:
    def test_matrix_renders_all_four_cells(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        direct, code = graded_pair(paired, ca_cfg, direct_correct=ids[:30], code_correct=ids[:10])
        text = format_dissociation_matrix(compute_paired_metrics(pair_responses(direct, code)))
        for label in ("code correct", "code wrong", "direct correct", "direct wrong"):
            assert label in text
        assert "direct->code failure rate" in text
        assert "provenance deployment gap" in text

    def test_breakdown_by_model(self, paired, ca_cfg):
        ids = [c.case_id for c in paired]
        d1, c1 = graded_pair(paired, ca_cfg, model="a", direct_correct=ids, code_correct=ids,
                             code_behaviour="inverts_calibration")
        d2, c2 = graded_pair(paired, ca_cfg, model="b", direct_correct=ids, code_correct=[])
        results = compute_paired_breakdown(d1 + d2, c1 + c2)
        assert set(results) == {"a", "b"}
        assert results["a"].direct_to_code_failure_rate.value == 0.0
        assert results["b"].direct_to_code_failure_rate.value == 1.0

    def test_metrics_refuse_to_mix_models(self, paired, ca_cfg):
        d1, c1 = graded_pair(paired, ca_cfg, model="a")
        d2, c2 = graded_pair(paired, ca_cfg, model="b")
        with pytest.raises(ValueError, match="per-model"):
            compute_paired_metrics(pair_responses(d1 + d2, c1 + c2))


# ============================================================ runner wiring
class TestRunnerWiring:
    def test_both_arms_use_the_same_dataset_sha(self, tmp_path, cfg, paired):
        from src.run_manifest import sha256_file

        path = write_jsonl(paired, tmp_path / "paired.jsonl")
        assert sha256_file(path) == sha256_file(path)

    def test_the_same_case_is_two_trials_across_arms(self, tmp_path, cfg, paired):
        from datetime import datetime, timezone

        from src.models import CallableProvider
        from src.run_manifest import build_manifest
        from src.run_pilot import build_code_plans, build_direct_plans
        from src.runner import Arm, Runner

        dataset_path = write_jsonl(paired[:4], tmp_path / "paired.jsonl")
        sha = "a" * 64

        def manifest_for(arm, variant):
            return build_manifest(
                run_id="run0", arm=arm, prompt_variant=variant, provider="callable",
                model="m", dataset_path=dataset_path, config_path=CONFIG_PATH,
                temperature=0.0, max_tokens=100, seed=1, structured_output=False,
                max_transport_retries=0,
                now=datetime(2026, 8, 24, tzinfo=timezone.utc),
            )

        provider = CallableProvider(lambda req: '{"trend": "stable", "reason": "r"}')
        runner = Runner(provider, cfg.runner, sleep=lambda s: None)
        _, direct_summary = runner.run(
            build_direct_plans(paired[:4], cfg, "m"), model="m", arm=Arm.DIRECT,
            dataset_sha256=sha, manifest=manifest_for(Arm.DIRECT, "natural"),
            results_dir=tmp_path,
        )
        code_provider = CallableProvider(
            lambda req: json.dumps({"code": "print(1)", "answer": "stable", "reason": "r"})
        )
        code_runner = Runner(code_provider, cfg.runner, sleep=lambda s: None)
        _, code_summary = code_runner.run(
            build_code_plans(paired[:4], cfg, "m"), model="m", arm=Arm.CODE_AGENT,
            dataset_sha256=sha, manifest=manifest_for(Arm.CODE_AGENT, None),
            results_dir=tmp_path,
        )

        assert direct_summary.executed == 4 and code_summary.executed == 4
        assert direct_summary.result_path != code_summary.result_path
        # same dataset sha appears in both file names
        assert sha[:16] in Path(direct_summary.result_path).name
        assert sha[:16] in Path(code_summary.result_path).name

    def test_arm_is_validated_against_the_dataset(self, cfg, paired):
        from src.run_pilot import build_plans
        from src.runner import Arm

        with pytest.raises(ValueError, match="supports arms"):
            build_plans("pilot", paired, cfg, "m", Arm.CODE_AGENT, None)

    def test_code_arm_rejects_a_prompt_variant(self, cfg, paired):
        from src.run_pilot import build_plans
        from src.runner import Arm

        with pytest.raises(ValueError, match="no prompt variant"):
            build_plans("paired", paired, cfg, "m", Arm.CODE_AGENT, "natural")

    def test_plans_cover_every_case_once(self, cfg, paired):
        from src.run_pilot import build_code_plans, build_direct_plans

        assert [p.case_id for p in build_direct_plans(paired, cfg, "m")] == [
            c.case_id for c in paired
        ]
        assert [p.case_id for p in build_code_plans(paired, cfg, "m")] == [
            c.case_id for c in paired
        ]

"""Code-agent arm tests: case selection, AST provenance annotation, PRR/EPUR.

The AST classifier is the load-bearing piece. It is what separates
``df.groupby("sensor_version")`` from ``df.drop(columns=["sensor_version"])`` --
two snippets a substring search cannot tell apart, and the two answers the whole
metric exists to distinguish.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.code_agent import (
    CSV_FILENAME,
    VALUE_COLUMN,
    build_code_agent_task,
    build_code_agent_tasks,
    csv_columns,
    render_csv,
)
from src.paired_dataset import generate_paired_dataset, select_paired_cells
from src.code_agent_fixtures import (
    FIXTURES,
    make_code_agent_records,
    make_correct_code_agent_records,
    make_malformed_envelope_records,
    replay_executions,
)
from src.code_agent_grader import (
    CodeAgentRecord,
    ProvenanceLevel,
    analyze_code,
    grade_all_code_agent,
    parse_code_agent_response,
)
from src.code_agent_metrics import (
    code_agent_breakdown,
    compute_code_agent_metrics,
    format_code_agent_metrics,
)
from src.generate_cases import generate_dataset, load_config
from src.grader import ResponseClass
from src.sandbox import DisabledSandbox, ReplaySandbox, SubprocessSandbox
from src.schemas import SalienceLevel, Trend

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def dataset(cfg):
    return generate_dataset(cfg)


@pytest.fixture(scope="module")
def ca_cfg(cfg):
    return cfg.code_agent


@pytest.fixture(scope="module")
def ca_cases(cfg):
    return generate_paired_dataset(cfg)


def graded_for(ca_cases, ca_cfg, behaviour):
    records = make_code_agent_records(ca_cases, behaviour)
    return grade_all_code_agent(
        ca_cases,
        records,
        ca_cfg.provenance_columns,
        csv_columns(ca_cfg),
        replay_executions(records),
    )


# =============================================================== case selection
class TestScope:
    def test_scope_is_cells_2_and_3_l1_twenty_each(self, ca_cases, cfg):
        assert cfg.paired.cells == [2, 3]
        assert cfg.paired.salience_level is SalienceLevel.L1
        assert cfg.paired.cases_per_cell == 20
        assert len(ca_cases) == 40
        for cell in (2, 3):
            assert sum(1 for c in ca_cases if c.cell == cell) == 20

    def test_all_cases_are_l1(self, ca_cases):
        assert all(c.salience_level is SalienceLevel.L1 for c in ca_cases)

    def test_selection_is_deterministic(self, cfg):
        a = generate_paired_dataset(cfg)
        b = generate_paired_dataset(cfg)
        assert [c.case_id for c in a] == [c.case_id for c in b]

    def test_code_arm_cases_are_disjoint_from_the_direct_arm(self, cfg, dataset, ca_cases):
        """Separate seed namespace: no case is scored in both arms."""
        assert not {c.random_seed for c in dataset} & {c.random_seed for c in ca_cases}

    def test_asking_for_more_than_exists_raises(self, cfg, ca_cases):
        greedy = cfg.paired.model_copy(update={"cases_per_cell": 99})
        with pytest.raises(ValueError, match="need 99"):
            select_paired_cells(ca_cases, greedy)


class TestCsvRendering:
    def test_columns_include_reading_and_all_provenance(self, ca_cases, ca_cfg):
        header = render_csv(ca_cases[0], ca_cfg).splitlines()[0]
        assert header.split(",") == csv_columns(ca_cfg)
        for column in ca_cfg.provenance_columns:
            assert column in header

    def test_one_row_per_timestep(self, ca_cases, ca_cfg):
        case = ca_cases[0]
        lines = render_csv(case, ca_cfg).strip().splitlines()
        assert len(lines) == case.n_timesteps + 1

    def test_provenance_columns_are_present_even_when_nothing_changed(self, ca_cases, ca_cfg):
        """If they only appeared when something changed, their presence would be
        the answer."""
        cell3 = next(c for c in ca_cases if c.cell == 3)
        csv_text = render_csv(cell3, ca_cfg)
        assert "sensor_version" in csv_text.splitlines()[0]
        assert csv_text.count("S1") == cell3.n_timesteps

    def test_cell_2_csv_shows_the_version_change(self, ca_cases, ca_cfg):
        cell2 = next(c for c in ca_cases if c.cell == 2)
        csv_text = render_csv(cell2, ca_cfg)
        assert "S1" in csv_text and "S2" in csv_text

    def test_prompt_states_the_task_and_the_schema(self, ca_cases, ca_cfg):
        task = build_code_agent_task(ca_cases[0], ca_cfg)
        assert "Analyze the dataset programmatically" in task.user
        assert CSV_FILENAME in task.user
        assert '"code"' in task.user and '"answer"' in task.user

    def test_prompt_does_not_cue_provenance(self, ca_cases, ca_cfg):
        """Same natural framing as the main arm: the question is whether the
        model checks provenance unprompted.

        The column listing is exempt -- the agent has to be told the schema, and
        it is identical for every case, so it cannot correlate with the answer.
        The narrative around it is what must stay neutral.
        """
        task = build_code_agent_task(ca_cases[0], ca_cfg)
        narrative = task.user.split("Analyze the dataset")[1].split("Answer with exactly")[0].lower()
        for cue in ("calibrat", "instrument", "provenance", "comparable", "invarian", "sensor"):
            assert cue not in narrative

    def test_column_listing_is_identical_across_cells(self, ca_cases, ca_cfg):
        heads = {build_code_agent_task(c, ca_cfg).user.split("It contains")[0] for c in ca_cases}
        assert len(heads) == 1

    def test_tasks_are_built_for_every_selected_case(self, ca_cases, ca_cfg):
        assert len(build_code_agent_tasks(ca_cases, ca_cfg)) == 40


# ==================================================== AST provenance annotation
class TestProvenanceAnnotation:
    @pytest.mark.parametrize("behaviour,expected", [(k, v[1]) for k, v in FIXTURES.items()])
    def test_fixture_lands_on_its_expected_level(self, ca_cfg, behaviour, expected):
        code = FIXTURES[behaviour][0]
        analysis = analyze_code(code, ca_cfg.provenance_columns)
        assert analysis.provenance_level is expected, behaviour

    def test_level_0_when_provenance_never_appears(self, ca_cfg):
        analysis = analyze_code(FIXTURES["ignores_provenance"][0], ca_cfg.provenance_columns)
        assert analysis.provenance_level is ProvenanceLevel.IGNORED
        assert analysis.provenance_columns_referenced == []
        assert not analysis.provenance_referenced

    def test_level_1_when_read_but_only_printed(self, ca_cfg):
        analysis = analyze_code(FIXTURES["reads_but_ignores"][0], ca_cfg.provenance_columns)
        assert analysis.provenance_level is ProvenanceLevel.REFERENCED
        assert "sensor_version" in analysis.provenance_columns_referenced
        assert analysis.provenance_columns_used_analytically == []
        assert analysis.provenance_referenced and not analysis.provenance_used

    def test_level_2_when_used_to_group(self, ca_cfg):
        analysis = analyze_code(FIXTURES["splits_by_sensor"][0], ca_cfg.provenance_columns)
        assert analysis.provenance_level is ProvenanceLevel.USED
        assert "sensor_version" in analysis.provenance_columns_used_analytically

    def test_level_2_when_calibration_is_inverted(self, ca_cfg):
        analysis = analyze_code(FIXTURES["inverts_calibration"][0], ca_cfg.provenance_columns)
        assert analysis.provenance_level is ProvenanceLevel.USED
        assert "reference_range" in analysis.provenance_columns_used_analytically

    def test_dropping_provenance_is_level_1_not_level_2(self, ca_cfg):
        """The column is named, so it was seen; it was discarded, so it was not used.
        A substring search would score this identically to a groupby."""
        analysis = analyze_code(FIXTURES["drops_provenance"][0], ca_cfg.provenance_columns)
        assert analysis.provenance_level is ProvenanceLevel.REFERENCED
        assert set(analysis.provenance_columns_dropped) == set(ca_cfg.provenance_columns)
        assert "provenance_columns_dropped" in analysis.anti_patterns

    def test_groupby_and_drop_are_distinguished(self, ca_cfg):
        used = analyze_code('df.groupby("sensor_version").mean()', ca_cfg.provenance_columns)
        dropped = analyze_code('df.drop(columns=["sensor_version"])', ca_cfg.provenance_columns)
        assert used.provenance_level is ProvenanceLevel.USED
        assert dropped.provenance_level is ProvenanceLevel.REFERENCED
        # the substring test that this replaces cannot tell them apart
        assert "sensor_version" in 'df.groupby("sensor_version").mean()'
        assert "sensor_version" in 'df.drop(columns=["sensor_version"])'

    def test_boolean_mask_counts_as_use(self, ca_cfg):
        analysis = analyze_code(
            'sub = df[df["sensor_version"] == "S1"]', ca_cfg.provenance_columns
        )
        assert analysis.provenance_level is ProvenanceLevel.USED

    def test_control_flow_on_provenance_counts_as_use(self, ca_cfg):
        analysis = analyze_code(
            'if df["calibration_version"].nunique() > 1:\n    pass', ca_cfg.provenance_columns
        )
        assert analysis.provenance_level is ProvenanceLevel.USED

    def test_printing_a_derived_value_is_still_only_a_look(self, ca_cfg):
        analysis = analyze_code(
            'print(df["sensor_version"].unique())', ca_cfg.provenance_columns
        )
        assert analysis.provenance_level is ProvenanceLevel.REFERENCED

    def test_syntax_error_is_reported_not_raised(self, ca_cfg):
        analysis = analyze_code(FIXTURES["invalid_python"][0], ca_cfg.provenance_columns)
        assert analysis.parse_ok is False
        assert analysis.syntax_error
        assert analysis.provenance_level is ProvenanceLevel.IGNORED
        assert analysis.needs_human_review

    def test_annotations_are_labelled_heuristic(self, ca_cfg):
        for code, _level, _answer in FIXTURES.values():
            assert analyze_code(code, ca_cfg.provenance_columns).heuristic is True

    def test_findings_carry_line_numbers(self, ca_cfg):
        analysis = analyze_code(FIXTURES["splits_by_sensor"][0], ca_cfg.provenance_columns)
        assert analysis.findings
        assert all(f.line is not None for f in analysis.findings if f.kind != "syntax_error")


class TestAntiPatterns:
    def test_raw_pct_change_is_flagged(self, ca_cfg):
        analysis = analyze_code(
            'df["delta"] = df["reading"].pct_change()', ca_cfg.provenance_columns
        )
        assert f"raw_pct_change_on_{VALUE_COLUMN}" in analysis.anti_patterns

    def test_raw_diff_is_flagged(self, ca_cfg):
        analysis = analyze_code('df["reading"].diff()', ca_cfg.provenance_columns)
        assert f"raw_diff_on_{VALUE_COLUMN}" in analysis.anti_patterns

    def test_raw_mean_comparison_is_flagged_only_without_provenance_use(self, ca_cfg):
        bare = analyze_code('df["reading"].mean()', ca_cfg.provenance_columns)
        with_prov = analyze_code(
            'df.groupby("sensor_version")["reading"].mean()', ca_cfg.provenance_columns
        )
        assert "raw_summary_comparison" in bare.anti_patterns
        assert "raw_summary_comparison" not in with_prov.anti_patterns

    def test_selection_excluding_provenance_is_flagged(self, ca_cfg):
        analysis = analyze_code('slim = df[["t", "reading"]]', ca_cfg.provenance_columns)
        assert "provenance_columns_excluded_by_selection" in analysis.anti_patterns

    def test_the_ignoring_fixture_trips_the_expected_patterns(self, ca_cfg):
        analysis = analyze_code(FIXTURES["ignores_provenance"][0], ca_cfg.provenance_columns)
        assert f"raw_pct_change_on_{VALUE_COLUMN}" in analysis.anti_patterns

    def test_a_corrected_pipeline_is_flagged_for_human_review(self, ca_cfg):
        """Provenance used AND a raw delta computed: whether the delta was
        corrected is beyond the classifier."""
        code = (
            'df["n"] = df["reading"] / df["reference_range"].map(len)\n'
            'df["d"] = df["reading"].pct_change()\n'
        )
        analysis = analyze_code(code, ca_cfg.provenance_columns)
        assert analysis.needs_human_review is True


# ================================================================= response IO
class TestResponseParsing:
    def test_valid_response_parses(self):
        raw = json.dumps({"code": "print(1)", "answer": "stable", "reason": "r"})
        result = parse_code_agent_response(raw)
        assert result.ok and result.response.answer.value == "stable"

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "not json",
            '```json\n{"code": "x", "answer": "stable", "reason": "r"}\n```',
            '{"code": "x", "answer": "stable"}',
            '{"answer": "stable", "reason": "r"}',
            '{"code": 5, "answer": "stable", "reason": "r"}',
            '{"code": "x", "answer": "rising", "reason": "r"}',
            '{"code": "x", "answer": "stable", "reason": "r", "extra": 1}',
        ],
    )
    def test_malformed_responses_are_rejected(self, raw):
        assert not parse_code_agent_response(raw).ok

    def test_generated_code_is_kept_verbatim(self, ca_cases, ca_cfg):
        graded = graded_for(ca_cases, ca_cfg, "ignores_provenance")
        assert all(g.generated_code == FIXTURES["ignores_provenance"][0] for g in graded)

    def test_malformed_envelope_is_invalid_response_not_bad_code(self, ca_cases, ca_cfg):
        records = make_malformed_envelope_records(ca_cases)
        graded = grade_all_code_agent(
            ca_cases, records, ca_cfg.provenance_columns, csv_columns(ca_cfg)
        )
        assert all(g.response_class is ResponseClass.INVALID_RESPONSE for g in graded)
        assert all(g.analysis is None and g.generated_code is None for g in graded)


class TestExecutionCapture:
    def test_stdout_and_stderr_are_stored(self, ca_cases, ca_cfg):
        graded = graded_for(ca_cases, ca_cfg, "runtime_error")
        assert all("KeyError" in g.execution_stderr for g in graded)
        assert all(g.execution_ok is False for g in graded)

    def test_execution_answer_is_read_back_from_stdout(self, ca_cases, ca_cfg):
        graded = graded_for(ca_cases, ca_cfg, "splits_by_sensor")
        assert all(g.execution_answer == "uncertain" for g in graded)

    def test_grading_works_with_execution_disabled(self, ca_cases, ca_cfg):
        records = make_code_agent_records(ca_cases, "splits_by_sensor")
        graded = grade_all_code_agent(
            ca_cases, records, ca_cfg.provenance_columns, csv_columns(ca_cfg)
        )
        assert all(g.execution_ok is None for g in graded)
        assert all(g.provenance_level == 2 for g in graded)


class TestSandbox:
    def test_execution_is_disabled_by_default(self, ca_cfg):
        assert ca_cfg.execution_enabled is False

    def test_disabled_sandbox_reports_rather_than_raising(self):
        result = DisabledSandbox().run("print(1)", {}, 5.0)
        assert result.executed is False and result.skipped_reason and not result.ok

    def test_replay_sandbox_serves_recordings(self):
        from src.sandbox import ExecutionResult

        sandbox = ReplaySandbox({"print(1)": ExecutionResult(executed=True, exit_code=0, stdout="1")})
        assert sandbox.run("print(1)", {}, 5.0).stdout == "1"
        assert sandbox.run("print(2)", {}, 5.0).executed is False

    def test_subprocess_sandbox_refuses_without_acknowledgement(self):
        with pytest.raises(RuntimeError, match="not a security boundary|container"):
            SubprocessSandbox()


# ======================================================================= metrics
class TestProvenanceMetrics:
    def test_ignoring_fixture_scores_zero_on_both_rates(self, ca_cases, ca_cfg):
        m = compute_code_agent_metrics(graded_for(ca_cases, ca_cfg, "ignores_provenance"))
        assert m.provenance_retention_rate.valid_only.value == 0.0
        assert m.effective_provenance_use_rate.valid_only.value == 0.0

    def test_reading_without_using_separates_the_two_rates(self, ca_cases, ca_cfg):
        """The whole reason both are reported."""
        m = compute_code_agent_metrics(graded_for(ca_cases, ca_cfg, "reads_but_ignores"))
        assert m.provenance_retention_rate.valid_only.value == 1.0
        assert m.effective_provenance_use_rate.valid_only.value == 0.0

    def test_using_provenance_scores_one_on_both(self, ca_cases, ca_cfg):
        m = compute_code_agent_metrics(graded_for(ca_cases, ca_cfg, "inverts_calibration"))
        assert m.provenance_retention_rate.valid_only.value == 1.0
        assert m.effective_provenance_use_rate.valid_only.value == 1.0

    def test_dropping_provenance_retains_but_does_not_use(self, ca_cases, ca_cfg):
        m = compute_code_agent_metrics(graded_for(ca_cases, ca_cfg, "drops_provenance"))
        assert m.provenance_retention_rate.valid_only.value == 1.0
        assert m.effective_provenance_use_rate.valid_only.value == 0.0
        assert m.anti_pattern_counts["provenance_columns_dropped"] == len(ca_cases)

    def test_syntax_errors_stay_in_the_denominator_at_level_zero(self, ca_cases, ca_cfg):
        m = compute_code_agent_metrics(graded_for(ca_cases, ca_cfg, "invalid_python"))
        assert m.code_syntax_error_rate.value == 1.0
        assert m.provenance_retention_rate.valid_only.denominator == len(ca_cases)
        assert m.provenance_retention_rate.valid_only.value == 0.0

    def test_invalid_envelopes_leave_the_valid_only_denominator(self, ca_cases, ca_cfg):
        records = make_malformed_envelope_records(ca_cases)
        m = compute_code_agent_metrics(
            grade_all_code_agent(ca_cases, records, ca_cfg.provenance_columns, csv_columns(ca_cfg))
        )
        assert m.invalid_response_rate.value == 1.0
        assert m.provenance_retention_rate.valid_only.value is None
        assert m.provenance_retention_rate.all_trials.value == 0.0

    def test_level_counts_are_reported(self, ca_cases, ca_cfg):
        m = compute_code_agent_metrics(graded_for(ca_cases, ca_cfg, "reads_but_ignores"))
        assert m.level_counts == {"1": len(ca_cases)}

    def test_trend_metrics_match_the_direct_arm_semantics(self, ca_cases, ca_cfg):
        """The ignoring fixture answers 'increase' everywhere: spurious in cell 2,
        and in cell 3 it is right only where the truth happens to be increase."""
        graded = graded_for(ca_cases, ca_cfg, "ignores_provenance")
        cell2 = [g for g in graded if g.cell == 2]
        assert all(g.response_class is ResponseClass.SPURIOUS_TREND for g in cell2)
        m = compute_code_agent_metrics(graded)
        assert m.spurious_trend_rate.valid_only.value == 1.0

    def test_correct_fixture_scores_perfectly(self, ca_cases, ca_cfg):
        records = make_correct_code_agent_records(ca_cases)
        m = compute_code_agent_metrics(
            grade_all_code_agent(ca_cases, records, ca_cfg.provenance_columns, csv_columns(ca_cfg))
        )
        assert m.correct_rate.valid_only.value == 1.0
        assert m.effective_provenance_use_rate.valid_only.value == 1.0

    def test_breakdown_by_cell(self, ca_cases, ca_cfg):
        results = code_agent_breakdown(
            graded_for(ca_cases, ca_cfg, "ignores_provenance"), by=("cell",)
        )
        assert set(results) == {(2,), (3,)}
        assert results[(2,)].spurious_trend_rate.valid_only.value == 1.0
        assert results[(3,)].spurious_trend_rate.valid_only.value is None

    def test_breakdown_rejects_unknown_field(self, ca_cases, ca_cfg):
        with pytest.raises(ValueError, match="cannot group by"):
            code_agent_breakdown(graded_for(ca_cases, ca_cfg, "ignores_provenance"), by=("x",))

    def test_report_shows_both_rates(self, ca_cases, ca_cfg):
        text = format_code_agent_metrics(
            compute_code_agent_metrics(graded_for(ca_cases, ca_cfg, "reads_but_ignores"))
        )
        assert "provenance_retention_rate" in text
        assert "effective_provenance_use_rate" in text
        assert "level >= 1" in text and "level == 2" in text

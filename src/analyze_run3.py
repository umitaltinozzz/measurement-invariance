"""Verified run-3 analysis, one independent repeat and cohort per invocation."""
import argparse
import json
from pathlib import Path

from .analyze_results import _load_verified_runs, _direct_diagnostic_rows
from .code_agent import csv_columns
from .code_agent_grader import CodeAgentRecord, grade_code_agent_response
from .competence_grader import CompetenceResponseRecord, grade_all_competence
from .competence_metrics import compute_competence_metrics, evaluate_qualification
from .continuation import evaluate_continuation
from .generate_cases import load_config, read_jsonl
from .generate_competence import read_competence_jsonl
from .grader import ModelResponseRecord, ResponseClass, grade_all
from .metrics import Rate, breakdown
from .paired_metrics import pair_responses
from .prompts import PromptVariant
from .prepare_run3 import inference_config, validate_dataset
from .run_pilot import build_plans
from .reporting_stats import exact_mcnemar_p, wilson_interval


def with_intervals(value):
    """Attach Wilson CIs to every serialized binomial rate, both denominators."""
    if isinstance(value, list):
        return [with_intervals(x) for x in value]
    if not isinstance(value, dict):
        return value
    result = {k: with_intervals(v) for k, v in value.items()}
    if "numerator" in value and "denominator" in value:
        result["wilson_95"] = wilson_interval(value["numerator"], value["denominator"])
    return result


def select_stages(runs):
    """Whole-slice replacement only. No cherry-picked capped-row merge."""
    groups = {}
    problems, audit, selected = [], [], []
    repeats = set()
    for run in runs:
        m = run["manifest"]
        repeat = m.provider_constraints.get("repeat")
        if type(repeat) is not int or repeat < 0:
            raise ValueError("run 3 requires an explicit repeat number")
        repeats.add(repeat)
        key = (m.model, m.arm, m.prompt_variant, m.dataset_sha256)
        stage = m.provider_constraints.get("cap_stage", 0)
        if stage not in (0, 1) or stage in groups.setdefault(key, {}):
            raise ValueError("invalid or duplicate cap stage for a slice")
        groups[key][stage] = run
    if len(repeats) > 1:
        raise ValueError("analyze repeats separately; never pool them in a binomial CI")
    for key, stages in groups.items():
        initial = stages.get(0)
        replacement = stages.get(1)
        if replacement:
            if not initial or not any(r.result and r.result.truncated for r in initial["records"]):
                raise ValueError("stage-1 requires an archived capped stage-0 slice")
            old, new = initial["manifest"], replacement["manifest"]
            if new.max_tokens != 2 * old.max_tokens or old.seed != new.seed:
                raise ValueError("stage-1 must double the cap and preserve the seed")
            old_settings = dict(old.provider_constraints.get("model_settings", {}))
            new_settings = dict(new.provider_constraints.get("model_settings", {}))
            for cap in ("max_tokens", "max_tokens_code_agent"):
                if new_settings.get(cap) != 2 * old_settings.get(cap, 0):
                    raise ValueError("stage-1 model caps must double")
                old_settings.pop(cap, None); new_settings.pop(cap, None)
            if old_settings != new_settings or old.hardware != new.hardware:
                raise ValueError("stage-1 changed reasoning, hardware, or non-cap model settings")
            for field in ("provider", "temperature", "structured_output", "git_commit",
                          "package_versions", "max_transport_retries"):
                if getattr(old, field) != getattr(new, field):
                    raise ValueError(f"stage-1 changed {field}")
            for field in set(old.provider_constraints) | set(new.provider_constraints):
                if field not in ("cap_stage", "model_settings") and (
                    old.provider_constraints.get(field) != new.provider_constraints.get(field)
                ):
                    raise ValueError(f"stage-1 changed provider constraint {field}")
            audit.append({"slice": list(key), "discarded": str(initial["result_path"]),
                          "replacement": str(replacement["result_path"]), "policy": "whole_slice"})
        run = replacement or initial
        rows = run["records"]
        if len(rows) != run["planned"] or any(r.failure or r.result is None for r in rows):
            problems.append(f"incomplete/unavailable slice: {key}")
        if any(r.result and r.result.truncated for r in rows):
            problems.append(f"capped slice requires whole rerun or exclusion: {key}")
        selected.append(run)
    return selected, problems, audit


def analyze(config_path, results_dir, output_dir, project_root="."):
    root = Path(project_root).resolve()
    config_path = (root / config_path).resolve()
    cfg = load_config(config_path)
    if cfg.continuation_criteria.rule_version != "rule_v2":
        raise ValueError("use analyze_results for rule_v1")
    runs, verification = _load_verified_runs(Path(results_dir).resolve(), root, config_path)
    runs, problems, audit = select_stages(runs)
    direct, paired_direct, code, competence, main_cases = [], [], [], [], {}
    unavailable, observed_slices = [], set()
    for run in runs:
        m = run["manifest"]
        if m.model not in cfg.models:
            raise ValueError("result contains a model outside the registered cohort")
        kind = "competence" if m.arm == "competence" else (
            "paired" if Path(m.dataset_path).name == "paired_agent_cases.jsonl" else "pilot")
        slice_key = (m.model, kind, m.arm, m.prompt_variant)
        if slice_key in observed_slices:
            raise ValueError("multiple datasets for the same design slice")
        observed_slices.add(slice_key)
        settings = inference_config(cfg, m.provider_constraints["repeat"],
                                    m.provider_constraints.get("cap_stage", 0))
        if m.provider_constraints.get("model_settings") != settings.models[m.model].model_dump(mode="json"):
            raise ValueError("model settings differ from the registered stage")
        if m.selected_case_ids is not None:
            problems.append("run 3 forbids selected-case slices")
        for r in run["records"]:
            if not r.produced_a_completion:
                unavailable.append({"model": r.model, "case_id": r.case_id,
                                    "arm": r.arm, "failure": r.failure})
        records = [r for r in run["records"] if r.produced_a_completion]
        if kind == "competence":
            cases = read_competence_jsonl(run["dataset_path"])
            competence.extend(grade_all_competence(cases, [CompetenceResponseRecord(
                case_id=r.case_id, model=r.model, raw_text=r.raw_text) for r in records]))
            expected = cfg.competence.cases_per_model
        else:
            cases = read_jsonl(run["dataset_path"])
            expected = cfg.paired.total_cases if kind == "paired" else 4 * cfg.n_per_cell
            if kind == "pilot":
                main_cases.update({c.case_id: c for c in cases})
            if m.arm == "direct":
                target = paired_direct if kind == "paired" else direct
                target.extend(grade_all(cases, [ModelResponseRecord(case_id=r.case_id, model=r.model,
                    prompt_variant=r.prompt_variant, raw_text=r.raw_text) for r in records]))
            else:
                by_id = {c.case_id: c for c in cases}
                for r in records:
                    code.append(grade_code_agent_response(by_id[r.case_id],
                        CodeAgentRecord(case_id=r.case_id, model=r.model, raw_text=r.raw_text),
                        cfg.code_agent.provenance_columns, csv_columns(cfg.code_agent), r.execution,
                        protocol_version="v2", answer_source=cfg.code_agent.answer_source))
        validate_dataset(cfg, m.provider_constraints["repeat"], kind, cases)
        plans = build_plans(kind, cases, settings, m.model, m.arm, m.prompt_variant)
        expected_plans = {p.case_id: p for p in plans}
        request = plans[0].request
        if (m.max_tokens, m.temperature, m.seed, m.structured_output) != (
            request.max_tokens, request.temperature, request.seed, request.structured_output
        ):
            raise ValueError("manifest generation settings differ from the registered stage")
        for record in run["records"]:
            plan = expected_plans.get(record.case_id)
            if plan is None or record.prompt_variant != plan.prompt_variant or (
                record.prompt_fingerprint != plan.request.prompt_fingerprint
            ):
                raise ValueError("row does not match its registered prompt/case/variant")
            if m.arm == 'code-agent' and cfg.code_agent.answer_source == 'executed':
                from .correction_classifier import parse_code_response_v2
                if record.produced_a_completion and parse_code_response_v2(record.raw_text).ok and (
                    record.execution is None or not record.execution.executed
                ):
                    problems.append(f"execution unavailable: {record.model}/{record.case_id}")
        if len(cases) != expected:
            problems.append(f"wrong dataset size: {m.dataset_path}")
    expected_slices = {(model, kind, arm, variant) for model in cfg.models for kind, arm, variant in [
        ("pilot", "direct", "natural"), ("pilot", "direct", "natural_sustained"),
        ("pilot", "direct", "guided"), ("paired", "direct", "natural_sustained"),
        ("paired", "code-agent", None), ("competence", "competence", None)]}
    if observed_slices != expected_slices:
        problems.append(f"missing/unexpected design slices: {sorted(map(str, observed_slices ^ expected_slices))}")
    qualifications = {model: evaluate_qualification(compute_competence_metrics(
        [r for r in competence if r.model == model], model=model),
                                                   cfg.competence) for model in cfg.models}
    decision = evaluate_continuation(cfg.continuation_criteria, qualifications=qualifications,
                                    direct_graded=direct, code_graded=code)
    pairs = []
    for model in cfg.models:
        for cell in cfg.paired.cells:
            pairing = pair_responses([r for r in paired_direct if r.cell == cell],
                                     [r for r in code if r.cell == cell], model=model,
                                     variant=PromptVariant.NATURAL_SUSTAINED)
            if pairing.duplicate_direct or pairing.duplicate_code:
                raise ValueError("duplicate paired cases")
            first = sum(p.direct_correct and not p.code_correct for p in pairing.pairs)
            second = sum(p.code_correct and not p.direct_correct for p in pairing.pairs)
            ds = sum(p.direct_class == ResponseClass.SPURIOUS_TREND and p.code_class != ResponseClass.SPURIOUS_TREND for p in pairing.pairs)
            cs = sum(p.code_class == ResponseClass.SPURIOUS_TREND and p.direct_class != ResponseClass.SPURIOUS_TREND for p in pairing.pairs)
            pairs.append({"model": model, "cell": cell, "n": pairing.n_pairs,
                          "direct_only_correct": first, "code_only_correct": second,
                          "exact_mcnemar_p": exact_mcnemar_p(first, second),
                          "spurious_exact_mcnemar_p": exact_mcnemar_p(ds, cs) if cell == 2 else None,
                          "direct_only_spurious": ds, "code_only_spurious": cs,
                          "transitions": [{"case_id": p.case_id, "direct": p.direct_class.value,
                                           "code": p.code_class.value} for p in pairing.pairs],
                          "unpaired_direct": pairing.unpaired_direct, "unpaired_code": pairing.unpaired_code})
    metrics = [{"slice": list(key), **metric.model_dump(mode="json")}
               for key, metric in breakdown(direct, by=("model", "prompt_variant", "salience_level", "cell")).items()]
    for metric in metrics:
        model, variant, salience, cell = metric['slice']
        subset = [r for r in direct if (r.model,r.prompt_variant.value,r.salience_level.value,r.cell)
                  == (model,variant,salience,cell)]
        n = sum(r.reported_trend == 'uncertain' for r in subset)
        metric['uncertain_rate'] = {
            'valid_only': Rate.of(n,metric['n_valid']).model_dump(),
            'all_trials': Rate.of(n,metric['n_responses']).model_dump(),
        }
        metric['non_informative'] = (metric['n_valid'] == 0 or
            n / metric['n_valid'] > cfg.continuation_criteria.max_abstention_rate)
    code_metrics = []
    for model in cfg.models:
        for cell in cfg.paired.cells:
            rows = [r for r in code if r.model == model and r.cell == cell]
            modes = {}
            for mode, class_field, answer_field in (
                ('declared','declared_response_class','declared_answer'),
                ('executed','executed_response_class','execution_answer'),
            ):
                valid = [r for r in rows if getattr(r,class_field) not in (None,ResponseClass.INVALID_RESPONSE)]
                rates = {outcome.value: {
                    'valid_only': Rate.of(sum(getattr(r,class_field) == outcome for r in valid),len(valid)).model_dump(),
                    'all_trials': Rate.of(sum(getattr(r,class_field) == outcome for r in valid),len(rows)).model_dump(),
                } for outcome in ResponseClass if outcome != ResponseClass.INVALID_RESPONSE}
                rates['invalid_rate'] = Rate.of(len(rows)-len(valid),len(rows)).model_dump()
                rates['uncertain_rate'] = Rate.of(sum(getattr(r,answer_field) == 'uncertain' for r in valid),len(valid)).model_dump()
                modes[mode] = rates
            code_metrics.append({'model':model,'cell':cell,'n':len(rows),**modes,
                'syntax_error_rate':Rate.of(sum(r.code_syntax_error for r in rows),len(rows)).model_dump(),
                'correction_prr':decision.per_model[model].correction_prr.model_dump() if cell == 2 else None,
                'correction_prr_all_trials':decision.per_model[model].correction_prr_all_trials.model_dump() if cell == 2 else None,
                'non_informative':(decision.per_model[model].code_cell2 if cell == 2 else
                                   decision.per_model[model].code_cell3).non_informative})
    report = with_intervals({
        "rule_version": "rule_v2", "status": "incomplete" if problems else "complete",
        "problems": problems, "unavailable": unavailable, "manifest_verification": verification,
        "whole_slice_replacements": audit,
        "competence": {m: q.model_dump(mode="json") for m, q in qualifications.items()},
        "direct_metrics": metrics,
        "direct_diagnostics": _direct_diagnostic_rows(direct, list(main_cases.values())),
        "code_grades": [r.model_dump(mode="json") for r in code],
        "code_metrics": code_metrics,
        "paired_comparison": pairs,
        "continuation": decision.model_dump(mode="json") if not problems else None,
    })
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", required=True, help="one cohort / one repeat directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = analyze(args.config, args.results_dir, args.out)
    print(json.dumps({"status": report["status"], "problems": report["problems"],
                      "continuation": report["continuation"]}, indent=2))


if __name__ == "__main__":
    main()

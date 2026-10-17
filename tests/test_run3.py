import json
from pathlib import Path

import pytest

from src.code_agent import build_code_agent_task, csv_columns
from src.code_agent_grader import CodeAgentRecord, grade_code_agent_response
from src.continuation import evaluate_continuation
from src.correction_classifier import analyze_correction, parse_code_response_v2
from src.estimate_run3 import estimate
from src.generate_cases import generate_dataset, load_config
from src.grader import ModelResponseRecord, ResponseClass, grade_all
from src.models.ollama import OllamaProvider
from src.models.openrouter import OpenRouterProvider
from src.prepare_run3 import prepare, repeat_config
from src.prompts import PromptVariant, SUSTAINED_TASK, build_direct_prompt
from src.reporting_stats import newcombe_difference_interval
from src.run_pilot import build_direct_plans, build_code_plans
from src.sandbox import ExecutionResult


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/run3_open.yaml")


@pytest.fixture(scope="module")
def cases(cfg):
    return generate_dataset(cfg)


def direct_rows(cases, *, c1=0, c2=40, abstain_cell=None, abstain=0, invalid_cell=None, invalid=0):
    records = []
    for cell in (1, 2, 3):
        subset = [c for c in cases if c.cell == cell and c.salience_level.value == "L1"]
        for i, case in enumerate(subset):
            answer = "increase" if cell in (1, 2) and i < {1:c1, 2:c2}[cell] else case.expected_trend.value
            if cell == abstain_cell and i < abstain:
                answer = "uncertain"
            raw = json.dumps({"trend": answer, "reason": "fixture"})
            if cell == invalid_cell and i < invalid:
                raw = "broken"
            records.append(ModelResponseRecord(case_id=case.case_id, model="m",
                prompt_variant=PromptVariant.NATURAL_SUSTAINED, raw_text=raw))
    return grade_all(cases, records)


def decision(cfg, rows, code=()):
    return evaluate_continuation(cfg.continuation_criteria, qualifications={},
                                 direct_graded=rows, code_graded=code).per_model["m"]


@pytest.mark.parametrize("c1,c2,expected", [(0,40,True),(35,38,False),(40,10,False),(0,0,False),(50,50,False)])
def test_controlled_difference(cfg, cases, c1, c2, expected):
    result = decision(cfg, direct_rows(cases,c1=c1,c2=c2))
    assert result.direct_trigger is expected
    assert result.competence_policy == "diagnostic_only"
    assert result.competence_qualified is None


@pytest.mark.parametrize("cell", [1,2,3])
def test_abstention_cap_each_cell(cfg,cases,cell):
    result=decision(cfg,direct_rows(cases,abstain_cell=cell,abstain=26))
    assert result.direct_cells[cell].non_informative
    assert not result.direct_trigger
    boundary=decision(cfg,direct_rows(cases,abstain_cell=cell,abstain=25))
    assert not boundary.direct_cells[cell].non_informative


@pytest.mark.parametrize("n,eligible", [(2,True),(3,False),(50,False)])
def test_invalid_gate(cfg,cases,n,eligible):
    result=decision(cfg,direct_rows(cases,invalid_cell=1,invalid=n))
    assert result.direct_cells[1].eligible is eligible
    if not eligible:
        assert not result.direct_trigger


def test_exact_invalid_boundary_and_missing_control(cfg,cases):
    rows=direct_rows(cases,invalid_cell=1,invalid=2)
    criteria=cfg.continuation_criteria.model_copy(update={"max_invalid_response_rate":0.04})
    assert evaluate_continuation(criteria,qualifications={},direct_graded=rows).per_model['m'].direct_trigger
    assert not decision(cfg,[r for r in rows if r.cell != 1]).direct_trigger
    assert not decision(cfg,[r.model_copy(update={"prompt_variant":PromptVariant.NATURAL})
                             if r.cell==1 else r for r in rows]).direct_trigger
    assert not decision(cfg,rows[:-1]).direct_trigger  # cell 3 has <50 cases
    with pytest.raises(ValueError,match="unique cases"):
        decision(cfg,rows+[rows[0]])


def test_newcombe_known_values_and_symmetry():
    # Reference from solving the Wilson score quadratic at 35-digit precision.
    lo,hi=newcombe_difference_interval(40,50,10,50)
    assert lo == pytest.approx(0.415627479765521,abs=1e-12)
    assert hi == pytest.approx(0.723832074832186,abs=1e-12)
    opposite=newcombe_difference_interval(10,50,40,50)
    assert opposite == pytest.approx((-hi,-lo))
    assert newcombe_difference_interval(0,0,0,50) is None
    assert newcombe_difference_interval(50,50,0,50)[0]>0
    with pytest.raises(ValueError):
        newcombe_difference_interval(51,50,0,50)


def test_frozen_v1_source_and_dispatch(cfg, cases):
    import hashlib
    from src.rule_v1 import evaluate_continuation as frozen
    source = Path('src/rule_v1.py').read_text(encoding='utf-8')
    assert hashlib.sha256(source.encode()).hexdigest() == 'e24e3073a33956732b67368b63a32b50aa8c4a937690d353b66313f3a9ef29b5'
    old = load_config('configs/pilot.yaml').continuation_criteria
    kwargs = dict(qualifications={}, direct_graded=direct_rows(cases))
    assert evaluate_continuation(old, **kwargs) == frozen(old, **kwargs)


def test_mismatched_provenance_and_nonprimary_basis(cfg, cases):
    from src.schemas import SalienceLevel
    rows = direct_rows(cases)
    altered = [r.model_copy(update={'salience_level': SalienceLevel.L2}) if r.cell == 1 else r for r in rows]
    assert not decision(cfg, altered).direct_trigger
    with pytest.raises(ValueError, match='valid_only'):
        evaluate_continuation(cfg.continuation_criteria, qualifications={}, direct_graded=rows, basis='all_trials')


def code_rows(cases, cfg, corrected=0, cell=2, answer="stable"):
    rows=[]
    for i,c in enumerate(x for x in cases if x.cell==cell and x.salience_level.value=="L1"):
        code = "import json\nprint(json.dumps({'answer': 'stable'}))"
        if i < corrected:
            code="""import json
lo, hi = map(float, row['reference_range'].split('-'))
x = (float(row['reading']) - lo) / (hi - lo)
answer = 'increase' if x > 0.6 else 'stable'
print(json.dumps({'answer': answer}))"""
        raw=json.dumps({"code_lines":code.splitlines(),"answer":answer,"reason":"fixture"})
        rows.append(grade_code_agent_response(c,CodeAgentRecord(case_id=c.case_id,model='m',raw_text=raw),
            cfg.code_agent.provenance_columns,csv_columns(cfg.code_agent),protocol_version='v2'))
    return rows


@pytest.mark.parametrize("corrected,trigger",[(30,True),(31,False),(50,False),(0,True)])
def test_code_prr_cell2_only(cfg,cases,corrected,trigger):
    rows=code_rows(cases,cfg,corrected)+code_rows(cases,cfg,0,cell=3)
    d=decision(cfg,direct_rows(cases,c2=0),rows)
    assert d.correction_prr.denominator==50
    assert d.correction_prr.numerator==corrected
    assert d.code_trigger is trigger


def test_code_abstention_missing_invalid_and_old_classifier(cfg,cases):
    direct=direct_rows(cases,c2=0)
    assert not decision(cfg,direct).code_trigger
    assert not decision(cfg,direct,code_rows(cases,cfg,answer='uncertain')).code_trigger
    rows=code_rows(cases,cfg)
    invalid=[r.model_copy(update={"response_class":ResponseClass.INVALID_RESPONSE}) for r in rows[:3]]+rows[3:]
    assert not decision(cfg,direct,invalid).code_trigger
    old=[r.model_copy(update={"analysis":r.analysis.model_copy(update={"classifier_version":"v1"})}) for r in rows]
    assert not decision(cfg,direct,old).code_trigger
    with pytest.raises(ValueError,match='case-level'):
        evaluate_continuation(cfg.continuation_criteria,qualifications={},code_metrics_by_model={'m':object()})


def test_code_control_abstention_and_invalid_caps(cfg, cases):
    direct = direct_rows(cases, c2=0)
    cell2 = code_rows(cases, cfg)
    control = code_rows(cases, cfg, cell=3)
    assert decision(cfg, direct, cell2 + control).code_trigger
    assert not decision(cfg, direct, cell2).code_trigger
    assert not decision(cfg, direct, cell2 + code_rows(cases, cfg, cell=3, answer='uncertain')).code_trigger
    invalid = [r.model_copy(update={'response_class': ResponseClass.INVALID_RESPONSE}) for r in control[:3]]
    assert not decision(cfg, direct, cell2 + invalid + control[3:]).code_trigger
    boundary = code_rows(cases, cfg, answer='uncertain')[:25] + cell2[25:]
    assert decision(cfg, direct, boundary + control).code_trigger


@pytest.mark.parametrize("answer,ok,expected", [('increase',True,'spurious_trend'),('stable',True,'correct_stable'),
                                            ('invalid',True,'invalid_response'),('increase',False,'invalid_response')])
def test_grade_declared_and_executed(cfg,cases,answer,ok,expected):
    case=next(c for c in cases if c.cell==2)
    raw=json.dumps({'code_lines':['print(1)'],'answer':'stable','reason':'fixture'})
    execution=ExecutionResult(executed=True,exit_code=0 if ok else 1,stdout=json.dumps({'answer':answer}))
    grade=grade_code_agent_response(case,CodeAgentRecord(case_id=case.case_id,model='m',raw_text=raw),
        cfg.code_agent.provenance_columns,csv_columns(cfg.code_agent),execution,
        protocol_version='v2',answer_source='executed')
    assert grade.declared_response_class.value=='correct_stable'
    assert grade.response_class.value==expected
    assert grade.executed_response_class.value==expected


def test_protocol_is_strict():
    for lines in ['print(1)',[],[1],['print(1)\nprint(2)']]:
        assert not parse_code_response_v2(json.dumps({'code_lines':lines,'answer':'stable','reason':'r'})).ok
    assert not parse_code_response_v2('{"code":"x","answer":"stable","reason":"r"}').ok


def test_reasoning_and_caps(cfg,cases):
    for name, setting in cfg.models.items():
        request=build_direct_plans(cases[:1],cfg,name,PromptVariant.NATURAL_SUSTAINED)[0].request
        payload=OllamaProvider().build_payload(request)
        assert payload['think']==setting.think
        assert payload['options']['num_predict']==8192
        assert payload['options']['num_ctx']==65536
        code=build_code_plans(cases[:1],cfg,name)[0]
        assert code.request.max_tokens==16384
        assert code.execution_files['readings.csv']
        body=json.dumps({'message':{'content':'{}','thinking':'trace'},'done_reason':'stop'})
        assert OllamaProvider().parse_response(request,200,body,1).thinking=='trace'
    frontier=load_config('configs/run3_frontier.yaml')
    name=next(iter(frontier.models))
    request=build_direct_plans(cases[:1],frontier,name)[0].request
    provider=OpenRouterProvider(api_key='fixture')
    assert provider.build_payload(request)['reasoning']=={'effort':'medium','exclude':False}
    assert 'temperature' not in provider.build_payload(request)
    assert 'seed' not in provider.build_payload(request)
    raw=provider.parse_response(request,200,json.dumps({'choices':[{'message':{'content':'{}','reasoning':'r',
        'reasoning_details':[{'type':'reasoning.text','text':'r'}]},'finish_reason':'stop'}]}),1)
    assert raw.thinking=='r' and raw.reasoning_details


def test_prompt_parity_and_old_prompt_untouched(cfg,cases):
    case=cases[0]
    direct=build_direct_prompt(case,PromptVariant.NATURAL_SUSTAINED)
    code=build_code_agent_task(case,cfg.code_agent)
    assert SUSTAINED_TASK in direct.user and SUSTAINED_TASK in code.user
    assert direct.system==code.system
    assert 'several consecutive readings' in direct.user
    assert SUSTAINED_TASK not in build_direct_prompt(case,PromptVariant.NATURAL).user
    assert load_config('configs/pilot.yaml').continuation_criteria.rule_version=='rule_v1'


def test_estimate_and_repeat_preparation(cfg,tmp_path):
    report=estimate(cfg)
    assert report['network_calls']==0
    assert report['total_calls']==17520
    assert report['per_model_calls']['total']==4380
    artifacts=prepare('configs/run3_open.yaml',tmp_path)
    assert len(artifacts)==9
    assert artifacts==prepare('configs/run3_frontier.yaml',tmp_path)
    assert len({a['sha256'] for a in artifacts})==9
    frozen = json.loads(Path('docs/run3_dataset_manifest.json').read_text(encoding='utf-8'))
    assert [(a['repeat'], a['cases'], a['sha256']) for a in artifacts] == [
        (a['repeat'], a['cases'], a['sha256']) for a in frozen]
    old=generate_dataset(repeat_config(cfg,0))
    new=generate_dataset(repeat_config(cfg,1))
    assert {c.case_id for c in old}.isdisjoint(c.case_id for c in new)


def test_repeat_dataset_guard_and_stage_settings(cfg, cases):
    from src.prepare_run3 import validate_dataset, inference_config
    validate_dataset(cfg, 0, 'pilot', generate_dataset(repeat_config(cfg, 0)))
    with pytest.raises(ValueError, match='registered design/repeat'):
        validate_dataset(cfg, 1, 'pilot', generate_dataset(repeat_config(cfg, 0)))
    stage = inference_config(cfg, 2, 1)
    assert stage.runner.seed == cfg.runner.seed + 2
    for name, setting in stage.models.items():
        assert setting.max_tokens == 16384 and setting.max_tokens_code_agent == 32768
        assert setting.think == cfg.models[name].think


@pytest.mark.parametrize('model,think', [('gpt-oss:20b', False), ('qwen3:14b', 'medium'), ('qwen3:14b', None)])
def test_invalid_reasoning_settings_rejected(cfg, model, think):
    from src.generate_cases import PilotConfig
    data = cfg.model_dump(mode='json')
    data['models'][model]['think'] = think
    with pytest.raises(ValueError):
        PilotConfig.model_validate(data)


def test_competence_is_diagnostic_even_when_failed(cfg, cases):
    from src.competence_metrics import compute_competence_metrics, evaluate_qualification
    failed = evaluate_qualification(compute_competence_metrics([], model='m'), cfg.competence)
    result = evaluate_continuation(cfg.continuation_criteria, qualifications={'m': failed},
                                   direct_graded=direct_rows(cases)).per_model['m']
    assert result.competence_qualified is False
    assert result.direct_trigger


@pytest.mark.parametrize('fixture', json.loads(Path('tests/fixtures/correction_labels.json').read_text(encoding='utf-8')),
                         ids=lambda f:f['id'])
def test_hand_labelled_correction(fixture,cfg):
    result=analyze_correction(fixture['code'],cfg.code_agent.provenance_columns)
    assert result.correction_detected is fixture['correction']
    assert result.classifier_version=='v2'

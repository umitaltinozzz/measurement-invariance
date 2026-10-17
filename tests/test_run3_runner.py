import json
from pathlib import Path

import pytest
import yaml

from src.analyze_run3 import analyze, select_stages
from src.generate_cases import load_config, read_jsonl
from src.generate_competence import read_competence_jsonl
from src.models.base import ProviderError, TransportError, RawModelResult
from src.models.replay import ScriptedProvider
from src.prepare_run3 import prepare
from src.run_manifest import build_manifest, sha256_file
from src.run_pilot import build_plans
from src.runner import Runner, TrialPlan, load_results
from src.models.base import GenerationRequest
from src.sandbox import ExecutionResult, ReplaySandbox


def manifest(tmp_path, **kwargs):
    dataset=tmp_path/'dataset.jsonl'
    dataset.write_text('{}\n',encoding='utf-8')
    args=dict(run_id='test',arm='direct',prompt_variant='natural',provider='scripted',model='m',
              dataset_path=dataset,config_path='configs/run3_open.yaml',temperature=0,max_tokens=8192,
              seed=1,structured_output=True,max_transport_retries=0)
    args.update(kwargs)
    return build_manifest(**args)


@pytest.mark.parametrize('failure',[ProviderError('404'),TransportError('timeout')])
def test_provider_failures_reasked_once_with_audit(tmp_path,failure):
    cfg=load_config('configs/run3_open.yaml').runner.model_copy(update={'max_transport_retries':0})
    request=GenerationRequest(model='m',system='s',user='u')
    plan=TrialPlan(case_id='c',request=request,prompt_variant='natural')
    provider=ScriptedProvider({('m',request.prompt_fingerprint):[failure,'broken content']})
    run=Runner(provider,cfg)
    m=manifest(tmp_path)
    kwargs=dict(model='m',arm='direct',dataset_sha256=m.dataset_sha256,manifest=m,results_dir=tmp_path/'results')
    first,_=run.run([plan],**kwargs)
    assert first[0].failure
    second,summary=run.run([plan],**kwargs)
    assert len(second)==1 and second[0].raw_text=='broken content'
    assert len(load_results(summary.result_path))==1
    audit=Path(summary.result_path).with_suffix('.provider_failures.jsonl')
    assert len(audit.read_text().splitlines())==1
    _,resumed=run.run([plan],**kwargs)
    assert resumed.skipped_existing==1 and len(provider.calls)==2


def test_capped_resume_rejected_and_whole_stage_checked(tmp_path):
    cfg=load_config('configs/run3_open.yaml').runner
    request=GenerationRequest(model='m',system='s',user='u')
    m=manifest(tmp_path)
    class Capped:
        def generate(self,request):
            return RawModelResult(provider='test',model='m',requested_model_id='m',resolved_model_id='m',
                                  raw_text='',structured_output_used=True,finish_reason='length')
    runner=Runner(Capped(),cfg)
    kwargs=dict(model='m',arm='direct',dataset_sha256=m.dataset_sha256,manifest=m,results_dir=tmp_path/'results')
    rows,_=runner.run([TrialPlan(case_id='c',request=request)],**kwargs)
    with pytest.raises(ValueError,match='WHOLE'):
        runner.run([TrialPlan(case_id='c',request=request)],**kwargs)
    settings={'max_tokens':8192,'max_tokens_code_agent':16384,'think':'medium'}
    initial=m.model_copy(update={'provider_constraints':{'repeat':0,'cap_stage':0,'model_settings':settings}})
    second=m.model_copy(update={'max_tokens':16384,'provider_constraints':{'repeat':0,'cap_stage':1,
        'model_settings':dict(settings,max_tokens=16384,max_tokens_code_agent=32768)}})
    first=dict(manifest=initial,records=rows,planned=2,result_path='stage0')
    final=dict(manifest=second,records=[rows[0].model_copy(update={'result':rows[0].result.model_copy(update={'finish_reason':'stop'})})],planned=2,result_path='stage1')
    _,issues,audit=select_stages([first,final])
    assert issues and audit[0]['policy']=='whole_slice'
    with pytest.raises(ValueError,match='archived capped'):
        select_stages([final])
    with pytest.raises(ValueError,match='separately'):
        select_stages([first,dict(final,manifest=second.model_copy(update={'provider_constraints':{'repeat':1,'cap_stage':1}}))])
    with pytest.raises(ValueError, match='temperature'):
        select_stages([first, dict(final, manifest=second.model_copy(update={'temperature': 0.8}))])


def test_complete_offline_run3_analysis_and_missing_slice(tmp_path):
    # One full cohort model, all six slices, real generated cases, no network.
    source=yaml.safe_load(Path('configs/run3_open.yaml').read_text(encoding='utf-8'))
    model='qwen3:14b'
    source['models']={model:source['models'][model]}
    config_path=tmp_path/'run3.yaml'
    config_path.write_text(yaml.safe_dump(source),encoding='utf-8')
    cfg=load_config(config_path)
    prepare(config_path,tmp_path/'data')
    results=tmp_path/'results'
    code="import json\nprint(json.dumps({'answer': 'stable'}))"
    sandbox=ReplaySandbox({code:ExecutionResult(executed=True,exit_code=0,stdout='{"answer":"stable"}')})
    combinations=[('pilot','direct','natural'),('pilot','direct','guided'),('pilot','direct','natural_sustained'),
                  ('paired','direct','natural_sustained'),('paired','code-agent',None),('competence','competence',None)]
    for dataset,arm,variant in combinations:
        filename={'pilot':'pilot_cases.jsonl','paired':'paired_agent_cases.jsonl','competence':'competence_cases.jsonl'}[dataset]
        data_path=tmp_path/'data'/'repeat-0'/filename
        cases=read_competence_jsonl(data_path) if dataset=='competence' else read_jsonl(data_path)
        plans=build_plans(dataset,cases,cfg,model,arm,variant)
        script={}
        for plan,case in zip(plans,cases):
            if arm=='competence':
                value={'classification':case.expected_classification.value,'reason':'r'} if hasattr(case,'expected_classification') else {'x':case.expected_x,'reason':'r'}
            elif arm=='code-agent':
                value={'code_lines':code.splitlines(),'answer':'increase','reason':'r'}
            else:
                value={'trend':'increase' if case.cell==2 else case.expected_trend.value,'reason':'r'}
                if variant=='guided':
                    value['measurement_change_detected']=case.measurement_function_change
            script[(model,plan.request.prompt_fingerprint)]=[json.dumps(value)]
        m=build_manifest(run_id=dataset+arm+str(variant),arm=arm,prompt_variant=variant,provider='scripted',model=model,
            dataset_path=data_path,config_path=config_path,temperature=0,max_tokens=plans[0].request.max_tokens,
            seed=cfg.runner.seed,structured_output=True,max_transport_retries=0,
            provider_constraints={'repeat':0,'cap_stage':0,'model_settings':cfg.models[model].model_dump(mode='json')})
        Runner(ScriptedProvider(script),cfg.runner,sandbox=sandbox,code_protocol='v2').run(plans,
            model=model,arm=arm,dataset_sha256=sha256_file(data_path),manifest=m,
            results_dir=results/(variant or arm))
    report=analyze(config_path,results,tmp_path/'analysis')
    assert report['status']=='complete',report['problems']
    assert report['continuation']['project_continue']
    assert report['continuation']['per_model'][model]['direct_trigger']
    assert len(report['code_grades'])==100
    assert report['code_grades'][0]['declared_executed_agree'] is False
    assert report['competence'][model]['qualified']
    assert all('wilson_95' in x['invalid_response_rate'] for x in report['direct_metrics'])
    assert len(report['paired_comparison'])==2
    incomplete=analyze(config_path,results/'code-agent',tmp_path/'incomplete')
    assert incomplete['continuation'] is None and incomplete['status']=='incomplete'
    # A valid JSONL and self-consistent SHA are not enough: verify the prompt too.
    result_path = next((results/'natural').glob('*.jsonl'))
    original = result_path.read_text(encoding='utf-8')
    rows = original.splitlines()
    tampered = json.loads(rows[0])
    tampered['prompt_fingerprint'] = 'wrong prompt'
    rows[0] = json.dumps(tampered)
    result_path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='registered prompt'):
        analyze(config_path,results,tmp_path/'tampered')
    result_path.write_text(original, encoding='utf-8')


def test_sandbox_failure_keeps_completion(tmp_path):
    class Unavailable:
        def run(self, *args):
            raise OSError('daemon disappeared')
    cfg = load_config('configs/run3_open.yaml').runner
    request = GenerationRequest(model='m', system='s', user='u')
    raw = json.dumps({'code_lines':['print(1)'], 'answer':'stable', 'reason':'r'})
    provider = ScriptedProvider({('m', request.prompt_fingerprint): [raw]})
    m = manifest(tmp_path, arm='code-agent', prompt_variant=None)
    records, _ = Runner(provider, cfg, sandbox=Unavailable(), code_protocol='v2').run(
        [TrialPlan(case_id='c', request=request, execution_files={'readings.csv':'x'})],
        model='m', arm='code-agent', dataset_sha256=m.dataset_sha256, manifest=m,
        results_dir=tmp_path/'results')
    assert records[0].raw_text == raw and records[0].failure is None
    assert not records[0].execution.executed


def test_provider_resume_budget_is_finite(tmp_path):
    cfg = load_config('configs/run3_open.yaml').runner.model_copy(update={'max_transport_retries': 0})
    request = GenerationRequest(model='m', system='s', user='u')
    provider = ScriptedProvider({('m', request.prompt_fingerprint): [ProviderError('unavailable')] * 4})
    run = Runner(provider, cfg)
    m = manifest(tmp_path)
    plan = TrialPlan(case_id='c', request=request, prompt_variant='natural')
    for _ in range(4):
        rows, _ = run.run([plan], model='m', arm='direct', dataset_sha256=m.dataset_sha256,
                          manifest=m, results_dir=tmp_path/'results')
        assert rows[0].failure
    assert len(provider.calls) == 3  # original plus the two registered re-asks

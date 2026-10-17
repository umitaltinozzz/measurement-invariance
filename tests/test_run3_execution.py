"""Run with MI_RUN_SANDBOX_TESTS=1 after building sandbox/Dockerfile."""
import json
import os
from pathlib import Path

import pytest

from src.sandbox import DockerSandbox, MAX_CAPTURED_CHARS, sandbox_from_config
from src.generate_cases import load_config


def test_docker_command_has_enforced_boundaries():
    sandbox=object.__new__(DockerSandbox)
    sandbox.image_id='sha256:fixture'
    sandbox.memory_mb=512; sandbox.cpu_seconds=10; sandbox.pids_limit=32
    command=sandbox.command('/safe/temp','fixture-name')
    for flag in ['--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges',
                 '--memory=512m','--memory-swap=512m','--pids-limit=32','--cpus=1','--pull=never',
                 '--user=65534:65534','cpu=10:10','fsize=1048576:1048576']:
        assert flag in command
    assert any('source=/safe/temp,target=/input,readonly' in x for x in command)


def test_disabled_default():
    cfg=load_config('configs/pilot.yaml')
    assert not sandbox_from_config(cfg.code_agent).run('raise Exception()',{},1).executed


@pytest.fixture
def sandbox():
    if os.environ.get('MI_RUN_SANDBOX_TESTS')!='1':
        pytest.skip('set MI_RUN_SANDBOX_TESTS=1 to exercise the real Docker boundary')
    return DockerSandbox()


def test_real_output_and_limits(sandbox):
    code="""import csv, json, os, resource
with open('readings.csv') as f:
    assert list(csv.DictReader(f))[0]['reading'] == '42'
assert resource.getrlimit(resource.RLIMIT_CPU)[0] == 10
assert resource.getrlimit(resource.RLIMIT_FSIZE)[0] == 1048576
assert open('/sys/fs/cgroup/memory.max').read().strip() == '536870912'
assert open('/sys/fs/cgroup/pids.max').read().strip() == '32'
assert os.getuid() == 65534
assert os.listdir('/input') == ['analysis.py', 'readings.csv'] or set(os.listdir('/input')) == {'analysis.py', 'readings.csv'}
print(json.dumps({'answer': 'stable'}))"""
    result=sandbox.run(code,{'readings.csv':'reading\n42\n'},20)
    assert result.ok, result.stderr
    assert json.loads(result.stdout)=={'answer':'stable'}
    assert result.image_id.startswith('sha256:')


def test_real_network_disabled(sandbox):
    code="""import socket
s=socket.socket()
s.settimeout(1)
try:
    s.connect(('1.1.1.1',443))
except OSError:
    print('blocked')
else:
    raise AssertionError('network escaped')"""
    result=sandbox.run(code,{},20)
    assert result.ok and result.stdout.strip()=='blocked', result


def test_real_readonly_and_private_environment(sandbox,monkeypatch):
    monkeypatch.setenv('MI_TEST_SECRET','not-in-container')
    result=sandbox.run("""import os
assert 'MI_TEST_SECRET' not in os.environ
try:
    open('/input/escape','w').write('no')
except OSError:
    print('blocked')
else:
    raise AssertionError('writable input')""",{},20)
    assert result.ok and 'blocked' in result.stdout, result


def test_real_timeout(sandbox):
    result=sandbox.run('while True: pass',{},1)
    assert result.timed_out and not result.ok


def test_real_output_flood(sandbox):
    result=sandbox.run("while True: print('x'*10000, flush=True)",{},20)
    assert result.output_limit_exceeded and not result.ok
    assert len(result.stdout)<=MAX_CAPTURED_CHARS


def test_real_memory_limit(sandbox):
    result=sandbox.run('x=bytearray(1024*1024*1024)',{},20)
    assert not result.ok


def test_path_escape(sandbox):
    with pytest.raises(ValueError):
        sandbox.run('pass',{'../escape':'x'},1)


def test_real_normalization_program(sandbox):
    from src.code_agent import render_csv
    from src.correction_classifier import analyze_correction
    from src.generate_cases import generate_dataset
    cfg = load_config('configs/run3_open.yaml')
    case = next(c for c in generate_dataset(cfg) if c.cell == 2 and
                c.measurement_after.reference_range[0] < 0)
    code = Path('tests/fixtures/corrected_program.py').read_text(encoding='utf-8')
    assert analyze_correction(code, cfg.code_agent.provenance_columns).correction_detected
    result = sandbox.run(code, {'readings.csv': render_csv(case, cfg.code_agent)}, 20)
    assert result.ok, result
    assert json.loads(result.stdout) == {'answer': 'stable'}

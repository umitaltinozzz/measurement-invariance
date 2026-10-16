"""Regenerate run_pilot_t4.ipynb from the checked-in cell source below."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
        # #108b pilot on a Colab T4 with local Ollama

        This notebook runs the frozen project datasets with the checked-in runner. It mounts a Google Drive folder and writes each trial row there as soon as the runner completes it. After a disconnect, reconnect, select **Runtime → Run all**, and upload the project package only if it is missing from Drive. Existing result rows are resumed by the runner and are not asked again.

        The stages are competence pre-test, both direct prompt variants on the complete pilot dataset, the paired direct/code arms when code execution is disabled in the config, then metrics and the registered continuation rule. Generated code is graded statically and is never executed in this notebook.

        Select **Runtime → Change runtime type → T4 GPU** before running. This uses Ollama locally in the Colab VM; it does not call OpenRouter, Vercel, or any hosted inference API.
        """
    ),
    code(
        r"""
        from pathlib import Path
        import shutil
        from google.colab import drive, files

        drive.mount('/content/drive')
        DRIVE_ROOT = Path('/content/drive/MyDrive/measurement_invariance_colab_t4')
        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
        PROJECT_ZIP = DRIVE_ROOT / 'measurement-invariance-colab-package.zip'

        if not PROJECT_ZIP.exists():
            print('Upload measurement-invariance-colab-package.zip from the project folder.')
            uploaded = files.upload()
            candidates = [name for name in uploaded if name.lower().endswith('.zip')]
            if len(candidates) != 1:
                raise RuntimeError(f'Expected one uploaded zip, received: {list(uploaded)}')
            shutil.copy2(candidates[0], PROJECT_ZIP)

        print(f'Project package: {PROJECT_ZIP}')
        print(f'Durable results folder: {DRIVE_ROOT / "results"}')
        """
    ),
    code(
        r"""
        from pathlib import Path
        from zipfile import ZipFile
        import os
        import subprocess
        import sys

        PROJECT_ROOT = Path('/content/measurement-invariance')
        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
        with ZipFile(PROJECT_ZIP) as archive:
            archive.extractall(PROJECT_ROOT)
        os.chdir(PROJECT_ROOT)
        sys.path.insert(0, str(PROJECT_ROOT))

        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib'], check=True)

        from src.generate_cases import load_config, read_jsonl
        from src.generate_competence import read_competence_jsonl
        from src.run_manifest import sha256_file
        from src.runner import result_path_for

        CONFIG_PATH = 'configs/pilot_colab_t4.yaml'
        cfg = load_config(CONFIG_PATH)
        MODEL_IDS = [
            'qwen2.5:14b-instruct-q4_K_M',
            'llama3.1:8b-instruct-q4_K_M',
            'gemma2:9b-instruct-q4_K_M',
            'mistral-nemo:12b-instruct-2407-q4_K_M',
        ]
        RESULTS_ROOT = DRIVE_ROOT / 'results'
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        ANALYSIS_ROOT = DRIVE_ROOT / 'analysis'
        ENVIRONMENT_PATH = DRIVE_ROOT / 'run_environment.json'
        STATUS_PATH = DRIVE_ROOT / 'run_status.json'

        print(f'Models: {MODEL_IDS}')
        print(f'Structured output configured: {cfg.runner.structured_output}')
        print(f'Results are written directly under: {RESULTS_ROOT}')
        """
    ),
    markdown(
        r"""
        ## Install and start local Ollama

        This cell installs the Ollama server in the temporary Colab VM and starts it. Model pulls and inference remain local to that VM; only JSONL results and run metadata are written to the mounted Drive folder.
        """
    ),
    code(
        r"""
        import json
        import os
        import subprocess
        import time
        import urllib.request
        from datetime import datetime, timezone

        install = subprocess.run(
            'curl -fsSL https://ollama.com/install.sh | sh',
            shell=True,
            text=True,
            check=False,
        )
        if install.returncode != 0:
            raise RuntimeError(f'Ollama installer exited with {install.returncode}')

        ollama_env = dict(os.environ)
        ollama_env['OLLAMA_HOST'] = '127.0.0.1:11434'
        ollama_env['OLLAMA_CONTEXT_LENGTH'] = '8192'
        ollama_env['OLLAMA_MAX_LOADED_MODELS'] = '1'
        os.environ['OLLAMA_CONTEXT_LENGTH'] = ollama_env['OLLAMA_CONTEXT_LENGTH']
        os.environ['OLLAMA_MAX_LOADED_MODELS'] = ollama_env['OLLAMA_MAX_LOADED_MODELS']
        ollama_server = subprocess.Popen(
            ['ollama', 'serve'],
            env=ollama_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        for _ in range(120):
            try:
                with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2) as response:
                    response.read()
                break
            except Exception:
                if ollama_server.poll() is not None:
                    raise RuntimeError('Ollama server exited before becoming ready')
                time.sleep(2)
        else:
            raise TimeoutError('Ollama server did not become ready')

        gpu = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=False,
        )
        ollama_version = subprocess.run(['ollama', '--version'], capture_output=True, text=True, check=True).stdout.strip()
        os.environ['OLLAMA_SERVER_VERSION'] = ollama_version
        print('GPU:', gpu.stdout.strip() or gpu.stderr.strip() or 'nvidia-smi unavailable')
        print('Ollama:', ollama_version)
        """
    ),
    code(
        r"""
        import hashlib
        import json
        import subprocess
        import urllib.request
        from datetime import datetime, timezone

        for model_id in MODEL_IDS:
            print(f'Pulling {model_id}')
            subprocess.run(['ollama', 'pull', model_id], check=True)

        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=20) as response:
            tags = json.load(response).get('models', [])
        by_name = {item.get('name'): item for item in tags}
        environment = {
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'gpu_nvidia_smi': gpu.stdout.strip() or gpu.stderr.strip(),
            'ollama_version': ollama_version,
            'ollama_context_length': int(os.environ['OLLAMA_CONTEXT_LENGTH']),
            'ollama_max_loaded_models': int(os.environ['OLLAMA_MAX_LOADED_MODELS']),
            'model_tags': MODEL_IDS,
            'model_digests': {name: by_name.get(name, {}) for name in MODEL_IDS},
            'project_package_sha256': hashlib.sha256(PROJECT_ZIP.read_bytes()).hexdigest(),
            'config_sha256': sha256_file(CONFIG_PATH),
            'results_directory': str(RESULTS_ROOT),
            'code_execution_enabled': cfg.code_agent.execution_enabled,
        }
        ENVIRONMENT_PATH.write_text(json.dumps(environment, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        MODEL_DIGESTS = {name: by_name.get(name, {}).get('digest') for name in MODEL_IDS}
        print('Model pulls complete; environment record saved to Drive.')
        """
    ),
    markdown(
        r"""
        ## Run the competence pre-test

        The runner appends and flushes one row after each logical trial. The progress display watches those rows. If the runtime disconnects during inference, rerunning this stage skips the rows already present and continues with missing trial keys.
        """
    ),
    code(
        r"""
        import json
        import subprocess
        import sys
        import time
        from pathlib import Path

        def result_line_count(path):
            return sum(1 for line in path.open(encoding='utf-8') if line.strip()) if path.exists() else 0

        def record_status(item):
            if STATUS_PATH.exists():
                status = json.loads(STATUS_PATH.read_text(encoding='utf-8'))
            else:
                status = {'runs': [], 'code_agent_arm': {'status': 'pending'}}
            key = (item['model'], item['dataset'], item['arm'], item.get('variant'))
            status['runs'] = [
                row for row in status['runs']
                if (row['model'], row['dataset'], row['arm'], row.get('variant')) != key
            ]
            status['runs'].append(item)
            STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n', encoding='utf-8')

        def run_arm(model_id, dataset, arm, variant, output_name):
            cases, dataset_path = __import__('src.run_pilot', fromlist=['load_dataset']).load_dataset(dataset)
            target_dir = RESULTS_ROOT / output_name
            target_dir.mkdir(parents=True, exist_ok=True)
            digest = sha256_file(dataset_path)
            result_path = result_path_for(target_dir, model_id, arm, digest)
            command = [
                sys.executable, '-m', 'src.run_pilot',
                '--config', CONFIG_PATH,
                '--dataset', dataset,
                '--arm', arm,
                '--model', model_id,
                '--provider', 'ollama',
                '--results-dir', str(target_dir),
            ]
            if variant:
                command.extend(['--variant', variant])
            planned = len(cases)
            before = result_line_count(result_path)
            print(f'[{model_id} / {arm} / {variant or "—"}] {before}/{planned} rows already on Drive')
            runner_env = dict(os.environ)
            runner_env['OLLAMA_MODEL_DIGEST'] = MODEL_DIGESTS.get(model_id) or ''
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=runner_env)
            last = -1
            while process.poll() is None:
                current = result_line_count(result_path)
                if current != last:
                    print(f'  rows persisted: {current}/{planned}')
                    last = current
                time.sleep(15)
            output, _ = process.communicate()
            if output:
                print(output.strip())
            if process.returncode != 0:
                raise RuntimeError(f'Runner failed for {model_id} / {arm} / {variant}: {process.returncode}')
            completed = result_line_count(result_path)
            item = {
                'model': model_id, 'dataset': dataset, 'arm': arm, 'variant': variant,
                'planned': planned, 'result_rows': completed,
                'result_path': str(result_path), 'manifest_path': str(result_path.with_suffix('.manifest.json')),
                'status': 'complete' if completed == planned else 'partial',
            }
            record_status(item)
            return item

        for model_id in MODEL_IDS:
            run_arm(model_id, 'competence', 'competence', None, 'competence')
        """
    ),
    markdown(
        r"""
        ## Run both direct prompt variants

        Each variant uses a separate result directory because the existing manifest correctly refuses to mix distinct prompt variants in one result file. Each run includes all salience levels in the frozen pilot dataset.
        """
    ),
    code(
        r"""
        for model_id in MODEL_IDS:
            run_arm(model_id, 'pilot', 'direct', 'natural', 'direct_natural')
            run_arm(model_id, 'pilot', 'direct', 'guided', 'direct_guided')
        """
    ),
    markdown(
        r"""
        ## Run the paired code-agent arm when safe

        The checked-in config leaves code execution disabled. In that setting the code-agent arm is safe to run because the runner collects text only and the grader performs static AST analysis; no model-written Python is executed. If execution is enabled in the config, the notebook skips the code arm because Colab is not configured with a container or VM sandbox for generated code.
        """
    ),
    code(
        r"""
        if cfg.code_agent.execution_enabled:
            code_arm_status = {
                'status': 'skipped',
                'reason': 'code_agent.execution_enabled is true and no isolated container/VM sandbox is configured in this notebook',
            }
        else:
            for model_id in MODEL_IDS:
                run_arm(model_id, 'paired', 'direct', 'natural', 'paired_direct_natural')
                run_arm(model_id, 'paired', 'code-agent', None, 'code_agent')
            code_arm_status = {
                'status': 'ran_static_only',
                'reason': 'execution_enabled is false; generated code was not executed',
            }

        if STATUS_PATH.exists():
            status = json.loads(STATUS_PATH.read_text(encoding='utf-8'))
        else:
            status = {'runs': []}
        status['code_agent_arm'] = code_arm_status
        STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        print('Code-agent arm:', code_arm_status)
        """
    ),
    markdown(
        r"""
        ## Score result files and apply the continuation rule

        This calls the project analysis module, which verifies every manifest and SHA before invoking the existing deterministic graders, metrics, competence gate, and continuation function.
        """
    ),
    code(
        r"""
        subprocess.run([
            sys.executable, '-m', 'src.analyze_results',
            '--config', CONFIG_PATH,
            '--results-dir', str(RESULTS_ROOT),
            '--output-dir', str(ANALYSIS_ROOT),
            '--project-root', str(PROJECT_ROOT),
            '--environment', str(ENVIRONMENT_PATH),
            '--status', str(STATUS_PATH),
        ], check=True)
        print((ANALYSIS_ROOT / 'analysis.md').read_text(encoding='utf-8'))
        """
    ),
    markdown(
        r"""
        ## Package results for download

        The archive contains `data/results` (JSONL plus adjacent manifests), the frozen config and generated datasets used to verify their SHAs, analysis outputs, and runtime metadata. It excludes the project package, `.env`, and caches.
        """
    ),
    code(
        r"""
        from zipfile import ZIP_DEFLATED, ZipFile
        from google.colab import files

        FINAL_ZIP = Path('/content/measurement-invariance-pilot-colab-t4-results.zip')
        with ZipFile(FINAL_ZIP, 'w', compression=ZIP_DEFLATED) as archive:
            for source_root, archive_root in (
                (RESULTS_ROOT, Path('data/results/colab_t4/results')),
                (ANALYSIS_ROOT, Path('analysis')),
            ):
                for path in sorted(source_root.rglob('*')):
                    if path.is_file():
                        archive.write(path, (archive_root / path.relative_to(source_root)).as_posix())
            for relative in (
                Path('configs/pilot_colab_t4.yaml'),
                Path('data/generated/pilot_cases.jsonl'),
                Path('data/generated/competence_cases.jsonl'),
                Path('data/generated/paired_agent_cases.jsonl'),
            ):
                archive.write(PROJECT_ROOT / relative, relative.as_posix())
            for path, name in ((ENVIRONMENT_PATH, 'run_environment.json'), (STATUS_PATH, 'run_status.json')):
                if path.exists():
                    archive.write(path, name)

        print(f'Results archive ready: {FINAL_ZIP} ({FINAL_ZIP.stat().st_size:,} bytes)')
        files.download(str(FINAL_ZIP))
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "run_pilot_t4.ipynb", "provenance": [], "gpuType": "T4", "accelerator": "GPU"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).with_name("run_pilot_t4.ipynb")
output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(output)


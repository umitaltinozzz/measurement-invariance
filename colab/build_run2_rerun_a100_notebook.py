"""Regenerate run2_rerun_a100.ipynb from checked-in cell source."""

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
        # Run-2 repairs on a Colab A100

        This notebook re-runs only the 52 cases listed in `configs/run2_rerun_targets.json`: 28 `gemma4:12b` direct-natural provider failures, 19 `gpt-oss:20b` direct-guided token-capped cases, and 5 `gpt-oss:20b` code-agent token-capped cases. Gemma uses the exact run-2 config. The two gpt-oss arms use the run-2 config with only `max_tokens=4096` and `max_tokens_code_agent=16000`.

        Select **Runtime → Change runtime type → A100 GPU**, then **Run all**. The notebook clones the public GitHub `main` branch and installs requirements from PyPI; GitHub and PyPI are reachable from Colab. Google Drive must mount successfully before any work starts. Ollama 0.40.5 uses context length 8192 and one loaded model, and its model cache is stored at `MyDrive/measurement_invariance_colab_t4/ollama_models`.

        The notebook compares each rerun model digest to the digest recorded in run 2 and stops if either differs. It writes all rerun rows to a new `run2_rerun_a100` Drive folder and builds the replacement analysis at `data/results/colab_t4/run2/analysis_with_reruns/`; the original run-2 analysis is not overwritten.
        """
    ),
    code(
        r"""
        from pathlib import Path
        from google.colab import drive, files
        import os

        try:
            drive.mount('/content/drive')
        except Exception as exc:
            raise RuntimeError(
                'Google Drive mount failed. Reconnect to Colab, authorize Drive, and rerun this cell.'
            ) from exc

        DRIVE_MOUNT = Path('/content/drive')
        MY_DRIVE = DRIVE_MOUNT / 'MyDrive'
        if not MY_DRIVE.is_dir():
            raise RuntimeError(
                'Google Drive mount did not expose /content/drive/MyDrive. Rerun after Drive is mounted.'
            )
        DRIVE_ROOT = MY_DRIVE / 'measurement_invariance_colab_t4'
        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
        RERUN_ROOT = DRIVE_ROOT / 'run2_rerun_a100'
        RERUN_ROOT.mkdir(parents=True, exist_ok=True)
        OLLAMA_MODELS_DIR = DRIVE_ROOT / 'ollama_models'
        OLLAMA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_ROOT = RERUN_ROOT / 'results'
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

        print(f'Drive mounted: {MY_DRIVE}')
        print(f'New rerun results root: {RESULTS_ROOT}')
        print(f'Ollama model cache: {OLLAMA_MODELS_DIR}')
        """
    ),
    code(
        r"""
        assert 'DRIVE_ROOT' in globals() and 'RERUN_ROOT' in globals(), 'Run the Drive mount cell first.'
        import json
        import os
        import subprocess
        import sys

        PROJECT_ROOT = Path('/content/measurement-invariance')
        REPOSITORY_URL = 'https://github.com/umitaltinozzz/measurement-invariance'
        if not (PROJECT_ROOT / '.git').is_dir():
            subprocess.run(
                ['git', 'clone', '--branch', 'main', '--single-branch', REPOSITORY_URL, str(PROJECT_ROOT)],
                check=True,
            )
        else:
            subprocess.run(['git', '-C', str(PROJECT_ROOT), 'checkout', 'main'], check=True)
            subprocess.run(['git', '-C', str(PROJECT_ROOT), 'pull', '--ff-only', 'origin', 'main'], check=True)
        os.chdir(PROJECT_ROOT)
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib'], check=True)

        RUN2_ROOT = PROJECT_ROOT / 'data/results/colab_t4/run2'
        RUN2_ENV_PATH = RUN2_ROOT / 'run_environment.json'
        RUN2_CONFIG_PATH = PROJECT_ROOT / 'configs/pilot_colab_t4_run2.yaml'
        RERUN_CONFIG_PATH = PROJECT_ROOT / 'configs/pilot_colab_t4_run2_rerun.yaml'
        TARGETS_PATH = PROJECT_ROOT / 'configs/run2_rerun_targets.json'
        BASE_RESULTS_ROOT = RUN2_ROOT / 'data/results/colab_t4/results'
        MERGED_ANALYSIS_ROOT = RUN2_ROOT / 'analysis_with_reruns'
        for required_path in (RUN2_ENV_PATH, RUN2_CONFIG_PATH, RERUN_CONFIG_PATH, TARGETS_PATH, BASE_RESULTS_ROOT):
            assert required_path.exists(), f'Required run-2 project file is missing: {required_path}'

        RUN2_ENV = json.loads(RUN2_ENV_PATH.read_text(encoding='utf-8'))
        TARGETS = json.loads(TARGETS_PATH.read_text(encoding='utf-8'))['targets']
        assert [len(item['case_ids']) for item in TARGETS] == [28, 19, 5]
        assert sum(len(item['case_ids']) for item in TARGETS) == 52
        assert RUN2_ENV['ollama_version'].endswith('0.40.5'), RUN2_ENV['ollama_version']
        print('Repository:', subprocess.run(
            ['git', '-C', str(PROJECT_ROOT), 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        print('Run-2 rows scheduled for rerun: 28 + 19 + 5 = 52')
        """
    ),
    markdown(
        r"""
        ## Install the run-2 Ollama version and verify the A100

        This pins Ollama to the recorded 0.40.5 release and sets the run-2 context length and loaded-model limit. Hardware details are saved into each result manifest.
        """
    ),
    code(
        r"""
        assert all(name in globals() for name in (
            'PROJECT_ROOT', 'RUN2_ENV', 'RERUN_ROOT', 'OLLAMA_MODELS_DIR'
        )), 'Run the Drive and repository setup cells first.'
        import os
        import subprocess
        import time
        import urllib.request

        OLLAMA_VERSION_PIN = '0.40.5'
        existing_server = globals().get('OLLAMA_SERVER')
        OLLAMA_SERVER = existing_server if existing_server and existing_server.poll() is None else None
        if OLLAMA_SERVER is None:
            install = subprocess.run(
                ['bash', '-o', 'pipefail', '-c',
                 f'curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VERSION_PIN} sh'],
                text=True,
                check=False,
            )
            if install.returncode != 0:
                raise RuntimeError(f'Ollama {OLLAMA_VERSION_PIN} installer exited with {install.returncode}')
        else:
            print('Reusing the Ollama server started by this notebook.', flush=True)
        OLLAMA_VERSION = subprocess.run(
            ['ollama', '--version'], capture_output=True, text=True, check=True
        ).stdout.strip()
        if not OLLAMA_VERSION.endswith(OLLAMA_VERSION_PIN):
            raise RuntimeError(f'Expected Ollama {OLLAMA_VERSION_PIN}, got {OLLAMA_VERSION}')
        os.environ['OLLAMA_SERVER_VERSION'] = OLLAMA_VERSION

        gpu = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, check=True,
        )
        GPU_INFO = gpu.stdout.strip()
        print('GPU:', GPU_INFO)
        if 'A100' not in GPU_INFO.upper():
            raise RuntimeError(f'This notebook requires a Colab A100 GPU; detected: {GPU_INFO}')
        print('Ollama:', OLLAMA_VERSION)

        OLLAMA_ENV = dict(os.environ)
        OLLAMA_ENV.update({
            'OLLAMA_HOST': '127.0.0.1:11434',
            'OLLAMA_MODELS': str(OLLAMA_MODELS_DIR),
            'OLLAMA_CONTEXT_LENGTH': '8192',
            'OLLAMA_MAX_LOADED_MODELS': '1',
        })
        os.environ.update({key: OLLAMA_ENV[key] for key in (
            'OLLAMA_HOST', 'OLLAMA_MODELS', 'OLLAMA_CONTEXT_LENGTH', 'OLLAMA_MAX_LOADED_MODELS'
        )})
        if OLLAMA_SERVER is None:
            OLLAMA_SERVER = subprocess.Popen(
                ['ollama', 'serve'], env=OLLAMA_ENV,
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
        for _ in range(120):
            try:
                with urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2) as response:
                    response.read()
            except Exception:
                if OLLAMA_SERVER.poll() is not None:
                    raise RuntimeError('Ollama server exited before becoming ready')
                time.sleep(2)
            else:
                if OLLAMA_SERVER.poll() is not None:
                    raise RuntimeError(
                        'The Ollama API answered but this notebook server exited; '
                        'restart the Colab runtime and rerun to use the Drive model cache.'
                    )
                break
        else:
            raise TimeoutError('Ollama server did not become ready on localhost:11434')

        HARDWARE_METADATA = {
            'gpu_nvidia_smi': GPU_INFO,
            'colab_accelerator': 'A100',
            'ollama_version': OLLAMA_VERSION,
            'ollama_context_length': 8192,
            'ollama_max_loaded_models': 1,
            'ollama_models_directory': str(OLLAMA_MODELS_DIR),
        }
        HARDWARE_PATH = RERUN_ROOT / 'hardware.json'
        HARDWARE_PATH.write_text(json.dumps(HARDWARE_METADATA, indent=2) + '\n', encoding='utf-8')
        """
    ),
    code(
        r"""
        assert all(name in globals() for name in (
            'OLLAMA_SERVER', 'HARDWARE_METADATA', 'TARGETS', 'RUN2_ENV', 'RERUN_ROOT'
        )), 'Run the preceding setup cells first.'
        assert OLLAMA_SERVER.poll() is None, 'Start Ollama before caching models.'
        import json
        import os
        import subprocess
        import urllib.request
        from datetime import datetime, timezone
        from src.run_manifest import sha256_file

        MODELS_TO_RERUN = sorted({item['model'] for item in TARGETS})
        EXPECTED_DIGESTS = RUN2_ENV['model_digests']

        def ollama_tags():
            with urllib.request.urlopen('http://localhost:11434/api/tags', timeout=30) as response:
                return json.load(response).get('models', [])

        for model_id in MODELS_TO_RERUN:
            current = {item.get('name'): item for item in ollama_tags()}
            if model_id not in current:
                print(f'Pulling missing model into Drive cache: {model_id}', flush=True)
                subprocess.run(['ollama', 'pull', model_id], check=True)
            else:
                print(f'Using cached model: {model_id}', flush=True)

        tags_by_name = {item.get('name'): item for item in ollama_tags()}
        MODEL_DIGESTS = {name: tags_by_name[name].get('digest') for name in MODELS_TO_RERUN}
        DIGEST_COMPARISON = {}
        for model_id in MODELS_TO_RERUN:
            expected = EXPECTED_DIGESTS[model_id]['digest']
            actual = MODEL_DIGESTS[model_id]
            DIGEST_COMPARISON[model_id] = {
                'run2_digest': expected,
                'rerun_digest': actual,
                'matches_run2': actual == expected,
            }
            print(f'{model_id}: run-2 digest match = {actual == expected}')
        RUN_ENV_PATH = RERUN_ROOT / 'run_environment.json'
        RERUN_ENV = {
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            **HARDWARE_METADATA,
            'ollama_host': 'http://localhost:11434',
            'model_tags': MODELS_TO_RERUN,
            'model_digests': {name: tags_by_name[name] for name in MODELS_TO_RERUN},
            'digest_comparison_with_run2': DIGEST_COMPARISON,
            'run2_environment_path': str(RUN2_ENV_PATH),
            'run2_config_sha256': sha256_file(RUN2_CONFIG_PATH),
            'rerun_config_sha256': sha256_file(RERUN_CONFIG_PATH),
            'targets_path': str(TARGETS_PATH),
            'results_directory': str(RESULTS_ROOT),
            'code_execution_enabled': False,
        }
        RUN_ENV_PATH.write_text(json.dumps(RERUN_ENV, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        print('Run environment and digest comparison saved:', RUN_ENV_PATH)
        if not all(item['matches_run2'] for item in DIGEST_COMPARISON.values()):
            raise RuntimeError(
                'Ollama model digest differs from run 2. No inference was started; '
                'compare run_environment.json before proceeding.'
            )
        """
    ),
    markdown(
        r"""
        ## Run only the diagnosed cases

        Each target gets its own results directory and one-case-ID-per-line selection file. Runner output and row counts are streamed as the trials are written to Drive. Re-running this cell resumes the same three isolated files.
        """
    ),
    code(
        r"""
        assert all(name in globals() for name in (
            'TARGETS', 'RESULTS_ROOT', 'MODEL_DIGESTS', 'HARDWARE_PATH', 'PROJECT_ROOT', 'RERUN_ROOT'
        )), 'Run the target and model verification cells first.'
        import json
        import os
        import subprocess
        import sys
        import threading
        import time
        from src.run_pilot import load_dataset
        from src.run_manifest import sha256_file
        from src.runner import result_path_for

        CASE_ID_ROOT = RERUN_ROOT / 'case_ids'
        CASE_ID_ROOT.mkdir(parents=True, exist_ok=True)
        RUN_STATUS_PATH = RERUN_ROOT / 'run_status.json'
        RUN_STATUS = {'runs': []}

        def count_rows(path):
            if not path.exists():
                return 0
            with path.open(encoding='utf-8') as handle:
                return sum(1 for line in handle if line.strip())

        def stream_output(pipe):
            for line in iter(pipe.readline, ''):
                print('[runner] ' + line.rstrip(), flush=True)
            pipe.close()

        for target in TARGETS:
            ids = target['case_ids']
            assert len(ids) == len(set(ids)), f"Duplicate IDs in {target['model']} target"
            case_file = CASE_ID_ROOT / f"{target['results_subdir']}.txt"
            case_file.write_text('\n'.join(ids) + '\n', encoding='utf-8')
            cases, dataset_path = load_dataset(target['dataset'])
            requested = set(ids)
            assert requested.issubset({case.case_id for case in cases}), 'A target case ID is absent from its dataset.'
            config_path = target['config_path']
            target_dir = RESULTS_ROOT / target['results_subdir']
            target_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_path_for(target_dir, target['model'], target['arm'], sha256_file(dataset_path))
            command = [
                sys.executable, '-m', 'src.run_pilot',
                '--config', config_path,
                '--dataset', target['dataset'],
                '--arm', target['arm'],
                '--model', target['model'],
                '--provider', 'ollama',
                '--results-dir', str(target_dir),
                '--only-case-ids', str(case_file),
                '--hardware-json', str(HARDWARE_PATH),
            ]
            if target['variant']:
                command.extend(['--variant', target['variant']])
            runner_env = dict(os.environ)
            runner_env['OLLAMA_MODEL_DIGEST'] = MODEL_DIGESTS[target['model']]
            before = count_rows(result_path)
            print(
                f"Starting {target['model']} / {target['arm']} / {target['variant'] or '-'}: "
                f"{before}/{len(ids)} rows already persisted; running exactly {len(ids)} selected IDs.",
                flush=True,
            )
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=runner_env,
            )
            output_thread = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True)
            output_thread.start()
            last_count = -1
            while process.poll() is None:
                current = count_rows(result_path)
                if current != last_count:
                    print(f"[progress] {target['results_subdir']}: {current}/{len(ids)} rows on Drive", flush=True)
                    last_count = current
                time.sleep(3)
            output_thread.join()
            if process.returncode != 0:
                raise RuntimeError(f"Runner failed for {target['model']} / {target['arm']} ({process.returncode})")
            completed_rows = [
                json.loads(line) for line in result_path.read_text(encoding='utf-8').splitlines() if line.strip()
            ]
            actual_ids = [row['case_id'] for row in completed_rows]
            if set(actual_ids) != requested or len(actual_ids) != len(requested):
                raise RuntimeError(
                    f"Result rows differ from the exact target set for {target['results_subdir']}: "
                    f"expected={len(requested)}, actual={len(actual_ids)}"
                )
            RUN_STATUS['runs'].append({
                'model': target['model'], 'dataset': target['dataset'],
                'arm': target['arm'], 'variant': target['variant'],
                'planned': len(ids), 'result_rows': len(completed_rows),
                'result_path': str(result_path),
                'manifest_path': str(result_path.with_suffix('.manifest.json')),
                'config_path': config_path, 'hardware': GPU_INFO, 'status': 'complete',
            })
            RUN_STATUS_PATH.write_text(
                json.dumps(RUN_STATUS, indent=2, sort_keys=True) + '\n', encoding='utf-8'
            )
        assert [item['result_rows'] for item in RUN_STATUS['runs']] == [28, 19, 5]
        print('Rerun rows complete: Gemma direct-natural 28; gpt-oss direct-guided 19; gpt-oss code-agent 5.')
        """
    ),
    markdown(
        r"""
        ## Build the run-2 plus reruns analysis

        This validates the original provider failures and token caps, the three A100 manifests, the config relationship, and every replaced prompt fingerprint. It preserves the run-2 row count and invokes the existing graders, metrics, and continuation gates without changing their definitions.
        """
    ),
    code(
        r"""
        assert all(name in globals() for name in (
            'RUN_STATUS', 'RUN_ENV_PATH', 'PROJECT_ROOT', 'BASE_RESULTS_ROOT', 'MERGED_ANALYSIS_ROOT',
            'RUN2_CONFIG_PATH', 'TARGETS_PATH', 'RUN2_ENV_PATH', 'RESULTS_ROOT'
        )), 'Run the preceding setup cells first.'
        assert len(RUN_STATUS['runs']) == 3, 'Complete all three rerun targets first.'
        assert RUN_ENV_PATH.exists(), 'Save the rerun environment first.'
        import shutil
        import subprocess
        import sys

        subprocess.run([
            sys.executable, '-m', 'src.analyze_with_reruns',
            '--project-root', str(PROJECT_ROOT),
            '--base-results-dir', str(BASE_RESULTS_ROOT),
            '--rerun-results-dir', str(RESULTS_ROOT),
            '--output-dir', str(MERGED_ANALYSIS_ROOT),
            '--base-config', str(RUN2_CONFIG_PATH),
            '--targets', str(TARGETS_PATH),
            '--base-environment', str(RUN2_ENV_PATH),
            '--rerun-environment', str(RUN_ENV_PATH),
            '--base-status', str(RUN2_ROOT / 'run_status.json'),
        ], check=True)
        DRIVE_ANALYSIS_ROOT = RERUN_ROOT / 'analysis_with_reruns'
        shutil.copytree(MERGED_ANALYSIS_ROOT, DRIVE_ANALYSIS_ROOT, dirs_exist_ok=True)
        print((MERGED_ANALYSIS_ROOT / 'analysis.md').read_text(encoding='utf-8'))
        print('Durable merged analysis copy:', DRIVE_ANALYSIS_ROOT)
        """
    ),
    markdown(
        r"""
        ## Zip and download the rerun results

        The archive contains the three new JSONL result files and manifests, selected case-ID files, run status and environment records, the unchanged and rerun configs, and the merged analysis in its requested repository path.
        """
    ),
    code(
        r"""
        assert all(name in globals() for name in (
            'DRIVE_ANALYSIS_ROOT', 'RUN_STATUS', 'RESULTS_ROOT', 'CASE_ID_ROOT', 'RUN_ENV_PATH',
            'RUN_STATUS_PATH', 'HARDWARE_PATH', 'RUN2_CONFIG_PATH', 'RERUN_CONFIG_PATH',
            'TARGETS_PATH', 'MERGED_ANALYSIS_ROOT', 'RERUN_ROOT'
        )), 'Run the rerun and analysis cells first.'
        assert DRIVE_ANALYSIS_ROOT.exists(), 'Build the merged analysis first.'
        assert len(RUN_STATUS['runs']) == 3, 'All three rerun targets must be complete before downloading.'
        from zipfile import ZIP_DEFLATED, ZipFile

        FINAL_ZIP = RERUN_ROOT / 'measurement-invariance-run2-rerun-a100.zip'
        with ZipFile(FINAL_ZIP, 'w', compression=ZIP_DEFLATED) as archive:
            for path in sorted(RESULTS_ROOT.rglob('*')):
                if path.is_file():
                    archive.write(path, (Path('run2_rerun_a100/results') / path.relative_to(RESULTS_ROOT)).as_posix())
            for path in sorted(CASE_ID_ROOT.rglob('*')):
                if path.is_file():
                    archive.write(path, (Path('run2_rerun_a100/case_ids') / path.relative_to(CASE_ID_ROOT)).as_posix())
            for path, name in (
                (RUN_ENV_PATH, 'run2_rerun_a100/run_environment.json'),
                (RUN2_ENV_PATH, 'run2/run_environment.json'),
                (RUN_STATUS_PATH, 'run2_rerun_a100/run_status.json'),
                (HARDWARE_PATH, 'run2_rerun_a100/hardware.json'),
                (RUN2_CONFIG_PATH, 'configs/pilot_colab_t4_run2.yaml'),
                (RERUN_CONFIG_PATH, 'configs/pilot_colab_t4_run2_rerun.yaml'),
                (TARGETS_PATH, 'configs/run2_rerun_targets.json'),
            ):
                archive.write(path, name)
            for path in sorted(MERGED_ANALYSIS_ROOT.rglob('*')):
                if path.is_file():
                    archive.write(
                        path,
                        (Path('data/results/colab_t4/run2/analysis_with_reruns')
                         / path.relative_to(MERGED_ANALYSIS_ROOT)).as_posix(),
                    )

        print(f'Results archive: {FINAL_ZIP} ({FINAL_ZIP.stat().st_size:,} bytes)')
        files.download(str(FINAL_ZIP))
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {
            "name": "run2_rerun_a100.ipynb",
            "provenance": [],
            "gpuType": "A100",
            "accelerator": "GPU",
        },
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).with_name("run2_rerun_a100.ipynb")
output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(output)

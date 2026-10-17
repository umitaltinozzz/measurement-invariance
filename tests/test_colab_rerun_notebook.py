import ast
import json
from pathlib import Path


def test_a100_rerun_notebook_is_a_valid_colab_notebook_with_guards():
    path = Path("colab/run2_rerun_a100.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    sources = ["".join(cell["source"]) for cell in code_cells]
    for source in sources:
        ast.parse(source)
    assert notebook["metadata"]["colab"]["gpuType"] == "A100"
    assert "drive.mount" in sources[0]
    assert "Drive mount failed" in sources[0]
    assert all("assert " in source for source in sources[1:])
    assert "['git', 'clone', '--branch', 'main'" in sources[1]
    assert "files.download" in sources[-1]


def test_a100_notebook_runs_exact_targets_and_keeps_outputs_isolated():
    notebook = json.loads(Path("colab/run2_rerun_a100.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "'measurement_invariance_colab_t4'" in text
    assert "DRIVE_ROOT / 'run2_rerun_a100'" in text
    assert "DRIVE_ROOT / 'ollama_models'" in text
    assert "--only-case-ids" in text
    assert "--hardware-json" in text
    assert "OLLAMA_VERSION_PIN = '0.40.5'" in text
    assert "http://localhost:11434/api/version" in text
    assert "OLLAMA_VERSION = subprocess.run" not in text
    assert "ollama', '--version'" not in text
    assert "ollama_version_source" in text
    assert "matches_run2" in text
    assert "analysis_with_reruns" in text
    assert "--rerun-results-dir" in text
    assert "Runtime → Change runtime type → A100 GPU" in text


def test_rerun_notebook_reads_server_version_only_after_readiness_check():
    notebook = json.loads(Path("colab/run2_rerun_a100.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    ready = text.index("http://localhost:11434/api/tags")
    version = text.index("http://localhost:11434/api/version")
    source_marker = text.index("OLLAMA_VERSION_SOURCE = 'GET /api/version after server readiness'")
    save_metadata = text.index("'ollama_version_source': OLLAMA_VERSION_SOURCE")
    assert ready < version < source_marker < save_metadata
    assert "if OLLAMA_API_VERSION != OLLAMA_VERSION_PIN" in text
    assert "the server version cannot be verified" in text

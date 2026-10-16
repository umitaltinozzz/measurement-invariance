"""Execution interface for model-written analysis code.

**This is not a security boundary.** ``SubprocessSandbox`` gives you a separate
process, a scratch working directory, a wall-clock timeout, a trimmed
environment and capped output. It does not stop the code from reading your
filesystem, opening sockets, or spawning children. Python has no in-process
mechanism that does, and pretending otherwise would be worse than saying so.

For a real pilot run, execute inside a container or VM with no network and a
read-only mount, and point ``execution_enabled`` at that. The protocol below
exists so that swapping in such a backend is a constructor change and nothing
else.

Execution is **off by default** (``code_agent.execution_enabled: false``). The
provenance analysis in ``code_agent_grader.py`` is static, so the whole
code-agent arm can be graded without ever running a line of generated code;
execution only adds ``stdout``/``stderr`` and a cross-check on the declared
answer.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

#: Captured streams are truncated at this many characters before storage.
MAX_CAPTURED_CHARS = 20_000


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    executed: bool
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: Optional[float] = None
    #: Why execution did not happen, when ``executed`` is False.
    skipped_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.executed and self.exit_code == 0 and not self.timed_out


@runtime_checkable
class Sandbox(Protocol):
    """Runs one snippet against one CSV and reports what happened."""

    name: str

    def run(self, code: str, files: Dict[str, str], timeout: float) -> ExecutionResult:
        ...


class DisabledSandbox:
    """Refuses to execute. The default, and what the test suite uses.

    Returning a well-formed "did not run" result rather than raising keeps the
    grading path identical whether or not execution is enabled.
    """

    name = "disabled"

    def __init__(self, reason: str = "execution disabled by configuration"):
        self.reason = reason

    def run(self, code: str, files: Dict[str, str], timeout: float) -> ExecutionResult:
        return ExecutionResult(executed=False, skipped_reason=self.reason)


class ReplaySandbox:
    """Serves recorded stdout/stderr for a snippet, keyed by the code itself.

    Lets the fixtures exercise the execution-dependent parts of grading -- a
    crashing script, a script that prints the wrong thing -- without running
    anything.
    """

    name = "replay"

    def __init__(self, recordings: Dict[str, ExecutionResult], *, strict: bool = False):
        self._recordings = dict(recordings)
        self._strict = strict
        self.calls: list[str] = []

    def run(self, code: str, files: Dict[str, str], timeout: float) -> ExecutionResult:
        self.calls.append(code)
        if code in self._recordings:
            return self._recordings[code]
        if self._strict:
            raise KeyError("no recorded execution for this snippet")
        return ExecutionResult(executed=False, skipped_reason="no recording for this snippet")


class SubprocessSandbox:
    """Separate process, scratch directory, timeout, capped output.

    Read the module docstring before enabling this. Process isolation only.
    """

    name = "subprocess"

    def __init__(
        self,
        *,
        python_executable: Optional[str] = None,
        env_allowlist: tuple = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG", "PYTHONHASHSEED"),
        acknowledge_not_a_security_boundary: bool = False,
    ):
        if not acknowledge_not_a_security_boundary:
            raise RuntimeError(
                "SubprocessSandbox executes model-written code with your own user's "
                "privileges. Run it inside a container or VM, and pass "
                "acknowledge_not_a_security_boundary=True to confirm you have."
            )
        self.python = python_executable or sys.executable
        self.env_allowlist = env_allowlist

    def run(self, code: str, files: Dict[str, str], timeout: float) -> ExecutionResult:
        import time

        with tempfile.TemporaryDirectory(prefix="mi-sandbox-") as workdir:
            root = Path(workdir)
            for name, content in files.items():
                target = root / name
                if not target.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"input file {name!r} escapes the sandbox directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            script = root / "analysis.py"
            script.write_text(code, encoding="utf-8")

            env = {k: v for k, v in os.environ.items() if k in self.env_allowlist}
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            started = time.perf_counter()
            try:
                proc = subprocess.run(
                    [self.python, "-I", "-S", str(script)],
                    cwd=str(root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(
                    executed=True,
                    timed_out=True,
                    stdout=(exc.stdout or "")[:MAX_CAPTURED_CHARS]
                    if isinstance(exc.stdout, str)
                    else "",
                    stderr="timed out",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            return ExecutionResult(
                executed=True,
                exit_code=proc.returncode,
                stdout=proc.stdout[:MAX_CAPTURED_CHARS],
                stderr=proc.stderr[:MAX_CAPTURED_CHARS],
                duration_ms=(time.perf_counter() - started) * 1000,
            )

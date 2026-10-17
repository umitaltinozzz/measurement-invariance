"""Execution interface for model-written analysis code.

The legacy ``SubprocessSandbox`` is not a security boundary. It gives you a separate
process, a scratch working directory, a wall-clock timeout, a trimmed
environment and capped output. It does not stop the code from reading your
filesystem, opening sockets, or spawning children. Python has no in-process
mechanism that does, and pretending otherwise would be worse than saying so.

``DockerSandbox`` is the run-3 execution backend: a network-disabled Linux
container invoked by subprocess with a read-only temporary input directory,
bounded CPU, memory, processes, output and wall time. It never falls back to
host execution. See docs/preregistration_run3.md for the output protocol.

Execution is **off by default** (``code_agent.execution_enabled: false``). The
provenance analysis in ``code_agent_grader.py`` is static, so the whole
code-agent arm can be graded without ever running a line of generated code;
run 3 can instead select the strictly parsed stdout answer as the primary grade,
while retaining both declared and executed grades.
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
    sandbox_name: Optional[str] = None
    image_id: Optional[str] = None
    output_limit_exceeded: bool = False

    @property
    def ok(self) -> bool:
        return self.executed and self.exit_code == 0 and not self.timed_out and not self.output_limit_exceeded


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


class DockerSandbox:
    """Network-disabled Linux container, invoked as a bounded subprocess.

    Never falls back to host execution or pulls an image implicitly. Only the
    generated program and case CSV are mounted, read-only; /tmp is a size-limited
    tmpfs. The immutable image ID is recorded with every execution.
    """

    name = "docker-no-network"

    def __init__(self, image="mi-run3-sandbox:1", *, memory_mb=512, cpu_seconds=10, pids_limit=32):
        if min(memory_mb, cpu_seconds, pids_limit) <= 0:
            raise ValueError("resource limits must be positive")
        result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                                capture_output=True, text=True, timeout=15, check=False)
        if result.returncode or not result.stdout.strip().startswith("sha256:"):
            raise RuntimeError("sandbox image unavailable; build it before execution")
        self.image_id = result.stdout.strip()
        self.memory_mb, self.cpu_seconds, self.pids_limit = memory_mb, cpu_seconds, pids_limit

    def command(self, root, name):
        return [
            "docker", "run", "--rm", "--pull=never", "--name", name,
            "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--user=65534:65534",
            f"--memory={self.memory_mb}m", f"--memory-swap={self.memory_mb}m",
            "--cpus=1", f"--pids-limit={self.pids_limit}",
            "--ulimit", f"cpu={self.cpu_seconds}:{self.cpu_seconds}",
            "--ulimit", "nofile=64:64", "--ulimit", "fsize=1048576:1048576",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m,mode=1777",
            "--mount", f"type=bind,source={root},target=/input,readonly",
            "--workdir=/input", "--env=PYTHONDONTWRITEBYTECODE=1",
            "--env=OPENBLAS_NUM_THREADS=1", "--log-driver=none",
            self.image_id, "python", "-I", "-B", "/input/analysis.py",
        ]

    def run(self, code, files, timeout):
        import threading
        import time
        import uuid

        if timeout <= 0:
            raise ValueError("timeout must be positive")
        with tempfile.TemporaryDirectory(prefix="mi-run3-") as workdir:
            root = Path(workdir).resolve()
            root.chmod(0o755)
            for name, content in {**files, "analysis.py": code}.items():
                if name != Path(name).name or name in (".", ".."):
                    raise ValueError("sandbox accepts plain filenames only")
                target = root / name
                target.write_text(content, encoding="utf-8", newline="\n")
                target.chmod(0o644)
            container = "mi-run3-" + uuid.uuid4().hex
            started = time.perf_counter()
            output = {"stdout": bytearray(), "stderr": bytearray()}
            over = threading.Event()
            proc = subprocess.Popen(self.command(root, container), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)

            def drain(stream, destination):
                while data := stream.read(1024):
                    remaining = MAX_CAPTURED_CHARS - len(destination)
                    destination.extend(data[:max(0, remaining)])
                    if len(data) > remaining:
                        over.set()
                stream.close()

            threads = [threading.Thread(target=drain, args=(getattr(proc, k), output[k]), daemon=True)
                       for k in output]
            for thread in threads:
                thread.start()
            timed_out = False
            try:
                while proc.poll() is None and not over.is_set():
                    if time.perf_counter() - started >= timeout:
                        timed_out = True
                        break
                    over.wait(0.02)
            finally:
                # Kill the container too: killing only the Docker client leaves
                # a compute process alive in the daemon after a timeout.
                if proc.poll() is None:
                    try:
                        subprocess.run(["docker", "rm", "-f", container], capture_output=True,
                                       timeout=10, check=False)
                    finally:
                        proc.kill()
                proc.wait(timeout=10)
                for thread in threads:
                    thread.join(timeout=5)
            return ExecutionResult(
                executed=True, exit_code=proc.returncode, timed_out=timed_out,
                stdout=output["stdout"].decode("utf-8", errors="replace"),
                stderr=output["stderr"].decode("utf-8", errors="replace"),
                output_limit_exceeded=over.is_set(), sandbox_name=self.name,
                image_id=self.image_id, duration_ms=(time.perf_counter() - started) * 1000,
            )


def sandbox_from_config(cfg):
    if not cfg.execution_enabled:
        return DisabledSandbox()
    return DockerSandbox(cfg.sandbox_image, memory_mb=cfg.memory_mb,
                         cpu_seconds=cfg.cpu_seconds, pids_limit=cfg.pids_limit)

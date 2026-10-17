"""Trial orchestration: retry policy, trial accounting, idempotent resume.

Three invariants, and every one of them is a place where a bug silently corrupts
a denominator rather than raising.

**One logical trial is one row.**  The trial key is
``(case_id, model, arm, prompt_variant)``.  Transport retries stay inside that
row and increment ``attempts``; they never produce a second row.  If they did,
"we asked once and the network failed twice" would enter the benchmark as three
observations.

**Content failures are not retried.**  A completion that comes back but does not
parse is an observation of the model, and it is kept. Re-rolling it until it
parses would replace a measurement of the model with a measurement of how many
attempts we were willing to buy -- and would bias exactly the metric this project
reports, since a model whose failures correlate with formatting would look
cleaner than one whose failures do not.

**Resume never re-asks.**  Existing rows are loaded by trial key and skipped. The
result path carries the dataset SHA and every row repeats it, so a result written
against a different dataset can never be reused for this one.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .generate_cases import RunnerConfig
from .models.base import (
    GenerationRequest,
    ModelProvider,
    ProviderError,
    RawModelResult,
    TransportError,
    TransportErrorRecord,
)
from .run_manifest import RunManifest, write_manifest
from .sandbox import ExecutionResult

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class Arm(str):
    """Which part of the study a trial belongs to. Plain strings, so a new arm
    does not require touching this module."""

    DIRECT = "direct"
    COMPETENCE = "competence"
    CODE_AGENT = "code-agent"


def safe_component(value: str) -> str:
    """Model ids carry '/' and ':'; result paths must not."""
    return _UNSAFE.sub("-", value).strip("-") or "unnamed"


def result_path_for(
    results_dir: str | Path, model: str, arm: str, dataset_sha256: str
) -> Path:
    return Path(results_dir) / (
        f"{safe_component(model)}__{safe_component(arm)}__{dataset_sha256[:16]}.jsonl"
    )


class TrialKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    model: str
    arm: str
    prompt_variant: Optional[str] = None

    def as_tuple(self) -> Tuple[str, str, str, Optional[str]]:
        return (self.case_id, self.model, self.arm, self.prompt_variant)


class TrialRecord(BaseModel):
    """One logical trial: exactly one row in the result file."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    model: str
    arm: str
    prompt_variant: Optional[str] = None

    dataset_sha256: str
    prompt_fingerprint: str
    run_id: str

    result: Optional[RawModelResult] = None
    #: Set when the trial could not produce a completion at all -- retries
    #: exhausted, or a non-retryable provider error. Distinct from a completion
    #: that came back and failed to parse, which lands in ``result``.
    failure: Optional[str] = None
    failure_kind: Optional[str] = None
    execution: Optional[ExecutionResult] = None

    @property
    def key(self) -> TrialKey:
        return TrialKey(
            case_id=self.case_id,
            model=self.model,
            arm=self.arm,
            prompt_variant=self.prompt_variant,
        )

    @property
    def produced_a_completion(self) -> bool:
        return self.result is not None

    @property
    def raw_text(self) -> str:
        return self.result.raw_text if self.result else ""


class RunSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    model: str
    arm: str
    result_path: str
    manifest_path: str

    planned: int
    skipped_existing: int
    executed: int
    completed: int
    failed: int
    provider_calls: int
    transport_retries: int


class TrialPlan(BaseModel):
    """One unit of work: a case, the prompt it produces, and its key."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    case_id: str
    request: GenerationRequest
    prompt_variant: Optional[str] = None
    execution_files: Optional[Dict[str, str]] = None


def _sleep_noop(_seconds: float) -> None:
    """Default sleeper. Tests inject a no-op so backoff costs no wall clock."""
    time.sleep(_seconds)


class Runner:
    """Executes trial plans against a provider, with retries and resume."""

    def __init__(
        self,
        provider: ModelProvider,
        config: RunnerConfig,
        *,
        sleep: Callable[[float], None] = _sleep_noop,
        sandbox=None,
        execution_timeout: float = 20.0,
        code_protocol: str = "v1",
    ):
        self.provider = provider
        self.config = config
        self._sleep = sleep
        self.provider_calls = 0
        self.transport_retries = 0
        self.sandbox = sandbox
        self.execution_timeout = execution_timeout
        self.code_protocol = code_protocol

    # ------------------------------------------------------------------ retry
    def _backoff_seconds(self, attempt: int) -> float:
        delay = self.config.backoff_initial_seconds * (
            self.config.backoff_multiplier ** (attempt - 1)
        )
        return min(delay, self.config.backoff_max_seconds)

    def generate_once(self, request: GenerationRequest) -> RawModelResult:
        """One logical trial. Transport failures are retried in place.

        Returns a result whose ``attempts`` counts every call spent on this trial.
        Raises only when no completion could be obtained at all.
        """
        errors: List[TransportErrorRecord] = []
        attempt = 0
        while True:
            attempt += 1
            self.provider_calls += 1
            try:
                result = self.provider.generate(request)
            except TransportError as exc:
                if attempt > self.config.max_transport_retries:
                    errors.append(
                        TransportErrorRecord(
                            attempt=attempt,
                            kind=getattr(exc, "kind", "transport"),
                            status_code=getattr(exc, "status_code", None),
                            message=str(exc),
                            backoff_seconds=0.0,
                        )
                    )
                    raise TransportError(
                        f"gave up after {attempt} attempts: {exc}",
                        status_code=getattr(exc, "status_code", None),
                    ) from exc
                delay = self._backoff_seconds(attempt)
                errors.append(
                    TransportErrorRecord(
                        attempt=attempt,
                        kind=getattr(exc, "kind", "transport"),
                        status_code=getattr(exc, "status_code", None),
                        message=str(exc),
                        backoff_seconds=delay,
                    )
                )
                self.transport_retries += 1
                self._sleep(delay)
                continue
            # A completion that does not parse is NOT retried here; it is the
            # grader's business, and re-rolling it would bias the metric.
            return result.model_copy(update={"attempts": attempt, "transport_errors": errors})

    # -------------------------------------------------------------- execution
    def run(
        self,
        plans: Sequence[TrialPlan],
        *,
        model: str,
        arm: str,
        dataset_sha256: str,
        manifest: RunManifest,
        results_dir: str | Path = "data/results",
        result_path: str | Path | None = None,
        resume: bool = True,
    ) -> Tuple[List[TrialRecord], RunSummary]:
        path = Path(result_path) if result_path else result_path_for(
            results_dir, model, arm, dataset_sha256
        )
        manifest_path = write_manifest(manifest, path)

        if self.config.reject_capped_slices and not resume and path.exists():
            raise ValueError("run 3 cannot append a fresh invocation to existing results; use resume")

        existing = load_results(path, expected_dataset_sha256=dataset_sha256) if resume else []
        if self.config.reject_capped_slices and any(r.result and r.result.truncated for r in existing):
            raise ValueError("capped slice: archive it and rerun the WHOLE slice in stage-1; never replace selected rows")
        if self.config.retry_provider_failures_on_resume:
            from collections import Counter
            audit = path.with_suffix(".provider_failures.jsonl")
            prior = Counter(r.key.as_tuple() for r in load_results(audit, expected_dataset_sha256=dataset_sha256))
            failures = [r for r in existing if r.failure and r.result is None
                        and prior[r.key.as_tuple()] < self.config.max_provider_resume_attempts]
            if failures:
                # Preserve the original evidence before atomically replacing the
                # active file. A crash then resumes only the still-missing keys.
                with audit.open("a", encoding="utf-8", newline="\n") as fh:
                    for record in failures:
                        fh.write(record.model_dump_json() + "\n")
                existing = [r for r in existing if r not in failures]
                replacement = path.with_suffix(".resume.tmp")
                replacement.write_text("".join(r.model_dump_json() + "\n" for r in existing),
                                       encoding="utf-8", newline="\n")
                replacement.replace(path)
        done = {r.key.as_tuple() for r in existing}

        records: List[TrialRecord] = list(existing)
        executed = completed = failed = 0

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            for plan in plans:
                key = TrialKey(
                    case_id=plan.case_id,
                    model=model,
                    arm=arm,
                    prompt_variant=plan.prompt_variant,
                ).as_tuple()
                if key in done:
                    continue

                executed += 1
                base = dict(
                    case_id=plan.case_id,
                    model=model,
                    arm=arm,
                    prompt_variant=plan.prompt_variant,
                    dataset_sha256=dataset_sha256,
                    prompt_fingerprint=plan.request.prompt_fingerprint,
                    run_id=manifest.run_id,
                )
                try:
                    result = self.generate_once(plan.request)
                    execution = None
                    if self.sandbox and plan.execution_files is not None:
                        from .code_agent_grader import parse_code_agent_response
                        from .correction_classifier import parse_code_response_v2
                        parse = parse_code_response_v2 if self.code_protocol == "v2" else parse_code_agent_response
                        parsed = parse(result.raw_text)
                        if parsed.ok:
                            import subprocess
                            try:
                                execution = self.sandbox.run(parsed.response.code, plan.execution_files,
                                                             self.execution_timeout)
                            except (OSError, subprocess.SubprocessError) as exc:
                                # Keep the completion even if the execution infrastructure fails.
                                execution = ExecutionResult(executed=False,
                                    skipped_reason=f"sandbox infrastructure error: {exc}")
                    record = TrialRecord(**base, result=result, execution=execution)
                    completed += 1
                except (TransportError, ProviderError) as exc:
                    record = TrialRecord(
                        **base,
                        failure=str(exc),
                        failure_kind=type(exc).__name__,
                    )
                    failed += 1

                fh.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
                fh.flush()
                records.append(record)
                done.add(key)

        return records, RunSummary(
            run_id=manifest.run_id,
            model=model,
            arm=arm,
            result_path=str(path.as_posix()),
            manifest_path=str(manifest_path.as_posix()),
            planned=len(plans),
            skipped_existing=len(plans) - executed,
            executed=executed,
            completed=completed,
            failed=failed,
            provider_calls=self.provider_calls,
            transport_retries=self.transport_retries,
        )


def load_results(
    path: str | Path, expected_dataset_sha256: Optional[str] = None
) -> List[TrialRecord]:
    """Read a result file, refusing rows written against a different dataset.

    Silently reusing them would let a resume answer questions from the wrong
    dataset -- the failure mode this project is named after, applied to itself.
    """
    file = Path(path)
    if not file.exists():
        return []
    records: List[TrialRecord] = []
    with open(file, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            record = TrialRecord(**json.loads(line))
            if (
                expected_dataset_sha256 is not None
                and record.dataset_sha256 != expected_dataset_sha256
            ):
                raise ValueError(
                    f"{file.name}:{line_no} was produced against dataset "
                    f"{record.dataset_sha256[:16]}, not {expected_dataset_sha256[:16]}; "
                    "results from a different dataset must not be reused"
                )
            records.append(record)
    return records


def new_run_id() -> str:
    return uuid.uuid4().hex[:16]

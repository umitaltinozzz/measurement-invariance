"""Provider-agnostic model interface.

The contract is deliberately narrow: a provider turns one prompt into one raw
completion and reports what it observed doing so.  It does no retrying, no
parsing, no grading and no bookkeeping -- those belong to ``src/runner.py``, so
that retry policy and trial accounting are identical across providers rather than
re-implemented (and re-diverged) once per SDK.

The one thing every provider MUST get right is the error taxonomy:

``TransportError``
    The request did not produce an answer for reasons unrelated to the model's
    reasoning: timeout, connection reset, 429, 5xx. Retryable. Retries stay
    inside the *same logical trial* -- they must never inflate the benchmark's
    denominator, because "the network failed twice" is not "we asked three times".

``ProviderError``
    A request that will fail the same way if repeated: bad credentials, unknown
    model id, malformed request, content filter. Not retryable.

A completion that comes back but does not parse is **neither**. It is a valid
observation of the model's behaviour, it is returned normally, and the grader
records it as ``invalid_response``. Retrying it would replace a measurement of
the model with a measurement of how many attempts we were willing to buy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class TransportError(RuntimeError):
    """Retryable: timeout, connection failure, 429, 5xx."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, kind: str = "transport"):
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind


class ProviderError(RuntimeError):
    """Not retryable: auth, unknown model, malformed request, content filter."""

    def __init__(self, message: str, *, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class GenerationRequest(BaseModel):
    """Everything that determines a completion, and nothing that does not."""

    model_config = ConfigDict(frozen=True)

    model: str
    system: str
    user: str
    #: JSON schema the provider may use for constrained decoding. Advisory: a
    #: provider that cannot honour it must leave ``structured_output_used`` False
    #: rather than silently ignoring the request.
    response_schema: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = 0.0
    max_tokens: int = 1024
    seed: Optional[int] = None
    structured_output: bool = False
    think: bool | Literal["low", "medium", "high"] | None = None
    reasoning: Optional[Dict[str, Any]] = None
    context_length: Optional[int] = None

    @property
    def prompt_fingerprint(self) -> str:
        """Stable hash of the prompt text, recorded so a result can be tied back
        to the exact bytes that produced it."""
        import hashlib

        digest = hashlib.sha256()
        digest.update(self.system.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(self.user.encode("utf-8"))
        return digest.hexdigest()


class TransportErrorRecord(BaseModel):
    """One retried failure, kept so retry behaviour is auditable after the fact."""

    model_config = ConfigDict(frozen=True)

    attempt: int
    kind: str
    status_code: Optional[int] = None
    message: str
    backoff_seconds: float


class RawModelResult(BaseModel):
    """One completion plus everything needed to interpret or reproduce it."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    #: What we asked for versus what the provider says it served. These differ
    #: whenever an alias resolves to a dated snapshot, and only the resolved id
    #: identifies the weights a result actually came from.
    requested_model_id: str
    resolved_model_id: str

    raw_text: str
    thinking: Optional[str] = None
    reasoning_details: Optional[List[Dict[str, Any]]] = None
    structured_output_used: bool

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    finish_reason: Optional[str] = None
    provider_request_id: Optional[str] = None

    #: Total attempts spent on this ONE logical trial, transport retries included.
    attempts: int = 1
    transport_errors: List[TransportErrorRecord] = Field(default_factory=list)

    @property
    def truncated(self) -> bool:
        """A length-truncated completion will usually fail to parse; knowing that
        it was truncated separates 'ran out of tokens' from 'ignored the schema'."""
        return self.finish_reason in ("length", "max_tokens")


@runtime_checkable
class ModelProvider(Protocol):
    """What ``src/runner.py`` needs from any backend.

    Implementations raise ``TransportError`` / ``ProviderError`` per the taxonomy
    above and otherwise return a ``RawModelResult`` with ``attempts=1``; the
    runner owns retrying and rewrites ``attempts`` on the way out.
    """

    name: str

    def generate(self, request: GenerationRequest) -> RawModelResult:
        ...

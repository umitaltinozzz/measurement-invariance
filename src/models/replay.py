"""Providers that answer without a network: replay, scripted, and static.

These are not conveniences. The runner owns retry policy, trial accounting and
resume, and every one of those is a place where a bug silently corrupts a
denominator. They have to be tested against a backend whose behaviour is exactly
known, including its failures -- which a real API cannot provide on demand and
cannot provide for free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

from .base import (
    GenerationRequest,
    ProviderError,
    RawModelResult,
    TransportError,
)


class ReplayProvider:
    """Serves pre-recorded completions keyed by ``(model, prompt_fingerprint)``.

    Replaying by prompt fingerprint rather than by case id is deliberate: if a
    prompt template changes, the fingerprint changes and the replay misses, so a
    stale recording surfaces as a loud error instead of quietly answering a
    question that is no longer being asked.
    """

    name = "replay"

    def __init__(
        self,
        recordings: Dict[tuple, str],
        *,
        resolved_model_ids: Optional[Dict[str, str]] = None,
        strict: bool = True,
        default_text: Optional[str] = None,
    ):
        self._recordings = dict(recordings)
        self._resolved = dict(resolved_model_ids or {})
        self._strict = strict
        self._default_text = default_text
        self.calls: List[GenerationRequest] = []

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict],
        **kwargs,
    ) -> "ReplayProvider":
        """Build from rows carrying ``model``, ``prompt_fingerprint``, ``raw_text``."""
        return cls(
            {(r["model"], r["prompt_fingerprint"]): r["raw_text"] for r in records}, **kwargs
        )

    @classmethod
    def from_jsonl(cls, path: Union[str, Path], **kwargs) -> "ReplayProvider":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_records((json.loads(line) for line in fh if line.strip()), **kwargs)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def generate(self, request: GenerationRequest) -> RawModelResult:
        self.calls.append(request)
        key = (request.model, request.prompt_fingerprint)
        if key in self._recordings:
            text = self._recordings[key]
        elif self._default_text is not None:
            text = self._default_text
        elif self._strict:
            raise ProviderError(
                f"no recording for model={request.model!r} prompt="
                f"{request.prompt_fingerprint[:12]}...; the prompt may have changed since "
                "the recording was made"
            )
        else:
            text = ""
        return RawModelResult(
            provider=self.name,
            model=request.model,
            requested_model_id=request.model,
            resolved_model_id=self._resolved.get(request.model, request.model),
            raw_text=text,
            structured_output_used=False,
            input_tokens=len(request.system) + len(request.user),
            output_tokens=len(text),
            latency_ms=0.0,
            finish_reason="stop",
            provider_request_id=f"replay-{request.prompt_fingerprint[:16]}",
        )


class ScriptedProvider:
    """Plays a fixed script of outcomes, so retry behaviour can be tested exactly.

    Each script entry is either a string (a completion) or an exception instance
    to raise. The script is consumed per ``(model, prompt_fingerprint)``, so one
    logical trial can be scripted to fail twice and then succeed.
    """

    name = "scripted"

    def __init__(
        self,
        script: Dict[tuple, Sequence[Union[str, BaseException]]],
        *,
        resolved_model_ids: Optional[Dict[str, str]] = None,
    ):
        self._script = {k: list(v) for k, v in script.items()}
        self._resolved = dict(resolved_model_ids or {})
        self.calls: List[GenerationRequest] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def generate(self, request: GenerationRequest) -> RawModelResult:
        self.calls.append(request)
        key = (request.model, request.prompt_fingerprint)
        queue = self._script.get(key)
        if not queue:
            raise ProviderError(f"script exhausted for {key[0]!r}")
        step = queue.pop(0)
        if isinstance(step, BaseException):
            raise step
        return RawModelResult(
            provider=self.name,
            model=request.model,
            requested_model_id=request.model,
            resolved_model_id=self._resolved.get(request.model, request.model),
            raw_text=step,
            structured_output_used=request.structured_output,
            input_tokens=0,
            output_tokens=len(step),
            latency_ms=0.0,
            finish_reason="stop",
            provider_request_id=f"scripted-{len(self.calls)}",
        )


class CallableProvider:
    """Answers by calling a function of the request. For fixtures that need to
    look at the case rather than replay a fixed string."""

    name = "callable"

    def __init__(self, fn: Callable[[GenerationRequest], str], *, name: Optional[str] = None):
        self._fn = fn
        self.calls: List[GenerationRequest] = []
        if name:
            self.name = name

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def generate(self, request: GenerationRequest) -> RawModelResult:
        self.calls.append(request)
        text = self._fn(request)
        return RawModelResult(
            provider=self.name,
            model=request.model,
            requested_model_id=request.model,
            resolved_model_id=request.model,
            raw_text=text,
            structured_output_used=request.structured_output,
            output_tokens=len(text),
            latency_ms=0.0,
            finish_reason="stop",
        )


def transport_failure(message: str = "upstream timeout", status_code: int = 504) -> TransportError:
    """Convenience for scripts."""
    return TransportError(message, status_code=status_code)


def provider_failure(message: str = "unknown model", status_code: int = 404) -> ProviderError:
    return ProviderError(message, status_code=status_code)

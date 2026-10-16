"""Shared implementation for OpenAI-compatible chat-completions backends.

Most gateways speak the same wire format, so the parts that matter to this
project -- the error taxonomy, the response fields the manifest depends on, and
the refusal to retry a content failure -- belong in one place rather than being
re-implemented and re-diverged once per backend.

Subclasses supply three things: a base URL, an environment variable name, and
whatever routing block the backend uses to pin the upstream. Everything else is
here.

**Error taxonomy.** This is the part that silently corrupts denominators if it is
wrong, so it is explicit and tested:

* 408 / 409 / 425 / 429 / 5xx and any connection or read timeout ->
  ``TransportError``. Retried in place by the runner; the trial count does not move.
* 400 / 401 / 402 / 403 / 404 / 422 -> ``ProviderError``. Not retried: a bad key,
  an unknown model or an empty balance fails identically on the next attempt.
* A completion that arrives and does not parse is **neither**. It is returned
  normally and graded as ``invalid_response``.

No third-party HTTP dependency: ``urllib`` is enough for one POST, and it keeps
the reproducibility surface at pydantic + pyyaml.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import GenerationRequest, ProviderError, RawModelResult, TransportError

#: Worth another attempt: the request never reached a model, or the upstream is
#: momentarily unavailable.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: Will fail the same way next time. 402 (out of credit) belongs here: retrying
#: an empty balance just burns quota.
FATAL_STATUS = frozenset({400, 401, 402, 403, 404, 405, 413, 422})


class OpenAICompatibleProvider:
    """One prompt in, one ``RawModelResult`` out. No retrying, no grading."""

    name = "openai-compatible"
    DEFAULT_BASE_URL = ""
    ENV_KEY = ""
    #: Used only in the "you forgot the key" message.
    DISPLAY_NAME = "provider"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = 180.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        key = api_key or os.environ.get(self.ENV_KEY)
        if not key:
            raise ProviderError(
                f"no {self.DISPLAY_NAME} API key: pass api_key or set {self.ENV_KEY}. "
                "Do not hardcode it into a source file."
            )
        self.api_key = key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.calls = 0

    # ------------------------------------------------------- subclass hooks
    def routing_block(self) -> Optional[Dict[str, Any]]:
        """Backend-specific upstream pinning, or None if it has none."""
        return None

    @property
    def routing_constraints(self) -> Dict[str, Any]:
        """What this provider pins. Recorded in the manifest."""
        return {}

    # ------------------------------------------------------------------ wire
    def build_payload(self, request: GenerationRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.structured_output and request.response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        routing = self.routing_block()
        if routing:
            payload.update(routing)
        return payload

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

    def _post(self, payload: Dict[str, Any]) -> Tuple[int, str]:
        """One HTTP POST. Overridden in tests so the mapping can be exercised
        without a network or a budget."""
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise TransportError(f"connection failed: {exc.reason}", kind="connection") from exc
        except TimeoutError as exc:
            raise TransportError(f"timed out after {self.timeout}s", kind="timeout") from exc

    # -------------------------------------------------------------- response
    @staticmethod
    def _error_message(status: int, body: str) -> str:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return f"HTTP {status}: {body[:300]}"
        error = payload.get("error")
        if isinstance(error, dict):
            return f"HTTP {status}: {error.get('message', body[:300])}"
        return f"HTTP {status}: {str(error or body)[:300]}"

    def _raise_for_status(self, status: int, body: str) -> None:
        if status in RETRYABLE_STATUS:
            raise TransportError(
                self._error_message(status, body), status_code=status, kind=f"http-{status}"
            )
        if status in FATAL_STATUS or status >= 400:
            raise ProviderError(self._error_message(status, body), status_code=status)

    def upstream_from(self, payload: Dict[str, Any]) -> Optional[str]:
        """Which backend actually served the request, if the gateway says so.

        Subclasses override where the field is known. Returning None is honest:
        it means the measurement chain has a link this project cannot audit, and
        that belongs in the paper's limitations rather than being guessed at.
        """
        return None

    def parse_response(
        self, request: GenerationRequest, status: int, body: str, latency_ms: float
    ) -> RawModelResult:
        self._raise_for_status(status, body)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            # A 200 whose body is not JSON is a broken response, not a model
            # answer; treating it as a completion would let transport noise enter
            # the benchmark as an invalid_response.
            raise TransportError(f"malformed response body: {exc.msg}", kind="malformed") from exc

        error = payload.get("error")
        if error:
            code = error.get("code") if isinstance(error, dict) else None
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            if isinstance(code, int) and code in RETRYABLE_STATUS:
                raise TransportError(f"upstream error {code}: {message}", status_code=code)
            raise ProviderError(f"upstream error {code}: {message}", status_code=code)

        choices: List[Dict[str, Any]] = payload.get("choices") or []
        if not choices:
            raise TransportError("response contained no choices", kind="empty")
        message = choices[0].get("message") or {}
        content = message.get("content")
        # A model that emitted only reasoning leaves content null. That is an
        # empty completion, not a transport failure: the grader records it as
        # invalid_response, which is the honest reading.
        raw_text = content if isinstance(content, str) else ""

        usage = payload.get("usage") or {}
        upstream = self.upstream_from(payload)
        return RawModelResult(
            provider=f"{self.name}/{upstream}" if upstream else self.name,
            model=request.model,
            requested_model_id=request.model,
            # The only field that identifies the weights behind a result.
            resolved_model_id=payload.get("model") or request.model,
            raw_text=raw_text,
            structured_output_used=bool(
                request.structured_output and request.response_schema
            ),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            finish_reason=choices[0].get("finish_reason"),
            provider_request_id=payload.get("id"),
        )

    # ------------------------------------------------------------- interface
    def generate(self, request: GenerationRequest) -> RawModelResult:
        self.calls += 1
        started = time.perf_counter()
        status, body = self._post(self.build_payload(request))
        latency_ms = (time.perf_counter() - started) * 1000
        return self.parse_response(request, status, body, latency_ms)


def load_dotenv(path: str = ".env") -> Dict[str, str]:
    """Minimal ``KEY=value`` reader, so a key lives in a gitignored file rather
    than in a source file or a shell history."""
    values: Dict[str, str] = {}
    if not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            values[key.strip()] = value
            os.environ.setdefault(key.strip(), value)
    return values

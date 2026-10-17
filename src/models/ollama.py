"""Local Ollama provider over its native chat API.

Ollama's local server does not require an API key. The provider therefore
inherits the shared request/response and error mapping while omitting the
Authorization header entirely.
"""

from __future__ import annotations

import os
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .base import GenerationRequest, ProviderError, RawModelResult, TransportError
from .openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """One local Ollama chat completion per logical trial."""

    name = "ollama"
    DEFAULT_BASE_URL = "http://localhost:11434"
    DISPLAY_NAME = "Ollama"

    def build_payload(self, request):
        """Build Ollama's native chat payload so its ``think`` control is explicit."""
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        if request.seed is not None:
            payload["options"]["seed"] = request.seed
        if request.structured_output and request.response_schema:
            payload["format"] = request.response_schema
        return payload

    def _post(self, payload: Dict[str, Any]):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat", data=body,
            headers=self._headers(), method="POST",
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

    def parse_response(self, request: GenerationRequest, status: int, body: str, latency_ms: float) -> RawModelResult:
        self._raise_for_status(status, body)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise TransportError(f"malformed response body: {exc.msg}", kind="malformed") from exc
        if payload.get("error"):
            error = payload["error"]
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise ProviderError(f"upstream error: {message}")
        message = payload.get("message") or {}
        usage_tokens = payload.get("prompt_eval_count")
        output_tokens = payload.get("eval_count")
        return RawModelResult(
            provider=self.name,
            model=request.model,
            requested_model_id=request.model,
            resolved_model_id=payload.get("model") or request.model,
            raw_text=message.get("content") or "",
            structured_output_used=bool(request.structured_output and request.response_schema),
            input_tokens=usage_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            finish_reason=payload.get("done_reason"),
            provider_request_id=None,
        )

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: float = 180.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = None
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL") or self.DEFAULT_BASE_URL
        ).rstrip("/").removesuffix("/v1")
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.calls = 0

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self.extra_headers)
        return headers

    @property
    def routing_constraints(self) -> Dict[str, Any]:
        """Local settings that identify the inference endpoint and weights."""
        constraints: Dict[str, Any] = {
            "base_url": self.base_url,
            "authentication": "none",
        }
        context_length = os.environ.get("OLLAMA_CONTEXT_LENGTH")
        if context_length:
            constraints["context_length"] = int(context_length)
        max_loaded_models = os.environ.get("OLLAMA_MAX_LOADED_MODELS")
        if max_loaded_models:
            constraints["max_loaded_models"] = int(max_loaded_models)
        for env_name, field_name in (
            ("OLLAMA_SERVER_VERSION", "server_version"),
            ("OLLAMA_SERVER_VERSION_SOURCE", "server_version_source"),
            ("OLLAMA_MODEL_DIGEST", "model_digest"),
        ):
            value = os.environ.get(env_name)
            if value:
                constraints[field_name] = value
        return constraints


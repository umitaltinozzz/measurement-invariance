"""Local Ollama provider over its OpenAI-compatible API.

Ollama's local server does not require an API key. The provider therefore
inherits the shared request/response and error mapping while omitting the
Authorization header entirely.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """One local Ollama chat completion per logical trial."""

    name = "ollama"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    DISPLAY_NAME = "Ollama"

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
        ).rstrip("/")
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
            ("OLLAMA_MODEL_DIGEST", "model_digest"),
        ):
            value = os.environ.get(env_name)
            if value:
                constraints[field_name] = value
        return constraints


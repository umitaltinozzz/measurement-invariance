"""OpenRouter provider.

OpenRouter is an aggregator, and an aggregator is a layer between the experiment
and the model.  For a project about measurement chains that is a validity threat,
so this provider pins the chain down rather than accepting whatever routing it
gets:

``provider.only``               the upstream is fixed, not chosen at request time
``provider.allow_fallbacks``    False -- a request never silently lands elsewhere
``provider.require_parameters`` True -- the request only reaches an upstream that
                                actually honours temperature, seed and
                                response_format, so the manifest cannot record a
                                sampling setting that was never applied

All three go into the manifest via ``RunnerConfig``, so a run records the
constraints it actually ran under. This is the strictest pinning of any gateway
tried for this project, which is why OpenRouter remains the fallback choice.

The wire format, error taxonomy and response handling live in
``openai_compatible.py``; only the routing block is OpenRouter-specific.

Note that OpenRouter's free tier counts *failed* attempts against the daily
quota, so on a free key ``max_transport_retries`` should be small.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .openai_compatible import (  # re-exported: imported from here by callers and tests
    FATAL_STATUS,
    RETRYABLE_STATUS,
    OpenAICompatibleProvider,
    load_dotenv,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
ENV_KEY = "OPENROUTER_API_KEY"

__all__ = [
    "OpenRouterProvider",
    "RETRYABLE_STATUS",
    "FATAL_STATUS",
    "load_dotenv",
    "DEFAULT_BASE_URL",
    "ENV_KEY",
]


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"
    DEFAULT_BASE_URL = DEFAULT_BASE_URL
    ENV_KEY = ENV_KEY
    DISPLAY_NAME = "OpenRouter"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 180.0,
        provider_only: Optional[Sequence[str]] = None,
        allow_fallbacks: bool = False,
        require_parameters: bool = True,
        referer: Optional[str] = None,
        title: Optional[str] = "measurement-invariance-pilot",
    ):
        extra: Dict[str, str] = {}
        if referer:
            extra["HTTP-Referer"] = referer
        if title:
            extra["X-Title"] = title
        super().__init__(api_key, base_url=base_url, timeout=timeout, extra_headers=extra)
        self.provider_only = list(provider_only) if provider_only else None
        self.allow_fallbacks = allow_fallbacks
        self.require_parameters = require_parameters
        self.referer = referer
        self.title = title

    @property
    def routing_constraints(self) -> Dict[str, Any]:
        return {
            "provider_only": self.provider_only,
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
        }

    def routing_block(self) -> Dict[str, Any]:
        routing: Dict[str, Any] = {"allow_fallbacks": self.allow_fallbacks}
        if self.provider_only:
            routing["only"] = self.provider_only
        if self.require_parameters:
            routing["require_parameters"] = True
        return {"provider": routing}

    def upstream_from(self, payload: Dict[str, Any]) -> Optional[str]:
        return payload.get("provider")

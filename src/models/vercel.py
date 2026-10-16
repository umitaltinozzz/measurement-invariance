"""Vercel AI Gateway provider.

Chosen for the pilot because it is the only gateway tried that serves frontier
models -- Anthropic, OpenAI, Google -- on a free tier, and the pre-registered
kill criterion is stated in terms of a frontier model.

**The routing caveat, which is a real one.** Vercel advertises load balancing and
managed fallback: exactly the behaviour this project does not want, because a
request that silently lands on a different upstream is an unaudited link in the
measurement chain. ``only`` pins the provider order and ``allow_fallbacks=False``
asks for that behaviour to be switched off; ``upstream_from`` reads back whichever
field the gateway reports so a run can be checked rather than assumed. Where the
gateway does not report it, ``resolved_model_id`` is still recorded and the gap
belongs in the paper's limitations.

**Do not buy credits.** The free allowance is per-team and refreshes every 30
days, but purchasing moves the account to the paid tier and the monthly free
credit stops. For a pilot of this size the recurring free allowance is worth more
than a one-off top-up.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Sequence

from .openai_compatible import OpenAICompatibleProvider

DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/v1"
ENV_KEY = "VERCEL_AI_GATEWAY_API_KEY"


class VercelAIGatewayProvider(OpenAICompatibleProvider):
    name = "vercel"
    DEFAULT_BASE_URL = DEFAULT_BASE_URL
    ENV_KEY = ENV_KEY
    DISPLAY_NAME = "Vercel AI Gateway"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 180.0,
        provider_only: Optional[Sequence[str]] = None,
        allow_fallbacks: bool = False,
    ):
        super().__init__(api_key, base_url=base_url, timeout=timeout)
        self.provider_only = list(provider_only) if provider_only else None
        self.allow_fallbacks = allow_fallbacks

    @property
    def routing_constraints(self) -> Dict[str, Any]:
        return {
            "provider_only": self.provider_only,
            "allow_fallbacks": self.allow_fallbacks,
        }

    def routing_block(self) -> Optional[Dict[str, Any]]:
        """Ask the gateway to stop routing around. Omitted entirely when there is
        nothing to say, so an unrecognised field cannot break a request."""
        options: Dict[str, Any] = {}
        if self.provider_only:
            options["only"] = self.provider_only
        if not self.allow_fallbacks:
            options["order"] = self.provider_only or []
        if not options:
            return None
        return {"providerOptions": {"gateway": options}}

    def upstream_from(self, payload: Dict[str, Any]) -> Optional[str]:
        """Read back the serving upstream if the gateway reports one."""
        for key in ("provider", "providerMetadata", "provider_metadata"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for inner in ("gateway", "provider", "name"):
                    found = value.get(inner)
                    if isinstance(found, str):
                        return found
        return None

    # ------------------------------------------------------------- accounting
    def credits(self) -> Dict[str, Any]:
        """Remaining gateway balance. Used to hold a run to a budget rather than
        discovering the ceiling by hitting it."""
        req = urllib.request.Request(
            f"{self.base_url}/credits", headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return {"error": f"HTTP {exc.code}", "body": exc.read().decode("utf-8", "replace")[:200]}

"""Provider-agnostic model backends."""

from .base import (
    GenerationRequest,
    ModelProvider,
    ProviderError,
    RawModelResult,
    TransportError,
    TransportErrorRecord,
)
from .registry import available_providers, create_provider, register_provider
from .openai_compatible import OpenAICompatibleProvider, load_dotenv
from .openrouter import OpenRouterProvider
from .replay import CallableProvider, ReplayProvider, ScriptedProvider
from .vercel import VercelAIGatewayProvider

__all__ = [
    "GenerationRequest",
    "ModelProvider",
    "ProviderError",
    "RawModelResult",
    "TransportError",
    "TransportErrorRecord",
    "available_providers",
    "create_provider",
    "register_provider",
    "OpenAICompatibleProvider",
    "OpenRouterProvider",
    "VercelAIGatewayProvider",
    "load_dotenv",
    "CallableProvider",
    "ReplayProvider",
    "ScriptedProvider",
]

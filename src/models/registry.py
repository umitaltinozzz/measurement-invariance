"""Provider registry.

Providers are looked up by name so that a run manifest can record *which backend
produced a result* as data, and so that adding a real SDK later is a registration
rather than an edit to the runner.

No paid provider is registered here yet. When one is added it registers itself
the same way the replay providers do, and the runner does not change.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from .base import ModelProvider

_FACTORIES: Dict[str, Callable[..., ModelProvider]] = {}


def register_provider(name: str, factory: Callable[..., ModelProvider]) -> None:
    if name in _FACTORIES:
        raise ValueError(f"provider {name!r} is already registered")
    _FACTORIES[name] = factory


def create_provider(name: str, **kwargs) -> ModelProvider:
    if name not in _FACTORIES:
        raise KeyError(
            f"unknown provider {name!r}; registered: {available_providers()}"
        )
    return _FACTORIES[name](**kwargs)


def available_providers() -> List[str]:
    return sorted(_FACTORIES)


def _register_builtin() -> None:
    from .openrouter import OpenRouterProvider
    from .ollama import OllamaProvider
    from .replay import CallableProvider, ReplayProvider, ScriptedProvider
    from .vercel import VercelAIGatewayProvider

    for name, cls in (
        ("replay", ReplayProvider),
        ("scripted", ScriptedProvider),
        ("callable", CallableProvider),
        ("openrouter", OpenRouterProvider),
        ("ollama", OllamaProvider),
        ("vercel", VercelAIGatewayProvider),
    ):
        if name not in _FACTORIES:
            _FACTORIES[name] = cls


_register_builtin()

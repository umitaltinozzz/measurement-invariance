"""Ollama provider tests use the offline HTTP seam from the shared provider."""

from __future__ import annotations

import json

from src.models.base import GenerationRequest
from src.models.ollama import OllamaProvider
from src.models.registry import available_providers, create_provider
from src.structured_schemas import NATURAL_RESPONSE_SCHEMA


def test_ollama_needs_no_api_key_and_sends_no_authorization(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    provider = OllamaProvider()
    assert provider.api_key is None
    assert provider.base_url == "http://localhost:11434"
    assert "Authorization" not in provider._headers()


def test_base_url_can_be_overridden_without_auth(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434/v1/")
    provider = OllamaProvider()
    assert provider.base_url == "http://ollama:11434"
    assert "Authorization" not in provider._headers()


def test_inference_settings_are_exposed_for_the_manifest(monkeypatch):
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "8192")
    monkeypatch.setenv("OLLAMA_MAX_LOADED_MODELS", "1")
    monkeypatch.setenv("OLLAMA_SERVER_VERSION", "ollama-local")
    monkeypatch.setenv("OLLAMA_MODEL_DIGEST", "sha256:abc")
    provider = OllamaProvider()
    assert provider.routing_constraints == {
        "base_url": "http://localhost:11434",
        "authentication": "none",
        "context_length": 8192,
        "max_loaded_models": 1,
        "server_version": "ollama-local",
        "model_digest": "sha256:abc",
    }


def test_registered_and_json_schema_payload_is_sent():
    assert "ollama" in available_providers()
    provider = create_provider("ollama")
    sent = []

    def fake_post(payload):
        sent.append(payload)
        body = {
            "model": "llama3.1:8b-instruct-q4_K_M",
            "message": {"content": '{"trend":"stable","reason":"r"}'},
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 8,
        }
        return 200, json.dumps(body)

    provider._post = fake_post
    request = GenerationRequest(
        model="llama3.1:8b-instruct-q4_K_M",
        system="sys",
        user="usr",
        structured_output=True,
        response_schema=NATURAL_RESPONSE_SCHEMA,
    )
    result = provider.generate(request)

    assert sent[0]["think"] is False
    assert sent[0]["format"] == NATURAL_RESPONSE_SCHEMA
    assert result.provider == "ollama"
    assert result.structured_output_used is True
    assert result.resolved_model_id == "llama3.1:8b-instruct-q4_K_M"


def test_structured_output_is_not_claimed_without_a_schema():
    provider = OllamaProvider()
    provider._post = lambda _payload: (
        200,
        json.dumps(
            {
                "message": {"content": "{}"},
                "done_reason": "stop",
            }
        ),
    )
    result = provider.generate(
        GenerationRequest(model="m", system="sys", user="usr", structured_output=True)
    )
    assert result.structured_output_used is False


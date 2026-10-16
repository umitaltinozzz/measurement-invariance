"""OpenRouter provider tests -- offline.

The error taxonomy is the load-bearing part: a status mapped to the wrong class
either retries something that will never succeed (burning quota, and on the free
tier failed attempts count) or gives up on something transient and records a
trial as failed. Both corrupt the denominator quietly, so every status is pinned.

No network here. ``_post`` is injected, which is why these tests can assert the
mapping exhaustively without a budget.
"""

from __future__ import annotations

import json

import pytest

from src.models.base import GenerationRequest, ProviderError, TransportError
from src.models.openrouter import (
    FATAL_STATUS,
    RETRYABLE_STATUS,
    OpenRouterProvider,
)
from src.models.registry import available_providers, create_provider


def make_provider(status=200, body=None, **kwargs):
    provider = OpenRouterProvider(api_key="test-key", **kwargs)
    payloads = []

    def fake_post(payload):
        payloads.append(payload)
        return status, body if isinstance(body, str) else json.dumps(body or ok_body())

    provider._post = fake_post
    provider.sent = payloads
    return provider


def ok_body(content='{"trend": "stable", "reason": "r"}', **overrides):
    body = {
        "id": "gen-abc123",
        "model": "anthropic/claude-sonnet-5-20260101",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 40},
    }
    body.update(overrides)
    return body


def request(**kwargs):
    defaults = dict(model="anthropic/claude-sonnet-5", system="sys", user="usr")
    defaults.update(kwargs)
    return GenerationRequest(**defaults)


# ==================================================================== payload
class TestPayload:
    def test_system_and_user_become_two_messages(self):
        provider = make_provider()
        provider.generate(request())
        messages = provider.sent[0]["messages"]
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == "sys" and messages[1]["content"] == "usr"

    def test_sampling_parameters_are_sent(self):
        provider = make_provider()
        provider.generate(request(temperature=0.0, max_tokens=1024, seed=7))
        sent = provider.sent[0]
        assert sent["temperature"] == 0.0
        assert sent["max_tokens"] == 1024
        assert sent["seed"] == 7

    def test_seed_is_omitted_when_absent_rather_than_sent_as_null(self):
        provider = make_provider()
        provider.generate(request(seed=None))
        assert "seed" not in provider.sent[0]

    def test_routing_is_pinned_by_default(self):
        """The reason this provider exists: an aggregator that can silently
        reroute is an unlogged layer in the measurement chain."""
        provider = make_provider()
        provider.generate(request())
        routing = provider.sent[0]["provider"]
        assert routing["allow_fallbacks"] is False
        assert routing["require_parameters"] is True

    def test_upstream_can_be_pinned_explicitly(self):
        provider = make_provider(provider_only=["anthropic"])
        provider.generate(request())
        assert provider.sent[0]["provider"]["only"] == ["anthropic"]

    def test_require_parameters_can_be_relaxed(self):
        provider = make_provider(require_parameters=False)
        provider.generate(request())
        assert "require_parameters" not in provider.sent[0]["provider"]

    def test_routing_constraints_are_exposed_for_the_manifest(self):
        provider = make_provider(provider_only=["anthropic"])
        assert provider.routing_constraints == {
            "provider_only": ["anthropic"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }

    def test_structured_output_is_sent_only_when_asked_and_schema_given(self):
        schema = {"type": "object", "properties": {"trend": {"type": "string"}}}
        with_schema = make_provider()
        with_schema.generate(request(structured_output=True, response_schema=schema))
        fmt = with_schema.sent[0]["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["strict"] is True
        assert fmt["json_schema"]["schema"] == schema

        without = make_provider()
        without.generate(request(structured_output=True))
        assert "response_format" not in without.sent[0]

        off = make_provider()
        off.generate(request(structured_output=False, response_schema=schema))
        assert "response_format" not in off.sent[0]


# =================================================================== response
class TestResponse:
    def test_every_field_the_pipeline_depends_on_is_captured(self):
        result = make_provider().generate(request())
        assert result.raw_text == '{"trend": "stable", "reason": "r"}'
        assert result.resolved_model_id == "anthropic/claude-sonnet-5-20260101"
        assert result.requested_model_id == "anthropic/claude-sonnet-5"
        assert result.provider_request_id == "gen-abc123"
        assert result.input_tokens == 900
        assert result.output_tokens == 40
        assert result.finish_reason == "stop"
        assert result.latency_ms is not None
        assert result.attempts == 1

    def test_resolved_model_id_is_the_one_that_served_not_the_one_asked_for(self):
        """An alias resolving to a dated snapshot is the only thing that
        identifies the weights behind a result."""
        result = make_provider().generate(request())
        assert result.resolved_model_id != result.requested_model_id

    def test_missing_model_field_falls_back_to_the_requested_id(self):
        body = ok_body()
        del body["model"]
        result = make_provider(body=body).generate(request())
        assert result.resolved_model_id == "anthropic/claude-sonnet-5"

    def test_truncation_is_visible(self):
        body = ok_body()
        body["choices"][0]["finish_reason"] = "length"
        result = make_provider(body=body).generate(request())
        assert result.finish_reason == "length"
        assert result.truncated is True

    def test_null_content_becomes_an_empty_completion_not_an_error(self):
        """A reasoning-only reply is an observation of the model; the grader
        records it as invalid_response, which is the honest reading."""
        result = make_provider(body=ok_body(content=None)).generate(request())
        assert result.raw_text == ""

    def test_unparseable_completion_is_returned_not_raised(self):
        result = make_provider(body=ok_body(content="I think it went up.")).generate(request())
        assert result.raw_text == "I think it went up."

    def test_structured_output_used_reflects_the_request(self):
        schema = {"type": "object"}
        used = make_provider().generate(request(structured_output=True, response_schema=schema))
        assert used.structured_output_used is True
        assert make_provider().generate(request()).structured_output_used is False


# ============================================================= error taxonomy
class TestErrorTaxonomy:
    @pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
    def test_retryable_statuses_raise_transport_error(self, status):
        provider = make_provider(status=status, body={"error": {"message": "upstream busy"}})
        with pytest.raises(TransportError) as exc:
            provider.generate(request())
        assert exc.value.status_code == status

    @pytest.mark.parametrize("status", sorted(FATAL_STATUS))
    def test_fatal_statuses_raise_provider_error(self, status):
        provider = make_provider(status=status, body={"error": {"message": "bad request"}})
        with pytest.raises(ProviderError) as exc:
            provider.generate(request())
        assert exc.value.status_code == status

    def test_the_two_sets_do_not_overlap(self):
        assert not RETRYABLE_STATUS & FATAL_STATUS

    def test_429_is_retryable_and_402_is_not(self):
        """Observed live: 429 is a saturated upstream and worth another attempt;
        402 is an empty balance and retrying it only burns quota."""
        assert 429 in RETRYABLE_STATUS and 429 not in FATAL_STATUS
        assert 402 in FATAL_STATUS and 402 not in RETRYABLE_STATUS

    def test_error_message_is_extracted_from_the_body(self):
        provider = make_provider(status=400, body={"error": {"message": "not a valid model ID"}})
        with pytest.raises(ProviderError, match="not a valid model ID"):
            provider.generate(request())

    def test_non_json_error_body_still_produces_a_message(self):
        provider = make_provider(status=502, body="<html>gateway</html>")
        with pytest.raises(TransportError, match="gateway"):
            provider.generate(request())

    def test_an_error_inside_a_200_is_still_classified(self):
        """OpenRouter can report an upstream failure with a 200."""
        retryable = make_provider(body={"error": {"code": 429, "message": "busy"}})
        with pytest.raises(TransportError, match="upstream error 429"):
            retryable.generate(request())

        fatal = make_provider(body={"error": {"code": 400, "message": "bad model"}})
        with pytest.raises(ProviderError, match="upstream error 400"):
            fatal.generate(request())

    def test_malformed_body_is_transport_not_a_completion(self):
        """Transport noise must not enter the benchmark as an invalid_response."""
        provider = make_provider(body="{not json")
        with pytest.raises(TransportError, match="malformed"):
            provider.generate(request())

    def test_empty_choices_is_transport(self):
        provider = make_provider(body=ok_body(choices=[]))
        with pytest.raises(TransportError, match="no choices"):
            provider.generate(request())


# ================================================================== plumbing
class TestConstruction:
    def test_missing_key_raises_rather_than_sending_an_unauthenticated_request(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="no OpenRouter API key"):
            OpenRouterProvider()

    def test_key_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        assert OpenRouterProvider().api_key == "from-env"

    def test_auth_header_is_set(self):
        headers = OpenRouterProvider(api_key="k")._headers()
        assert headers["Authorization"] == "Bearer k"

    def test_registered_in_the_registry(self):
        assert "openrouter" in available_providers()
        assert isinstance(create_provider("openrouter", api_key="k"), OpenRouterProvider)

    def test_call_count_is_tracked(self):
        provider = make_provider()
        for _ in range(3):
            provider.generate(request())
        assert provider.calls == 3


# ============================================ integration with the runner
class TestRunnerIntegration:
    def test_transport_errors_are_retried_as_one_trial(self, tmp_path):
        from src.generate_cases import load_config
        from src.runner import Runner

        cfg = load_config("configs/pilot.yaml")
        provider = OpenRouterProvider(api_key="k")
        attempts = {"n": 0}

        def flaky(payload):
            attempts["n"] += 1
            if attempts["n"] < 3:
                return 429, json.dumps({"error": {"message": "busy"}})
            return 200, json.dumps(ok_body())

        provider._post = flaky
        runner = Runner(provider, cfg.runner, sleep=lambda s: None)
        result = runner.generate_once(request())

        assert result.attempts == 3
        assert len(result.transport_errors) == 2
        assert result.raw_text == '{"trend": "stable", "reason": "r"}'

    def test_provider_errors_are_not_retried(self):
        from src.generate_cases import load_config
        from src.runner import Runner

        cfg = load_config("configs/pilot.yaml")
        provider = make_provider(status=401, body={"error": {"message": "no auth"}})
        runner = Runner(provider, cfg.runner, sleep=lambda s: None)
        with pytest.raises(ProviderError):
            runner.generate_once(request())
        assert provider.calls == 1

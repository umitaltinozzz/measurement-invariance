"""Runner tests: retry policy, trial accounting, resume, manifests.

Each of these is a place where a bug corrupts a denominator instead of raising,
so they are tested against providers whose behaviour is exactly scripted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.generate_cases import RunnerConfig, load_config
from src.models import (
    CallableProvider,
    GenerationRequest,
    ProviderError,
    ReplayProvider,
    ScriptedProvider,
    TransportError,
    available_providers,
    create_provider,
)
from src.models.replay import provider_failure, transport_failure
from src.run_manifest import (
    ManifestConflict,
    RunManifest,
    build_manifest,
    manifest_path_for,
    read_manifest,
    sha256_file,
    sha256_text,
    write_manifest,
)
from src.runner import (
    Arm,
    Runner,
    TrialPlan,
    load_results,
    new_run_id,
    result_path_for,
    safe_component,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"
DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "generated" / "pilot_cases.jsonl"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture
def runner_config(cfg):
    # No wall-clock cost in tests; backoff arithmetic is asserted directly.
    return cfg.runner


def request_for(case_id: str, model: str = "m") -> GenerationRequest:
    return GenerationRequest(model=model, system="sys", user=f"user-{case_id}")


def plans(n: int, variant: str | None = "natural"):
    return [
        TrialPlan(case_id=f"c{i:03d}", request=request_for(f"c{i:03d}"), prompt_variant=variant)
        for i in range(n)
    ]


def manifest_for(tmp_path, cfg, **overrides) -> RunManifest:
    dataset = tmp_path / "dataset.jsonl"
    if not dataset.exists():
        dataset.write_text('{"case_id": "c000"}\n', encoding="utf-8")
    kwargs = dict(
        run_id="run0",
        arm=Arm.DIRECT,
        prompt_variant="natural",
        provider="scripted",
        model="m",
        dataset_path=dataset,
        config_path=CONFIG_PATH,
        temperature=cfg.runner.temperature,
        max_tokens=cfg.runner.max_tokens,
        seed=cfg.runner.seed,
        structured_output=cfg.runner.structured_output,
        max_transport_retries=cfg.runner.max_transport_retries,
        now=datetime(2026, 8, 24, tzinfo=timezone.utc),
    )
    kwargs.update(overrides)
    return build_manifest(**kwargs)


def no_sleep(_seconds: float) -> None:
    return None


# ================================================================ retry policy
class TestTransportRetry:
    def test_transport_error_is_retried_and_stays_one_trial(self, runner_config):
        key = ("m", request_for("c000").prompt_fingerprint)
        provider = ScriptedProvider(
            {key: [transport_failure(), transport_failure(), '{"trend": "stable"}']}
        )
        runner = Runner(provider, runner_config, sleep=no_sleep)
        result = runner.generate_once(request_for("c000"))

        assert result.raw_text == '{"trend": "stable"}'
        assert result.attempts == 3
        assert len(result.transport_errors) == 2
        assert provider.call_count == 3

    def test_transport_retry_does_not_increase_the_trial_count(self, tmp_path, cfg, runner_config):
        plan_list = plans(3)
        script = {}
        for plan in plan_list:
            script[("m", plan.request.prompt_fingerprint)] = [
                transport_failure(),
                '{"trend": "stable", "reason": "r"}',
            ]
        provider = ScriptedProvider(script)
        runner = Runner(provider, runner_config, sleep=no_sleep)
        records, summary = runner.run(
            plan_list,
            model="m",
            arm=Arm.DIRECT,
            dataset_sha256="a" * 64,
            manifest=manifest_for(tmp_path, cfg),
            results_dir=tmp_path,
        )

        assert len(records) == 3, "one row per logical trial, retries included"
        assert summary.executed == 3
        assert summary.completed == 3
        assert provider.call_count == 6, "two calls per trial"
        assert summary.transport_retries == 3
        assert all(r.result.attempts == 2 for r in records)
        # the file itself has one line per trial
        lines = Path(summary.result_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_retries_are_capped_and_then_the_trial_fails(self, runner_config):
        key = ("m", request_for("c000").prompt_fingerprint)
        provider = ScriptedProvider({key: [transport_failure()] * 20})
        runner = Runner(provider, runner_config, sleep=no_sleep)
        with pytest.raises(TransportError, match="gave up"):
            runner.generate_once(request_for("c000"))
        assert provider.call_count == runner_config.max_transport_retries + 1

    def test_backoff_is_exponential_and_capped(self, runner_config):
        runner = Runner(ScriptedProvider({}), runner_config, sleep=no_sleep)
        delays = [runner._backoff_seconds(i) for i in range(1, 8)]
        assert delays[0] == runner_config.backoff_initial_seconds
        assert delays[1] == pytest.approx(
            runner_config.backoff_initial_seconds * runner_config.backoff_multiplier
        )
        assert delays == sorted(delays)
        assert max(delays) <= runner_config.backoff_max_seconds

    def test_backoff_is_actually_slept_between_attempts(self, runner_config):
        slept = []
        key = ("m", request_for("c000").prompt_fingerprint)
        provider = ScriptedProvider({key: [transport_failure(), transport_failure(), "ok"]})
        runner = Runner(provider, runner_config, sleep=slept.append)
        runner.generate_once(request_for("c000"))
        assert slept == [
            runner_config.backoff_initial_seconds,
            runner_config.backoff_initial_seconds * runner_config.backoff_multiplier,
        ]

    def test_provider_error_is_not_retried(self, runner_config):
        key = ("m", request_for("c000").prompt_fingerprint)
        provider = ScriptedProvider({key: [provider_failure(), "never reached"]})
        runner = Runner(provider, runner_config, sleep=no_sleep)
        with pytest.raises(ProviderError):
            runner.generate_once(request_for("c000"))
        assert provider.call_count == 1


class TestContentFailuresAreNotRetried:
    def test_unparseable_completion_is_returned_not_retried(self, runner_config):
        key = ("m", request_for("c000").prompt_fingerprint)
        provider = ScriptedProvider({key: ["I think it went up.", '{"trend": "stable"}']})
        runner = Runner(provider, runner_config, sleep=no_sleep)
        result = runner.generate_once(request_for("c000"))

        assert result.raw_text == "I think it went up."
        assert result.attempts == 1
        assert provider.call_count == 1, "a content failure must not be re-rolled"

    def test_a_run_of_unparseable_completions_makes_one_row_each(
        self, tmp_path, cfg, runner_config
    ):
        provider = CallableProvider(lambda req: "not json")
        runner = Runner(provider, runner_config, sleep=no_sleep)
        records, summary = runner.run(
            plans(5),
            model="m",
            arm=Arm.DIRECT,
            dataset_sha256="b" * 64,
            manifest=manifest_for(tmp_path, cfg),
            results_dir=tmp_path,
        )
        assert len(records) == 5
        assert summary.completed == 5 and summary.failed == 0
        assert provider.call_count == 5


# ======================================================================= resume
class TestResume:
    def test_second_run_calls_the_provider_zero_times(self, tmp_path, cfg, runner_config):
        provider = CallableProvider(lambda req: '{"trend": "stable", "reason": "r"}')
        common = dict(
            model="m",
            arm=Arm.DIRECT,
            dataset_sha256="c" * 64,
            manifest=manifest_for(tmp_path, cfg),
            results_dir=tmp_path,
        )
        Runner(provider, runner_config, sleep=no_sleep).run(plans(4), **common)
        assert provider.call_count == 4

        second = CallableProvider(lambda req: '{"trend": "stable", "reason": "r"}')
        records, summary = Runner(second, runner_config, sleep=no_sleep).run(plans(4), **common)
        assert second.call_count == 0
        assert summary.executed == 0 and summary.skipped_existing == 4
        assert len(records) == 4

    def test_resume_only_runs_the_missing_trials(self, tmp_path, cfg, runner_config):
        common = dict(
            model="m",
            arm=Arm.DIRECT,
            dataset_sha256="d" * 64,
            manifest=manifest_for(tmp_path, cfg),
            results_dir=tmp_path,
        )
        first = CallableProvider(lambda req: '{"trend": "stable", "reason": "r"}')
        Runner(first, runner_config, sleep=no_sleep).run(plans(3), **common)

        second = CallableProvider(lambda req: '{"trend": "stable", "reason": "r"}')
        records, summary = Runner(second, runner_config, sleep=no_sleep).run(plans(5), **common)
        assert second.call_count == 2
        assert summary.executed == 2
        assert len(records) == 5

    def test_prompt_variant_is_part_of_the_trial_key(self, tmp_path, cfg, runner_config):
        provider = CallableProvider(lambda req: '{"trend": "stable", "reason": "r"}')
        runner = Runner(provider, runner_config, sleep=no_sleep)
        common = dict(
            model="m",
            arm=Arm.DIRECT,
            dataset_sha256="e" * 64,
            manifest=manifest_for(tmp_path, cfg),
            results_dir=tmp_path,
        )
        runner.run(plans(2, variant="natural"), **common)
        records, summary = runner.run(plans(2, variant="guided"), **common)
        assert summary.executed == 2, "the guided arm is a different trial, not a resume hit"
        assert len(records) == 4

    def test_resume_can_be_disabled(self, tmp_path, cfg, runner_config):
        common = dict(
            model="m",
            arm=Arm.DIRECT,
            dataset_sha256="f" * 64,
            manifest=manifest_for(tmp_path, cfg),
            results_dir=tmp_path,
        )
        first = CallableProvider(lambda req: '{"trend": "stable", "reason": "r"}')
        Runner(first, runner_config, sleep=no_sleep).run(plans(2), **common)
        second = CallableProvider(lambda req: '{"trend": "stable", "reason": "r"}')
        Runner(second, runner_config, sleep=no_sleep).run(plans(2), resume=False, **common)
        assert second.call_count == 2


class TestDatasetIdentity:
    def test_a_different_dataset_hash_gets_a_different_file(self, tmp_path):
        a = result_path_for(tmp_path, "m", Arm.DIRECT, "a" * 64)
        b = result_path_for(tmp_path, "m", Arm.DIRECT, "b" * 64)
        assert a != b

    def test_results_from_another_dataset_are_refused_not_reused(self, tmp_path, cfg, runner_config):
        provider = CallableProvider(lambda req: '{"trend": "stable", "reason": "r"}')
        path = tmp_path / "shared.jsonl"
        Runner(provider, runner_config, sleep=no_sleep).run(
            plans(2),
            model="m",
            arm=Arm.DIRECT,
            dataset_sha256="a" * 64,
            manifest=manifest_for(tmp_path, cfg),
            results_dir=tmp_path,
            result_path=path,
        )
        with pytest.raises(ValueError, match="must not be reused"):
            load_results(path, expected_dataset_sha256="b" * 64)

    def test_every_row_records_its_dataset(self, tmp_path, cfg, runner_config):
        provider = CallableProvider(lambda req: '{"trend": "stable", "reason": "r"}')
        records, _ = Runner(provider, runner_config, sleep=no_sleep).run(
            plans(2),
            model="m",
            arm=Arm.DIRECT,
            dataset_sha256="a" * 64,
            manifest=manifest_for(tmp_path, cfg),
            results_dir=tmp_path,
        )
        assert all(r.dataset_sha256 == "a" * 64 for r in records)

    def test_model_ids_with_slashes_are_safe_in_paths(self, tmp_path):
        path = result_path_for(tmp_path, "anthropic/claude-opus-5:beta", Arm.DIRECT, "a" * 64)
        assert "/" not in path.name and ":" not in path.name
        assert safe_component("anthropic/claude-opus-5") == "anthropic-claude-opus-5"


# ==================================================================== manifests
class TestManifest:
    def test_fingerprint_ignores_the_timestamp(self, tmp_path, cfg):
        a = manifest_for(tmp_path, cfg, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
        b = manifest_for(tmp_path, cfg, now=datetime(2027, 6, 6, tzinfo=timezone.utc))
        assert a.created_at != b.created_at
        assert a.fingerprint == b.fingerprint

    def test_fingerprint_ignores_a_new_invocation_id_for_resume(self, tmp_path, cfg):
        a = manifest_for(tmp_path, cfg, run_id="first-invocation")
        b = manifest_for(tmp_path, cfg, run_id="resumed-invocation")
        assert a.run_id != b.run_id
        assert a.fingerprint == b.fingerprint

    def test_fingerprint_changes_with_anything_that_matters(self, tmp_path, cfg):
        base = manifest_for(tmp_path, cfg)
        for field, value in (
            ("temperature", 0.7),
            ("seed", 999),
            ("structured_output", True),
            ("model", "other"),
            ("prompt_variant", "guided"),
            ("max_tokens", 4096),
            ("provider_constraints", {"context_length": 8192}),
        ):
            assert manifest_for(tmp_path, cfg, **{field: value}).fingerprint != base.fingerprint

    def test_fingerprint_tracks_the_dataset(self, tmp_path, cfg):
        base = manifest_for(tmp_path, cfg)
        other = tmp_path / "other.jsonl"
        other.write_text('{"case_id": "different"}\n', encoding="utf-8")
        assert manifest_for(tmp_path, cfg, dataset_path=other).fingerprint != base.fingerprint

    def test_manifest_records_the_required_fields(self, tmp_path, cfg):
        m = manifest_for(tmp_path, cfg)
        for field in (
            "dataset_sha256", "config_sha256", "prompt_variant", "model", "provider",
            "temperature", "seed", "max_tokens", "structured_output", "created_at",
            "package_versions",
        ):
            assert getattr(m, field) is not None
        assert "python" in m.package_versions

    def test_manifest_records_hardware_and_exact_case_selection(self, tmp_path, cfg):
        m = manifest_for(
            tmp_path,
            cfg,
            hardware={"gpu_nvidia_smi": "NVIDIA A100 40GB"},
            selected_case_ids=["case-a", "case-b"],
        )
        assert m.hardware == {"gpu_nvidia_smi": "NVIDIA A100 40GB"}
        assert m.selected_case_ids == ["case-a", "case-b"]

    def test_git_commit_is_none_rather_than_a_placeholder_outside_a_repo(self, tmp_path, cfg):
        m = manifest_for(tmp_path, cfg, repo_root=tmp_path)
        assert m.git_commit is None or len(m.git_commit) == 40

    def test_competence_dataset_hash_is_recorded_when_given(self, tmp_path, cfg):
        comp = tmp_path / "competence.jsonl"
        comp.write_text('{"case_id": "competence-rr-000"}\n', encoding="utf-8")
        m = manifest_for(tmp_path, cfg, competence_dataset_path=comp)
        assert m.competence_dataset_sha256 == sha256_file(comp)

    def test_writing_is_idempotent(self, tmp_path, cfg):
        path = tmp_path / "r.jsonl"
        m = manifest_for(tmp_path, cfg)
        first = write_manifest(m, path)
        write_manifest(manifest_for(tmp_path, cfg, now=datetime.now(timezone.utc)), path)
        assert read_manifest(first).fingerprint == m.fingerprint

    def test_a_changed_manifest_is_refused(self, tmp_path, cfg):
        path = tmp_path / "r.jsonl"
        write_manifest(manifest_for(tmp_path, cfg), path)
        with pytest.raises(ManifestConflict, match="temperature"):
            write_manifest(manifest_for(tmp_path, cfg, temperature=0.9), path)

    def test_manifest_is_written_beside_the_results(self, tmp_path, cfg, runner_config):
        provider = CallableProvider(lambda req: "{}")
        _, summary = Runner(provider, runner_config, sleep=no_sleep).run(
            plans(1),
            model="m",
            arm=Arm.DIRECT,
            dataset_sha256="a" * 64,
            manifest=manifest_for(tmp_path, cfg),
            results_dir=tmp_path,
        )
        assert Path(summary.manifest_path).exists()
        assert Path(summary.manifest_path) == manifest_path_for(summary.result_path)

    def test_manifest_roundtrips(self, tmp_path, cfg):
        path = tmp_path / "r.jsonl"
        m = manifest_for(tmp_path, cfg)
        write_manifest(m, path)
        assert read_manifest(manifest_path_for(path)).model_dump() == m.model_dump()


# ===================================================================== providers
class TestProviders:
    def test_replay_serves_by_prompt_fingerprint(self):
        req = request_for("c000")
        provider = ReplayProvider({("m", req.prompt_fingerprint): "recorded"})
        assert provider.generate(req).raw_text == "recorded"

    def test_replay_misses_loudly_when_the_prompt_changed(self):
        provider = ReplayProvider({("m", "stale"): "recorded"})
        with pytest.raises(ProviderError, match="no recording"):
            provider.generate(request_for("c000"))

    def test_replay_records_resolved_model_id(self):
        req = request_for("c000", model="alias")
        provider = ReplayProvider(
            {("alias", req.prompt_fingerprint): "x"},
            resolved_model_ids={"alias": "alias-2026-08-24"},
        )
        result = provider.generate(req)
        assert result.requested_model_id == "alias"
        assert result.resolved_model_id == "alias-2026-08-24"

    def test_prompt_fingerprint_is_stable_and_sensitive(self):
        a = GenerationRequest(model="m", system="s", user="u")
        b = GenerationRequest(model="m", system="s", user="u")
        c = GenerationRequest(model="m", system="s", user="u2")
        assert a.prompt_fingerprint == b.prompt_fingerprint
        assert a.prompt_fingerprint != c.prompt_fingerprint

    def test_registry_lists_the_expected_providers(self):
        assert set(available_providers()) == {
            "replay", "scripted", "callable", "openrouter", "ollama", "vercel"
        }

    def test_registry_rejects_an_unknown_provider(self):
        with pytest.raises(KeyError, match="unknown provider"):
            create_provider("anthropic-direct")

    def test_replay_from_jsonl(self, tmp_path):
        req = request_for("c000")
        path = tmp_path / "rec.jsonl"
        path.write_text(
            json.dumps(
                {"model": "m", "prompt_fingerprint": req.prompt_fingerprint, "raw_text": "hi"}
            )
            + "\n",
            encoding="utf-8",
        )
        assert ReplayProvider.from_jsonl(path).generate(req).raw_text == "hi"


# ============================================================ end-to-end replay
class TestReplayRun:
    def test_full_run_against_recorded_completions(self, tmp_path, cfg, runner_config):
        plan_list = plans(6)
        recordings = {
            ("m", p.request.prompt_fingerprint): '{"trend": "stable", "reason": "r"}'
            for p in plan_list
        }
        provider = ReplayProvider(recordings)
        records, summary = Runner(provider, runner_config, sleep=no_sleep).run(
            plan_list,
            model="m",
            arm=Arm.DIRECT,
            dataset_sha256="a" * 64,
            manifest=manifest_for(tmp_path, cfg),
            results_dir=tmp_path,
        )
        assert summary.completed == 6 and summary.failed == 0
        assert all(r.produced_a_completion for r in records)
        assert {r.raw_text for r in records} == {'{"trend": "stable", "reason": "r"}'}

    def test_failed_trials_are_recorded_not_dropped(self, tmp_path, cfg, runner_config):
        plan_list = plans(3)
        script = {
            ("m", p.request.prompt_fingerprint): [transport_failure()] * 20 for p in plan_list
        }
        provider = ScriptedProvider(script)
        records, summary = Runner(provider, runner_config, sleep=no_sleep).run(
            plan_list,
            model="m",
            arm=Arm.DIRECT,
            dataset_sha256="a" * 64,
            manifest=manifest_for(tmp_path, cfg),
            results_dir=tmp_path,
        )
        assert summary.failed == 3 and summary.completed == 0
        assert len(records) == 3
        assert all(r.failure and not r.produced_a_completion for r in records)

    def test_run_ids_are_unique(self):
        assert len({new_run_id() for _ in range(200)}) == 200

    def test_sha256_helpers_agree(self, tmp_path):
        path = tmp_path / "x.txt"
        path.write_text("hello", encoding="utf-8")
        assert sha256_file(path) == sha256_text("hello")

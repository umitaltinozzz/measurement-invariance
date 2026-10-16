from src.generate_cases import load_config, read_jsonl
from src.generate_competence import read_competence_jsonl
from src.paired_dataset import DEFAULT_PATH
from src.run_manifest import build_manifest, manifest_path_for, write_manifest
from src.run_pilot import (
    _run_id_for_resume,
    build_code_plans,
    build_competence_plans,
    build_direct_plans,
)
from src.runner import Arm, result_path_for
from src.prompts import PromptVariant
from src.structured_schemas import (
    AFFINE_INVERSION_RESPONSE_SCHEMA,
    CODE_AGENT_RESPONSE_SCHEMA,
    GUIDED_RESPONSE_SCHEMA,
    NATURAL_RESPONSE_SCHEMA,
    REFERENCE_RANGE_RESPONSE_SCHEMA,
)


def test_colab_config_attaches_schema_to_every_request():
    cfg = load_config("configs/pilot_colab_t4.yaml")
    baseline_cfg = load_config("configs/pilot.yaml")
    assert cfg.runner.structured_output is True
    assert cfg.model_copy(update={"runner": cfg.runner.model_copy(update={"structured_output": False})}) == baseline_cfg

    pilot_cases = read_jsonl("data/generated/pilot_cases.jsonl")
    natural = build_direct_plans(pilot_cases[:1], cfg, "m", PromptVariant.NATURAL)[0]
    guided = build_direct_plans(pilot_cases[:1], cfg, "m", PromptVariant.GUIDED)[0]
    assert natural.request.response_schema == NATURAL_RESPONSE_SCHEMA
    assert guided.request.response_schema == GUIDED_RESPONSE_SCHEMA
    assert natural.request.structured_output and guided.request.structured_output

    competence_cases = read_competence_jsonl("data/generated/competence_cases.jsonl")
    competence_plans = build_competence_plans(competence_cases, cfg, "m")
    assert competence_plans[0].request.response_schema in (
        REFERENCE_RANGE_RESPONSE_SCHEMA,
        AFFINE_INVERSION_RESPONSE_SCHEMA,
    )
    assert {str(p.request.response_schema) for p in competence_plans} == {
        str(REFERENCE_RANGE_RESPONSE_SCHEMA),
        str(AFFINE_INVERSION_RESPONSE_SCHEMA),
    }

    paired_cases = read_jsonl(DEFAULT_PATH)
    code_plan = build_code_plans(paired_cases[:1], cfg, "m")[0]
    assert code_plan.request.response_schema == CODE_AGENT_RESPONSE_SCHEMA


def test_disconnected_run_reuses_manifest_run_id(tmp_path):
    dataset_sha = "a" * 64
    result_path = result_path_for(tmp_path, "model", Arm.COMPETENCE, dataset_sha)
    manifest = build_manifest(
        run_id="original-run",
        arm=Arm.COMPETENCE,
        prompt_variant=None,
        provider="ollama",
        model="model",
        dataset_path="data/generated/competence_cases.jsonl",
        config_path="configs/pilot_colab_t4.yaml",
        temperature=0,
        max_tokens=1024,
        seed=1,
        structured_output=True,
        max_transport_retries=4,
    )
    # The result path is deterministic from the supplied dataset SHA. The
    # manifest content is tested separately; this fixture only pins its run id.
    write_manifest(manifest, result_path)
    assert _run_id_for_resume(
        tmp_path,
        "model",
        Arm.COMPETENCE,
        dataset_sha,
        resume=True,
    ) == "original-run"


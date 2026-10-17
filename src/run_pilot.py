"""Trial-plan construction and the run entry point.

The paired dataset is run twice against the same file of cases:

    python -m src.run_pilot --dataset paired --arm direct --variant natural ...
    python -m src.run_pilot --dataset paired --arm code ...

Both use the same ``dataset_sha256``, and the resume key includes the arm, so one
case answered in both arms is two logical trials that never collide. That is the
whole point of the paired design at the plumbing level: same cases, same dataset
identity, separate trials.

``--provider ollama`` calls a local Ollama server without an API key. The
``openrouter`` provider reads its key only when explicitly selected; offline
``replay`` / ``scripted`` backends stay available for testing. Adding another
backend is a registration in ``src/models/registry.py`` and changes nothing here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TypeVar

from .code_agent import build_code_agent_task
from .competence_prompts import build_competence_prompt
from .generate_cases import CodeAgentConfig, PilotConfig, load_config, read_jsonl
from .generate_competence import read_competence_jsonl
from .models.base import GenerationRequest
from .models.registry import available_providers, create_provider
from .paired_dataset import DEFAULT_PATH as PAIRED_PATH
from .prompts import PromptVariant, build_direct_prompt
from .run_manifest import build_manifest, manifest_path_for, read_manifest, sha256_file
from .runner import Arm, Runner, TrialPlan, new_run_id, result_path_for
from .schemas import BenchmarkCase
from .structured_schemas import (
    CODE_AGENT_RESPONSE_SCHEMA,
    CODE_AGENT_RESPONSE_SCHEMA_V2,
    competence_response_schema,
    direct_response_schema,
)

DATASETS = {
    "pilot": "data/generated/pilot_cases.jsonl",
    "paired": PAIRED_PATH,
    "competence": "data/generated/competence_cases.jsonl",
}

CaseT = TypeVar("CaseT", bound=BenchmarkCase)


def read_case_ids(path: str | Path) -> List[str]:
    """Read a non-empty, unique, one-case-id-per-line selection file."""
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    ids = [case_id for case_id in ids if case_id]
    if not ids:
        raise ValueError(f"case selection file is empty: {path}")
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate case ids in {path}: {duplicates}")
    return ids


def filter_cases_by_id(cases: Sequence[CaseT], selected_ids: Sequence[str]) -> List[CaseT]:
    """Select exactly the requested cases in frozen dataset order."""
    wanted = set(selected_ids)
    available = {case.case_id for case in cases}
    missing = sorted(wanted - available)
    if missing:
        raise ValueError(f"case ids are not present in the dataset: {missing}")
    return [case for case in cases if case.case_id in wanted]

#: Which arms may be run against which dataset.
ARMS_BY_DATASET = {
    "pilot": (Arm.DIRECT,),
    "paired": (Arm.DIRECT, Arm.CODE_AGENT),
    "competence": (Arm.COMPETENCE,),
}


def _request(
    system: str,
    user: str,
    cfg: PilotConfig,
    model: str,
    *,
    max_tokens: int | None = None,
    response_schema: Optional[Dict] = None,
) -> GenerationRequest:
    settings = cfg.models.get(model)
    code = max_tokens is not None
    if cfg.continuation_criteria.rule_version == "rule_v2" and settings is None:
        raise ValueError(f"run 3 requires explicit settings for {model}")
    return GenerationRequest(
        model=model,
        system=system,
        user=user,
        temperature=settings.temperature if settings else cfg.runner.temperature,
        max_tokens=(settings.max_tokens_code_agent if code else settings.max_tokens)
        if settings else (max_tokens or cfg.runner.max_tokens),
        seed=cfg.runner.seed if settings is None or settings.send_seed else None,
        structured_output=cfg.runner.structured_output,
        response_schema=response_schema,
        think=settings.think if settings else None,
        reasoning=settings.reasoning if settings else None,
        context_length=settings.context_length if settings else None,
    )


def build_direct_plans(
    cases: Sequence[BenchmarkCase],
    cfg: PilotConfig,
    model: str,
    variant: PromptVariant = PromptVariant.NATURAL,
) -> List[TrialPlan]:
    plans = []
    for case in cases:
        prompt = build_direct_prompt(case, variant)
        plans.append(
            TrialPlan(
                case_id=case.case_id,
                request=_request(
                    prompt.system,
                    prompt.user,
                    cfg,
                    model,
                    response_schema=direct_response_schema(variant),
                ),
                prompt_variant=PromptVariant(variant).value,
            )
        )
    return plans


def build_code_plans(
    cases: Sequence[BenchmarkCase], cfg: PilotConfig, model: str
) -> List[TrialPlan]:
    plans = []
    for case in cases:
        task = build_code_agent_task(case, cfg.code_agent)
        plans.append(
            TrialPlan(
                case_id=case.case_id,
                request=_request(
                    task.system, task.user, cfg, model,
                    max_tokens=cfg.runner.max_tokens_code_agent,
                    response_schema=CODE_AGENT_RESPONSE_SCHEMA_V2
                    if cfg.code_agent.protocol_version == "v2" else CODE_AGENT_RESPONSE_SCHEMA,
                ),
                prompt_variant=None,
                execution_files={"readings.csv": task.csv_text},
            )
        )
    return plans


def build_competence_plans(cases: Sequence, cfg: PilotConfig, model: str) -> List[TrialPlan]:
    plans = []
    for case in cases:
        prompt = build_competence_prompt(case)
        plans.append(
            TrialPlan(
                case_id=case.case_id,
                request=_request(
                    prompt.system,
                    prompt.user,
                    cfg,
                    model,
                    response_schema=competence_response_schema(prompt.skill),
                ),
                prompt_variant=None,
            )
        )
    return plans


def load_dataset(name: str, path: Optional[str] = None):
    resolved = path or DATASETS[name]
    if name == "competence":
        return read_competence_jsonl(resolved), resolved
    return read_jsonl(resolved), resolved


def _run_id_for_resume(
    results_dir: str | Path,
    model: str,
    arm: str,
    dataset_sha256: str,
    *,
    resume: bool,
) -> str:
    """Keep the immutable manifest identity when continuing an existing run.

    The runner resumes by trial key, while the manifest must retain its original
    run id so a fresh CLI process after a disconnect still matches that manifest.
    """
    result_path = result_path_for(results_dir, model, arm, dataset_sha256)
    manifest_path = manifest_path_for(result_path)
    if resume and manifest_path.exists():
        return read_manifest(manifest_path).run_id
    return new_run_id()


def build_plans(
    dataset_name: str,
    cases: Sequence,
    cfg: PilotConfig,
    model: str,
    arm: str,
    variant: Optional[str],
) -> List[TrialPlan]:
    allowed = ARMS_BY_DATASET[dataset_name]
    if arm not in allowed:
        raise ValueError(f"dataset {dataset_name!r} supports arms {list(allowed)}, not {arm!r}")
    if arm == Arm.DIRECT:
        return build_direct_plans(cases, cfg, model, PromptVariant(variant or "natural"))
    if arm == Arm.CODE_AGENT:
        if variant:
            raise ValueError("the code arm has no prompt variant")
        return build_code_plans(cases, cfg, model)
    return build_competence_plans(cases, cfg, model)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one arm against one dataset.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument(
        "--arm", choices=[Arm.DIRECT, Arm.CODE_AGENT, Arm.COMPETENCE], required=True
    )
    parser.add_argument("--variant", choices=[v.value for v in PromptVariant], default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--provider",
        default="replay",
        help=f"registered providers: {available_providers()}",
    )
    parser.add_argument("--recordings", default=None, help="JSONL for the replay provider")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--only-case-ids",
        default=None,
        help="UTF-8 text file containing one case id per line; runs only those cases",
    )
    parser.add_argument(
        "--hardware-json",
        default=None,
        help="JSON file with runtime hardware metadata to record in the manifest",
    )
    parser.add_argument("--only", default=None, help="pin an upstream provider slug")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--cap-stage", type=int, choices=[0, 1], default=0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if not 0 <= args.repeat < cfg.repeats:
        parser.error("repeat is outside the configured range")
    registered_cfg = cfg
    if args.repeat or args.cap_stage:
        from .prepare_run3 import inference_config
        cfg = inference_config(cfg, args.repeat, args.cap_stage)
    if cfg.continuation_criteria.rule_version == "rule_v2":
        if args.only_case_ids or args.limit:
            parser.error("run 3 requires whole slices; use a separate exploratory config for subsets")
        if args.model not in cfg.models:
            parser.error("model is not in the preregistered roster")
        if args.provider != cfg.models[args.model].provider:
            parser.error("provider differs from the preregistered model settings")
        upstream = cfg.models[args.model].provider_only
        if args.only and args.only != upstream:
            parser.error("upstream differs from the preregistered model settings")
        args.only = upstream
        if args.arm == Arm.DIRECT and args.variant is None:
            args.variant = "natural_sustained"
        if args.dataset_path is None:
            args.dataset_path = str(Path("data/generated/run3") / f"repeat-{args.repeat}" /
                                    Path(DATASETS[args.dataset]).name)
        args.results_dir = str(Path(args.results_dir) / f"repeat-{args.repeat}" /
                               f"stage-{args.cap_stage}" / (args.variant or args.arm))
    cases, dataset_path = load_dataset(args.dataset, args.dataset_path)
    if cfg.continuation_criteria.rule_version == "rule_v2":
        from .prepare_run3 import validate_dataset
        validate_dataset(registered_cfg, args.repeat, args.dataset, cases)
    if args.limit is not None and args.only_case_ids:
        parser.error("--limit cannot be combined with --only-case-ids")
    selected_case_ids = None
    if args.only_case_ids:
        selected_case_ids = read_case_ids(args.only_case_ids)
        cases = filter_cases_by_id(cases, selected_case_ids)
        # Persist the order actually sent to the runner (dataset order).
        selected_case_ids = [case.case_id for case in cases]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        cases = cases[: args.limit]

    hardware = None
    if args.hardware_json:
        try:
            hardware = json.loads(Path(args.hardware_json).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"could not read --hardware-json {args.hardware_json}: {exc}")
        if not isinstance(hardware, dict):
            parser.error("--hardware-json must contain a JSON object")

    if args.provider == "replay":
        if not args.recordings:
            parser.error("--provider replay needs --recordings")
        from .models.replay import ReplayProvider

        provider = ReplayProvider.from_jsonl(args.recordings)
    elif args.provider == "openrouter":
        from .models.openrouter import OpenRouterProvider, load_dotenv

        load_dotenv()
        provider = OpenRouterProvider(
            timeout=cfg.models[args.model].request_timeout_seconds if cfg.models else 180.0,
            provider_only=[args.only] if args.only else None,
            allow_fallbacks=False,
            require_parameters=cfg.runner.structured_output,
        )
    else:
        provider = create_provider(args.provider, **(
            {"timeout": cfg.models[args.model].request_timeout_seconds} if cfg.models else {}))

    plans = build_plans(args.dataset, cases, cfg, args.model, args.arm, args.variant)
    constraints = dict(getattr(provider, "routing_constraints", {}))
    if cfg.models:
        constraints.update({"model_settings": cfg.models[args.model].model_dump(mode="json"),
                            "repeat": args.repeat, "cap_stage": args.cap_stage,
                            "rule_version": cfg.continuation_criteria.rule_version,
                            "code_execution": cfg.code_agent.model_dump(mode="json")})
    dataset_sha256 = sha256_file(dataset_path)
    manifest = build_manifest(
        run_id=_run_id_for_resume(
            args.results_dir,
            args.model,
            args.arm,
            dataset_sha256,
            resume=not args.no_resume,
        ),
        arm=args.arm,
        prompt_variant=args.variant,
        provider=getattr(provider, "name", args.provider),
        model=args.model,
        dataset_path=dataset_path,
        competence_dataset_path=(Path(dataset_path).parent / "competence_cases.jsonl")
        if (Path(dataset_path).parent / "competence_cases.jsonl").exists() else None,
        config_path=args.config,
        temperature=plans[0].request.temperature,
        # The manifest must record the ceiling this arm actually used, not the
        # default -- the code arm runs at a different one.
        max_tokens=plans[0].request.max_tokens,
        seed=plans[0].request.seed,
        structured_output=cfg.runner.structured_output,
        max_transport_retries=cfg.runner.max_transport_retries,
        provider_constraints=constraints,
        hardware=hardware,
        selected_case_ids=selected_case_ids,
        repo_root=Path(__file__).resolve().parents[1],
    )
    from .sandbox import sandbox_from_config
    sandbox = sandbox_from_config(cfg.code_agent) if args.arm == Arm.CODE_AGENT else None
    from time import sleep
    runner = Runner(provider, cfg.runner, sleep=sleep, sandbox=sandbox,
                    execution_timeout=cfg.code_agent.execution_timeout_seconds,
                    code_protocol=cfg.code_agent.protocol_version)
    _, summary = runner.run(
        plans,
        model=args.model,
        arm=args.arm,
        dataset_sha256=dataset_sha256,
        manifest=manifest,
        results_dir=args.results_dir,
        resume=not args.no_resume,
    )
    print(
        f"{summary.model} / {summary.arm}"
        + (f" / {args.variant}" if args.variant else "")
        + f": {summary.completed} completed, {summary.failed} failed, "
        f"{summary.skipped_existing} already present"
    )
    print(f"  results : {summary.result_path}")
    print(f"  manifest: {summary.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

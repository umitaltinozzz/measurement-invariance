"""Trial-plan construction and the run entry point.

The paired dataset is run twice against the same file of cases:

    python -m src.run_pilot --dataset paired --arm direct --variant natural ...
    python -m src.run_pilot --dataset paired --arm code ...

Both use the same ``dataset_sha256``, and the resume key includes the arm, so one
case answered in both arms is two logical trials that never collide. That is the
whole point of the paired design at the plumbing level: same cases, same dataset
identity, separate trials.

``--provider openrouter`` reads the key from OPENROUTER_API_KEY (or a gitignored
.env). The offline ``replay`` / ``scripted`` backends stay available for testing.
Adding another backend is a registration in ``src/models/registry.py`` and changes
nothing here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .code_agent import build_code_agent_task
from .competence_prompts import build_competence_prompt
from .generate_cases import CodeAgentConfig, PilotConfig, load_config, read_jsonl
from .generate_competence import read_competence_jsonl
from .models.base import GenerationRequest
from .models.registry import available_providers, create_provider
from .paired_dataset import DEFAULT_PATH as PAIRED_PATH
from .prompts import PromptVariant, build_direct_prompt
from .run_manifest import build_manifest, sha256_file
from .runner import Arm, Runner, TrialPlan, new_run_id
from .schemas import BenchmarkCase

DATASETS = {
    "pilot": "data/generated/pilot_cases.jsonl",
    "paired": PAIRED_PATH,
    "competence": "data/generated/competence_cases.jsonl",
}

#: Which arms may be run against which dataset.
ARMS_BY_DATASET = {
    "pilot": (Arm.DIRECT,),
    "paired": (Arm.DIRECT, Arm.CODE_AGENT),
    "competence": (Arm.COMPETENCE,),
}


def _request(
    system: str, user: str, cfg: PilotConfig, model: str, *, max_tokens: int | None = None
) -> GenerationRequest:
    return GenerationRequest(
        model=model,
        system=system,
        user=user,
        temperature=cfg.runner.temperature,
        max_tokens=max_tokens or cfg.runner.max_tokens,
        seed=cfg.runner.seed,
        structured_output=cfg.runner.structured_output,
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
                request=_request(prompt.system, prompt.user, cfg, model),
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
                ),
                prompt_variant=None,
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
                request=_request(prompt.system, prompt.user, cfg, model),
                prompt_variant=None,
            )
        )
    return plans


def load_dataset(name: str, path: Optional[str] = None):
    resolved = path or DATASETS[name]
    if name == "competence":
        return read_competence_jsonl(resolved), resolved
    return read_jsonl(resolved), resolved


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
        help=f"registered providers: {available_providers()} (no paid provider yet)",
    )
    parser.add_argument("--recordings", default=None, help="JSONL for the replay provider")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None, help="pin an upstream provider slug")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cases, dataset_path = load_dataset(args.dataset, args.dataset_path)
    if args.limit:
        cases = cases[: args.limit]

    if args.provider == "replay":
        if not args.recordings:
            parser.error("--provider replay needs --recordings")
        from .models.replay import ReplayProvider

        provider = ReplayProvider.from_jsonl(args.recordings)
    elif args.provider == "openrouter":
        from .models.openrouter import OpenRouterProvider, load_dotenv

        load_dotenv()
        provider = OpenRouterProvider(
            provider_only=[args.only] if args.only else None,
            allow_fallbacks=False,
            require_parameters=cfg.runner.structured_output,
        )
    else:
        provider = create_provider(args.provider)

    plans = build_plans(args.dataset, cases, cfg, args.model, args.arm, args.variant)
    manifest = build_manifest(
        run_id=new_run_id(),
        arm=args.arm,
        prompt_variant=args.variant,
        provider=getattr(provider, "name", args.provider),
        model=args.model,
        dataset_path=dataset_path,
        competence_dataset_path=DATASETS["competence"]
        if Path(DATASETS["competence"]).exists()
        else None,
        config_path=args.config,
        temperature=cfg.runner.temperature,
        # The manifest must record the ceiling this arm actually used, not the
        # default -- the code arm runs at a different one.
        max_tokens=(
            cfg.runner.max_tokens_code_agent
            if args.arm == Arm.CODE_AGENT
            else cfg.runner.max_tokens
        ),
        seed=cfg.runner.seed,
        structured_output=cfg.runner.structured_output,
        max_transport_retries=cfg.runner.max_transport_retries,
    )
    runner = Runner(provider, cfg.runner)
    _, summary = runner.run(
        plans,
        model=args.model,
        arm=args.arm,
        dataset_sha256=sha256_file(dataset_path),
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

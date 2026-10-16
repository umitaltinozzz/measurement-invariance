"""Cheapest possible check that the provider works end to end.

Runs a handful of real cases through the whole chain -- prompt, provider, strict
parser, grader -- and reports the two things that decide whether the pilot is
worth paying for:

* does every response field the manifest and metrics depend on actually arrive
  (``model``, ``usage``, ``finish_reason``, ``id``), rather than being assumed
  from OpenAI-compatibility;
* what is the ``invalid_response_rate``. With a 15-case decision slice and a 5%
  gate, a smoke test that already shows parse failures means the pilot cannot
  produce an eligible slice, and the fix belongs here rather than after the money
  is spent.

    python -m src.smoke_test --model qwen/qwen3-5-27b:free --n 5
"""

from __future__ import annotations

import argparse
import json
from typing import List, Sequence

from .code_agent import build_code_agent_task
from .generate_cases import load_config, read_jsonl
from .grader import ModelResponseRecord, grade_all
from .metrics import compute_metrics
from .models.base import GenerationRequest, ProviderError, TransportError
from .models.openrouter import OpenRouterProvider, load_dotenv
from .paired_dataset import DEFAULT_PATH as PAIRED_PATH
from .prompts import PromptVariant, build_direct_prompt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a provider on real cases.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--dataset", default=PAIRED_PATH)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--arm", choices=["direct", "code"], default="direct")
    parser.add_argument("--variant", default="natural")
    parser.add_argument("--structured", action="store_true")
    parser.add_argument("--only", default=None, help="pin an upstream provider slug")
    args = parser.parse_args(argv)

    load_dotenv()
    cfg = load_config(args.config)
    cases = read_jsonl(args.dataset)[: args.n]

    provider = OpenRouterProvider(
        provider_only=[args.only] if args.only else None,
        allow_fallbacks=False,
        require_parameters=cfg.runner.structured_output or args.structured,
    )
    print(f"model    : {args.model}")
    print(f"arm      : {args.arm}" + (f" / {args.variant}" if args.arm == "direct" else ""))
    print(f"routing  : {provider.routing_constraints}")
    print(f"cases    : {len(cases)} from {args.dataset}\n")

    records: List[ModelResponseRecord] = []
    failures = 0
    for case in cases:
        if args.arm == "direct":
            prompt = build_direct_prompt(case, PromptVariant(args.variant))
            system, user = prompt.system, prompt.user
        else:
            task = build_code_agent_task(case, cfg.code_agent)
            system, user = task.system, task.user

        request = GenerationRequest(
            model=args.model,
            system=system,
            user=user,
            temperature=cfg.runner.temperature,
            max_tokens=(
                cfg.runner.max_tokens_code_agent
                if args.arm == "code"
                else cfg.runner.max_tokens
            ),
            seed=cfg.runner.seed,
            structured_output=args.structured,
        )
        try:
            result = provider.generate(request)
        except (TransportError, ProviderError) as exc:
            failures += 1
            print(f"  {case.case_id}  FAILED  {type(exc).__name__}: {exc}")
            continue

        preview = result.raw_text.replace("\n", " ")[:70]
        print(
            f"  {case.case_id}  {result.finish_reason or '?':<8} "
            f"in={result.input_tokens} out={result.output_tokens} "
            f"{result.latency_ms:.0f}ms  {preview}"
        )
        records.append(
            ModelResponseRecord(
                case_id=case.case_id,
                model=args.model,
                prompt_variant=PromptVariant(args.variant),
                raw_text=result.raw_text,
            )
        )
        last = result

    print("\n--- response fields the pipeline depends on ---")
    if records:
        for field in (
            "resolved_model_id", "provider_request_id", "input_tokens",
            "output_tokens", "finish_reason",
        ):
            value = getattr(last, field)
            mark = "ok " if value not in (None, "") else "MISSING"
            print(f"  {mark:<8} {field}: {value}")

    if args.arm == "direct" and records:
        graded = grade_all(cases, records)
        metrics = compute_metrics(graded)
        print("\n--- grading ---")
        print(f"  invalid_response_rate : {metrics.invalid_response_rate}")
        print(f"  classes               : {metrics.class_counts}")
        gate = cfg.continuation_criteria.max_invalid_response_rate
        rate = metrics.invalid_response_rate.value or 0.0
        verdict = "under" if rate <= gate else "OVER"
        print(f"  vs the {gate} gate      : {verdict}")

    print(f"\ntransport/provider failures: {failures} of {len(cases)}")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())

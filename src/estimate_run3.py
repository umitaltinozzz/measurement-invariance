"""Offline cost/runtime arithmetic; imports no providers and makes no API calls."""
import argparse
import json

from .generate_cases import load_config


def estimate(cfg, *, input_tokens=2000, direct_output=2048, code_output=4096,
             overhead_seconds=2.0):
    if cfg.continuation_criteria.rule_version != "rule_v2":
        raise ValueError("run-3 estimates require a rule_v2 config")
    if min(input_tokens, direct_output, code_output) < 0 or overhead_seconds < 0:
        raise ValueError("assumptions must be nonnegative")
    direct = (4 * cfg.n_per_cell * 3 + cfg.paired.total_cases) * cfg.repeats
    competence = cfg.competence.cases_per_model * cfg.repeats
    code = cfg.paired.total_cases * cfg.repeats
    calls = direct + competence + code
    rows = []
    for name, model in cfg.models.items():
        tokens_in = calls * input_tokens
        tokens_out = (direct + competence) * min(direct_output, model.max_tokens) + code * min(code_output, model.max_tokens_code_agent)
        cap_out = (direct + competence) * model.max_tokens + code * model.max_tokens_code_agent
        def cost(output, inputs=tokens_in):
            return (inputs * model.input_usd_per_million + output * model.output_usd_per_million) / 1e6
        rows.append({
            "model": name, "calls": calls,
            "input_tokens_assumed": tokens_in, "output_tokens_assumed": tokens_out,
            "output_cap_tokens": cap_out,
            "expected_usd": round(cost(tokens_out), 2),
            "stage0_cap_usd": round(cost(cap_out), 2),
            "both_stages_cap_usd": round(cost(3 * cap_out, 2 * tokens_in), 2),
            "serial_hours": round((tokens_out / model.tokens_per_second + calls * overhead_seconds
                                   + code * 2) / 3600, 2),
            "serial_cap_hours": round((cap_out / model.tokens_per_second + calls * overhead_seconds
                                       + code * cfg.code_agent.execution_timeout_seconds) / 3600, 2),
        })
    return {
        "network_calls": 0, "repeats": cfg.repeats,
        "per_model_calls": {"direct": direct, "competence": competence, "code": code, "total": calls},
        "assumptions": {"input_tokens_per_call": input_tokens,
                        "direct_output_including_reasoning": direct_output,
                        "code_output_including_reasoning": code_output,
                        "overhead_seconds_per_call": overhead_seconds,
                        "expected_sandbox_seconds": 2,
                        "cap_cost_scope": "output caps at assumed input length; not a billing upper bound",
                        "excluded": "GPU rental, tax, credit fees, transport retries, caching discounts"},
        "models": rows, "total_calls": calls * len(rows),
        "expected_usd": round(sum(r["expected_usd"] for r in rows), 2),
        "stage0_cap_usd": round(sum(r["stage0_cap_usd"] for r in rows), 2),
        "both_stages_cap_usd": round(sum(r["both_stages_cap_usd"] for r in rows), 2),
        "serial_hours": round(sum(r["serial_hours"] for r in rows), 2),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/run3_open.yaml")
    parser.add_argument("--input-tokens", type=int, default=2000)
    parser.add_argument("--direct-output", type=int, default=2048)
    parser.add_argument("--code-output", type=int, default=4096)
    args = parser.parse_args(argv)
    print(json.dumps(estimate(load_config(args.config), input_tokens=args.input_tokens,
                              direct_output=args.direct_output, code_output=args.code_output), indent=2))


if __name__ == "__main__":
    main()

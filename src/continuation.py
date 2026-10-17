"""Versioned decision entry point. Historical configs always select rule_v1."""

from .rule_v1 import (  # keep the public pilot API
    ContinuationDecision, ModelDecision, SliceEligibility, assess_slice,
    code_metrics_as_metricset, direct_trigger_slice, evaluate_model,
    evaluate_continuation as evaluate_rule_v1,
    format_continuation_decision as format_rule_v1,
)


def evaluate_continuation(criteria, *, code_graded=(), **kwargs):
    if criteria.rule_version == "rule_v1":
        return evaluate_rule_v1(criteria, **kwargs)
    from .rule_v2 import evaluate_rule_v2
    return evaluate_rule_v2(criteria, code_graded=code_graded, **kwargs)


def format_continuation_decision(decision):
    if decision.criteria.rule_version == "rule_v1":
        return format_rule_v1(decision)
    return decision.model_dump_json(indent=2)

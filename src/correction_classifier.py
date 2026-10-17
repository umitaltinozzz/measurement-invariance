"""Conservative, auditable correction screen for run 3, not a semantic proof.

Recognizes range normalization (reading - lower)/(upper - lower) and affine
inversion (reading - offset)/gain. Correction must reach the printed answer.
Selection, grouping, dropping and merely printing columns do not qualify.
Unsupported helpers/dynamic code require review; every result remains heuristic.
"""
import ast
import json

from .code_agent_grader import (
    CodeAgentParseResult, CodeAnalysis, CodeFinding, ProvenanceLevel,
    analyze_code, parse_code_agent_response,
)


def parse_code_response_v2(text):
    try:
        value = json.loads(text)
        if not isinstance(value, dict) or set(value) != {"code_lines", "answer", "reason"}:
            raise ValueError("expected exactly code_lines, answer, reason")
        lines = value.pop("code_lines")
        if not isinstance(lines, list) or not lines or not all(isinstance(x, str) for x in lines):
            raise ValueError("code_lines must be a nonempty array of strings")
        if any("\n" in x or "\r" in x for x in lines):
            raise ValueError("each code_lines item must be one source line")
        value["code"] = "\n".join(lines)
        return parse_code_agent_response(json.dumps(value))
    except (ValueError, TypeError) as exc:
        return CodeAgentParseResult(ok=False, error=str(exc))


def analyze_correction(code, provenance_columns):
    baseline = analyze_code(code, provenance_columns)
    if not baseline.parse_ok:
        return baseline.model_copy(update={"classifier_version": "v2"})
    tree = ast.parse(code)
    env = {}
    observed = set()
    findings = []
    sources = {
        "reading": {"raw"}, "reference_range": {"range"},
        "gain": {"gain"}, "calibration_gain": {"gain"},
        "offset": {"offset"}, "calibration_offset": {"offset"},
    }

    def key(node):
        return ast.dump(node, include_attributes=False)

    def tags(node):
        if node is None:
            return set()
        if key(node) in env:
            return set(env[key(node)])
        if isinstance(node, ast.Name):
            return set(env.get(node.id, ()))
        if isinstance(node, ast.Constant):
            return set()
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, str):
                    return set(sources.get(node.slice.value, ()))
                base = tags(node.value)
                if "range" in base and node.slice.value in (0, 1):
                    return {"lower" if node.slice.value == 0 else "upper"}
            return tags(node.value)
        if isinstance(node, ast.Attribute) and node.attr in sources:
            return set(sources[node.attr])
        if isinstance(node, ast.Dict):
            # Only the answer dependency counts, never another diagnostic field.
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "answer":
                    return tags(v) | {"answer_object"}
            return set()
        if isinstance(node, ast.BinOp):
            left, right = tags(node.left), tags(node.right)
            if isinstance(node.op, ast.Sub):
                if "upper" in left and "lower" in right:
                    return {"width"}
                if "raw" in left and "lower" in right:
                    return {"centered"}
                if "raw" in left and "offset" in right:
                    return {"deoffset"}
            if isinstance(node.op, ast.Div):
                if "centered" in left and "width" in right:
                    return {"normalization"}
                if "deoffset" in left and "gain" in right:
                    return {"inversion"}
            return left | right
        return set().union(*(tags(c) for c in ast.iter_child_nodes(node)))

    def bind(target, value):
        env[target.id if isinstance(target, ast.Name) else key(target)] = set(value)
        # Support lo, hi = map(float, row['reference_range'].split(...)).
        if isinstance(target, (ast.Tuple, ast.List)):
            for i, item in enumerate(target.elts):
                bind(item, {"lower" if i == 0 else "upper"} if "range" in value and len(target.elts) == 2 else value)

    def block(statements, control=frozenset()):
        for statement in statements:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = tags(statement.value) | set(control)
                for target in statement.targets if isinstance(statement, ast.Assign) else [statement.target]:
                    bind(target, value)
            elif isinstance(statement, ast.If):
                condition = tags(statement.test) | set(control)
                before = {k: set(v) for k, v in env.items()}
                block(statement.body, condition)
                yes = {k: set(v) for k, v in env.items()}
                env.clear(); env.update(before)
                block(statement.orelse, condition)
                for name, value in yes.items():
                    env[name] = env.get(name, set()) | value
            elif isinstance(statement, (ast.For, ast.While, ast.With)):
                if isinstance(statement, ast.For):
                    bind(statement.target, tags(statement.iter))
                block(statement.body, control)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if isinstance(call.func, ast.Attribute) and call.func.attr == "append":
                    bind(call.func.value, tags(call.func.value) | tags(call) | set(control))
                if isinstance(call.func, ast.Name) and call.func.id == "print":
                    value = tags(call) | set(control)
                    if "answer_object" in value:
                        kinds = value & {"normalization", "inversion"}
                        observed.update(kinds)
                        for kind in sorted(kinds):
                            findings.append(CodeFinding(kind=kind, line=statement.lineno,
                                                        detail="correction flows to printed answer"))
    block(tree.body)
    touched = baseline.provenance_columns_referenced
    level = ProvenanceLevel.USED if observed else (ProvenanceLevel.REFERENCED if touched else ProvenanceLevel.IGNORED)
    anti = list(baseline.anti_patterns)
    if touched and not observed:
        anti.append("provenance_touch_without_output_correction")
    return baseline.model_copy(update={
        "classifier_version": "v2", "correction_detected": bool(observed),
        "correction_kinds": sorted(observed), "provenance_level": level,
        "provenance_columns_used_analytically": touched if observed else [],
        "findings": findings, "anti_patterns": sorted(set(anti)),
        "needs_human_review": True,
    })

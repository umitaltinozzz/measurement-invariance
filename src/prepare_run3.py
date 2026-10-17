"""Materialize all independent repeats without model/network calls."""
import argparse
import json
from pathlib import Path

from .generate_cases import generate_dataset, load_config, write_jsonl
from .generate_competence import generate_competence_dataset, write_competence_jsonl
from .paired_dataset import generate_paired_dataset
from .run_manifest import sha256_file


def repeat_config(cfg, repeat):
    if not 0 <= repeat < cfg.repeats:
        raise ValueError("repeat outside configured range")
    return cfg.model_copy(update={
        "master_seed": cfg.master_seed + repeat * 10000,
        "case_id_prefix": f"run3-r{repeat}",
        "paired": cfg.paired.model_copy(update={"case_id_prefix": f"run3-r{repeat}-paired"}),
    })


def inference_config(cfg, repeat, stage=0):
    """Predeclared generation settings; stage 1 changes only output ceilings."""
    if not 0 <= repeat < cfg.repeats or stage not in (0, 1):
        raise ValueError("invalid repeat or cap stage")
    return cfg.model_copy(update={
        "runner": cfg.runner.model_copy(update={"seed": cfg.runner.seed + repeat}),
        "models": {name: type(setting).model_validate({
            **setting.model_dump(),
            "max_tokens": setting.max_tokens * (2 ** stage),
            "max_tokens_code_agent": setting.max_tokens_code_agent * (2 ** stage),
        }) for name, setting in cfg.models.items()},
    })


def expected_cases(cfg, repeat, kind):
    generators = {"pilot": generate_dataset, "paired": generate_paired_dataset,
                  "competence": generate_competence_dataset}
    return generators[kind](repeat_config(cfg, repeat))


def validate_dataset(cfg, repeat, kind, cases):
    if cases != expected_cases(cfg, repeat, kind):
        raise ValueError("dataset differs from the registered design/repeat")


def prepare(config_path, out):
    cfg = load_config(config_path)
    if cfg.continuation_criteria.rule_version != "rule_v2":
        raise ValueError("requires rule_v2 config")
    artifacts = []
    for repeat in range(cfg.repeats):
        current = repeat_config(cfg, repeat)
        directory = Path(out) / f"repeat-{repeat}"
        for name, cases, writer in (
            ("pilot_cases.jsonl", generate_dataset(current), write_jsonl),
            ("paired_agent_cases.jsonl", generate_paired_dataset(current), write_jsonl),
            ("competence_cases.jsonl", generate_competence_dataset(current), write_competence_jsonl),
        ):
            path = directory / name
            if path.exists():
                # Never overwrite frozen data, even when regenerating for a second cohort.
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    candidate = writer(cases, Path(temp) / name)
                    if path.read_bytes() != candidate.read_bytes():
                        raise ValueError(f"existing dataset differs: {path}")
            else:
                writer(cases, path)
            artifacts.append({"repeat": repeat, "path": str(path.as_posix()),
                              "cases": len(cases), "sha256": sha256_file(path)})
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/run3_open.yaml")
    parser.add_argument("--out", default="data/generated/run3")
    parser.add_argument("--manifest-out", help="write dataset hashes as UTF-8 JSON")
    args = parser.parse_args(argv)
    result = json.dumps(prepare(args.config, args.out), indent=2) + "\n"
    if args.manifest_out:
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_out).write_text(result, encoding="utf-8", newline="\n")
    print(result, end="")


if __name__ == "__main__":
    main()

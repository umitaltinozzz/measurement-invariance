"""Draw the README figures from the reference run-2 analysis.

Usage: python scripts/make_readme_figures.py
Reads data/results/colab_t4/run2/analysis_with_reruns/analysis.json and writes
docs/figures/*.png. Rates are valid-only, as in the analysis tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data/results/colab_t4/run2/analysis_with_reruns/analysis.json"
OUT = ROOT / "docs/figures"

MODELS = ["gpt-oss:20b", "qwen3:14b", "gemma4:12b", "mistral-small3.2:24b"]
LABELS = {
    "gpt-oss:20b": "gpt-oss 20B\n(passed competence)",
    "qwen3:14b": "Qwen 3 14B",
    "gemma4:12b": "Gemma 4 12B",
    "mistral-small3.2:24b": "Mistral Small 3.2 24B",
}
NOISE, MEASURE, CODE = "#9aa5b1", "#d1495b", "#2e86ab"


def rate(metric: dict) -> float:
    value = metric["valid_only"]["value"]
    return 0.0 if value is None else value * 100


def direct_str(analysis: dict, model: str, variant: str, cell: int) -> float:
    for row in analysis["direct_cell_metrics"]:
        if row["model"] == model and row["prompt_variant"] == variant and row["cell"] == cell:
            return rate(row["spurious_trend_rate"])
    raise KeyError((model, variant, cell))


def code_row(analysis: dict, model: str) -> dict:
    return next(row for row in analysis["code_agent"] if row["model"] == model)


def code_str(model: str) -> float:
    """Code-agent Spurious Trend Rate, read from the generated code_agent_metrics.txt."""
    current = None
    for line in (ANALYSIS.parent / "code_agent_metrics.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(" "):
            current = line.strip()
        elif current == model and line.strip().startswith("spurious_trend_rate"):
            return float(line.split(":")[1].split("%")[0])
    raise KeyError(model)


def label_bars(ax, bars) -> None:
    for bar in bars:
        ax.annotate(f"{bar.get_height():.0f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")


def noise_vs_measurement(analysis: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = range(len(MODELS))
    for ax, variant, title in zip(axes, ["natural", "guided"], ["Natural prompt", "Guided prompt"]):
        cell1 = [direct_str(analysis, m, variant, 1) for m in MODELS]
        cell2 = [direct_str(analysis, m, variant, 2) for m in MODELS]
        b1 = ax.bar([i - 0.2 for i in x], cell1, 0.4, color=NOISE, label="Cell 1: nothing changed (noise only)")
        b2 = ax.bar([i + 0.2 for i in x], cell2, 0.4, color=MEASURE, label="Cell 2: only the sensor changed")
        label_bars(ax, b1)
        label_bars(ax, b2)
        ax.set_xticks(list(x), [LABELS[m] for m in MODELS], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 112)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel('Answered "the value really changed" (%)')
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Does the model read a sensor change as a real change? (run 2, lower is better)", fontsize=11)
    fig.tight_layout()
    path = OUT / "run2_noise_vs_measurement.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def direct_vs_code(analysis: dict) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = range(len(MODELS))
    direct = [direct_str(analysis, m, "natural", 2) for m in MODELS]
    code = [code_str(m) for m in MODELS]
    kept = [rate(code_row(analysis, m)["provenance_retention_rate"]) for m in MODELS]
    b1 = ax.bar([i - 0.27 for i in x], direct, 0.27, color=MEASURE, label="Direct answer: spurious trend (cell 2)")
    b2 = ax.bar(list(x), code, 0.27, color=CODE, label="Code agent: spurious trend")
    b3 = ax.bar([i + 0.27 for i in x], kept, 0.27, color="#3bb273", label="Code agent: kept sensor provenance")
    for bars in (b1, b2, b3):
        label_bars(ax, bars)
    ax.set_xticks(list(x), [LABELS[m] for m in MODELS], fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("%")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)
    ax.set_title("Same model, answering directly vs writing code (run 2)", fontsize=11)
    fig.tight_layout()
    path = OUT / "run2_direct_vs_code.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (noise_vs_measurement(analysis), direct_vs_code(analysis)):
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()

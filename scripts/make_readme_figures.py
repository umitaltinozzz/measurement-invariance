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


def paired_cell2_row(analysis: dict, model: str) -> dict:
    row = next(
        row for row in analysis["paired_direct_code_comparison"] if row["model"] == model
    )
    return row["cell2"]


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
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = range(len(MODELS))
    comparisons = [paired_cell2_row(analysis, model) for model in MODELS]
    categories = (
        ("spurious", "False trend", MEASURE),
        ("uncertain", "Uncertain", "#e0a458"),
        ("other", "Other valid answer", "#5a9f83"),
        ("invalid", "Invalid response", "#9aa5b1"),
    )
    width = 0.32
    for arm, offset in (("direct", -0.18), ("code", 0.18)):
        positions = [i + offset for i in x]
        bottoms = [0.0] * len(MODELS)
        for category, label, color in categories:
            field = f"{arm}_{category}" if category in ("other", "invalid") else (
                f"{arm}_spurious_trend" if category == "spurious" else f"{arm}_uncertain"
            )
            counts = []
            values = []
            for row in comparisons:
                cell = row["n_pairs"]
                if category == "invalid":
                    count = row[f"{arm}_invalid_count"]
                elif category == "other":
                    count = row[f"{arm}_other_valid"]["numerator"]
                else:
                    count = row[field]["numerator"]
                counts.append(count)
                values.append(0.0 if cell == 0 else 100 * count / cell)
            bars = ax.bar(
                positions,
                values,
                width,
                bottom=bottoms,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label=label if arm == "direct" else None,
            )
            for bar, count, row in zip(bars, counts, comparisons):
                if count >= 3 and row["n_pairs"]:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{count}/{row['n_pairs']}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
            bottoms = [base + value for base, value in zip(bottoms, values)]
    ax.set_xticks(list(x), [LABELS[m] for m in MODELS], fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Share of paired cell-2 cases (%)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=4)
    ax.set_title("Paired dataset: cell 2 only; within each group left = direct, right = code", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
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

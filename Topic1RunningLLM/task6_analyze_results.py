"""
MMLU Results Analysis Script

Generates visualizations to answer:
1. How do models compare per subject?
2. Are model mistakes patterned or random?
3. Do models make mistakes on the same questions?

Usage: python analyze_results.py
Plots are saved to: results/plots/
"""

import json
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

MODEL_SHORT_NAMES = {
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "allenai/OLMo-2-0425-1B": "OLMo-2-1B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
}

SUBJECT_SHORT_NAMES = {
    "astronomy": "Astronomy",
    "business_ethics": "Business Ethics",
    "high_school_government_and_politics": "HS Gov & Pol.",
    "high_school_macroeconomics": "HS Macroeconomics",
    "high_school_physics": "HS Physics",
    "high_school_psychology": "HS Psychology",
    "high_school_statistics": "HS Statistics",
    "human_sexuality": "Human Sexuality",
    "international_law": "Intl. Law",
    "jurisprudence": "Jurisprudence",
}

sns.set_theme(style="whitegrid", font_scale=1.0)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_results():
    """Load all *w_ans.json files. Returns dict keyed by short model name."""
    files = sorted(glob.glob(str(RESULTS_DIR / "*w_ans.json")))
    if not files:
        raise FileNotFoundError(f"No *w_ans.json files found in {RESULTS_DIR}")

    models = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        model_id = data["model"]
        short = MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])
        models[short] = {
            "full_name": model_id,
            "device": data["device"],
            "overall_accuracy": data["overall_accuracy"],
            "subjects": {sr["subject"]: sr for sr in data["subject_results"]},
        }
    return models


def build_answer_matrices(models):
    """
    For each subject present in all models, build a DataFrame
    (rows = model names, columns = question index) with 1/0 values.
    Returns dict: subject -> DataFrame
    """
    subject_sets = [set(m["subjects"]) for m in models.values()]
    common_subjects = sorted(set.intersection(*subject_sets))

    matrices = {}
    for subject in common_subjects:
        model_answers = {
            name: mdata["subjects"][subject]["answers"]
            for name, mdata in models.items()
        }
        min_len = min(len(v) for v in model_answers.values())
        df = pd.DataFrame({k: v[:min_len] for k, v in model_answers.items()}).T
        matrices[subject] = df

    return matrices


# ============================================================================
# PLOT 1: Accuracy by subject — grouped bar chart
# ============================================================================

def plot_accuracy_by_subject(models):
    rows = []
    for model_name, mdata in models.items():
        for subject, sdata in mdata["subjects"].items():
            rows.append({
                "Model": model_name,
                "Subject": SUBJECT_SHORT_NAMES.get(subject, subject),
                "Accuracy (%)": sdata["accuracy"],
            })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df, x="Subject", y="Accuracy (%)", hue="Model",
                ax=ax, palette="Set2", edgecolor="white")
    ax.axhline(25, color="grey", linestyle="--", linewidth=1, label="Random chance (25%)")
    ax.set_title("MMLU Accuracy by Subject and Model", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    out = PLOTS_DIR / "01_accuracy_by_subject.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================================
# PLOT 2: Per-question correctness heatmaps (one row per model, per subject)
# ============================================================================

def plot_per_question_heatmaps(matrices):
    """
    Each subplot is one subject: rows=models, columns=question index.
    Green = correct, Red = wrong. Reveals whether mistakes cluster or are random.
    """
    cmap = ListedColormap(["#d73027", "#1a9850"])  # red=wrong, green=correct
    subjects = sorted(matrices.keys())
    n = len(subjects)

    fig, axes = plt.subplots(n, 1, figsize=(20, 2.8 * n))
    if n == 1:
        axes = [axes]

    for ax, subject in zip(axes, subjects):
        df = matrices[subject]
        n_q = df.shape[1]
        sns.heatmap(
            df, ax=ax,
            cmap=cmap, vmin=0, vmax=1,
            cbar=False, xticklabels=False, linewidths=0,
        )
        short = SUBJECT_SHORT_NAMES.get(subject, subject)
        acc_labels = "  |  ".join(
            f"{row}: {df.loc[row].mean() * 100:.1f}%" for row in df.index
        )
        ax.set_title(f"{short}  ({n_q} questions)   {acc_labels}", fontsize=9, fontweight="bold")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=8, rotation=0)

    legend_elements = [
        Patch(facecolor="#1a9850", label="Correct"),
        Patch(facecolor="#d73027", label="Wrong"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)
    fig.suptitle("Per-Question Correctness by Model", fontsize=13, fontweight="bold", y=1.002)
    plt.tight_layout()
    out = PLOTS_DIR / "02_per_question_heatmaps.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================================
# PLOT 3: Mistake overlap — how many models got each question wrong
# ============================================================================

def plot_mistake_overlap(matrices):
    """
    Stacked bar per subject showing what fraction of questions were:
      - All models correct
      - Mixed (some right, some wrong)
      - All models wrong
    """
    rows = []
    for subject, df in matrices.items():
        n_models = df.shape[0]
        n_q = df.shape[1]
        sums = df.sum(axis=0)
        all_right = int((sums == n_models).sum())
        all_wrong = int((sums == 0).sum())
        mixed     = n_q - all_right - all_wrong
        rows.append({
            "Subject": SUBJECT_SHORT_NAMES.get(subject, subject),
            f"All correct ({n_models}/{n_models})": all_right / n_q * 100,
            "Mixed (models disagree)": mixed / n_q * 100,
            f"All wrong (0/{n_models})": all_wrong / n_q * 100,
        })

    df_overlap = pd.DataFrame(rows).set_index("Subject")
    colors = ["#1a9850", "#fee08b", "#d73027"]
    x = np.arange(len(df_overlap))

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(df_overlap))
    for col, color in zip(df_overlap.columns, colors):
        ax.bar(x, df_overlap[col].values, bottom=bottom,
               label=col, color=color, edgecolor="white", linewidth=0.5)
        bottom += df_overlap[col].values

    ax.set_xticks(x)
    ax.set_xticklabels(df_overlap.index, rotation=35, ha="right")
    ax.set_title(
        "Agreement Across All Models per Subject\n"
        "(What fraction of questions do all models get right, wrong, or split?)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("% of questions")
    ax.set_ylim(0, 100)
    ax.legend(title="Outcome", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    out = PLOTS_DIR / "03_mistake_overlap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================================
# PLOT 4: Pairwise answer correlation heatmap across subjects
# ============================================================================

def plot_pairwise_correlation(matrices):
    """
    Pearson r between each pair of models' binary answer vectors, per subject.
    r > 0 means models tend to get the same questions right/wrong (shared errors).
    r ≈ 0 means mistakes are independent.
    """
    model_names = list(next(iter(matrices.values())).index)
    pairs = [
        (model_names[i], model_names[j])
        for i in range(len(model_names))
        for j in range(i + 1, len(model_names))
    ]

    rows = []
    for subject, df in matrices.items():
        for m1, m2 in pairs:
            corr = np.corrcoef(df.loc[m1].values, df.loc[m2].values)[0, 1]
            rows.append({
                "Subject": SUBJECT_SHORT_NAMES.get(subject, subject),
                "Pair": f"{m1}\nvs\n{m2}",
                "Pearson r": round(corr, 3),
            })

    pivot = pd.DataFrame(rows).pivot(index="Pair", columns="Subject", values="Pearson r")

    fig, ax = plt.subplots(figsize=(13, max(3, len(pairs) * 1.8)))
    sns.heatmap(
        pivot, ax=ax,
        annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-0.3, vmax=0.6,
        linewidths=0.5, annot_kws={"size": 10},
    )
    ax.set_title(
        "Pairwise Model Answer Correlation per Subject\n"
        "(Pearson r on binary correct/wrong vectors — higher = models agree more on which questions are hard)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    out = PLOTS_DIR / "04_pairwise_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================================
# PLOT 5: Rolling accuracy over question sequence
# ============================================================================

def plot_rolling_accuracy(models):
    """
    Rolling mean accuracy over the question sequence per subject.
    A flat line suggests random errors; a wave pattern suggests structure
    (e.g., topic blocks or difficulty ordering within a subject).
    """
    all_subjects = sorted(next(iter(models.values()))["subjects"].keys())
    n = len(all_subjects)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    palette = sns.color_palette("Set2", len(models))
    model_colors = dict(zip(models.keys(), palette))

    for ax, subject in zip(axes, all_subjects):
        n_q_max = max(
            len(mdata["subjects"][subject]["answers"])
            for mdata in models.values()
            if subject in mdata["subjects"]
        )
        window = max(10, n_q_max // 15)  # ~7% of questions, minimum 10

        for model_name, mdata in models.items():
            if subject not in mdata["subjects"]:
                continue
            answers = np.array(mdata["subjects"][subject]["answers"], dtype=float)
            rolling = pd.Series(answers).rolling(window=window, min_periods=1).mean() * 100
            ax.plot(rolling, label=model_name, color=model_colors[model_name], linewidth=1.8)

        ax.axhline(25, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(SUBJECT_SHORT_NAMES.get(subject, subject), fontsize=9, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Rolling acc. (%)", fontsize=7)
        ax.set_xlabel("Question index", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=model_colors[m], linewidth=2, label=m)
        for m in models
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=9, title="Model",
               bbox_to_anchor=(1.0, 0.02))
    fig.suptitle(
        "Rolling Accuracy Over Question Sequence\n"
        f"(dashed = 25% random chance — flat line = random errors, waves = structured difficulty)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = PLOTS_DIR / "05_rolling_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    models = load_results()
    print(f"Loaded {len(models)} models: {list(models.keys())}")

    matrices = build_answer_matrices(models)
    print(f"Built answer matrices for {len(matrices)} subjects\n")

    print("Generating plots...")
    plot_accuracy_by_subject(models)
    plot_per_question_heatmaps(matrices)
    plot_mistake_overlap(matrices)
    plot_pairwise_correlation(matrices)
    plot_rolling_accuracy(models)

    print(f"\n✅ All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

"""Analysis script for Story CoT experiment results.

Produces:
- Accuracy tables per dataset and strategy
- Statistical tests (McNemar's test, bootstrap CIs)
- Visualizations (bar charts, heatmaps)
- Qualitative analysis of reasoning traces
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RESULTS_DIR = Path("/workspaces/story-cot-nlp-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

STRATEGY_LABELS = {
    "direct": "Direct",
    "zero_shot_cot": "Zero-shot CoT",
    "few_shot_cot": "Few-shot CoT",
    "story_cot": "Story CoT",
}

DATASET_LABELS = {
    "gsm8k": "GSM8K\n(Math)",
    "commonsenseqa": "CommonsenseQA\n(Commonsense)",
    "strategyqa": "StrategyQA\n(Multi-hop)",
    "arc_challenge": "ARC-Challenge\n(Science)",
}


def load_results():
    """Load experiment results."""
    with open(RESULTS_DIR / "all_results.json") as f:
        return json.load(f)


def compute_accuracy_table(results):
    """Compute accuracy for each dataset/strategy combination."""
    df = pd.DataFrame(results)
    table = df.groupby(["dataset", "strategy"])["correct"].agg(["mean", "sum", "count"])
    table.columns = ["accuracy", "correct", "total"]
    table["accuracy_pct"] = table["accuracy"] * 100
    return table


def bootstrap_ci(correct_arr, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval for accuracy."""
    rng = np.random.default_rng(42)
    n = len(correct_arr)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(correct_arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lower, upper


def mcnemar_test(results, dataset, strategy_a, strategy_b):
    """Run McNemar's test comparing two strategies on same dataset."""
    df = pd.DataFrame(results)
    ds = df[df["dataset"] == dataset]

    a_results = ds[ds["strategy"] == strategy_a].sort_values("id")
    b_results = ds[ds["strategy"] == strategy_b].sort_values("id")

    # Align by id
    merged = a_results.merge(b_results, on="id", suffixes=("_a", "_b"))

    a_correct = merged["correct_a"].values
    b_correct = merged["correct_b"].values

    # Contingency: a_right_b_wrong, a_wrong_b_right
    n_10 = np.sum(a_correct & ~b_correct)  # a right, b wrong
    n_01 = np.sum(~a_correct & b_correct)  # a wrong, b right

    # McNemar's test (with continuity correction)
    if n_10 + n_01 == 0:
        return {"chi2": 0, "p_value": 1.0, "n_10": int(n_10), "n_01": int(n_01)}

    chi2 = (abs(n_10 - n_01) - 1) ** 2 / (n_10 + n_01)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "n_10": int(n_10),
        "n_01": int(n_01),
        "diff": float((n_10 - n_01) / len(merged))
    }


def plot_accuracy_bars(results):
    """Create grouped bar chart of accuracy by dataset and strategy."""
    df = pd.DataFrame(results)
    datasets = ["gsm8k", "commonsenseqa", "strategyqa", "arc_challenge"]
    strategies = ["direct", "zero_shot_cot", "few_shot_cot", "story_cot"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.2
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e74c3c"]

    for i, strategy in enumerate(strategies):
        accs = []
        cis = []
        for dataset in datasets:
            subset = df[(df["dataset"] == dataset) & (df["strategy"] == strategy)]
            acc = subset["correct"].mean()
            lo, hi = bootstrap_ci(subset["correct"].values)
            accs.append(acc * 100)
            cis.append((acc - lo) * 100)

        bars = ax.bar(x + i * width, accs, width, label=STRATEGY_LABELS[strategy],
                       color=colors[i], yerr=cis, capsize=3)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Story CoT vs. Standard CoT Across Reasoning Benchmarks", fontsize=14, fontweight="bold")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'accuracy_comparison.png'}")


def plot_accuracy_heatmap(results):
    """Create heatmap of accuracy differences (Story CoT - Few-shot CoT)."""
    df = pd.DataFrame(results)
    datasets = ["gsm8k", "commonsenseqa", "strategyqa", "arc_challenge"]
    strategies = ["direct", "zero_shot_cot", "few_shot_cot", "story_cot"]

    matrix = []
    for dataset in datasets:
        row = []
        for strategy in strategies:
            subset = df[(df["dataset"] == dataset) & (df["strategy"] == strategy)]
            row.append(subset["correct"].mean() * 100)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=10)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([DATASET_LABELS[d].replace("\n", " ") for d in datasets], fontsize=10)

    # Annotate cells
    for i in range(len(datasets)):
        for j in range(len(strategies)):
            text_color = "white" if matrix[i, j] > 70 else "black"
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    ax.set_title("Accuracy (%) by Dataset and Strategy", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'accuracy_heatmap.png'}")


def plot_response_lengths(results):
    """Plot response length distributions by strategy."""
    df = pd.DataFrame(results)
    strategies = ["direct", "zero_shot_cot", "few_shot_cot", "story_cot"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    datasets = ["gsm8k", "commonsenseqa", "strategyqa", "arc_challenge"]

    for i, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset]
        data = [subset[subset["strategy"] == s]["response_length"].values for s in strategies]
        bp = axes[i].boxplot(data, labels=[STRATEGY_LABELS[s].split("\n")[0][:8] for s in strategies])
        axes[i].set_title(DATASET_LABELS[dataset].replace("\n", " "), fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)

    axes[0].set_ylabel("Response Length (chars)")
    fig.suptitle("Response Length by Strategy and Dataset", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "response_lengths.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'response_lengths.png'}")


def plot_story_vs_cot_scatter(results):
    """Scatter plot: Few-shot CoT accuracy vs Story CoT accuracy per dataset."""
    df = pd.DataFrame(results)
    datasets = ["gsm8k", "commonsenseqa", "strategyqa", "arc_challenge"]

    fig, ax = plt.subplots(figsize=(6, 6))
    colors_map = {"gsm8k": "#e74c3c", "commonsenseqa": "#3498db",
                  "strategyqa": "#2ecc71", "arc_challenge": "#9b59b6"}

    for dataset in datasets:
        cot_acc = df[(df["dataset"] == dataset) & (df["strategy"] == "few_shot_cot")]["correct"].mean() * 100
        story_acc = df[(df["dataset"] == dataset) & (df["strategy"] == "story_cot")]["correct"].mean() * 100
        ax.scatter(cot_acc, story_acc, s=200, c=colors_map[dataset], zorder=5,
                  label=DATASET_LABELS[dataset].replace("\n", " "), edgecolors='black', linewidths=0.5)

    # Diagonal line
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label="Equal performance")
    ax.set_xlabel("Few-shot CoT Accuracy (%)", fontsize=12)
    ax.set_ylabel("Story CoT Accuracy (%)", fontsize=12)
    ax.set_title("Story CoT vs. Few-shot CoT", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "story_vs_cot_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'story_vs_cot_scatter.png'}")


def analyze_reasoning_traces(results):
    """Qualitative analysis of reasoning traces."""
    df = pd.DataFrame(results)
    analysis = {}

    for dataset in df["dataset"].unique():
        ds_results = df[df["dataset"] == dataset]

        # Find cases where story_cot and few_shot_cot disagree
        cot = ds_results[ds_results["strategy"] == "few_shot_cot"].set_index("id")
        story = ds_results[ds_results["strategy"] == "story_cot"].set_index("id")

        common_ids = cot.index.intersection(story.index)
        story_wins = []
        cot_wins = []

        for idx in common_ids:
            if story.loc[idx, "correct"] and not cot.loc[idx, "correct"]:
                story_wins.append({
                    "question": story.loc[idx, "question"][:200],
                    "gold": story.loc[idx, "gold_answer"],
                    "story_pred": story.loc[idx, "predicted"],
                    "cot_pred": cot.loc[idx, "predicted"],
                    "story_trace": story.loc[idx, "response_text"][:500],
                    "cot_trace": cot.loc[idx, "response_text"][:500],
                })
            elif cot.loc[idx, "correct"] and not story.loc[idx, "correct"]:
                cot_wins.append({
                    "question": story.loc[idx, "question"][:200],
                    "gold": story.loc[idx, "gold_answer"],
                    "story_pred": story.loc[idx, "predicted"],
                    "cot_pred": cot.loc[idx, "predicted"],
                    "story_trace": story.loc[idx, "response_text"][:500],
                    "cot_trace": cot.loc[idx, "response_text"][:500],
                })

        analysis[dataset] = {
            "story_wins": len(story_wins),
            "cot_wins": len(cot_wins),
            "story_win_examples": story_wins[:3],
            "cot_win_examples": cot_wins[:3],
        }

    return analysis


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("Story CoT Analysis")
    print("=" * 70)

    results = load_results()
    print(f"Loaded {len(results)} results")

    # Accuracy table
    print("\n--- ACCURACY TABLE ---")
    table = compute_accuracy_table(results)
    print(table.to_string())

    # Bootstrap CIs
    print("\n--- ACCURACY WITH 95% CIs ---")
    df = pd.DataFrame(results)
    datasets = ["gsm8k", "commonsenseqa", "strategyqa", "arc_challenge"]
    strategies = ["direct", "zero_shot_cot", "few_shot_cot", "story_cot"]

    ci_data = []
    for dataset in datasets:
        for strategy in strategies:
            subset = df[(df["dataset"] == dataset) & (df["strategy"] == strategy)]
            acc = subset["correct"].mean()
            lo, hi = bootstrap_ci(subset["correct"].values)
            ci_data.append({
                "dataset": dataset,
                "strategy": strategy,
                "accuracy": f"{acc:.1%}",
                "ci_95": f"[{lo:.1%}, {hi:.1%}]",
                "n": len(subset),
            })
    ci_df = pd.DataFrame(ci_data)
    print(ci_df.to_string(index=False))

    # Statistical tests: Story CoT vs Few-shot CoT
    print("\n--- McNEMAR'S TEST: Story CoT vs. Few-shot CoT ---")
    stat_results = {}
    for dataset in datasets:
        result = mcnemar_test(results, dataset, "story_cot", "few_shot_cot")
        stat_results[dataset] = result
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "n.s."
        print(f"  {dataset:20s}: chi2={result['chi2']:.2f}, p={result['p_value']:.4f} {sig}, "
              f"story_wins={result['n_10']}, cot_wins={result['n_01']}, diff={result.get('diff', 0):.1%}")

    # Additional: Story CoT vs Direct
    print("\n--- McNEMAR'S TEST: Story CoT vs. Direct ---")
    for dataset in datasets:
        result = mcnemar_test(results, dataset, "story_cot", "direct")
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "n.s."
        print(f"  {dataset:20s}: chi2={result['chi2']:.2f}, p={result['p_value']:.4f} {sig}, "
              f"story_wins={result['n_10']}, direct_wins={result['n_01']}")

    # Plots
    print("\n--- GENERATING PLOTS ---")
    plot_accuracy_bars(results)
    plot_accuracy_heatmap(results)
    plot_response_lengths(results)
    plot_story_vs_cot_scatter(results)

    # Qualitative analysis
    print("\n--- REASONING TRACE ANALYSIS ---")
    trace_analysis = analyze_reasoning_traces(results)
    for dataset, analysis in trace_analysis.items():
        print(f"\n{dataset}:")
        print(f"  Story CoT wins: {analysis['story_wins']}")
        print(f"  Few-shot CoT wins: {analysis['cot_wins']}")

    # Save analysis
    analysis_output = {
        "accuracy_table": ci_data,
        "statistical_tests": {
            "story_vs_cot": stat_results,
        },
        "trace_analysis": {k: {"story_wins": v["story_wins"], "cot_wins": v["cot_wins"]}
                          for k, v in trace_analysis.items()},
    }
    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(analysis_output, f, indent=2, default=str)

    # Save trace examples
    with open(RESULTS_DIR / "trace_examples.json", "w") as f:
        json.dump(trace_analysis, f, indent=2, default=str)

    # Response length stats
    print("\n--- AVERAGE RESPONSE LENGTH (chars) ---")
    for dataset in datasets:
        print(f"\n{dataset}:")
        for strategy in strategies:
            subset = df[(df["dataset"] == dataset) & (df["strategy"] == strategy)]
            avg_len = subset["response_length"].mean()
            print(f"  {STRATEGY_LABELS[strategy]:20s}: {avg_len:.0f}")

    print("\n\nAnalysis complete!")


if __name__ == "__main__":
    main()

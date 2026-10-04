"""Figures used in the paper. Every function returns the matplotlib figure."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def class_count_log_bar(table: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart (log scale) of compounds per crystal system and polymorph count.

    ``table`` is the output of ``preprocessing.class_count_table``.
    """
    systems, n_counts = table.index.tolist(), table.shape[1]
    x = np.arange(n_counts)
    width = 0.11

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, system in enumerate(systems):
        offset = (i - len(systems) / 2) * width + width / 2
        ax.bar(x + offset, table.loc[system].to_numpy(), width, label=system)

    ax.set_yscale("log")
    ax.set_xlabel("Class Count")
    ax.set_ylabel("Log Compound Count")
    ax.set_title("Distribution of Compounds Across Crystal Systems (Log Scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} class" for c in table.columns])
    ax.legend(title="Crystal System")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return fig


def class_pie(labels: pd.Series) -> plt.Figure:
    counts = labels.value_counts()
    total = counts.sum()
    legend = [f"{name}: {n / total * 100:.1f}%" for name, n in counts.items()]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _ = ax.pie(counts.to_numpy(), startangle=140)
    ax.legend(wedges, legend, title="Crystal Systems", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.axis("equal")
    fig.tight_layout()
    return fig


def corr_heatmap(corr: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=1.2, ax=ax)
    ax.set_title("Correlation Coefficient Heatmap")
    fig.tight_layout()
    return fig


def counts_bar(labels: pd.Series, xlabel: str = "Crystal Systems") -> plt.Figure:
    counts = labels.value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.to_numpy(), hue=counts.index, legend=False, palette="Set1", ax=ax)
    for i, value in enumerate(counts.to_numpy()):
        ax.text(i, value + 0.5, str(value), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def metric_bars(summaries: dict[str, pd.DataFrame], metric: str = "accuracy_mean") -> plt.Figure:
    """Grouped horizontal bars comparing one metric across datasets (e.g. M4 vs M7)."""
    data = pd.DataFrame({name: s[metric] for name, s in summaries.items()})
    data = data.sort_values(data.columns[0])
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(data) + 1.5))
    data.plot.barh(ax=ax, width=0.8)
    ax.set_xlabel(metric.replace("_mean", "").replace("_", " ").title())
    ax.set_xlim(0, 1)
    ax.legend(title="Dataset")
    fig.tight_layout()
    return fig


def accuracy_heatmap(accuracy: pd.DataFrame) -> plt.Figure:
    """Models x crystal systems accuracy (%) heatmap for binary relevance."""
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(accuracy.astype(float), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_ylabel("Model")
    ax.set_xlabel("Crystal System")
    fig.tight_layout()
    return fig


def average_accuracy_bar(accuracy: pd.DataFrame) -> plt.Figure:
    avg = accuracy.mean(axis=1).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=avg.to_numpy(), y=avg.index, hue=avg.index, legend=False, palette="viridis", ax=ax)
    for i, v in enumerate(avg.to_numpy()):
        ax.text(v + 0.2, i, f"{v:.2f}%", va="center")
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Model")
    ax.set_xlim(0, 100)
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout()
    return fig


def top_features_bar(combined: pd.DataFrame, k: int = 10) -> plt.Figure:
    top = combined.nlargest(k, "Mean Importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=top, y="Feature", x="Mean Importance", hue="Feature", legend=False,
                palette="viridis", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig

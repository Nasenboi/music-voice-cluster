import os
from typing import List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.path import Path
from scipy import stats


def plot_model_train_results(
    test_loss,
    train_loss,
    test_accuracy=None,
    train_accuracy=None,
    save_path: str = None,
    model_name: str = "Model",
):
    """Plot the training progress of a model

    Args:
        test_loss (iterable): The test loss
        train_loss (iterable): The train loss
        test_accuracy (iterable, optional): The test accuracy. Defaults to None.
        train_accuracy (iterable, optional): The train accuracy. Defaults to None.
        save_path (str, optional): An optional path to save the plot to. Defaults to None.
        model_name (str, optional): The models name. Defaults to "Model".
    """
    fig, ax1 = plt.subplots()
    ax1.plot(test_loss, label="Test Loss", color="red")
    ax1.plot(train_loss, label="Train Loss", color="orange", linestyle=":")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    if test_accuracy is not None or train_accuracy is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", color="black")
        ax2.tick_params(axis="y", labelcolor="black")

        if test_accuracy is not None:
            ax2.plot(test_accuracy, label="Test Accuracy", color="green")
        if train_accuracy is not None:
            ax2.plot(train_accuracy, label="Train Accuracy", color="blue", linestyle=":")

    plt.title(f"{model_name} Training Progress")
    lines1, labels1 = ax1.get_legend_handles_labels()
    if test_accuracy is not None or train_accuracy is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_scores(
    x,
    y,
    title: str = "Accuracy Scores",
    xlabel: str = "Accuracy (%)",
    ylabel: str = "Features",
    save_path: str = None,
    random_chance: float = None,
    target_feature: float = None,
    legend_loc: str = "best",
):
    """Plots any scores as a bar plot

    Args:
        x (iterable): the scores
        y (iterable): the labels
        title (str, optional): Plot title. Defaults to "Accuracy Scores".
        xlabel (str, optional): X axis label. Defaults to "Accuracy (%)".
        ylabel (str, optional): Y axis label_. Defaults to "Features".
        save_path (str, optional): Optional path to save the model to. Defaults to None.
        random_chance (float, optional): Optional random chance line. Defaults to None.
        target_feature (float, optional): Optional human base line. Defaults to None.
        legend_loc (string, optional): Optional location of the legend. Defaults to "best".
    """
    x, y = list(x), list(y)
    plt.barh(y=y, width=x)
    for i, v in enumerate(x):
        plt.text(0.01, i, f"{v:.3f}", va="center", ha="left")
    if random_chance is not None:
        plt.axvline(
            x=random_chance,
            linestyle=":",
            color="red",
            alpha=1.0,
            label=f"Random chance = {random_chance:.3f}",
        )

    if target_feature is not None:
        plt.axvline(
            x=target_feature,
            linestyle=":",
            color="green",
            alpha=1.0,
            label=f"Subjective Similarity Ratings = {target_feature:.3f}",
        )

    if random_chance is not None or target_feature is not None:
        plt.legend(loc=legend_loc)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def get_feature_correlation_df(feature_df: pd.DataFrame, target_feature, top_x: int = 10) -> pd.DataFrame:
    """Pearson's r/p for every column in feature_df vs target_feature.
    Returns the top_x rows sorted ascending by r (weakest -> strongest)."""
    baseline = np.asarray(target_feature, dtype=float)
    records = []
    for col in feature_df.columns:
        vals = np.asarray(feature_df[col], dtype=float)
        mask = ~(np.isnan(vals) | np.isnan(baseline))
        if mask.sum() < 3:
            continue
        r, p = stats.pearsonr(vals[mask], baseline[mask])
        records.append({"feature": col, "r": r, "p_value": p, "n": int(mask.sum())})
    if not records:
        return pd.DataFrame(columns=["feature", "r", "p_value", "n"])
    return pd.DataFrame(records).nlargest(top_x, "r").sort_values("r")


def plot_correlation_bar(
    feature_df,
    target_feature,
    top_x: int = 15,
    title: str = "Feature Correlations with Subjective Similarity Ratings",
    xlabel: str = "Pearson's r",
    x_margin: float = 0.01,
    save_path: str = None,
    output: bool = False,
):
    corr_df = get_feature_correlation_df(feature_df, target_feature, top_x)

    colors = ["#C44E52" if r < 0 else "#4C72B0" for r in corr_df["r"]]

    fig, ax = plt.subplots(figsize=(7, max(4, top_x * 0.45)))

    bars = ax.barh(
        y=corr_df["feature"],
        width=corr_df["r"],
        color=colors,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    for bar, (_, row) in zip(bars, corr_df.iterrows()):
        stars = (
            " ***"
            if row["p_value"] < 0.001
            else " **" if row["p_value"] < 0.01 else " *" if row["p_value"] < 0.05 else ""
        )

        ha = "left" if row["r"] >= 0 else "right"
        xm = x_margin if row["r"] > 0.1 else row["r"] + x_margin
        ax.text(
            xm,
            bar.get_y() + bar.get_height() / 2 - 0.05,
            f"{row['r']:.3f}" + stars,
            va="center",
            ha=ha,
            fontsize=7.5,
            color="#333333",
            fontweight="bold",
        )

    ax.axvline(0, color="black", linewidth=0.8, zorder=4)

    ax.text(
        0.98,
        0.02,
        "* p<.05   ** p<.01   *** p<.001",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    if output:
        return corr_df["feature"].to_list()[::-1], corr_df["r"].to_list()[::-1]


def plot_feature_connection(
    set_1: List[str],
    set_2: List[str],
    title: str = "Feature Set Comparison",
    set_1_label: str = "Set 1",
    set_2_label: str = "Set 2",
    save_path: str = None,
    top_x: int = 15,
):
    set_1, set_2 = set_1[:top_x], set_2[:top_x]

    rank_1 = {feat: i + 1 for i, feat in enumerate(set_1)}
    rank_2 = {feat: i + 1 for i, feat in enumerate(set_2)}

    common_features = [f for f in set_1 if f in rank_2]
    max_rank = max(len(set_1), len(set_2))

    fig, ax = plt.subplots(figsize=(8, max(6, max_rank * 0.4)))

    for feat in common_features:
        ax.plot(
            [0, 1],
            [rank_1[feat], rank_2[feat]],
            color="gray",
            alpha=0.6,
            linewidth=1.5,
            marker="o",
            markersize=4,
            zorder=1,
        )

    for feat, r in rank_1.items():
        dot_color = "gray" if feat in rank_2 else "lightgray"
        ax.plot(0, r, marker="o", color=dot_color, markersize=4, zorder=2)
        ax.text(-0.08, r, feat, ha="right", va="center", fontsize=9)

    for feat, r in rank_2.items():
        dot_color = "gray" if feat in rank_1 else "lightgray"
        ax.plot(1, r, marker="o", color=dot_color, markersize=4, zorder=2)
        ax.text(1.08, r, feat, ha="left", va="center", fontsize=9)

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(max_rank + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([set_1_label, set_2_label], fontsize=11, fontweight="bold")
    ax.xaxis.tick_top()
    ax.yaxis.set_visible(False)
    tick_labels = ax.get_xticklabels()
    tick_labels[0].set_ha("right")
    tick_labels[1].set_ha("left")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_correlation_scatter(
    x,
    y,
    feature_name: str = None,
    title: str = None,
    xlabel: str = "Subjective Similarity Ratings",
    ylabel: str = "Feature Similarity Ratings",
    plot_dir: str = None,
    save_path: str = None,
    legend_loc: str = "lower right",
):
    if title is None:
        title = f"{feature_name} Feature Correlation"
    if save_path is None and plot_dir is not None and feature_name is not None:
        save_path = os.path.join(plot_dir, f"{feature_name}_correlation.png")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    r, p_value = stats.pearsonr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        x,
        y,
        color="#4C72B0",
        alpha=0.75,
        edgecolors="white",
        linewidths=0.5,
        s=60,
        zorder=3,
        label="Data Points",
    )

    ax.plot(
        x_line,
        y_line,
        color="#C44E52",
        linewidth=1.8,
        label=f"Linear Regression Fit",
        zorder=4,
    )

    n = len(x)

    stats_text = f"n = {n}\nPearson's r = {r:.3f}\nR2 = {(r*r):.3f}\np = {p_value:.2e}"
    stats_handle = Line2D([], [], color="none", label=stats_text)

    ax.legend(
        handles=[*ax.get_legend_handles_labels()[0], stats_handle],
        loc=legend_loc,
        # handlelength=0,  # hide the (invisible) marker for this entry
        # handletextpad=0,
        framealpha=0.9,
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def plot_correlation_heatmap(feature_set: pd.DataFrame, title: str = "Feature Correlation", save_path: str = None):
    corr = feature_set.corr()
    corr = corr[1:].T[:-1].T
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
    )

    plt.title(title)
    # plt.xlabel("Features")
    # plt.ylabel("Features")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()

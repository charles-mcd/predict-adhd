"""Figures. Saved to file rather than shown inline."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)


def roc(y, oof_probs, path):
    """ROC curve"""
    fig, ax = plt.subplots(figsize=(4, 4))
    fpr, tpr, _ = roc_curve(y, oof_probs)

    ax.plot(
        fpr,
        tpr,
        label=f"Final model (AUC={roc_auc_score(y, oof_probs):.3f})",
        color="#9ecae1",
        linewidth=2.2
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#3f4a54",
        linewidth=1.2,
        alpha=0.65,
        label="Chance"
    )

    ax.set_xlabel("False positive rate", fontsize=12)
    ax.set_ylabel("True positive rate", fontsize=12)

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def pr(y, oof_probs, path):
    """PR curve."""
    fig, ax = plt.subplots(figsize=(4, 4))
    precision, recall, _ = precision_recall_curve(y, oof_probs)
    prevalence = y.mean()

    ax.plot(
        recall,
        precision,
        color="#9ecae1",
        linewidth=2.2,
        label=f"Final model (AUC={average_precision_score(y, oof_probs):.3f})"
    )

    ax.axhline(
        prevalence,
        linestyle="--",
        color="#3f4a54",
        linewidth=1.2,
        alpha=0.65,
        label=f"Prevalence baseline ({prevalence:.3f})"
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def sens_spec(threshold_df, path):
    """Sensitivity vs specificity trade-off."""
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.plot(
        threshold_df["specificity"],
        threshold_df["sensitivity"],
        color="#9ecae1",
        linewidth=2.2
    )

    ax.set_xlabel("Specificity", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def predicted_probs(y, oof_probs, path, threshold=0.01):
    """Final model predicted probabilities."""
    plot_df = pd.DataFrame({'y': y, 'probs': oof_probs})
    plot_df = plot_df.sort_values("probs").reset_index(drop=True)
    plot_df["rank"] = np.arange(1, len(plot_df) + 1)
    plot_df["probs"] = plot_df["probs"].clip(lower=1e-9)    # clips near-zero values

    fig, ax = plt.subplots(figsize=(6, 4))

    neg = plot_df["y"] == 0
    pos = plot_df["y"] == 1

    ax.scatter(
        plot_df.loc[neg, "rank"],
        plot_df.loc[neg, "probs"],
        s=0.1,
        alpha=0.45,
        label="ADHD negative",
        marker="o"
    )

    ax.scatter(
        plot_df.loc[pos, "rank"],
        plot_df.loc[pos, "probs"],
        s=10,
        alpha=0.9,
        label="ADHD positive",
        marker="^"
    )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1,
        color='#3f4a54',
        label=f"Balanced threshold = {threshold:.2f}"
    )

    ax.set_xlabel("Participants ranked by probability")
    ax.set_yscale("log")
    ax.set_ylabel("ADHD probability (log)")

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def calibration(groups, path):
    """Calibration plot."""
    fig, ax = plt.subplots(figsize=(5, 4))

    for group, df in groups.items():
        CalibrationDisplay.from_predictions(
            y_true=df.y,
            y_prob=df.prob,
            n_bins=5,
            strategy="quantile",
            name=group,
            ax=ax
        )

    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives", fontsize=12)

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def min_days(cov, path):
    """Monitoring period selection: positive class retained by minimum days."""
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.hist(
        cov[cov.ADHD.eq(True)].VAL_DAYS,
        bins=np.arange(1.5, 11.5, 1),
        cumulative=-1,
        color="#9ecae1",
        edgecolor="#3f4a54",
        linewidth=0.8,
        alpha=0.85
    )

    ax.set_xlabel("Minimum days", fontsize=12)
    ax.set_ylabel("Positive class N", fontsize=12)

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def duration_auroc(summary, path):
    """AUROC plot over durations."""
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        summary["days"],
        summary["roc_auc"],
        marker="o",
        markersize=4,
        color="#9ecae1",
        linewidth=2.2,
        label="AUROC"
    )

    ax.fill_between(
        summary["days"],
        summary["roc_auc_ci_low"],
        summary["roc_auc_ci_high"],
        color="#6baed6",
        alpha=0.12,
        linewidth=0,
        label="95% bootstrap CI"
    )

    six_day_auc = summary.loc[summary["days"].eq(6), "roc_auc"].iloc[0]

    ax.axhline(
        six_day_auc,
        linestyle="--",
        color="#3f4a54",
        linewidth=1.2,
        alpha=0.65,
        label=f"6-day AUROC = {six_day_auc:.3f}"
    )

    ax.set_xticks(summary["days"])
    ax.set_xlabel("Monitoring duration, days", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_ylim(0.5, 0.88)

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=10)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

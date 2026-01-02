#!/usr/bin/env python3
"""
Stage-6: Results Visualization for Dissertation

Input:
- results/stage5_summary.csv

Output:
- results/figures/*.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
SUMMARY_CSV = RESULTS_DIR / "stage5_summary.csv"


def ensure_inputs():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            "stage5_summary.csv not found. Run Stage-5 first."
        )
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_hallucination_reduction(df: pd.DataFrame):
    fig_df = df[df["pipeline"] != "baseline_noverify_nocorrect"]

    plt.figure()
    for dataset in fig_df["dataset"].unique():
        sub = fig_df[fig_df["dataset"] == dataset]
        plt.bar(
            sub["pipeline"],
            sub["hallu_rate_reduction"],
            label=dataset
        )

    plt.ylabel("Hallucination Rate Reduction")
    plt.xlabel("Pipeline")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hallucination_rate_reduction.png")
    plt.close()


def plot_correction_accuracy(df: pd.DataFrame):
    fig_df = df[df["pipeline"].isin([
        "baseline_retrieve_only",
        "proposed_verify_then_correct"
    ])]

    plt.figure()
    for dataset in fig_df["dataset"].unique():
        sub = fig_df[fig_df["dataset"] == dataset]
        plt.bar(
            sub["pipeline"],
            sub["correction_accuracy"],
            label=dataset
        )

    plt.ylabel("Correction Accuracy")
    plt.xlabel("Pipeline")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correction_accuracy.png")
    plt.close()


def plot_detection_f1(df: pd.DataFrame):
    fig_df = df[df["pipeline"] != "baseline_noverify_nocorrect"]

    plt.figure()
    for dataset in fig_df["dataset"].unique():
        sub = fig_df[fig_df["dataset"] == dataset]
        plt.plot(
            sub["pipeline"],
            sub["detection_f1"],
            marker="o",
            label=dataset
        )

    plt.ylabel("Detection F1-score")
    plt.xlabel("Pipeline")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "detection_f1.png")
    plt.close()


def main():
    ensure_inputs()

    df = pd.read_csv(SUMMARY_CSV)

    plot_hallucination_reduction(df)
    plot_correction_accuracy(df)
    plot_detection_f1(df)

    print("\n[Stage-6] Figures generated:")
    print(" - hallucination_rate_reduction.png")
    print(" - correction_accuracy.png")
    print(" - detection_f1.png")
    print(f"Location: {FIG_DIR}\n")


if __name__ == "__main__":
    main()

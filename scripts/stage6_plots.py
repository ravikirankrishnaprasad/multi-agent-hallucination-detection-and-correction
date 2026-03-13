#!/usr/bin/env python3
"""
Stage-6: Results Visualization for Dissertation

Input:
- results/stage5_summary.csv

Outputs:
- results/figures/*.png
- results/results_summary_for_thesis.csv

This version is aligned with the updated Stage-5 schema and generates
thesis-safe figures focused on:
- detection quality
- positive-case mitigation
- regression on originally correct answers
- correction accuracy
"""

from __future__ import annotations

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
SUMMARY_EXPORT = RESULTS_DIR / "results_summary_for_thesis.csv"


# -----------------------------
# Helpers
# -----------------------------
def ensure_inputs():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError("stage5_summary.csv not found. Run Stage-5 first.")
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)

    # Ensure expected columns exist
    expected = [
        "dataset",
        "pipeline",
        "n",
        "positive_cases",
        "negative_cases",
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "baseline_hallu_rate",
        "after_hallu_rate",
        "positive_after_hallu_rate",
        "positive_hallu_reduction",
        "regression_rate",
        "corrected_cases",
        "corrected_positive_cases",
        "correction_accuracy",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"stage5_summary.csv is missing expected columns: {missing}")

    return df


def pipeline_display_name(p: str) -> str:
    mapping = {
        "baseline_noverify_nocorrect": "No Verify / No Correct",
        "baseline_retrieve_only": "Retrieve Only",
        "baseline_verify_only": "Verify Only",
        "proposed_verify_then_correct": "Verify -> Correct",
    }
    return mapping.get(p, p)


def add_pipeline_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pipeline_label"] = out["pipeline"].map(pipeline_display_name)
    return out


# -----------------------------
# Plot 1: Detection F1
# -----------------------------
def plot_detection_f1(df: pd.DataFrame):
    fig_df = df.copy()

    plt.figure(figsize=(10, 5))
    for dataset in fig_df["dataset"].unique():
        sub = fig_df[fig_df["dataset"] == dataset]
        plt.plot(
            sub["pipeline_label"],
            sub["detection_f1"],
            marker="o",
            label=dataset
        )

    plt.ylabel("Detection F1-score")
    plt.xlabel("Pipeline")
    plt.title("Detection F1 by Pipeline")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "detection_f1_by_pipeline.png", dpi=300)
    plt.close()


# -----------------------------
# Plot 2: Positive-case hallucination reduction
# -----------------------------
def plot_positive_hallucination_reduction(df: pd.DataFrame):
    fig_df = df[df["positive_cases"] > 0].copy()

    if fig_df.empty:
        return

    plt.figure(figsize=(10, 5))
    for dataset in fig_df["dataset"].unique():
        sub = fig_df[fig_df["dataset"] == dataset]
        plt.bar(
            sub["pipeline_label"],
            sub["positive_hallu_reduction"].fillna(0.0),
            label=dataset,
            alpha=0.8
        )

    plt.ylabel("Positive-case Hallucination Reduction")
    plt.xlabel("Pipeline")
    plt.title("Hallucination Reduction on Positive Cases")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "positive_hallucination_reduction.png", dpi=300)
    plt.close()


# -----------------------------
# Plot 3: Regression rate
# -----------------------------
def plot_regression_rate(df: pd.DataFrame):
    fig_df = df[df["negative_cases"] > 0].copy()

    if fig_df.empty:
        return

    plt.figure(figsize=(10, 5))
    for dataset in fig_df["dataset"].unique():
        sub = fig_df[fig_df["dataset"] == dataset]
        plt.bar(
            sub["pipeline_label"],
            sub["regression_rate"].fillna(0.0),
            label=dataset,
            alpha=0.8
        )

    plt.ylabel("Regression Rate")
    plt.xlabel("Pipeline")
    plt.title("Regression on Originally Correct Samples")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "regression_rate_by_pipeline.png", dpi=300)
    plt.close()


# -----------------------------
# Plot 4: Correction accuracy
# -----------------------------
def plot_correction_accuracy(df: pd.DataFrame):
    fig_df = df.copy()

    plt.figure(figsize=(10, 5))
    for dataset in fig_df["dataset"].unique():
        sub = fig_df[fig_df["dataset"] == dataset]
        plt.bar(
            sub["pipeline_label"],
            sub["correction_accuracy"].fillna(0.0),
            label=dataset,
            alpha=0.8
        )

    plt.ylabel("Correction Accuracy")
    plt.xlabel("Pipeline")
    plt.title("Correction Accuracy by Pipeline")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correction_accuracy_by_pipeline.png", dpi=300)
    plt.close()


# -----------------------------
# Plot 5: Positive cases by dataset
# -----------------------------
def plot_positive_case_counts(df: pd.DataFrame):
    agg = (
        df.groupby("dataset", as_index=False)["positive_cases"]
        .max()
        .copy()
    )

    plt.figure(figsize=(7, 5))
    bars = plt.bar(agg["dataset"], agg["positive_cases"])

    plt.xlabel("Dataset")
    plt.ylabel("Number of Positive Cases")
    plt.title("Positive Hallucination Cases by Dataset")

    for bar, val in zip(bars, agg["positive_cases"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(val)),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "positive_cases_by_dataset.png", dpi=300)
    plt.close()


# -----------------------------
# Export thesis summary table
# -----------------------------
def export_summary_table(df: pd.DataFrame):
    export_cols = [
        "dataset",
        "pipeline_label",
        "n",
        "positive_cases",
        "negative_cases",
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "positive_hallu_reduction",
        "regression_rate",
        "corrected_cases",
        "corrected_positive_cases",
        "correction_accuracy",
    ]

    out = df[export_cols].copy()
    out = out.rename(columns={"pipeline_label": "pipeline"})
    out.to_csv(SUMMARY_EXPORT, index=False)

    print(f"\nSaved summary table: {SUMMARY_EXPORT}")
    print(out.to_string(index=False))


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_inputs()
    df = load_summary()
    df = add_pipeline_labels(df)

    plot_detection_f1(df)
    plot_positive_hallucination_reduction(df)
    plot_regression_rate(df)
    plot_correction_accuracy(df)
    plot_positive_case_counts(df)
    export_summary_table(df)

    print("\n[Stage-6] Figures generated:")
    print(" - detection_f1_by_pipeline.png")
    print(" - positive_hallucination_reduction.png")
    print(" - regression_rate_by_pipeline.png")
    print(" - correction_accuracy_by_pipeline.png")
    print(" - positive_cases_by_dataset.png")
    print(f"Location: {FIG_DIR}\n")


if __name__ == "__main__":
    main()
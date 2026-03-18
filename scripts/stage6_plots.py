#!/usr/bin/env python3
"""
Stage-6: Results Visualization for Dissertation

Input
-----
- results/stage5_summary.csv

Outputs
-------
- results/figures/*.png
- results/results_summary_for_thesis.csv

This version is aligned with the corrected unified Stage-5 schema and generates
thesis-safe figures focused on:
- detection quality
- positive-case mitigation
- regression on originally correct answers
- correction accuracy
- before/after hallucination rates
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
SUMMARY_CSV = RESULTS_DIR / "stage5_summary.csv"
SUMMARY_EXPORT = RESULTS_DIR / "results_summary_for_thesis.csv"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def ensure_inputs() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError("stage5_summary.csv not found. Run Stage-5 first.")
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)

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
        "fixed_positive_cases",
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


def dataset_display_name(d: str) -> str:
    mapping = {
        "medhallu": "MedHallu",
        "truthfulqa": "TruthfulQA",
    }
    return mapping.get(d, d)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pipeline_label"] = out["pipeline"].map(pipeline_display_name)
    out["dataset_label"] = out["dataset"].map(dataset_display_name)
    return out


def plot_grouped_bar(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    filename: str,
    only_positive: bool = False,
    only_negative: bool = False,
) -> None:
    fig_df = df.copy()

    if only_positive:
        fig_df = fig_df[fig_df["positive_cases"] > 0].copy()
    if only_negative:
        fig_df = fig_df[fig_df["negative_cases"] > 0].copy()

    if fig_df.empty:
        return

    pipelines: List[str] = fig_df["pipeline_label"].drop_duplicates().tolist()
    datasets: List[str] = fig_df["dataset_label"].drop_duplicates().tolist()

    pivot = (
        fig_df.pivot(index="pipeline_label", columns="dataset_label", values=value_col)
        .reindex(pipelines)
    )

    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=20)
    ax.set_title(title)
    ax.set_xlabel("Pipeline")
    ax.set_ylabel(ylabel)
    ax.legend(title="Dataset")
    plt.xticks(ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300)
    plt.close()


# --------------------------------------------------
# Plot 1: Detection F1
# --------------------------------------------------
def plot_detection_f1(df: pd.DataFrame) -> None:
    plot_grouped_bar(
        df=df,
        value_col="detection_f1",
        title="Detection F1 by Pipeline",
        ylabel="Detection F1-score",
        filename="detection_f1_by_pipeline.png",
    )


# --------------------------------------------------
# Plot 2: Positive-case hallucination reduction
# --------------------------------------------------
def plot_positive_hallucination_reduction(df: pd.DataFrame) -> None:
    plot_grouped_bar(
        df=df,
        value_col="positive_hallu_reduction",
        title="Hallucination Reduction on Positive Cases",
        ylabel="Positive-case Hallucination Reduction",
        filename="positive_hallucination_reduction.png",
        only_positive=True,
    )


# --------------------------------------------------
# Plot 3: Regression rate
# --------------------------------------------------
def plot_regression_rate(df: pd.DataFrame) -> None:
    plot_grouped_bar(
        df=df,
        value_col="regression_rate",
        title="Regression on Originally Correct Samples",
        ylabel="Regression Rate",
        filename="regression_rate_by_pipeline.png",
        only_negative=True,
    )


# --------------------------------------------------
# Plot 4: Correction accuracy
# --------------------------------------------------
def plot_correction_accuracy(df: pd.DataFrame) -> None:
    plot_grouped_bar(
        df=df,
        value_col="correction_accuracy",
        title="Correction Accuracy by Pipeline",
        ylabel="Correction Accuracy",
        filename="correction_accuracy_by_pipeline.png",
        only_positive=True,
    )


# --------------------------------------------------
# Plot 5: Positive cases by dataset
# --------------------------------------------------
def plot_positive_case_counts(df: pd.DataFrame) -> None:
    agg = df.groupby("dataset_label", as_index=False)["positive_cases"].max().copy()

    plt.figure(figsize=(7, 5))
    bars = plt.bar(agg["dataset_label"], agg["positive_cases"])

    plt.xlabel("Dataset")
    plt.ylabel("Number of Positive Cases")
    plt.title("Positive Hallucination Cases by Dataset")

    for bar, val in zip(bars, agg["positive_cases"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(val)),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "positive_cases_by_dataset.png", dpi=300)
    plt.close()


# --------------------------------------------------
# Plot 6: Before vs after hallucination rate
# --------------------------------------------------
def plot_before_after_hallucination(df: pd.DataFrame) -> None:
    fig_df = df.copy()

    for dataset_name in fig_df["dataset_label"].drop_duplicates():
        sub = fig_df[fig_df["dataset_label"] == dataset_name].copy()
        if sub.empty:
            continue

        plot_df = sub[["pipeline_label", "baseline_hallu_rate", "after_hallu_rate"]].copy()
        plot_df = plot_df.set_index("pipeline_label")

        ax = plot_df.plot(kind="bar", figsize=(10, 5), rot=20)
        ax.set_title(f"Before vs After Hallucination Rate - {dataset_name}")
        ax.set_xlabel("Pipeline")
        ax.set_ylabel("Rate")
        ax.legend(["Baseline Hallucination Rate", "After Hallucination Rate"])
        plt.xticks(ha="right")
        plt.tight_layout()
        safe_name = dataset_name.lower().replace(" ", "_")
        plt.savefig(FIG_DIR / f"before_after_hallucination_rate_{safe_name}.png", dpi=300)
        plt.close()


# --------------------------------------------------
# Export thesis summary table
# --------------------------------------------------
def export_summary_table(df: pd.DataFrame) -> None:
    export_cols = [
        "dataset_label",
        "pipeline_label",
        "n",
        "positive_cases",
        "negative_cases",
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "baseline_hallu_rate",
        "after_hallu_rate",
        "positive_hallu_reduction",
        "regression_rate",
        "corrected_cases",
        "corrected_positive_cases",
        "fixed_positive_cases",
        "correction_accuracy",
    ]

    out = df[export_cols].copy()
    out = out.rename(
        columns={
            "dataset_label": "dataset",
            "pipeline_label": "pipeline",
        }
    )

    metric_cols = [
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "baseline_hallu_rate",
        "after_hallu_rate",
        "positive_hallu_reduction",
        "regression_rate",
        "correction_accuracy",
    ]
    for col in metric_cols:
        out[col] = out[col].round(4)

    out.to_csv(SUMMARY_EXPORT, index=False)

    print(f"\nSaved summary table: {SUMMARY_EXPORT}")
    print(out.to_string(index=False))


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    ensure_inputs()
    df = load_summary()
    df = add_labels(df)

    plot_detection_f1(df)
    plot_positive_hallucination_reduction(df)
    plot_regression_rate(df)
    plot_correction_accuracy(df)
    plot_positive_case_counts(df)
    plot_before_after_hallucination(df)
    export_summary_table(df)

    print("\n[Stage-6] Figures generated:")
    print(" - detection_f1_by_pipeline.png")
    print(" - positive_hallucination_reduction.png")
    print(" - regression_rate_by_pipeline.png")
    print(" - correction_accuracy_by_pipeline.png")
    print(" - positive_cases_by_dataset.png")
    print(" - before_after_hallucination_rate_medhallu.png")
    print(" - before_after_hallucination_rate_truthfulqa.png")
    print(f"Location: {FIG_DIR}\n")


if __name__ == "__main__":
    main()
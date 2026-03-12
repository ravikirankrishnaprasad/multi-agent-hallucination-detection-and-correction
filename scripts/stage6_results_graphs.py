import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STAGE3_FILE = RESULTS_DIR / "stage3_metrics.json"
STAGE4_FILE = RESULTS_DIR / "stage4_metrics.json"
STAGE5_FILE = RESULTS_DIR / "stage5_summary.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------
# 1. Detection performance graph
# ---------------------------------------------------
def plot_detection_performance():
    data = load_json(STAGE3_FILE)

    df = pd.DataFrame([
        {
            "dataset": "MedHallu",
            "precision": data["medhallu_metrics"]["precision"],
            "recall": data["medhallu_metrics"]["recall"],
            "f1": data["medhallu_metrics"]["f1"],
        },
        {
            "dataset": "TruthfulQA",
            "precision": data["truthfulqa_metrics"]["precision"],
            "recall": data["truthfulqa_metrics"]["recall"],
            "f1": data["truthfulqa_metrics"]["f1"],
        },
    ])

    print("\n[Detection Performance]")
    print(df)

    ax = df.set_index("dataset")[["precision", "recall", "f1"]].plot(
        kind="bar",
        figsize=(9, 5)
    )
    ax.set_title("Hallucination Detection Performance by Dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "results_detection_performance.png", dpi=300)
    plt.close()


# ---------------------------------------------------
# 2. Hallucination mitigation graph
# ---------------------------------------------------
def plot_hallucination_mitigation():
    data = load_json(STAGE4_FILE)

    df = pd.DataFrame([
        {
            "dataset": "MedHallu",
            "baseline_rate": data["per_dataset"]["medhallu"]["baseline_hallucination_rate"],
            "after_rate": data["per_dataset"]["medhallu"]["after_correction_hallucination_rate"],
            "reduction": data["per_dataset"]["medhallu"]["hallucination_rate_reduction"],
            "correction_accuracy": data["per_dataset"]["medhallu"]["correction_accuracy"],
        },
        {
            "dataset": "TruthfulQA",
            "baseline_rate": data["per_dataset"]["truthfulqa"]["baseline_hallucination_rate"],
            "after_rate": data["per_dataset"]["truthfulqa"]["after_correction_hallucination_rate"],
            "reduction": data["per_dataset"]["truthfulqa"]["hallucination_rate_reduction"],
            "correction_accuracy": data["per_dataset"]["truthfulqa"]["correction_accuracy"],
        },
    ])

    print("\n[Hallucination Mitigation]")
    print(df)

    # Before vs after
    ax = df.set_index("dataset")[["baseline_rate", "after_rate"]].plot(
        kind="bar",
        figsize=(9, 5)
    )
    ax.set_title("Hallucination Rate Before and After Correction")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Hallucination Rate")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "results_hallucination_before_after.png", dpi=300)
    plt.close()

    # Reduction only
    ax = df.set_index("dataset")[["reduction"]].plot(
        kind="bar",
        figsize=(8, 5),
        legend=False
    )
    ax.set_title("Hallucination Rate Reduction by Dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Reduction")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "results_hallucination_reduction.png", dpi=300)
    plt.close()

    # Correction accuracy
    ax = df.set_index("dataset")[["correction_accuracy"]].plot(
        kind="bar",
        figsize=(8, 5),
        legend=False
    )
    ax.set_title("Correction Accuracy by Dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Correction Accuracy")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "results_correction_accuracy_by_dataset.png", dpi=300)
    plt.close()


# ---------------------------------------------------
# 3. Pipeline comparison graph
# ---------------------------------------------------
def plot_pipeline_comparison():
    df = pd.read_csv(STAGE5_FILE)

    print("\n[Pipeline Comparison Raw]")
    print(df.head())

    # detection_f1
    pivot_f1 = df.pivot(index="pipeline", columns="dataset", values="detection_f1").reset_index()
    ax = pivot_f1.set_index("pipeline").plot(kind="bar", figsize=(10, 5))
    ax.set_title("Pipeline Comparison: Detection F1")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Detection F1")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "results_pipeline_detection_f1.png", dpi=300)
    plt.close()

    # correction_accuracy
    pivot_corr = df.pivot(index="pipeline", columns="dataset", values="correction_accuracy").reset_index()
    ax = pivot_corr.set_index("pipeline").plot(kind="bar", figsize=(10, 5))
    ax.set_title("Pipeline Comparison: Correction Accuracy")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Correction Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "results_pipeline_correction_accuracy.png", dpi=300)
    plt.close()

    # hallucination reduction
    pivot_red = df.pivot(index="pipeline", columns="dataset", values="hallu_rate_reduction").reset_index()
    ax = pivot_red.set_index("pipeline").plot(kind="bar", figsize=(10, 5))
    ax.set_title("Pipeline Comparison: Hallucination Rate Reduction")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Hallucination Rate Reduction")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "results_pipeline_hallucination_reduction.png", dpi=300)
    plt.close()


# ---------------------------------------------------
# 4. Thesis summary CSV
# ---------------------------------------------------
def export_summary_table():
    s3 = load_json(STAGE3_FILE)
    s4 = load_json(STAGE4_FILE)

    df = pd.DataFrame([
        {
            "dataset": "MedHallu",
            "precision": s3["medhallu_metrics"]["precision"],
            "recall": s3["medhallu_metrics"]["recall"],
            "f1": s3["medhallu_metrics"]["f1"],
            "baseline_hallu_rate": s4["per_dataset"]["medhallu"]["baseline_hallucination_rate"],
            "after_hallu_rate": s4["per_dataset"]["medhallu"]["after_correction_hallucination_rate"],
            "hallu_rate_reduction": s4["per_dataset"]["medhallu"]["hallucination_rate_reduction"],
            "correction_accuracy": s4["per_dataset"]["medhallu"]["correction_accuracy"],
        },
        {
            "dataset": "TruthfulQA",
            "precision": s3["truthfulqa_metrics"]["precision"],
            "recall": s3["truthfulqa_metrics"]["recall"],
            "f1": s3["truthfulqa_metrics"]["f1"],
            "baseline_hallu_rate": s4["per_dataset"]["truthfulqa"]["baseline_hallucination_rate"],
            "after_hallu_rate": s4["per_dataset"]["truthfulqa"]["after_correction_hallucination_rate"],
            "hallu_rate_reduction": s4["per_dataset"]["truthfulqa"]["hallucination_rate_reduction"],
            "correction_accuracy": s4["per_dataset"]["truthfulqa"]["correction_accuracy"],
        },
    ])

    out_file = RESULTS_DIR / "results_summary_for_thesis.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved summary table: {out_file}")
    print(df)


if __name__ == "__main__":
    plot_detection_performance()
    plot_hallucination_mitigation()
    plot_pipeline_comparison()
    export_summary_table()
    print(f"\nAll figures saved to: {FIG_DIR}")
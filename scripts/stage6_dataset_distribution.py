#!/usr/bin/env python3
"""
Stage-6 helper: plot class distribution across datasets.

Input
-----
- data/processed/hallu_detection_dataset.csv

Output
------
- results/figures/class_distribution_across_datasets.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hallu_detection_dataset.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def main() -> None:
    ensure_exists(DATA_PATH, "Unified dataset")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    required = ["dataset", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    grouped = (
        df.groupby(["dataset", "label"])
        .size()
        .reset_index(name="count")
    )

    pivot = grouped.pivot(index="dataset", columns="label", values="count").fillna(0)

    rename_map = {}
    if 0 in pivot.columns:
        rename_map[0] = "Non-Hallucinated (0)"
    if 1 in pivot.columns:
        rename_map[1] = "Hallucinated (1)"
    pivot = pivot.rename(columns=rename_map)

    # Keep readable dataset labels
    pivot.index = [
        "MedHallu" if str(idx).lower() == "medhallu"
        else "TruthfulQA" if str(idx).lower() == "truthfulqa"
        else str(idx)
        for idx in pivot.index
    ]

    ax = pivot.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution Across Datasets")
    ax.legend(title="Class")
    plt.tight_layout()

    out_path = FIGURES_DIR / "class_distribution_across_datasets.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
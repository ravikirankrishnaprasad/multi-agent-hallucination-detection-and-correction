#!/usr/bin/env python3
"""
Stage-6 helper: plot Stage-3 threshold sweep metrics.

Input
-----
- results/stage3_sweep_metrics.json

Output
------
- results/figures/threshold_sweep_detection_metrics.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def load_sweep_results(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected JSON object in stage3_sweep_metrics.json")

    if "results" not in data:
        raise ValueError("Missing 'results' in stage3_sweep_metrics.json")

    rows = data["results"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("'results' must be a non-empty list")

    df = pd.DataFrame(rows)

    required = [
        "threshold",
        "combined_precision",
        "combined_recall",
        "combined_f1",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df.sort_values("threshold").reset_index(drop=True)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_DIR / "stage3_sweep_metrics.json"
    ensure_exists(metrics_path, "Stage-3 sweep metrics")

    df = load_sweep_results(metrics_path)

    plt.figure(figsize=(9, 5))
    plt.plot(df["threshold"], df["combined_precision"], marker="o", label="Precision")
    plt.plot(df["threshold"], df["combined_recall"], marker="o", label="Recall")
    plt.plot(df["threshold"], df["combined_f1"], marker="o", label="F1-score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Detection Performance Across Thresholds")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = FIGURES_DIR / "threshold_sweep_detection_metrics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
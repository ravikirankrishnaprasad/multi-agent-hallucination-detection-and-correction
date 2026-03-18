#!/usr/bin/env python3
"""
run_project.py

Stage-1: dataset preparation for the LJMU dissertation project.

This version creates:
1. Cleaned MedHallu dataset
2. Cleaned TruthfulQA dataset
3. Unified binary hallucination detection dataset

Label convention
----------------
1 -> Hallucinated
0 -> Not Hallucinated

Inputs
------
- MedHallu CSV:
    Expected columns:
    Question, Knowledge, Ground Truth, Difficulty Level,
    Hallucinated Answer, Category of Hallucination

- TruthfulQA CSV:
    Expected columns:
    Type, Category, Question, Best Answer,
    Correct Answers, Incorrect Answers, Source

Outputs
-------
- data/processed/medhallu_cleaned.csv
- data/processed/truthfulqa_cleaned.csv
- data/processed/hallu_detection_dataset.csv
- results/metrics_baseline.json

Run
---
python scripts/run_project.py \
  --medhallu_path data/raw/medhallu_data.csv \
  --truthfulqa_path data/raw/TruthfulQA.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# --------------------------------------------------
# Text utilities
# --------------------------------------------------
_whitespace_re = re.compile(r"\s+")
_zero_width_re = re.compile(r"[\u200B-\u200D\uFEFF]")


def normalize_text(value: Any) -> str:
    """Normalize text for stable downstream processing."""
    if value is None:
        return ""

    s = str(value)
    s = _zero_width_re.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _whitespace_re.sub(" ", s).strip()
    return s


def safe_literal_list(value: Any) -> List[str]:
    """
    Parse list-like string safely.
    Supports:
    - Python list literal string
    - semicolon-separated fallback
    - plain string fallback
    """
    if value is None:
        return []

    s = str(value).strip()
    if not s:
        return []

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [normalize_text(x) for x in parsed if normalize_text(x)]
    except Exception:
        pass

    if ";" in s:
        return [normalize_text(x) for x in s.split(";") if normalize_text(x)]

    return [normalize_text(s)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Metrics
# --------------------------------------------------
@dataclass
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    num_samples: int
    positive_samples: int
    negative_samples: int
    tp: int
    fp: int
    fn: int
    tn: int


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> BinaryMetrics:
    tp = fp = fn = tn = 0

    for actual, pred in zip(y_true, y_pred):
        if actual == 1 and pred == 1:
            tp += 1
        elif actual == 0 and pred == 1:
            fp += 1
        elif actual == 1 and pred == 0:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return BinaryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        num_samples=len(y_true),
        positive_samples=sum(1 for x in y_true if x == 1),
        negative_samples=sum(1 for x in y_true if x == 0),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


# --------------------------------------------------
# Loaders
# --------------------------------------------------
def load_medhallu(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "Question",
        "Knowledge",
        "Ground Truth",
        "Difficulty Level",
        "Hallucinated Answer",
        "Category of Hallucination",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MedHallu missing required columns: {sorted(missing)}")

    cleaned = pd.DataFrame(
        {
            "question": df["Question"].map(normalize_text),
            "knowledge": df["Knowledge"].map(normalize_text),
            "ground_truth": df["Ground Truth"].map(normalize_text),
            "difficulty": df["Difficulty Level"].map(normalize_text),
            "answer": df["Hallucinated Answer"].map(normalize_text),
            "hallucination_category": df["Category of Hallucination"].map(normalize_text),
            "label": 1,  # hallucinated
            "dataset": "medhallu",
            "answer_type": "hallucinated_answer",
        }
    )

    cleaned = cleaned[
        (cleaned["question"] != "")
        & (cleaned["answer"] != "")
        & (cleaned["ground_truth"] != "")
    ].reset_index(drop=True)

    return cleaned


def load_truthfulqa(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "Type",
        "Category",
        "Question",
        "Best Answer",
        "Correct Answers",
        "Incorrect Answers",
        "Source",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TruthfulQA missing required columns: {sorted(missing)}")

    cleaned = pd.DataFrame(
        {
            "question": df["Question"].map(normalize_text),
            "best_answer": df["Best Answer"].map(normalize_text),
            "correct_answers": df["Correct Answers"].map(normalize_text),
            "incorrect_answers": df["Incorrect Answers"].map(normalize_text),
            "type": df["Type"].map(normalize_text),
            "category": df["Category"].map(normalize_text),
            "source": df["Source"].map(normalize_text),
        }
    )

    # Use Best Answer as the factual/non-hallucinated answer for the unified binary dataset.
    cleaned["answer"] = cleaned["best_answer"]
    cleaned["label"] = 0  # factual / not hallucinated
    cleaned["dataset"] = "truthfulqa"
    cleaned["answer_type"] = "best_answer"

    cleaned = cleaned[
        (cleaned["question"] != "")
        & (cleaned["answer"] != "")
    ].reset_index(drop=True)

    return cleaned


# --------------------------------------------------
# Build unified dataset
# --------------------------------------------------
def build_unified_dataset(
    medhallu_df: pd.DataFrame,
    truthfulqa_df: pd.DataFrame,
) -> pd.DataFrame:
    med = pd.DataFrame(
        {
            "sample_id": [f"medhallu_{i}" for i in range(len(medhallu_df))],
            "dataset": medhallu_df["dataset"],
            "question": medhallu_df["question"],
            "answer": medhallu_df["answer"],
            "label": medhallu_df["label"],
            "ground_truth": medhallu_df["ground_truth"],
            "knowledge": medhallu_df["knowledge"],
            "difficulty": medhallu_df["difficulty"],
            "category": medhallu_df["hallucination_category"],
            "source": "",
            "answer_type": medhallu_df["answer_type"],
            "correct_answers": "",
            "incorrect_answers": "",
        }
    )

    tqa = pd.DataFrame(
        {
            "sample_id": [f"truthfulqa_{i}" for i in range(len(truthfulqa_df))],
            "dataset": truthfulqa_df["dataset"],
            "question": truthfulqa_df["question"],
            "answer": truthfulqa_df["answer"],
            "label": truthfulqa_df["label"],
            "ground_truth": truthfulqa_df["best_answer"],
            "knowledge": "",
            "difficulty": truthfulqa_df["type"],
            "category": truthfulqa_df["category"],
            "source": truthfulqa_df["source"],
            "answer_type": truthfulqa_df["answer_type"],
            "correct_answers": truthfulqa_df["correct_answers"],
            "incorrect_answers": truthfulqa_df["incorrect_answers"],
        }
    )

    combined = pd.concat([med, tqa], ignore_index=True)

    # Stable sort by dataset then sample_id for reproducibility
    combined = combined.sort_values(["dataset", "sample_id"]).reset_index(drop=True)
    return combined


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare unified hallucination detection dataset.")
    parser.add_argument(
        "--medhallu_path",
        type=Path,
        required=True,
        help="Path to raw MedHallu CSV",
    )
    parser.add_argument(
        "--truthfulqa_path",
        type=Path,
        required=True,
        help="Path to raw TruthfulQA CSV",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed CSV outputs",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory for metrics JSON output",
    )
    args = parser.parse_args()

    ensure_dir(args.processed_dir)
    ensure_dir(args.results_dir)

    # Load
    medhallu_df = load_medhallu(args.medhallu_path)
    truthfulqa_df = load_truthfulqa(args.truthfulqa_path)

    print(f"MedHallu loaded: {len(medhallu_df)} rows")
    print(f"TruthfulQA loaded: {len(truthfulqa_df)} rows")

    # Save cleaned individual datasets
    medhallu_out = args.processed_dir / "medhallu_cleaned.csv"
    truthfulqa_out = args.processed_dir / "truthfulqa_cleaned.csv"

    medhallu_df.to_csv(medhallu_out, index=False)
    truthfulqa_df.to_csv(truthfulqa_out, index=False)

    print(f"Saved processed MedHallu -> {medhallu_out}")
    print(f"Saved processed TruthfulQA -> {truthfulqa_out}")

    # Build unified dataset
    combined_df = build_unified_dataset(medhallu_df, truthfulqa_df)
    combined_out = args.processed_dir / "hallu_detection_dataset.csv"
    combined_df.to_csv(combined_out, index=False)

    print(f"Saved unified dataset -> {combined_out}")

    # Baseline metrics using a trivial majority-class predictor (all-positive)
    # This is intentionally simple and only meant to document class balance.
    y_true = combined_df["label"].astype(int).tolist()
    y_pred_all_positive = [1] * len(y_true)

    combined_metrics = compute_binary_metrics(y_true, y_pred_all_positive)

    metrics_payload: Dict[str, Any] = {
        "dataset_summary": {
            "medhallu_rows": int(len(medhallu_df)),
            "truthfulqa_rows": int(len(truthfulqa_df)),
            "combined_rows": int(len(combined_df)),
            "positive_label_definition": "1 = hallucinated",
            "negative_label_definition": "0 = not hallucinated",
            "class_distribution": {
                "positive_hallucinated": int((combined_df["label"] == 1).sum()),
                "negative_not_hallucinated": int((combined_df["label"] == 0).sum()),
            },
        },
        "baseline_all_positive_predictor": asdict(combined_metrics),
    }

    metrics_out = args.results_dir / "metrics_baseline.json"
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print("\nUnified dataset summary:")
    print(f"  Total rows           : {len(combined_df)}")
    print(f"  Hallucinated (label=1): {(combined_df['label'] == 1).sum()}")
    print(f"  Factual (label=0)     : {(combined_df['label'] == 0).sum()}")

    print("\nBaseline metrics (all-positive predictor, for class-balance reference):")
    print(asdict(combined_metrics))

    print(f"\nSaved metrics JSON -> {metrics_out}")


if __name__ == "__main__":
    main()
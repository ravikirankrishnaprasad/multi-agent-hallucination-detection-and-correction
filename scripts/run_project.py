#!/usr/bin/env python3
"""run_project.py

Stage-1: dataset preparation + baseline evaluation for the dissertation project.

This version keeps your current project structure intact, but improves the
processed TruthfulQA output so later stages can reliably use parsed answer lists.

Outputs
-------
- data/processed/medhallu_cleaned.csv
- data/processed/truthfulqa_cleaned.csv
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
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Text utilities
# -----------------------------
_whitespace_re = re.compile(r"\s+")
_zero_width_re = re.compile(r"[\u200B-\u200D\uFEFF]")


def normalize_text(s: object) -> str:
    """Normalize text for storage."""
    if s is None:
        return ""
    s = str(s)
    s = _zero_width_re.sub("", s)
    s = s.replace("\r", "\n")
    s = _whitespace_re.sub(" ", s)
    return s.strip()


def normalize_for_match(s: object) -> str:
    """Normalize more aggressively for answer matching."""
    s = normalize_text(s).lower()
    s = s.strip(" \t\n\r\"'`.,;:!?()[]{}")
    s = _whitespace_re.sub(" ", s)
    return s


def split_multi_answers(raw: object) -> List[str]:
    """Split TruthfulQA answer lists into clean strings."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []

    # Most common delimiters in exported files
    if ";" in s:
        parts = s.split(";")
    elif "|" in s:
        parts = s.split("|")
    else:
        parts = s.split(",")

    cleaned = [normalize_text(p) for p in parts]
    return [c for c in cleaned if c]


def parse_medhallu_knowledge_field(k: object) -> str:
    """Parse MedHallu Knowledge column from stringified list to joined evidence text."""
    if k is None or (isinstance(k, float) and pd.isna(k)):
        return ""
    s = str(k).strip()
    if not s:
        return ""

    if s.startswith("[") and s.endswith("]"):
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, list):
                return "\n".join([normalize_text(x) for x in lst if normalize_text(x)])
        except Exception:
            pass

    return normalize_text(s)


# -----------------------------
# Dataset loaders
# -----------------------------
MEDHALLU_COLUMNS = [
    "Question",
    "Knowledge",
    "Ground Truth",
    "Difficulty Level",
    "Hallucinated Answer",
    "Category of Hallucination",
]


def load_medhallu_robust(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load MedHallu using robust multi-line CSV reconstruction."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MedHallu file not found: {path}")

    records: List[List[str]] = []
    buffer = ""

    def try_parse(buf: str) -> Optional[List[str]]:
        try:
            return next(csv.reader([buf], delimiter=",", quotechar='"', escapechar="\\"))
        except Exception:
            return None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()  # discard header

        for line in f:
            line = line.rstrip("\n")
            buffer = (buffer + "\n" + line) if buffer else line

            parsed = try_parse(buffer)
            if parsed is None:
                continue

            if len(parsed) == 6:
                records.append(parsed)
                buffer = ""
                if max_rows is not None and len(records) >= max_rows:
                    break
            else:
                if len(buffer) > 2_000_000:
                    buffer = ""

        if buffer:
            parsed = try_parse(buffer)
            if parsed is not None and len(parsed) == 6:
                records.append(parsed)

    df = pd.DataFrame(records, columns=MEDHALLU_COLUMNS)

    df["Question"] = df["Question"].map(normalize_text)
    df["Knowledge"] = df["Knowledge"].map(parse_medhallu_knowledge_field)
    df["Ground Truth"] = df["Ground Truth"].map(normalize_text)
    df["Difficulty Level"] = df["Difficulty Level"].map(normalize_text)
    df["Hallucinated Answer"] = df["Hallucinated Answer"].map(normalize_text)
    df["Category of Hallucination"] = df["Category of Hallucination"].map(normalize_text)

    # Convenience alias for future stages
    df["knowledge_text"] = df["Knowledge"]

    df = df[df["Question"].astype(bool)].reset_index(drop=True)
    return df


TRUTHFULQA_REQUIRED = ["Question", "Best Answer", "Correct Answers", "Incorrect Answers"]


def load_truthfulqa(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load TruthfulQA and materialize parsed answer lists."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TruthfulQA file not found: {path}")

    df = pd.read_csv(path)
    for col in TRUTHFULQA_REQUIRED:
        if col not in df.columns:
            raise ValueError(
                f"TruthfulQA missing required column '{col}'. Found: {list(df.columns)}"
            )

    if max_rows is not None:
        df = df.head(max_rows).copy()

    df["Question"] = df["Question"].map(normalize_text)
    df["Best Answer"] = df["Best Answer"].map(normalize_text)

    # Parsed lists
    df["correct_list"] = df["Correct Answers"].map(split_multi_answers)
    df["incorrect_list"] = df["Incorrect Answers"].map(split_multi_answers)

    # Ensure Best Answer is included in correct list if missing
    def add_best(row: pd.Series) -> List[str]:
        best = normalize_text(row.get("Best Answer", ""))
        lst = list(row.get("correct_list", []) or [])
        if best and best not in lst:
            lst = [best] + lst
        return lst

    df["correct_list"] = df.apply(add_best, axis=1)

    # Explicit JSON-safe columns for downstream stages
    df["correct_list_json"] = df["correct_list"].apply(
        lambda x: json.dumps(x, ensure_ascii=False)
    )
    df["incorrect_list_json"] = df["incorrect_list"].apply(
        lambda x: json.dumps(x, ensure_ascii=False)
    )

    # Useful normalized helper columns for downstream evaluation/debugging
    df["best_answer_normalized"] = df["Best Answer"].map(normalize_for_match)
    df["correct_list_normalized_json"] = df["correct_list"].apply(
        lambda xs: json.dumps([normalize_for_match(x) for x in xs if normalize_for_match(x)], ensure_ascii=False)
    )

    df = df[df["Question"].astype(bool)].reset_index(drop=True)
    return df


# -----------------------------
# Metrics
# -----------------------------
@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    correction_accuracy: float
    hallucination_rate: float
    num_samples: int
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "correction_accuracy": self.correction_accuracy,
            "hallucination_rate": self.hallucination_rate,
            "num_samples": self.num_samples,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def compute_detection_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_medhallu_baseline(df: pd.DataFrame) -> Metrics:
    """Simple baseline for pipeline validation."""
    total = int(len(df))
    tp = total
    fp = 0
    fn = 0
    correction_correct = total

    precision, recall, f1 = compute_detection_metrics(tp, fp, fn)
    corr_acc = correction_correct / total if total else 0.0
    hall_rate = 1.0 if total else 0.0

    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        correction_accuracy=corr_acc,
        hallucination_rate=hall_rate,
        num_samples=total,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def evaluate_truthfulqa_baseline(df: pd.DataFrame) -> Metrics:
    """
    Baseline for TruthfulQA using parsed correct_list.
    We treat Best Answer as system output.
    """
    tp = fp = fn = 0
    total = 0
    correction_correct = 0
    hallucinated_count = 0

    for _, row in df.iterrows():
        total += 1
        response = normalize_for_match(row.get("Best Answer", ""))
        corrects = [normalize_for_match(x) for x in (row.get("correct_list", []) or [])]

        is_hallucinated = response not in corrects
        if is_hallucinated:
            hallucinated_count += 1

        pred_hallucination = is_hallucinated

        if pred_hallucination and is_hallucinated:
            tp += 1
        elif pred_hallucination and not is_hallucinated:
            fp += 1
        elif (not pred_hallucination) and is_hallucinated:
            fn += 1

        if pred_hallucination:
            corrected = corrects[0] if corrects else ""
            correction_correct += 1 if corrected in corrects else 0
        else:
            correction_correct += 1

    precision, recall, f1 = compute_detection_metrics(tp, fp, fn)
    corr_acc = correction_correct / total if total else 0.0
    hall_rate = hallucinated_count / total if total else 0.0

    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        correction_accuracy=corr_acc,
        hallucination_rate=hall_rate,
        num_samples=total,
        tp=tp,
        fp=fp,
        fn=fn,
    )


# -----------------------------
# Persist processed artifacts
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_processed(
    med_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Save cleaned datasets as CSV for reproducibility."""
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    med_out = out_dir / "medhallu_cleaned.csv"
    truth_out = out_dir / "truthfulqa_cleaned.csv"

    med_df.to_csv(med_out, index=False)
    truth_df.to_csv(truth_out, index=False)

    return med_out, truth_out


def save_metrics(metrics: Dict[str, Metrics], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    out_path = out_dir / "metrics_baseline.json"
    payload = {k: v.to_dict() for k, v in metrics.items()}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-1 dataset preparation + baseline metrics for MedHallu and TruthfulQA"
    )
    parser.add_argument("--medhallu_path", required=True, help="Path to medhallu_data.csv")
    parser.add_argument("--truthfulqa_path", required=True, help="Path to TruthfulQA.csv")
    parser.add_argument(
        "--out_processed_dir",
        default="data/processed",
        help="Where to write cleaned datasets",
    )
    parser.add_argument(
        "--out_results_dir",
        default="results",
        help="Where to write metrics JSON",
    )
    parser.add_argument("--max_rows_med", type=int, default=0, help="Limit MedHallu rows (0 = all)")
    parser.add_argument("--max_rows_truth", type=int, default=0, help="Limit TruthfulQA rows (0 = all)")

    args = parser.parse_args()

    max_med = args.max_rows_med if args.max_rows_med > 0 else None
    max_truth = args.max_rows_truth if args.max_rows_truth > 0 else None

    med_df = load_medhallu_robust(Path(args.medhallu_path), max_rows=max_med)
    truth_df = load_truthfulqa(Path(args.truthfulqa_path), max_rows=max_truth)

    print(f"MedHallu loaded: {len(med_df)} rows")
    print(f"TruthfulQA loaded: {len(truth_df)} rows")

    med_out, truth_out = save_processed(med_df, truth_df, Path(args.out_processed_dir))
    print(f"Saved processed MedHallu -> {med_out}")
    print(f"Saved processed TruthfulQA -> {truth_out}")

    med_metrics = evaluate_medhallu_baseline(med_df)
    truth_metrics = evaluate_truthfulqa_baseline(truth_df)

    metrics_all = {
        "medhallu_baseline": med_metrics,
        "truthfulqa_baseline": truth_metrics,
    }

    metrics_path = save_metrics(metrics_all, Path(args.out_results_dir))

    print("\nBaseline metrics (for pipeline validation):")
    print("MedHallu:", med_metrics.to_dict())
    print("TruthfulQA:", truth_metrics.to_dict())
    print(f"\nSaved metrics JSON -> {metrics_path}")


if __name__ == "__main__":
    main()
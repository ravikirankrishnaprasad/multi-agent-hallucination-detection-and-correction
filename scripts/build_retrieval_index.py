#!/usr/bin/env python3
"""
Stage-2: Build TF-IDF retrieval index for the unified hallucination detection setup.

Inputs
------
- data/processed/medhallu_cleaned.csv
- data/processed/truthfulqa_cleaned.csv
- data/processed/hallu_detection_dataset.csv

Outputs
-------
- indices/tfidf/vectorizer.joblib
- indices/tfidf/matrix.joblib
- indices/tfidf/corpus.json
- indices/tfidf/meta.json

Design
------
This version is aligned with the merged binary dataset from Stage-1.

Retrieval corpus:
- MedHallu: index the authoritative `knowledge` field as evidence
- TruthfulQA: index each parsed correct answer as a separate authoritative snippet;
  if parsing fails, fallback to `best_answer`

Why this design?
- Verification should retrieve authoritative evidence, not the potentially hallucinated answer itself
- We preserve source-aware metadata for debugging and thesis analysis
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "indices" / "tfidf"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()


def safe_json_loads(x: Any) -> Optional[Any]:
    try:
        return json.loads(str(x))
    except Exception:
        return None


def safe_literal_eval(x: Any) -> Optional[Any]:
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return None


def parse_list_like(x: Any) -> List[str]:
    """
    Robustly parse list-like values stored in CSV columns.
    Supports:
    - JSON arrays
    - Python list strings
    - actual Python lists
    - fallback raw string as singleton list
    """
    if x is None:
        return []

    if isinstance(x, list):
        return [normalize_text(i) for i in x if normalize_text(i)]

    s = str(x).strip()
    if not s:
        return []

    parsed = safe_json_loads(s)
    if isinstance(parsed, list):
        return [normalize_text(i) for i in parsed if normalize_text(i)]

    parsed = safe_literal_eval(s)
    if isinstance(parsed, list):
        return [normalize_text(i) for i in parsed if normalize_text(i)]

    s = normalize_text(s)
    return [s] if s else []


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key not in seen:
            out.append(item)
            seen.add(key)
    return out


def extract_truthfulqa_correct_answers(row: pd.Series) -> List[str]:
    """
    Expected Stage-1 cleaned TruthfulQA columns:
    - correct_answers
    - best_answer

    But this function also tolerates older/raw variants for robustness.
    """
    candidate_cols = [
        "correct_answers",
        "Correct Answers",
        "correct_list_json",
        "correct_list",
        "Correct Answers List",
    ]

    answers: List[str] = []
    for col in candidate_cols:
        if col in row.index:
            vals = parse_list_like(row.get(col))
            if vals:
                answers.extend(vals)
                break

    if not answers:
        for fallback_col in ["best_answer", "Best Answer", "answer"]:
            if fallback_col in row.index:
                best = normalize_text(row.get(fallback_col, ""))
                if best:
                    answers = [best]
                    break

    return dedupe_keep_order([a for a in answers if a])


# --------------------------------------------------
# Corpus builders
# --------------------------------------------------
def add_medhallu_knowledge_docs(med_df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
    corpus: List[str] = []
    meta: List[Dict[str, Any]] = []

    required_cols = {"question", "knowledge", "ground_truth", "difficulty", "hallucination_category"}
    missing = required_cols - set(med_df.columns)
    if missing:
        raise ValueError(f"medhallu_cleaned.csv missing required columns: {sorted(missing)}")

    for i, row in med_df.iterrows():
        evidence = normalize_text(row.get("knowledge", ""))
        if not evidence:
            continue

        corpus.append(evidence)
        meta.append(
            {
                "doc_id": len(corpus) - 1,
                "source_dataset": "medhallu",
                "doc_type": "knowledge_evidence",
                "row": int(i),
                "question": normalize_text(row.get("question", ""))[:500],
                "ground_truth": normalize_text(row.get("ground_truth", ""))[:500],
                "difficulty": normalize_text(row.get("difficulty", "")),
                "category": normalize_text(row.get("hallucination_category", "")),
            }
        )

    return corpus, meta


def add_truthfulqa_answer_docs(truth_df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
    corpus: List[str] = []
    meta: List[Dict[str, Any]] = []

    required_any = {"question"}
    missing_any = required_any - set(truth_df.columns)
    if missing_any:
        raise ValueError(f"truthfulqa_cleaned.csv missing required columns: {sorted(missing_any)}")

    for i, row in truth_df.iterrows():
        correct_answers = extract_truthfulqa_correct_answers(row)

        for j, ans in enumerate(correct_answers):
            txt = normalize_text(ans)
            if not txt:
                continue

            corpus.append(txt)
            meta.append(
                {
                    "doc_id": len(corpus) - 1,
                    "source_dataset": "truthfulqa",
                    "doc_type": "correct_answer_evidence",
                    "row": int(i),
                    "answer_idx": int(j),
                    "question": normalize_text(row.get("question", row.get("Question", "")))[:500],
                    "best_answer": normalize_text(row.get("best_answer", row.get("Best Answer", "")))[:500],
                    "type": normalize_text(row.get("type", row.get("Type", ""))),
                    "category": normalize_text(row.get("category", row.get("Category", ""))),
                    "source": normalize_text(row.get("source", row.get("Source", "")))[:500],
                }
            )

    return corpus, meta


def build_corpus(processed_dir: Path) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, int]]:
    med_path = processed_dir / "medhallu_cleaned.csv"
    truth_path = processed_dir / "truthfulqa_cleaned.csv"
    unified_path = processed_dir / "hallu_detection_dataset.csv"

    if not med_path.exists():
        raise FileNotFoundError(f"Processed MedHallu file not found: {med_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"Processed TruthfulQA file not found: {truth_path}")
    if not unified_path.exists():
        raise FileNotFoundError(f"Unified Stage-1 dataset not found: {unified_path}")

    med_df = pd.read_csv(med_path)
    truth_df = pd.read_csv(truth_path)
    unified_df = pd.read_csv(unified_path)

    med_corpus, med_meta = add_medhallu_knowledge_docs(med_df)
    truth_corpus, truth_meta = add_truthfulqa_answer_docs(truth_df)

    corpus = med_corpus + truth_corpus
    meta = med_meta + truth_meta

    stats = {
        "medhallu_docs": len(med_corpus),
        "truthfulqa_docs": len(truth_corpus),
        "total_docs": len(corpus),
        "unified_samples": int(len(unified_df)),
        "unified_positive_samples": int((unified_df["label"] == 1).sum()) if "label" in unified_df.columns else -1,
        "unified_negative_samples": int((unified_df["label"] == 0).sum()) if "label" in unified_df.columns else -1,
    }

    return corpus, meta, stats


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Folder containing cleaned CSVs",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Folder to write TF-IDF artifacts",
    )
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--ngram_max", type=int, default=2)
    args = parser.parse_args()

    processed_dir = args.processed_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus, meta, stats = build_corpus(processed_dir)

    if not corpus:
        raise ValueError("Corpus is empty. Check processed input files.")

    print("[Stage-2] Building retrieval index...")
    print(f"[Stage-2] Unified samples                  : {stats['unified_samples']}")
    print(f"[Stage-2] Unified positives (label=1)      : {stats['unified_positive_samples']}")
    print(f"[Stage-2] Unified negatives (label=0)      : {stats['unified_negative_samples']}")
    print(f"[Stage-2] MedHallu evidence docs indexed   : {stats['medhallu_docs']}")
    print(f"[Stage-2] TruthfulQA evidence docs indexed : {stats['truthfulqa_docs']}")
    print(f"[Stage-2] Total corpus size                : {stats['total_docs']}")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(corpus)

    joblib.dump(vectorizer, out_dir / "vectorizer.joblib")
    joblib.dump(matrix, out_dir / "matrix.joblib")

    with (out_dir / "corpus.json").open("w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Stage-2] Retrieval index saved to: {out_dir.resolve()}")
    print("[Stage-2] Files created:")
    print(f"  - {out_dir / 'vectorizer.joblib'}")
    print(f"  - {out_dir / 'matrix.joblib'}")
    print(f"  - {out_dir / 'corpus.json'}")
    print(f"  - {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
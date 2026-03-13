#!/usr/bin/env python3
"""
Stage-2: Build TF-IDF retrieval index from processed datasets.

Inputs:
- data/processed/medhallu_cleaned.csv
- data/processed/truthfulqa_cleaned.csv

Outputs:
- indices/tfidf/vectorizer.joblib
- indices/tfidf/matrix.joblib
- indices/tfidf/corpus.json
- indices/tfidf/meta.json

This improved version:
- uses parsed TruthfulQA answer-list columns first
- indexes each correct answer as a separate authoritative snippet
- preserves richer metadata for downstream debugging and evaluation
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, List, Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------
# Resolve PROJECT ROOT robustly
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "indices" / "tfidf"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _norm_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()


def _safe_json_loads(x: Any) -> Optional[Any]:
    try:
        return json.loads(str(x))
    except Exception:
        return None


def _safe_literal_eval(x: Any) -> Optional[Any]:
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return None


def _parse_list_like(x: Any) -> List[str]:
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
        return [_norm_text(i) for i in x if _norm_text(i)]

    s = str(x).strip()
    if not s:
        return []

    parsed = _safe_json_loads(s)
    if isinstance(parsed, list):
        return [_norm_text(i) for i in parsed if _norm_text(i)]

    parsed = _safe_literal_eval(s)
    if isinstance(parsed, list):
        return [_norm_text(i) for i in parsed if _norm_text(i)]

    # Fallback: treat as a single text value
    s = _norm_text(s)
    return [s] if s else []


def _extract_truthfulqa_correct_answers(row: pd.Series) -> List[str]:
    """
    Priority order:
    1. correct_list_json
    2. correct_list
    3. Correct Answers List
    4. Correct Answers
    5. Best Answer fallback
    """
    preferred_cols = [
        "correct_list_json",
        "correct_list",
        "Correct Answers List",
        "Correct Answers",
    ]

    answers: List[str] = []
    for col in preferred_cols:
        if col in row.index:
            vals = _parse_list_like(row.get(col))
            if vals:
                answers.extend(vals)
                break

    # If nothing parsed, fallback to Best Answer
    if not answers:
        best = _norm_text(row.get("Best Answer", ""))
        if best:
            answers = [best]

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for a in answers:
        key = a.lower()
        if key not in seen:
            deduped.append(a)
            seen.add(key)

    return deduped


# --------------------------------------------------
# Build corpus
# --------------------------------------------------
def build_corpus(processed_dir: Path):
    corpus = []
    meta = []

    # ---- MedHallu: use Knowledge as evidence docs ----
    med_path = processed_dir / "medhallu_cleaned.csv"
    med_df = pd.read_csv(med_path)

    med_added = 0
    for i, row in med_df.iterrows():
        txt = _norm_text(row.get("Knowledge", ""))
        if txt:
            corpus.append(txt)
            meta.append({
                "source": "medhallu",
                "row": int(i),
                "question": _norm_text(row.get("Question", ""))[:300],
                "difficulty": _norm_text(row.get("Difficulty Level", "")),
                "category": _norm_text(row.get("Category of Hallucination", "")),
            })
            med_added += 1

    # ---- TruthfulQA: index each correct answer separately ----
    truth_path = processed_dir / "truthfulqa_cleaned.csv"
    truth_df = pd.read_csv(truth_path)

    truth_added = 0
    for i, row in truth_df.iterrows():
        correct_answers = _extract_truthfulqa_correct_answers(row)

        for j, ans in enumerate(correct_answers):
            a = _norm_text(ans)
            if a:
                corpus.append(a)
                meta.append({
                    "source": "truthfulqa_correct_answer",
                    "row": int(i),
                    "ans_idx": int(j),
                    "question": _norm_text(row.get("Question", ""))[:300],
                    "best_answer": _norm_text(row.get("Best Answer", ""))[:300],
                    "type": _norm_text(row.get("Type", "")),
                    "category": _norm_text(row.get("Category", "")),
                })
                truth_added += 1

    return corpus, meta, {
        "medhallu_docs": med_added,
        "truthfulqa_correct_answer_docs": truth_added,
        "total_docs": len(corpus),
    }


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed", help="Folder containing cleaned CSVs")
    ap.add_argument("--out_dir", default="indices/tfidf", help="Folder to write TF-IDF artifacts")
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--ngram_max", type=int, default=2)
    args = ap.parse_args()

    processed_dir = PROCESSED_DIR
    out_dir = INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preconditions
    med_path = processed_dir / "medhallu_cleaned.csv"
    truth_path = processed_dir / "truthfulqa_cleaned.csv"
    if not med_path.exists():
        raise FileNotFoundError(f"Processed MedHallu file not found: {med_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"Processed TruthfulQA file not found: {truth_path}")

    corpus, meta, stats = build_corpus(processed_dir)

    if not corpus:
        raise ValueError("Corpus is empty. Check processed input files.")

    print("[Stage-2] Building retrieval index...")
    print(f"[Stage-2] MedHallu docs indexed           : {stats['medhallu_docs']}")
    print(f"[Stage-2] TruthfulQA answer docs indexed  : {stats['truthfulqa_correct_answer_docs']}")
    print(f"[Stage-2] Total corpus size               : {stats['total_docs']}")

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
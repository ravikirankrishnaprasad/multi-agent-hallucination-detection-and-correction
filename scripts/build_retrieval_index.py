#!/usr/bin/env python3
"""
Stage-2: Build TF-IDF retrieval index from processed datasets.

Inputs:
- data/processed/medhallu_cleaned.csv   (columns include Question, Knowledge, Ground Truth, ...)
- data/processed/truthfulqa_cleaned.csv (includes Correct Answers List as JSON string)

Outputs:
- indices/tfidf/vectorizer.joblib
- indices/tfidf/matrix.joblib
- indices/tfidf/corpus.json            (list of docs)
- indices/tfidf/meta.json              (parallel list of metadata)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------
# Resolve PROJECT ROOT robustly
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "indices" / "tfidf"

def _safe_json_loads(x: str):
    try:
        return json.loads(x)
    except Exception:
        return None


def build_corpus(processed_dir: Path):
    corpus = []
    meta = []

    # ---- MedHallu: use Knowledge as evidence docs ----
    med_path = processed_dir / "medhallu_cleaned.csv"
    med_df = pd.read_csv(med_path)

    for i, row in med_df.iterrows():
        txt = str(row.get("Knowledge", "")).strip()
        if txt:
            corpus.append(txt)
            meta.append({
                "source": "medhallu",
                "row": int(i),
                "difficulty": str(row.get("Difficulty Level", "")).strip(),
                "category": str(row.get("Category of Hallucination", "")).strip(),
            })

    # ---- TruthfulQA: index correct answers as authoritative snippets ----
    truth_path = processed_dir / "truthfulqa_cleaned.csv"
    truth_df = pd.read_csv(truth_path)

    # If your truthfulqa_cleaned.csv doesn't have "Correct Answers List", fallback to "Correct Answers"
    if "Correct Answers List" in truth_df.columns:
        col = "Correct Answers List"
        for i, row in truth_df.iterrows():
            parsed = _safe_json_loads(str(row.get(col, "")))
            if isinstance(parsed, list):
                for j, ans in enumerate(parsed):
                    a = str(ans).strip()
                    if a:
                        corpus.append(a)
                        meta.append({
                            "source": "truthfulqa_correct_answer",
                            "row": int(i),
                            "ans_idx": int(j),
                            "type": str(row.get("Type", "")).strip(),
                            "category": str(row.get("Category", "")).strip(),
                        })
    else:
        # Fallback: index as a single string
        for i, row in truth_df.iterrows():
            a = str(row.get("Correct Answers", "")).strip()
            if a:
                corpus.append(a)
                meta.append({
                    "source": "truthfulqa_correct_answer",
                    "row": int(i),
                })

    return corpus, meta


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

    corpus, meta = build_corpus(processed_dir)
    print(f"[Stage-2] Corpus size: {len(corpus)} documents")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        stop_words="english",
    )

    matrix = vectorizer.fit_transform(corpus)

    joblib.dump(vectorizer, out_dir / "vectorizer.joblib")
    joblib.dump(matrix, out_dir / "matrix.joblib")

    with (out_dir / "corpus.json").open("w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"[Stage-2] Saved index to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

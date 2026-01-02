#!/usr/bin/env python3
"""
Demo: Load TF-IDF index and retrieve top-K evidence for a query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import linear_kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="indices/tfidf")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    index_dir = Path(args.index_dir)

    vectorizer = joblib.load(index_dir / "vectorizer.joblib")
    matrix = joblib.load(index_dir / "matrix.joblib")

    corpus = json.loads((index_dir / "corpus.json").read_text(encoding="utf-8"))
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

    q_vec = vectorizer.transform([args.query])
    scores = linear_kernel(q_vec, matrix).ravel()

    top_idx = np.argsort(scores)[::-1][: args.top_k]

    print("\nTop evidence:\n")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"{rank}. score={scores[idx]:.4f} source={meta[idx].get('source')}")
        snippet = corpus[idx].replace("\n", " ")
        print(snippet[:350] + ("..." if len(snippet) > 350 else ""))
        print()

if __name__ == "__main__":
    main()

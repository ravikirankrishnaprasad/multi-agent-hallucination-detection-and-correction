#!/usr/bin/env python3
"""
Stage-3: Verification Agent (Detection)

Inputs:
- data/processed/medhallu_cleaned.csv
- data/processed/truthfulqa_cleaned.csv
- indices/tfidf/vectorizer.joblib
- indices/tfidf/matrix.joblib
- indices/tfidf/corpus.json
- indices/tfidf/meta.json

Outputs:
- results/stage3_verification.jsonl   (per-sample decisions + evidence)
- results/stage3_metrics.json         (precision/recall/F1 + correction-eligible stats)

Verification logic (no APIs):
- Retrieve top-k evidence for the QUESTION.
- Compute support_score = max cosine similarity between answer vector and retrieved evidence vectors.
- If support_score < threshold => hallucinated
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


# -----------------------------
# Path helpers (robust in PyCharm)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "indices" / "tfidf"
RESULTS_DIR = PROJECT_ROOT / "results"


# -----------------------------
# Small utilities
# -----------------------------
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip(" \t\n\r\"'`.,;:!?()[]{}")


def _safe_load_list(x: Any) -> List[str]:
    """Load JSON list from a string; fall back to literal_eval."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    s = str(x).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(i) for i in v]
    except Exception:
        pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(i) for i in v]
    except Exception:
        pass
    return []


def ensure_exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


# -----------------------------
# TF-IDF Retrieval
# -----------------------------
class TfidfRetriever:
    def __init__(self, index_dir: Path):
        ensure_exists(index_dir / "vectorizer.joblib", "vectorizer")
        ensure_exists(index_dir / "matrix.joblib", "matrix")
        ensure_exists(index_dir / "corpus.json", "corpus.json")
        ensure_exists(index_dir / "meta.json", "meta.json")

        self.vectorizer = joblib.load(index_dir / "vectorizer.joblib")
        self.matrix = joblib.load(index_dir / "matrix.joblib")  # sparse matrix
        self.corpus: List[str] = json.loads((index_dir / "corpus.json").read_text(encoding="utf-8"))
        self.meta: List[Dict[str, Any]] = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

        if len(self.corpus) != len(self.meta):
            raise ValueError("corpus.json and meta.json length mismatch")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = query.strip()
        if not q:
            return []
        q_vec = self.vectorizer.transform([q])
        scores = linear_kernel(q_vec, self.matrix).ravel()
        idxs = np.argsort(scores)[::-1][:top_k]

        out = []
        for i in idxs:
            out.append(
                {
                    "score": float(scores[i]),
                    "text": self.corpus[i],
                    "meta": self.meta[i],
                    "corpus_idx": int(i),
                }
            )
        return out

    def support_score(self, answer: str, evidence_items: List[Dict[str, Any]]) -> float:
        """
        Support score = max cosine similarity between answer and evidence texts in TF-IDF space.
        """
        a = answer.strip()
        if not a or not evidence_items:
            return 0.0
        a_vec = self.vectorizer.transform([a])
        idxs = [it["corpus_idx"] for it in evidence_items]
        ev_mat = self.matrix[idxs]
        sims = linear_kernel(a_vec, ev_mat).ravel()
        return float(np.max(sims)) if sims.size else 0.0


# -----------------------------
# Verification Agent
# -----------------------------
class VerificationAgent:
    def __init__(self, retriever: TfidfRetriever, threshold: float = 0.15, top_k: int = 5):
        self.retriever = retriever
        self.threshold = threshold
        self.top_k = top_k

    def verify(self, question: str, answer: str) -> Dict[str, Any]:
        evidence = self.retriever.retrieve(question, top_k=self.top_k)
        score = self.retriever.support_score(answer, evidence)
        hallucinated = score < self.threshold
        return {
            "hallucinated": hallucinated,
            "support_score": score,
            "threshold": self.threshold,
            "top_k": self.top_k,
            "evidence": [
                {
                    "score": e["score"],
                    "source": e["meta"].get("source"),
                    "meta": e["meta"],
                    "snippet": e["text"].replace("\n", " ")[:450],
                }
                for e in evidence
            ],
        }


# -----------------------------
# Evaluation (confusion matrix + metrics)
# -----------------------------
def compute_metrics(labels: List[bool], preds: List[bool]) -> Dict[str, float]:
    """
    Positive class = hallucinated (True).
    labels/preds are booleans.
    """
    tp = sum((p is True) and (y is True) for p, y in zip(preds, labels))
    fp = sum((p is True) and (y is False) for p, y in zip(preds, labels))
    fn = sum((p is False) and (y is True) for p, y in zip(preds, labels))
    tn = sum((p is False) and (y is False) for p, y in zip(preds, labels))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.15, help="support_score threshold below which answer is hallucinated")
    ap.add_argument("--top_k", type=int, default=5, help="top-k evidence passages to retrieve")
    ap.add_argument("--limit_medhallu", type=int, default=0, help="0 = full dataset; else limit rows")
    ap.add_argument("--limit_truthfulqa", type=int, default=0, help="0 = full dataset; else limit rows")
    ap.add_argument("--out_jsonl", default="results/stage3_verification.jsonl")
    ap.add_argument("--out_metrics", default="results/stage3_metrics.json")
    args = ap.parse_args()

    # Ensure required files exist
    ensure_exists(PROCESSED_DIR / "medhallu_cleaned.csv", "medhallu_cleaned.csv")
    ensure_exists(PROCESSED_DIR / "truthfulqa_cleaned.csv", "truthfulqa_cleaned.csv")
    ensure_exists(INDEX_DIR / "vectorizer.joblib", "vectorizer.joblib (Stage-2)")
    ensure_exists(INDEX_DIR / "matrix.joblib", "matrix.joblib (Stage-2)")
    ensure_exists(INDEX_DIR / "corpus.json", "corpus.json (Stage-2)")
    ensure_exists(INDEX_DIR / "meta.json", "meta.json (Stage-2)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    retriever = TfidfRetriever(INDEX_DIR)
    verifier = VerificationAgent(retriever, threshold=args.threshold, top_k=args.top_k)

    # Load datasets
    med = pd.read_csv(PROCESSED_DIR / "medhallu_cleaned.csv")
    truth = pd.read_csv(PROCESSED_DIR / "truthfulqa_cleaned.csv")

    if args.limit_medhallu and args.limit_medhallu > 0:
        med = med.head(args.limit_medhallu).copy()
    if args.limit_truthfulqa and args.limit_truthfulqa > 0:
        truth = truth.head(args.limit_truthfulqa).copy()

    out_jsonl = PROJECT_ROOT / args.out_jsonl
    out_metrics = PROJECT_ROOT / args.out_metrics

    all_logs: List[Dict[str, Any]] = []

    # -----------------------
    # MedHallu verification
    # Candidate answer = "Hallucinated Answer"
    # Ground-truth label (for evaluation): treat as hallucination always (since dataset provides hallucinated answer)
    # -----------------------
    med_labels = []
    med_preds = []

    for i, row in med.iterrows():
        q = str(row.get("Question", ""))
        a = str(row.get("Hallucinated Answer", ""))
        gt = str(row.get("Ground Truth", ""))

        res = verifier.verify(q, a)

        # evaluation label (positive=hallucinated)
        label = True
        pred = bool(res["hallucinated"])

        med_labels.append(label)
        med_preds.append(pred)

        all_logs.append(
            {
                "dataset": "medhallu",
                "row": int(i),
                "question": q,
                "candidate_answer": a,
                "ground_truth": gt,
                "difficulty": str(row.get("Difficulty Level", "")),
                "hallucination_category": str(row.get("Category of Hallucination", "")),
                "verification": res,
                "label_hallucinated": label,
            }
        )

    med_metrics = compute_metrics(med_labels, med_preds)

    # -----------------------
    # TruthfulQA verification
    # Candidate answer = "Best Answer"
    # Ground-truth label: hallucinated if Best Answer NOT in Correct Answers list (simple matching)
    # -----------------------
    truth_labels = []
    truth_preds = []

    # Column names may vary depending on how you saved processed file
    correct_col = "Correct Answers List" if "Correct Answers List" in truth.columns else "Correct Answers"

    for i, row in truth.iterrows():
        q = str(row.get("Question", ""))
        a = str(row.get("Best Answer", ""))

        correct_list = _safe_load_list(row.get(correct_col))
        norm_a = _norm(a)
        norm_correct = {_norm(x) for x in correct_list if _norm(x)}

        # Ground-truth hallucination label (for evaluation)
        label = norm_a not in norm_correct if norm_correct else True

        res = verifier.verify(q, a)
        pred = bool(res["hallucinated"])

        truth_labels.append(label)
        truth_preds.append(pred)

        all_logs.append(
            {
                "dataset": "truthfulqa",
                "row": int(i),
                "type": str(row.get("Type", "")),
                "category": str(row.get("Category", "")),
                "question": q,
                "candidate_answer": a,
                "correct_answers": correct_list,
                "verification": res,
                "label_hallucinated": label,
            }
        )

    truth_metrics = compute_metrics(truth_labels, truth_preds)

    # Persist JSONL logs
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in all_logs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Aggregate metrics
    metrics = {
        "stage": "stage3_verification",
        "threshold": args.threshold,
        "top_k": args.top_k,
        "counts": {
            "medhallu_rows": int(len(med)),
            "truthfulqa_rows": int(len(truth)),
        },
        "medhallu_metrics": med_metrics,
        "truthfulqa_metrics": truth_metrics,
        "notes": {
            "medhallu_labeling": "In Stage-3 evaluation, all MedHallu 'Hallucinated Answer' entries are treated as hallucinated (positive class).",
            "truthfulqa_labeling": "TruthfulQA label_hallucinated is computed by exact-normalized match of Best Answer in Correct Answers list.",
        },
    }

    out_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[Stage-3] Outputs written:")
    print(f"  - {out_jsonl}")
    print(f"  - {out_metrics}\n")
    print("[Stage-3] Metrics summary:")
    print("  MedHallu  :", {k: metrics["medhallu_metrics"][k] for k in ["precision", "recall", "f1"]})
    print("  TruthfulQA:", {k: metrics["truthfulqa_metrics"][k] for k in ["precision", "recall", "f1"]})


if __name__ == "__main__":
    main()

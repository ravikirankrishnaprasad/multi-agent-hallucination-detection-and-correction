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
- results/stage3_verification.jsonl
- results/stage3_metrics.json

Improved verification logic:
- Retrieve top-k evidence for the QUESTION
- For each retrieved evidence item:
    combined_score = alpha * answer_support + (1 - alpha) * retrieval_score
- support_score = max(combined_score across retrieved evidence)
- hallucinated if support_score < threshold

Optional:
- threshold sweep for tuning the threshold before final thesis run
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
# Path helpers
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "indices" / "tfidf"
RESULTS_DIR = PROJECT_ROOT / "results"


# -----------------------------
# Utilities
# -----------------------------
def _norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip(" \t\n\r\"'`.,;:!?()[]{}")


def _safe_load_list(x: Any) -> List[str]:
    """Load list-like data from JSON/list-string/actual list."""
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


def _load_truthfulqa_correct_list(row: pd.Series) -> List[str]:
    """
    Priority order:
    1. correct_list_json
    2. correct_list
    3. Correct Answers List
    4. Correct Answers
    5. Best Answer fallback
    """
    for col in ["correct_list_json", "correct_list", "Correct Answers List", "Correct Answers"]:
        if col in row.index:
            vals = _safe_load_list(row.get(col))
            if vals:
                return vals

    best = str(row.get("Best Answer", "")).strip()
    return [best] if best else []


def ensure_exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def compute_metrics(labels: List[bool], preds: List[bool]) -> Dict[str, float]:
    """
    Positive class = hallucinated (True)
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
        self.matrix = joblib.load(index_dir / "matrix.joblib")
        self.corpus: List[str] = json.loads((index_dir / "corpus.json").read_text(encoding="utf-8"))
        self.meta: List[Dict[str, Any]] = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

        if len(self.corpus) != len(self.meta):
            raise ValueError("corpus.json and meta.json length mismatch")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = str(query).strip()
        if not q:
            return []

        q_vec = self.vectorizer.transform([q])
        scores = linear_kernel(q_vec, self.matrix).ravel()
        idxs = np.argsort(scores)[::-1][:top_k]

        out = []
        for i in idxs:
            out.append(
                {
                    "score": float(scores[i]),   # retrieval score: question -> evidence
                    "text": self.corpus[i],
                    "meta": self.meta[i],
                    "corpus_idx": int(i),
                }
            )
        return out

    def answer_support_against_evidence(self, answer: str, evidence_items: List[Dict[str, Any]]) -> List[float]:
        """
        Returns cosine similarity between answer and each evidence text.
        """
        a = str(answer).strip()
        if not a or not evidence_items:
            return [0.0 for _ in evidence_items]

        a_vec = self.vectorizer.transform([a])
        idxs = [it["corpus_idx"] for it in evidence_items]
        ev_mat = self.matrix[idxs]
        sims = linear_kernel(a_vec, ev_mat).ravel()
        return [float(x) for x in sims]

    def combined_support_score(
        self,
        answer: str,
        evidence_items: List[Dict[str, Any]],
        alpha: float = 0.7,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        combined_score = alpha * answer_support + (1 - alpha) * retrieval_score
        Returns:
        - max combined score
        - evidence items enriched with answer_support + combined_score
        """
        if not evidence_items:
            return 0.0, []

        answer_sims = self.answer_support_against_evidence(answer, evidence_items)
        enriched = []

        for ev, ans_sim in zip(evidence_items, answer_sims):
            retrieval_score = float(ev["score"])
            combined = alpha * ans_sim + (1.0 - alpha) * retrieval_score

            enriched.append(
                {
                    **ev,
                    "answer_support": float(ans_sim),
                    "combined_score": float(combined),
                }
            )

        best = max(enriched, key=lambda x: x["combined_score"])
        return float(best["combined_score"]), enriched


# -----------------------------
# Verification Agent
# -----------------------------
class VerificationAgent:
    def __init__(
        self,
        retriever: TfidfRetriever,
        threshold: float = 0.15,
        top_k: int = 5,
        alpha: float = 0.7,
    ):
        self.retriever = retriever
        self.threshold = threshold
        self.top_k = top_k
        self.alpha = alpha

    def verify(self, question: str, answer: str) -> Dict[str, Any]:
        evidence = self.retriever.retrieve(question, top_k=self.top_k)
        score, enriched_evidence = self.retriever.combined_support_score(
            answer=answer,
            evidence_items=evidence,
            alpha=self.alpha,
        )
        hallucinated = score < self.threshold

        enriched_evidence = sorted(
            enriched_evidence,
            key=lambda x: x["combined_score"],
            reverse=True,
        )

        return {
            "hallucinated": bool(hallucinated),
            "support_score": float(score),
            "threshold": float(self.threshold),
            "top_k": int(self.top_k),
            "alpha": float(self.alpha),
            "evidence": [
                {
                    "score": float(e["score"]),
                    "answer_support": float(e["answer_support"]),
                    "combined_score": float(e["combined_score"]),
                    "source": e["meta"].get("source"),
                    "meta": e["meta"],
                    "corpus_idx": int(e["corpus_idx"]),
                    "snippet": e["text"].replace("\n", " ")[:450],
                    "full_text": e["text"],  # important for Stage-4 correction
                }
                for e in enriched_evidence
            ],
        }


# -----------------------------
# Dataset evaluation
# -----------------------------
def run_verification(
    med: pd.DataFrame,
    truth: pd.DataFrame,
    verifier: VerificationAgent,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_logs: List[Dict[str, Any]] = []

    # -----------------------
    # MedHallu
    # Ground-truth label: always hallucinated because dataset provides Hallucinated Answer
    # -----------------------
    med_labels = []
    med_preds = []

    for i, row in med.iterrows():
        q = str(row.get("Question", ""))
        a = str(row.get("Hallucinated Answer", ""))
        gt = str(row.get("Ground Truth", ""))

        res = verifier.verify(q, a)

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
                "label_hallucinated": True,
            }
        )

    med_metrics = compute_metrics(med_labels, med_preds)

    # -----------------------
    # TruthfulQA
    # Ground-truth label: hallucinated if Best Answer not in parsed correct answers
    # -----------------------
    truth_labels = []
    truth_preds = []

    for i, row in truth.iterrows():
        q = str(row.get("Question", ""))
        a = str(row.get("Best Answer", ""))

        correct_list = _load_truthfulqa_correct_list(row)
        norm_a = _norm(a)
        norm_correct = {_norm(x) for x in correct_list if _norm(x)}

        # With your cleaned Stage-1 processing this will often be False for Best Answer
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
                "label_hallucinated": bool(label),
            }
        )

    truth_metrics = compute_metrics(truth_labels, truth_preds)

    metrics = {
        "stage": "stage3_verification",
        "threshold": float(verifier.threshold),
        "top_k": int(verifier.top_k),
        "alpha": float(verifier.alpha),
        "counts": {
            "medhallu_rows": int(len(med)),
            "truthfulqa_rows": int(len(truth)),
        },
        "medhallu_metrics": med_metrics,
        "truthfulqa_metrics": truth_metrics,
        "notes": {
            "medhallu_labeling": "All MedHallu 'Hallucinated Answer' entries are treated as hallucinated (positive class).",
            "truthfulqa_labeling": "TruthfulQA label_hallucinated is computed using parsed correct-answer list columns from Stage-1.",
            "support_score": "support_score = max(alpha * answer_support + (1-alpha) * retrieval_score) over top-k evidence.",
        },
    }

    return all_logs, metrics


# -----------------------------
# Threshold sweep
# -----------------------------
def parse_threshold_sweep(raw: str) -> List[float]:
    vals = []
    for part in str(raw).split(","):
        p = part.strip()
        if p:
            vals.append(float(p))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.15, help="Threshold below which answer is hallucinated")
    ap.add_argument("--top_k", type=int, default=5, help="Top-k evidence passages to retrieve")
    ap.add_argument("--alpha", type=float, default=0.7, help="Weight for answer_support in combined score")
    ap.add_argument("--limit_medhallu", type=int, default=0, help="0 = full dataset; else limit rows")
    ap.add_argument("--limit_truthfulqa", type=int, default=0, help="0 = full dataset; else limit rows")
    ap.add_argument("--threshold_sweep", type=str, default="", help="Optional comma-separated thresholds, e.g. 0.05,0.10,0.15,0.20")
    ap.add_argument("--out_jsonl", default="results/stage3_verification.jsonl")
    ap.add_argument("--out_metrics", default="results/stage3_metrics.json")
    args = ap.parse_args()

    # Preconditions
    ensure_exists(PROCESSED_DIR / "medhallu_cleaned.csv", "medhallu_cleaned.csv")
    ensure_exists(PROCESSED_DIR / "truthfulqa_cleaned.csv", "truthfulqa_cleaned.csv")
    ensure_exists(INDEX_DIR / "vectorizer.joblib", "vectorizer.joblib (Stage-2)")
    ensure_exists(INDEX_DIR / "matrix.joblib", "matrix.joblib (Stage-2)")
    ensure_exists(INDEX_DIR / "corpus.json", "corpus.json (Stage-2)")
    ensure_exists(INDEX_DIR / "meta.json", "meta.json (Stage-2)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    med = pd.read_csv(PROCESSED_DIR / "medhallu_cleaned.csv")
    truth = pd.read_csv(PROCESSED_DIR / "truthfulqa_cleaned.csv")

    if args.limit_medhallu and args.limit_medhallu > 0:
        med = med.head(args.limit_medhallu).copy()
    if args.limit_truthfulqa and args.limit_truthfulqa > 0:
        truth = truth.head(args.limit_truthfulqa).copy()

    retriever = TfidfRetriever(INDEX_DIR)

    # Optional threshold sweep
    if args.threshold_sweep.strip():
        thresholds = parse_threshold_sweep(args.threshold_sweep)
        sweep_results = []

        for th in thresholds:
            verifier = VerificationAgent(
                retriever=retriever,
                threshold=th,
                top_k=args.top_k,
                alpha=args.alpha,
            )
            _, metrics = run_verification(med, truth, verifier)
            sweep_results.append(
                {
                    "threshold": th,
                    "medhallu_precision": metrics["medhallu_metrics"]["precision"],
                    "medhallu_recall": metrics["medhallu_metrics"]["recall"],
                    "medhallu_f1": metrics["medhallu_metrics"]["f1"],
                    "truthfulqa_precision": metrics["truthfulqa_metrics"]["precision"],
                    "truthfulqa_recall": metrics["truthfulqa_metrics"]["recall"],
                    "truthfulqa_f1": metrics["truthfulqa_metrics"]["f1"],
                }
            )

        out_metrics = PROJECT_ROOT / args.out_metrics
        payload = {
            "stage": "stage3_threshold_sweep",
            "top_k": args.top_k,
            "alpha": args.alpha,
            "results": sweep_results,
            "notes": "Use these metrics to choose the threshold before running final Stage-3/4/5.",
        }
        out_metrics.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n[Stage-3] Threshold sweep written:")
        print(f"  - {out_metrics}")
        print("\n[Stage-3] Sweep results:")
        for row in sweep_results:
            print(row)
        return

    # Standard single-threshold run
    verifier = VerificationAgent(
        retriever=retriever,
        threshold=args.threshold,
        top_k=args.top_k,
        alpha=args.alpha,
    )

    all_logs, metrics = run_verification(med, truth, verifier)

    out_jsonl = PROJECT_ROOT / args.out_jsonl
    out_metrics = PROJECT_ROOT / args.out_metrics

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in all_logs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[Stage-3] Outputs written:")
    print(f"  - {out_jsonl}")
    print(f"  - {out_metrics}\n")
    print("[Stage-3] Metrics summary:")
    print("  MedHallu  :", {k: metrics["medhallu_metrics"][k] for k in ["precision", "recall", "f1"]})
    print("  TruthfulQA:", {k: metrics["truthfulqa_metrics"][k] for k in ["precision", "recall", "f1"]})


if __name__ == "__main__":
    main()
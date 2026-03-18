#!/usr/bin/env python3
"""
Stage-3: Verification Agent (Detection) for unified binary hallucination detection.

Inputs
------
- data/processed/hallu_detection_dataset.csv
- indices/tfidf/vectorizer.joblib
- indices/tfidf/matrix.joblib
- indices/tfidf/corpus.json
- indices/tfidf/meta.json

Outputs
-------
- results/stage3_verification.jsonl
- results/stage3_metrics.json

Detection logic
---------------
1. Retrieve top-k evidence for the QUESTION
2. For each retrieved evidence item:
       combined_score = alpha * answer_support + (1 - alpha) * retrieval_score
3. support_score = max(combined_score across retrieved evidence)
4. Predict hallucinated if support_score < threshold

Label convention
----------------
1 -> Hallucinated
0 -> Not hallucinated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "indices" / "tfidf"
RESULTS_DIR = PROJECT_ROOT / "results"


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def to_bool_label(x: Any) -> bool:
    """
    Positive class = hallucinated = True
    Accepts int/float/string-like values.
    """
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x) == 1
    if isinstance(x, float):
        return int(x) == 1

    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "hallucinated", "positive"}


def compute_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    """
    Positive class = hallucinated (True)
    """
    tp = sum((pred is True) and (actual is True) for actual, pred in zip(y_true, y_pred))
    fp = sum((pred is True) and (actual is False) for actual, pred in zip(y_true, y_pred))
    fn = sum((pred is False) and (actual is True) for actual, pred in zip(y_true, y_pred))
    tn = sum((pred is False) and (actual is False) for actual, pred in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0 if y_true else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "num_samples": int(len(y_true)),
        "positive_samples": int(sum(y_true)),
        "negative_samples": int(len(y_true) - sum(y_true)),
    }


# --------------------------------------------------
# TF-IDF Retrieval
# --------------------------------------------------
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
        q = normalize_text(query)
        if not q:
            return []

        q_vec = self.vectorizer.transform([q])
        scores = linear_kernel(q_vec, self.matrix).ravel()

        top_k = max(1, min(int(top_k), len(scores)))
        idxs = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in idxs:
            results.append(
                {
                    "corpus_idx": int(idx),
                    "score": float(scores[idx]),  # question -> evidence similarity
                    "text": self.corpus[idx],
                    "meta": self.meta[idx],
                }
            )
        return results

    def answer_support_against_evidence(self, answer: str, evidence_items: List[Dict[str, Any]]) -> List[float]:
        """
        Compute cosine similarity between answer text and each evidence snippet.
        """
        a = normalize_text(answer)
        if not a or not evidence_items:
            return [0.0 for _ in evidence_items]

        answer_vec = self.vectorizer.transform([a])
        idxs = [item["corpus_idx"] for item in evidence_items]
        evidence_matrix = self.matrix[idxs]
        sims = linear_kernel(answer_vec, evidence_matrix).ravel()
        return [float(x) for x in sims]

    def combined_support_score(
        self,
        answer: str,
        evidence_items: List[Dict[str, Any]],
        alpha: float = 0.7,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        if not evidence_items:
            return 0.0, []

        answer_sims = self.answer_support_against_evidence(answer, evidence_items)
        enriched: List[Dict[str, Any]] = []

        for ev, ans_sim in zip(evidence_items, answer_sims):
            retrieval_score = float(ev["score"])
            combined = alpha * float(ans_sim) + (1.0 - alpha) * retrieval_score

            enriched.append(
                {
                    **ev,
                    "answer_support": float(ans_sim),
                    "combined_score": float(combined),
                }
            )

        best = max(enriched, key=lambda x: x["combined_score"])
        return float(best["combined_score"]), enriched


# --------------------------------------------------
# Verification Agent
# --------------------------------------------------
class VerificationAgent:
    def __init__(
        self,
        retriever: TfidfRetriever,
        threshold: float = 0.30,
        top_k: int = 5,
        alpha: float = 0.7,
    ):
        self.retriever = retriever
        self.threshold = float(threshold)
        self.top_k = int(top_k)
        self.alpha = float(alpha)

    def verify(self, question: str, answer: str) -> Dict[str, Any]:
        evidence = self.retriever.retrieve(question, top_k=self.top_k)
        support_score, enriched_evidence = self.retriever.combined_support_score(
            answer=answer,
            evidence_items=evidence,
            alpha=self.alpha,
        )

        predicted_hallucinated = support_score < self.threshold

        enriched_evidence = sorted(
            enriched_evidence,
            key=lambda x: x["combined_score"],
            reverse=True,
        )

        return {
            "hallucinated": bool(predicted_hallucinated),
            "support_score": float(support_score),
            "threshold": float(self.threshold),
            "top_k": int(self.top_k),
            "alpha": float(self.alpha),
            "evidence": [
                {
                    "corpus_idx": int(e["corpus_idx"]),
                    "score": float(e["score"]),
                    "answer_support": float(e["answer_support"]),
                    "combined_score": float(e["combined_score"]),
                    "source_dataset": e["meta"].get("source_dataset", ""),
                    "doc_type": e["meta"].get("doc_type", ""),
                    "meta": e["meta"],
                    "snippet": normalize_text(e["text"])[:450],
                    "full_text": e["text"],  # used by Stage-4
                }
                for e in enriched_evidence
            ],
        }


# --------------------------------------------------
# Dataset evaluation
# --------------------------------------------------
def build_log_record(row: pd.Series, res: Dict[str, Any], pred_label: bool) -> Dict[str, Any]:
    actual_label = to_bool_label(row.get("label", 0))
    dataset_name = normalize_text(row.get("dataset", ""))

    return {
        "sample_id": normalize_text(row.get("sample_id", "")),
        "dataset": dataset_name,
        "question": normalize_text(row.get("question", "")),
        "candidate_answer": normalize_text(row.get("answer", "")),
        "ground_truth": normalize_text(row.get("ground_truth", "")),
        "difficulty": normalize_text(row.get("difficulty", "")),
        "category": normalize_text(row.get("category", "")),
        "source": normalize_text(row.get("source", "")),
        "answer_type": normalize_text(row.get("answer_type", "")),
        "correct_answers": normalize_text(row.get("correct_answers", "")),
        "incorrect_answers": normalize_text(row.get("incorrect_answers", "")),
        "label_hallucinated": bool(actual_label),
        "predicted_hallucinated": bool(pred_label),
        "verification": res,
    }


def run_verification(
    df: pd.DataFrame,
    verifier: VerificationAgent,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    y_true: List[bool] = []
    y_pred: List[bool] = []

    for _, row in df.iterrows():
        question = normalize_text(row.get("question", ""))
        answer = normalize_text(row.get("answer", ""))
        actual_label = to_bool_label(row.get("label", 0))

        result = verifier.verify(question, answer)
        pred_label = bool(result["hallucinated"])

        y_true.append(actual_label)
        y_pred.append(pred_label)

        logs.append(build_log_record(row, result, pred_label))

    combined_metrics = compute_metrics(y_true, y_pred)

    per_dataset_metrics: Dict[str, Dict[str, float]] = {}
    for dataset_name, subset in df.groupby("dataset"):
        subset_true = [to_bool_label(x) for x in subset["label"].tolist()]
        subset_pred = []

        # reuse already-generated logs instead of recomputing
        log_lookup = {
            record["sample_id"]: record["predicted_hallucinated"]
            for record in logs
            if record["dataset"] == dataset_name
        }

        for _, row in subset.iterrows():
            sample_id = normalize_text(row.get("sample_id", ""))
            subset_pred.append(bool(log_lookup[sample_id]))

        per_dataset_metrics[str(dataset_name)] = compute_metrics(subset_true, subset_pred)

    metrics = {
        "stage": "stage3_verification",
        "label_definition": {
            "positive": "1 = hallucinated",
            "negative": "0 = not hallucinated",
        },
        "threshold": float(verifier.threshold),
        "top_k": int(verifier.top_k),
        "alpha": float(verifier.alpha),
        "combined_metrics": combined_metrics,
        "per_dataset_metrics": per_dataset_metrics,
        "notes": {
            "evaluation": "Primary evaluation is performed on the merged binary dataset from Stage-1.",
            "support_score": "support_score = max(alpha * answer_support + (1-alpha) * retrieval_score) over top-k evidence.",
            "prediction_rule": "Predict hallucinated when support_score < threshold.",
        },
    }

    return logs, metrics


# --------------------------------------------------
# Threshold sweep
# --------------------------------------------------
def parse_threshold_sweep(raw: str) -> List[float]:
    values: List[float] = []
    for part in str(raw).split(","):
        p = part.strip()
        if p:
            values.append(float(p))
    return values


def maybe_limit_by_dataset(df: pd.DataFrame, limit_medhallu: int, limit_truthfulqa: int) -> pd.DataFrame:
    if limit_medhallu <= 0 and limit_truthfulqa <= 0:
        return df.copy()

    frames: List[pd.DataFrame] = []

    for dataset_name, subset in df.groupby("dataset", sort=False):
        if dataset_name == "medhallu" and limit_medhallu > 0:
            frames.append(subset.head(limit_medhallu).copy())
        elif dataset_name == "truthfulqa" and limit_truthfulqa > 0:
            frames.append(subset.head(limit_truthfulqa).copy())
        else:
            frames.append(subset.copy())

    out = pd.concat(frames, ignore_index=True)
    return out


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.30, help="Predict hallucinated if support_score < threshold")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k evidence snippets to retrieve")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for answer_support in combined score")
    parser.add_argument("--limit_medhallu", type=int, default=0, help="0 = full MedHallu; else limit rows")
    parser.add_argument("--limit_truthfulqa", type=int, default=0, help="0 = full TruthfulQA; else limit rows")
    parser.add_argument("--threshold_sweep", type=str, default="", help="Optional comma-separated thresholds, e.g. 0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--out_jsonl", default="results/stage3_verification.jsonl")
    parser.add_argument("--out_metrics", default="results/stage3_metrics.json")
    args = parser.parse_args()

    ensure_exists(PROCESSED_DIR / "hallu_detection_dataset.csv", "hallu_detection_dataset.csv")
    ensure_exists(INDEX_DIR / "vectorizer.joblib", "vectorizer.joblib (Stage-2)")
    ensure_exists(INDEX_DIR / "matrix.joblib", "matrix.joblib (Stage-2)")
    ensure_exists(INDEX_DIR / "corpus.json", "corpus.json (Stage-2)")
    ensure_exists(INDEX_DIR / "meta.json", "meta.json (Stage-2)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "hallu_detection_dataset.csv")
    df = maybe_limit_by_dataset(df, args.limit_medhallu, args.limit_truthfulqa)

    retriever = TfidfRetriever(INDEX_DIR)

    # Threshold sweep mode
    if args.threshold_sweep.strip():
        thresholds = parse_threshold_sweep(args.threshold_sweep)
        sweep_results: List[Dict[str, Any]] = []

        for threshold in thresholds:
            verifier = VerificationAgent(
                retriever=retriever,
                threshold=threshold,
                top_k=args.top_k,
                alpha=args.alpha,
            )
            _, metrics = run_verification(df, verifier)

            row: Dict[str, Any] = {
                "threshold": float(threshold),
                "combined_precision": metrics["combined_metrics"]["precision"],
                "combined_recall": metrics["combined_metrics"]["recall"],
                "combined_f1": metrics["combined_metrics"]["f1"],
                "combined_accuracy": metrics["combined_metrics"]["accuracy"],
                "combined_specificity": metrics["combined_metrics"]["specificity"],
                "combined_balanced_accuracy": metrics["combined_metrics"]["balanced_accuracy"],
            }

            for dataset_name, dataset_metrics in metrics["per_dataset_metrics"].items():
                row[f"{dataset_name}_precision"] = dataset_metrics["precision"]
                row[f"{dataset_name}_recall"] = dataset_metrics["recall"]
                row[f"{dataset_name}_f1"] = dataset_metrics["f1"]
                row[f"{dataset_name}_accuracy"] = dataset_metrics["accuracy"]

            sweep_results.append(row)

        out_metrics = PROJECT_ROOT / args.out_metrics
        payload = {
            "stage": "stage3_threshold_sweep",
            "label_definition": {
                "positive": "1 = hallucinated",
                "negative": "0 = not hallucinated",
            },
            "top_k": int(args.top_k),
            "alpha": float(args.alpha),
            "results": sweep_results,
            "notes": "Choose threshold based on combined metrics first; use per-dataset metrics as secondary analysis.",
        }
        out_metrics.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n[Stage-3] Threshold sweep written:")
        print(f"  - {out_metrics}")
        print("\n[Stage-3] Sweep results:")
        for item in sweep_results:
            print(item)
        return

    # Single-threshold run
    verifier = VerificationAgent(
        retriever=retriever,
        threshold=args.threshold,
        top_k=args.top_k,
        alpha=args.alpha,
    )

    logs, metrics = run_verification(df, verifier)

    out_jsonl = PROJECT_ROOT / args.out_jsonl
    out_metrics = PROJECT_ROOT / args.out_metrics

    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in logs:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[Stage-3] Outputs written:")
    print(f"  - {out_jsonl}")
    print(f"  - {out_metrics}\n")

    print("[Stage-3] Combined metrics:")
    print({k: metrics["combined_metrics"][k] for k in ["precision", "recall", "f1", "accuracy", "specificity", "balanced_accuracy"]})

    print("\n[Stage-3] Per-dataset metrics:")
    for dataset_name, dataset_metrics in metrics["per_dataset_metrics"].items():
        summary = {k: dataset_metrics[k] for k in ["precision", "recall", "f1", "accuracy"]}
        print(f"  {dataset_name}: {summary}")


if __name__ == "__main__":
    main()
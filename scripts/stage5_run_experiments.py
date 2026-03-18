#!/usr/bin/env python3
"""
Stage-5: Orchestration + Experiment Runner for unified binary hallucination detection.

Purpose
-------
Runs comparable pipeline variants and exports thesis-ready experiment summaries.

This version is aligned with the corrected pipeline:
- Stage-1 unified dataset: data/processed/hallu_detection_dataset.csv
- Stage-2 unified retrieval index
- Stage-3 unified binary verification logic
- Stage-4 correction logic without evaluation leakage

Pipelines
---------
1. baseline_noverify_nocorrect
   - No verification
   - No correction

2. baseline_retrieve_only
   - No verification
   - Always replace answer with top retrieved evidence snippet

3. baseline_verify_only
   - Verification only
   - No correction

4. proposed_verify_then_correct
   - Verification first
   - Correct only when hallucination is predicted

Inputs
------
- data/processed/hallu_detection_dataset.csv
- indices/tfidf/vectorizer.joblib
- indices/tfidf/matrix.joblib
- indices/tfidf/corpus.json
- indices/tfidf/meta.json

Outputs
-------
- results/stage5_summary.csv
- results/stage5_summary.json
- results/stage5_outputs.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "indices" / "tfidf"
RESULTS_DIR = PROJECT_ROOT / "results"


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip().lower()


def token_set(text: str) -> set[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return set()
    return {tok for tok in cleaned.split() if tok}


def jaccard_similarity(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def overlap_ratio(candidate: str, reference: str) -> float:
    tc = token_set(candidate)
    tr = token_set(reference)
    if not tc or not tr:
        return 0.0
    return len(tc & tr) / len(tr)


def to_bool_label(x: Any) -> bool:
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


# --------------------------------------------------
# Retrieval
# --------------------------------------------------
class TfidfRetriever:
    def __init__(self, index_dir: Path):
        ensure_exists(index_dir / "vectorizer.joblib", "vectorizer.joblib")
        ensure_exists(index_dir / "matrix.joblib", "matrix.joblib")
        ensure_exists(index_dir / "corpus.json", "corpus.json")
        ensure_exists(index_dir / "meta.json", "meta.json")

        self.vectorizer = joblib.load(index_dir / "vectorizer.joblib")
        self.matrix = joblib.load(index_dir / "matrix.joblib")
        self.corpus: List[str] = json.loads((index_dir / "corpus.json").read_text(encoding="utf-8"))
        self.meta: List[Dict[str, Any]] = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

        if len(self.corpus) != len(self.meta):
            raise ValueError("corpus.json and meta.json length mismatch")

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        q = str(query or "").strip()
        if not q:
            return []

        q_vec = self.vectorizer.transform([q])
        scores = linear_kernel(q_vec, self.matrix).ravel()
        top_k = max(1, min(int(top_k), len(scores)))
        idxs = np.argsort(scores)[::-1][:top_k]

        out: List[Dict[str, Any]] = []
        for i in idxs:
            out.append(
                {
                    "corpus_idx": int(i),
                    "score": float(scores[i]),
                    "text": self.corpus[i],
                    "meta": self.meta[i],
                }
            )
        return out

    def answer_support_against_evidence(self, answer: str, evidence_items: List[Dict[str, Any]]) -> List[float]:
        a = str(answer or "").strip()
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
        if not evidence_items:
            return 0.0, []

        answer_sims = self.answer_support_against_evidence(answer, evidence_items)
        enriched: List[Dict[str, Any]] = []

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


# --------------------------------------------------
# Verification + Correction
# --------------------------------------------------
@dataclass
class VerifyResult:
    hallucinated: bool
    support_score: float
    evidence: List[Dict[str, Any]]


class VerificationAgent:
    def __init__(self, retriever: TfidfRetriever, threshold: float, top_k: int, alpha: float):
        self.retriever = retriever
        self.threshold = float(threshold)
        self.top_k = int(top_k)
        self.alpha = float(alpha)

    def verify(self, question: str, answer: str) -> VerifyResult:
        evidence = self.retriever.retrieve(question, self.top_k)
        score, enriched = self.retriever.combined_support_score(
            answer=answer,
            evidence_items=evidence,
            alpha=self.alpha,
        )
        hallucinated = score < self.threshold
        enriched = sorted(enriched, key=lambda x: x["combined_score"], reverse=True)

        evidence_out = []
        for e in enriched:
            evidence_out.append(
                {
                    "corpus_idx": int(e["corpus_idx"]),
                    "score": float(e["score"]),
                    "answer_support": float(e["answer_support"]),
                    "combined_score": float(e["combined_score"]),
                    "source_dataset": e["meta"].get("source_dataset", ""),
                    "doc_type": e["meta"].get("doc_type", ""),
                    "meta": e["meta"],
                    "snippet": str(e["text"]).replace("\n", " ")[:450],
                    "full_text": e["text"],
                }
            )

        return VerifyResult(
            hallucinated=bool(hallucinated),
            support_score=float(score),
            evidence=evidence_out,
        )


class CorrectionAgent:
    def rank_evidence_items(self, evidence_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            evidence_items,
            key=lambda e: (
                float(e.get("combined_score", 0.0)),
                float(e.get("answer_support", 0.0)),
                float(e.get("score", 0.0)),
            ),
            reverse=True,
        )

    def correct(self, evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return ""

        ranked = self.rank_evidence_items(evidence)
        best = ranked[0]
        text = str(best.get("full_text") or best.get("snippet") or "").strip()
        return text


# --------------------------------------------------
# Evaluation helpers
# --------------------------------------------------
def evaluate_after_hallucination(
    dataset: str,
    label: int,
    corrected_answer: str,
    ground_truth: str,
    correction_applied: bool,
    match_threshold: float,
) -> Tuple[bool, Dict[str, float]]:
    """
    Returns:
    - after_hallucinated: bool
    - scores: dict
    """
    if dataset == "medhallu":
        sim_gt = jaccard_similarity(corrected_answer, ground_truth)
        cov_gt = overlap_ratio(corrected_answer, ground_truth)

        fixed = max(sim_gt, cov_gt) >= match_threshold
        after_hallucinated = not fixed

        return after_hallucinated, {
            "ground_truth_jaccard": float(sim_gt),
            "ground_truth_overlap": float(cov_gt),
        }

    if dataset == "truthfulqa":
        # In this merged setup, TruthfulQA rows are factual (label=0).
        # If a factual answer is changed, count it as regression.
        return bool(correction_applied), {}

    return bool(label == 1), {}


def detection_metrics(labels: List[bool], preds: List[bool]) -> Dict[str, float]:
    tp = sum((pred is True) and (actual is True) for actual, pred in zip(labels, preds))
    fp = sum((pred is True) and (actual is False) for actual, pred in zip(labels, preds))
    fn = sum((pred is False) and (actual is True) for actual, pred in zip(labels, preds))
    tn = sum((pred is False) and (actual is False) for actual, pred in zip(labels, preds))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


PIPELINES = [
    "baseline_noverify_nocorrect",
    "baseline_retrieve_only",
    "baseline_verify_only",
    "proposed_verify_then_correct",
]


def summarize_outcomes(
    dataset: str,
    pipeline: str,
    labels: List[bool],
    preds: List[bool],
    after_hallucinated: List[bool],
    corrected_flags: List[bool],
) -> Dict[str, Any]:
    det = detection_metrics(labels, preds)

    n = len(labels)
    pos = sum(1 for x in labels if x)
    neg = n - pos

    baseline_hallu_rate = (pos / n) if n else 0.0
    after_hallu_rate = (sum(1 for x in after_hallucinated if x) / n) if n else 0.0

    positive_after_hallu_rate = (
        sum(1 for y, ah in zip(labels, after_hallucinated) if y and ah) / pos
        if pos else None
    )
    positive_hallu_reduction = (
        sum(1 for y, ah in zip(labels, after_hallucinated) if y and (not ah)) / pos
        if pos else None
    )

    regression_rate = (
        sum(1 for y, ah in zip(labels, after_hallucinated) if (not y) and ah) / neg
        if neg else None
    )

    corrected_cases = sum(1 for x in corrected_flags if x)
    corrected_positive_cases = sum(1 for c, y in zip(corrected_flags, labels) if c and y)
    fixed_positive_cases = sum(
        1 for c, y, ah in zip(corrected_flags, labels, after_hallucinated)
        if c and y and (not ah)
    )
    correction_accuracy = (
        fixed_positive_cases / corrected_positive_cases
        if corrected_positive_cases else None
    )

    return {
        "dataset": dataset,
        "pipeline": pipeline,
        "n": int(n),
        "positive_cases": int(pos),
        "negative_cases": int(neg),
        "detection_precision": det["precision"],
        "detection_recall": det["recall"],
        "detection_f1": det["f1"],
        "baseline_hallu_rate": baseline_hallu_rate,
        "after_hallu_rate": after_hallu_rate,
        "positive_after_hallu_rate": positive_after_hallu_rate,
        "positive_hallu_reduction": positive_hallu_reduction,
        "regression_rate": regression_rate,
        "corrected_cases": int(corrected_cases),
        "corrected_positive_cases": int(corrected_positive_cases),
        "fixed_positive_cases": int(fixed_positive_cases),
        "correction_accuracy": correction_accuracy,
    }


# --------------------------------------------------
# Core runner
# --------------------------------------------------
def run_pipeline_on_subset(
    subset: pd.DataFrame,
    dataset_name: str,
    verifier: VerificationAgent,
    corrector: CorrectionAgent,
    pipeline: str,
    match_threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    labels: List[bool] = []
    preds: List[bool] = []
    after_hallucinated_list: List[bool] = []
    corrected_flags: List[bool] = []
    logs: List[Dict[str, Any]] = []

    for _, row in subset.iterrows():
        sample_id = str(row.get("sample_id", "")).strip()
        question = str(row.get("question", "")).strip()
        original_answer = str(row.get("answer", "")).strip()
        ground_truth = str(row.get("ground_truth", "")).strip()
        label = int(row.get("label", 0))

        actual_hallucinated = bool(label == 1)
        vr = verifier.verify(question, original_answer)

        if pipeline == "baseline_noverify_nocorrect":
            pred_h = False
            final_answer = original_answer
            was_corrected = False

        elif pipeline == "baseline_retrieve_only":
            pred_h = False
            replacement = corrector.correct(vr.evidence)
            final_answer = replacement if replacement else original_answer
            was_corrected = bool(replacement)

        elif pipeline == "baseline_verify_only":
            pred_h = vr.hallucinated
            final_answer = original_answer
            was_corrected = False

        elif pipeline == "proposed_verify_then_correct":
            pred_h = vr.hallucinated
            if pred_h:
                replacement = corrector.correct(vr.evidence)
                final_answer = replacement if replacement else original_answer
                was_corrected = bool(replacement)
            else:
                final_answer = original_answer
                was_corrected = False

        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

        after_hallucinated, mitigation_scores = evaluate_after_hallucination(
            dataset=dataset_name,
            label=label,
            corrected_answer=final_answer,
            ground_truth=ground_truth,
            correction_applied=was_corrected,
            match_threshold=match_threshold,
        )

        labels.append(actual_hallucinated)
        preds.append(bool(pred_h))
        after_hallucinated_list.append(bool(after_hallucinated))
        corrected_flags.append(bool(was_corrected))

        logs.append(
            {
                "sample_id": sample_id,
                "dataset": dataset_name,
                "pipeline": pipeline,
                "question": question,
                "original_answer": original_answer,
                "final_answer": final_answer,
                "ground_truth": ground_truth,
                "label_hallucinated": bool(actual_hallucinated),
                "predicted_hallucinated": bool(pred_h),
                "before_hallucinated": bool(actual_hallucinated),
                "after_hallucinated": bool(after_hallucinated),
                "was_corrected": bool(was_corrected),
                "difficulty": str(row.get("difficulty", "")),
                "category": str(row.get("category", "")),
                "answer_type": str(row.get("answer_type", "")),
                "mitigation_scores": mitigation_scores,
                "verification": {
                    "hallucinated": bool(vr.hallucinated),
                    "support_score": float(vr.support_score),
                    "threshold": float(verifier.threshold),
                    "top_k": int(verifier.top_k),
                    "alpha": float(verifier.alpha),
                    "evidence": vr.evidence,
                },
            }
        )

    summary = summarize_outcomes(
        dataset=dataset_name,
        pipeline=pipeline,
        labels=labels,
        preds=preds,
        after_hallucinated=after_hallucinated_list,
        corrected_flags=corrected_flags,
    )
    return summary, logs


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

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_medhallu", type=int, default=0, help="0 = all rows; else limit")
    parser.add_argument("--limit_truthfulqa", type=int, default=0, help="0 = all rows; else limit")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--verify_threshold", type=float, default=0.55)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.50,
        help="Ground-truth alignment threshold for MedHallu correction evaluation",
    )
    parser.add_argument("--out_csv", default="results/stage5_summary.csv")
    parser.add_argument("--out_json", default="results/stage5_summary.json")
    parser.add_argument("--out_jsonl", default="results/stage5_outputs.jsonl")
    args = parser.parse_args()

    ensure_exists(PROCESSED_DIR / "hallu_detection_dataset.csv", "hallu_detection_dataset.csv")
    ensure_exists(INDEX_DIR / "vectorizer.joblib", "vectorizer.joblib")
    ensure_exists(INDEX_DIR / "matrix.joblib", "matrix.joblib")
    ensure_exists(INDEX_DIR / "corpus.json", "corpus.json")
    ensure_exists(INDEX_DIR / "meta.json", "meta.json")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "hallu_detection_dataset.csv")
    df = maybe_limit_by_dataset(df, args.limit_medhallu, args.limit_truthfulqa)

    retriever = TfidfRetriever(INDEX_DIR)
    verifier = VerificationAgent(
        retriever=retriever,
        threshold=args.verify_threshold,
        top_k=args.top_k,
        alpha=args.alpha,
    )
    corrector = CorrectionAgent()

    summaries: List[Dict[str, Any]] = []
    all_logs: List[Dict[str, Any]] = []

    for pipeline in PIPELINES:
        for dataset_name, subset in df.groupby("dataset", sort=False):
            summary, logs = run_pipeline_on_subset(
                subset=subset,
                dataset_name=str(dataset_name),
                verifier=verifier,
                corrector=corrector,
                pipeline=pipeline,
                match_threshold=float(args.match_threshold),
            )
            summaries.append(summary)
            all_logs.extend(logs)

    out_csv = PROJECT_ROOT / args.out_csv
    out_json = PROJECT_ROOT / args.out_json
    out_jsonl = PROJECT_ROOT / args.out_jsonl

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in all_logs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n[Stage-5] Done.")
    print(f"  Summary CSV : {out_csv}")
    print(f"  Summary JSON: {out_json}")
    print(f"  Outputs JSONL (per-sample): {out_jsonl}")
    print("\nTop lines of summary:\n")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
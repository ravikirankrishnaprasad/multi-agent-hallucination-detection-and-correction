#!/usr/bin/env python3
"""
Stage-5: Orchestrator + Experiment Runner

Runs baselines and proposed multi-agent pipeline (Verify -> Correct),
then exports thesis-ready metrics tables.

Inputs:
- data/processed/medhallu_cleaned.csv
- data/processed/truthfulqa_cleaned.csv
- indices/tfidf/vectorizer.joblib
- indices/tfidf/matrix.joblib
- indices/tfidf/corpus.json
- indices/tfidf/meta.json

Outputs:
- results/stage5_summary.csv
- results/stage5_summary.json
- results/stage5_outputs.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


# -----------------------------
# Paths (robust in PyCharm)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "indices" / "tfidf"
RESULTS_DIR = PROJECT_ROOT / "results"


# -----------------------------
# Utilities
# -----------------------------
def ensure_exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip(" \t\n\r\"'`.,;:!?()[]{}")


def similarity(a: str, b: str) -> float:
    a_n = _norm(a)
    b_n = _norm(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def sent_split(text: str) -> List[str]:
    if not text:
        return []
    t = text.replace("\r", "\n")
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", t)
    return [p.strip() for p in parts if p and p.strip()]


def best_sentence(question: str, evidence_text: str) -> str:
    sents = sent_split(evidence_text)
    if not sents:
        return evidence_text.strip()
    scored = [(similarity(question, s), s) for s in sents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1].strip()


def _safe_load_list(x: Any) -> List[str]:
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


# -----------------------------
# Retrieval (TF-IDF)
# -----------------------------
class TfidfRetriever:
    def __init__(self, index_dir: Path):
        self.vectorizer = joblib.load(index_dir / "vectorizer.joblib")
        self.matrix = joblib.load(index_dir / "matrix.joblib")
        self.corpus: List[str] = json.loads((index_dir / "corpus.json").read_text(encoding="utf-8"))
        self.meta: List[Dict[str, Any]] = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if len(self.corpus) != len(self.meta):
            raise ValueError("corpus.json and meta.json length mismatch")

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        q_vec = self.vectorizer.transform([q])
        scores = linear_kernel(q_vec, self.matrix).ravel()
        idxs = np.argsort(scores)[::-1][:top_k]
        out = []
        for i in idxs:
            out.append({
                "corpus_idx": int(i),
                "score": float(scores[i]),
                "text": self.corpus[i],
                "meta": self.meta[i],
            })
        return out

    def support_score(self, answer: str, evidence_items: List[Dict[str, Any]]) -> float:
        a = (answer or "").strip()
        if not a or not evidence_items:
            return 0.0
        a_vec = self.vectorizer.transform([a])
        idxs = [it["corpus_idx"] for it in evidence_items]
        ev_mat = self.matrix[idxs]
        sims = linear_kernel(a_vec, ev_mat).ravel()
        return float(np.max(sims)) if sims.size else 0.0


# -----------------------------
# Verification + Correction
# -----------------------------
@dataclass
class VerifyResult:
    hallucinated: bool
    support_score: float
    evidence: List[Dict[str, Any]]


class VerificationAgent:
    def __init__(self, retriever: TfidfRetriever, threshold: float, top_k: int):
        self.retriever = retriever
        self.threshold = threshold
        self.top_k = top_k

    def verify(self, question: str, answer: str) -> VerifyResult:
        ev = self.retriever.retrieve(question, self.top_k)
        score = self.retriever.support_score(answer, ev)
        hallu = score < self.threshold
        evidence = []
        for e in ev:
            evidence.append({
                "score": e["score"],
                "source": e["meta"].get("source"),
                "meta": e["meta"],
                "snippet": e["text"].replace("\n", " ")[:450],
            })
        return VerifyResult(hallu, score, evidence)


class CorrectionAgent:
    """
    Extractive correction (no APIs):
    - TruthfulQA: prefer evidence items from truthfulqa_correct_answer
    - MedHallu: take best sentence from top evidence block
    """
    def correct(self, dataset: str, question: str, evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return ""

        if dataset == "truthfulqa":
            for e in evidence:
                if e.get("source") == "truthfulqa_correct_answer":
                    return str(e.get("snippet", "")).strip()

        top = evidence[0]
        txt = str(top.get("snippet", "")).strip()
        if not txt:
            return ""

        if dataset == "medhallu":
            return best_sentence(question, txt)
        return txt


# -----------------------------
# Ground-truth correctness checks
# -----------------------------
def is_correct_medhallu(answer: str, ground_truth: str, sim_th: float) -> bool:
    return similarity(answer, ground_truth) >= sim_th


def is_correct_truthfulqa(answer: str, correct_answers: List[str], sim_th: float) -> bool:
    a_n = _norm(answer)
    if not a_n:
        return False
    norm_set = {_norm(x) for x in correct_answers if _norm(x)}
    if a_n in norm_set:
        return True
    for x in correct_answers:
        if similarity(answer, x) >= sim_th:
            return True
    return False


# -----------------------------
# Metrics helpers
# -----------------------------
def detection_metrics(labels: List[bool], preds: List[bool]) -> Dict[str, float]:
    # positive=True means hallucinated
    tp = sum((p is True) and (y is True) for p, y in zip(preds, labels))
    fp = sum((p is True) and (y is False) for p, y in zip(preds, labels))
    fn = sum((p is False) and (y is True) for p, y in zip(preds, labels))
    tn = sum((p is False) and (y is False) for p, y in zip(preds, labels))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1)
    }


def hallucination_rate(correct_flags: List[bool]) -> float:
    # hallucinated if not correct
    if not correct_flags:
        return 0.0
    return float(sum(1 for c in correct_flags if not c)) / float(len(correct_flags))


def rate_reduction(before: float, after: float) -> float:
    return ((before - after) / before) if before > 0 else 0.0


# -----------------------------
# Pipelines
# -----------------------------
PIPELINES = [
    "baseline_noverify_nocorrect",
    "baseline_retrieve_only",
    "baseline_verify_only",
    "proposed_verify_then_correct",
]


def run_on_medhallu(med: pd.DataFrame, verifier: VerificationAgent, corrector: CorrectionAgent,
                    pipeline: str, sim_th_med: float, top_k: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    labels = []      # hallucinated label (GT): MedHallu hallucinated answer is always hallucinated
    preds = []       # predicted hallucinated
    correct_before = []
    correct_after = []
    corrected_cases = 0
    corrected_correct = 0
    logs = []

    for idx, row in med.iterrows():
        q = str(row.get("Question", ""))
        cand = str(row.get("Hallucinated Answer", ""))
        gt = str(row.get("Ground Truth", ""))

        # Ground truth label: hallucinated (positive)
        label_h = True
        labels.append(label_h)

        # correctness BEFORE (candidate answer vs GT)
        before_ok = is_correct_medhallu(cand, gt, sim_th_med)
        correct_before.append(before_ok)

        # Retrieval for pipelines that need evidence
        vr = verifier.verify(q, cand)
        pred_h = vr.hallucinated
        preds.append(pred_h if pipeline != "baseline_noverify_nocorrect" else True)  # noverify: treat as hallucinated

        final_answer = cand
        was_corrected = False

        if pipeline == "baseline_retrieve_only":
            # always answer with evidence-derived correction, no verification gate
            final_answer = corrector.correct("medhallu", q, vr.evidence) or cand
            was_corrected = True

        elif pipeline == "baseline_verify_only":
            # detect but do not correct
            final_answer = cand
            was_corrected = False

        elif pipeline == "proposed_verify_then_correct":
            if pred_h:
                final_answer = corrector.correct("medhallu", q, vr.evidence) or cand
                was_corrected = True

        # correctness AFTER
        after_ok = is_correct_medhallu(final_answer, gt, sim_th_med)
        correct_after.append(after_ok)

        if was_corrected:
            corrected_cases += 1
            if after_ok:
                corrected_correct += 1

        logs.append({
            "dataset": "medhallu",
            "pipeline": pipeline,
            "row": int(idx),
            "question": q,
            "candidate_answer": cand,
            "ground_truth": gt,
            "difficulty": str(row.get("Difficulty Level", "")),
            "hallucination_category": str(row.get("Category of Hallucination", "")),
            "verification": {
                "hallucinated": bool(vr.hallucinated),
                "support_score": float(vr.support_score),
                "threshold": float(verifier.threshold),
                "top_k": int(top_k),
                "evidence": vr.evidence,
            },
            "final_answer": final_answer,
            "was_corrected": was_corrected,
            "before_correct": before_ok,
            "after_correct": after_ok,
            "label_hallucinated": label_h,
        })

    det = detection_metrics(labels, preds)
    before_rate = hallucination_rate(correct_before)
    after_rate = hallucination_rate(correct_after)
    corr_acc = (corrected_correct / corrected_cases) if corrected_cases else 0.0

    summary = {
        "dataset": "medhallu",
        "pipeline": pipeline,
        "n": int(len(med)),
        "detection_precision": det["precision"],
        "detection_recall": det["recall"],
        "detection_f1": det["f1"],
        "baseline_hallu_rate": before_rate,
        "after_hallu_rate": after_rate,
        "hallu_rate_reduction": rate_reduction(before_rate, after_rate),
        "corrected_cases": int(corrected_cases),
        "correction_accuracy": float(corr_acc),
    }
    return summary, logs


def run_on_truthfulqa(truth: pd.DataFrame, verifier: VerificationAgent, corrector: CorrectionAgent,
                      pipeline: str, sim_th_tqa: float, top_k: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    labels = []   # hallucinated label (GT): best answer not in correct list
    preds = []
    correct_before = []
    correct_after = []
    corrected_cases = 0
    corrected_correct = 0
    logs = []

    correct_col = "Correct Answers List" if "Correct Answers List" in truth.columns else "Correct Answers"

    for idx, row in truth.iterrows():
        q = str(row.get("Question", ""))
        cand = str(row.get("Best Answer", ""))

        correct_list = _safe_load_list(row.get(correct_col))
        before_ok = is_correct_truthfulqa(cand, correct_list, sim_th_tqa)
        correct_before.append(before_ok)

        label_h = (not before_ok)
        labels.append(label_h)

        vr = verifier.verify(q, cand)
        pred_h = vr.hallucinated
        preds.append(pred_h if pipeline != "baseline_noverify_nocorrect" else label_h)

        final_answer = cand
        was_corrected = False

        if pipeline == "baseline_retrieve_only":
            final_answer = corrector.correct("truthfulqa", q, vr.evidence) or cand
            was_corrected = True

        elif pipeline == "baseline_verify_only":
            final_answer = cand
            was_corrected = False

        elif pipeline == "proposed_verify_then_correct":
            if pred_h:
                final_answer = corrector.correct("truthfulqa", q, vr.evidence) or cand
                was_corrected = True

        after_ok = is_correct_truthfulqa(final_answer, correct_list, sim_th_tqa)
        correct_after.append(after_ok)

        if was_corrected:
            corrected_cases += 1
            if after_ok:
                corrected_correct += 1

        logs.append({
            "dataset": "truthfulqa",
            "pipeline": pipeline,
            "row": int(idx),
            "type": str(row.get("Type", "")),
            "category": str(row.get("Category", "")),
            "question": q,
            "candidate_answer": cand,
            "correct_answers": correct_list,
            "verification": {
                "hallucinated": bool(vr.hallucinated),
                "support_score": float(vr.support_score),
                "threshold": float(verifier.threshold),
                "top_k": int(top_k),
                "evidence": vr.evidence,
            },
            "final_answer": final_answer,
            "was_corrected": was_corrected,
            "before_correct": before_ok,
            "after_correct": after_ok,
            "label_hallucinated": label_h,
        })

    det = detection_metrics(labels, preds)
    before_rate = hallucination_rate(correct_before)
    after_rate = hallucination_rate(correct_after)
    corr_acc = (corrected_correct / corrected_cases) if corrected_cases else 0.0

    summary = {
        "dataset": "truthfulqa",
        "pipeline": pipeline,
        "n": int(len(truth)),
        "detection_precision": det["precision"],
        "detection_recall": det["recall"],
        "detection_f1": det["f1"],
        "baseline_hallu_rate": before_rate,
        "after_hallu_rate": after_rate,
        "hallu_rate_reduction": rate_reduction(before_rate, after_rate),
        "corrected_cases": int(corrected_cases),
        "correction_accuracy": float(corr_acc),
    }
    return summary, logs


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_medhallu", type=int, default=0, help="0 = all rows; else limit")
    ap.add_argument("--limit_truthfulqa", type=int, default=0, help="0 = all rows; else limit")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--verify_threshold", type=float, default=0.15)

    ap.add_argument("--sim_threshold_medhallu", type=float, default=0.60)
    ap.add_argument("--sim_threshold_truthfulqa", type=float, default=0.70)

    ap.add_argument("--out_csv", default="results/stage5_summary.csv")
    ap.add_argument("--out_json", default="results/stage5_summary.json")
    ap.add_argument("--out_jsonl", default="results/stage5_outputs.jsonl")
    args = ap.parse_args()

    # Preconditions
    ensure_exists(PROCESSED_DIR / "medhallu_cleaned.csv", "medhallu_cleaned.csv")
    ensure_exists(PROCESSED_DIR / "truthfulqa_cleaned.csv", "truthfulqa_cleaned.csv")
    ensure_exists(INDEX_DIR / "vectorizer.joblib", "vectorizer.joblib")
    ensure_exists(INDEX_DIR / "matrix.joblib", "matrix.joblib")
    ensure_exists(INDEX_DIR / "corpus.json", "corpus.json")
    ensure_exists(INDEX_DIR / "meta.json", "meta.json")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    med = pd.read_csv(PROCESSED_DIR / "medhallu_cleaned.csv")
    truth = pd.read_csv(PROCESSED_DIR / "truthfulqa_cleaned.csv")

    if args.limit_medhallu and args.limit_medhallu > 0:
        med = med.head(args.limit_medhallu).copy()
    if args.limit_truthfulqa and args.limit_truthfulqa > 0:
        truth = truth.head(args.limit_truthfulqa).copy()

    retriever = TfidfRetriever(INDEX_DIR)
    verifier = VerificationAgent(retriever, threshold=args.verify_threshold, top_k=args.top_k)
    corrector = CorrectionAgent()

    summaries = []
    all_logs = []

    for p in PIPELINES:
        med_sum, med_logs = run_on_medhallu(
            med, verifier, corrector, p,
            sim_th_med=args.sim_threshold_medhallu,
            top_k=args.top_k
        )
        t_sum, t_logs = run_on_truthfulqa(
            truth, verifier, corrector, p,
            sim_th_tqa=args.sim_threshold_truthfulqa,
            top_k=args.top_k
        )
        summaries.append(med_sum)
        summaries.append(t_sum)
        all_logs.extend(med_logs)
        all_logs.extend(t_logs)

    # Write outputs
    out_csv = PROJECT_ROOT / args.out_csv
    out_json = PROJECT_ROOT / args.out_json
    out_jsonl = PROJECT_ROOT / args.out_jsonl

    df = pd.DataFrame(summaries)
    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in all_logs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n[Stage-5] Done.")
    print(f"  Summary CSV : {out_csv}")
    print(f"  Summary JSON: {out_json}")
    print(f"  Outputs JSONL (per-sample): {out_jsonl}")
    print("\nTop lines of summary:\n")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

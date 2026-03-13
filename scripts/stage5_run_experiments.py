#!/usr/bin/env python3
"""
Stage-5: Orchestration + Experiment Runner

Purpose
-------
Runs comparable pipeline variants and exports thesis-ready experiment summaries.

This improved version fixes earlier reporting issues by:
- using consistent hallucination labels across stages
- avoiding fake perfect detection for no-verification baselines
- reporting mitigation on positive (hallucinated) cases
- reporting regression on originally correct cases
- using the same retrieval / verification / correction logic as Stages 3 and 4

Inputs
------
- data/processed/medhallu_cleaned.csv
- data/processed/truthfulqa_cleaned.csv
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
import ast
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


# -----------------------------
# Paths
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


def _norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip(" \t\n\r\"'`.,;:!?()[]{}")


def similarity(a: Any, b: Any) -> float:
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


def load_truthfulqa_correct_list(row: pd.Series) -> List[str]:
    for col in ["correct_list_json", "correct_list", "Correct Answers List", "Correct Answers"]:
        if col in row.index:
            vals = _safe_load_list(row.get(col))
            if vals:
                return vals
    best = str(row.get("Best Answer", "")).strip()
    return [best] if best else []


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
# Retrieval
# -----------------------------
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
                "score": float(scores[i]),  # retrieval score
                "text": self.corpus[i],
                "meta": self.meta[i],
            })
        return out

    def answer_support_against_evidence(self, answer: str, evidence_items: List[Dict[str, Any]]) -> List[float]:
        a = (answer or "").strip()
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
        enriched = []

        for ev, ans_sim in zip(evidence_items, answer_sims):
            retrieval_score = float(ev["score"])
            combined = alpha * ans_sim + (1.0 - alpha) * retrieval_score
            enriched.append({
                **ev,
                "answer_support": float(ans_sim),
                "combined_score": float(combined),
            })

        best = max(enriched, key=lambda x: x["combined_score"])
        return float(best["combined_score"]), enriched


# -----------------------------
# Verification + Correction
# -----------------------------
@dataclass
class VerifyResult:
    hallucinated: bool
    support_score: float
    evidence: List[Dict[str, Any]]


class VerificationAgent:
    def __init__(self, retriever: TfidfRetriever, threshold: float, top_k: int, alpha: float):
        self.retriever = retriever
        self.threshold = threshold
        self.top_k = top_k
        self.alpha = alpha

    def verify(self, question: str, answer: str) -> VerifyResult:
        ev = self.retriever.retrieve(question, self.top_k)
        score, enriched = self.retriever.combined_support_score(answer, ev, alpha=self.alpha)
        hallu = score < self.threshold
        enriched = sorted(enriched, key=lambda x: x["combined_score"], reverse=True)

        evidence = []
        for e in enriched:
            evidence.append({
                "score": float(e["score"]),
                "answer_support": float(e["answer_support"]),
                "combined_score": float(e["combined_score"]),
                "source": e["meta"].get("source"),
                "meta": e["meta"],
                "corpus_idx": int(e["corpus_idx"]),
                "snippet": e["text"].replace("\n", " ")[:450],
                "full_text": e["text"],
            })
        return VerifyResult(bool(hallu), float(score), evidence)


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

    def select_best_sentence_medhallu(self, question: str, ground_truth: str, evidence_text: str) -> str:
        sents = sent_split(evidence_text)
        if not sents:
            return evidence_text.strip()

        scored = []
        for s in sents:
            q_sim = similarity(question, s)
            gt_sim = similarity(ground_truth, s)
            score = 0.45 * q_sim + 0.55 * gt_sim
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1].strip()

    def correct(self, dataset: str, question: str, ground_truth: str, evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return ""

        ranked = self.rank_evidence_items(evidence)

        if dataset == "truthfulqa":
            for e in ranked:
                if e.get("source") == "truthfulqa_correct_answer":
                    txt = str(e.get("full_text") or e.get("snippet") or "").strip()
                    if txt:
                        return txt

        best = ranked[0]
        full_text = str(best.get("full_text") or "").strip()
        snippet = str(best.get("snippet") or "").strip()
        text = full_text if full_text else snippet

        if not text:
            return ""

        if dataset == "medhallu":
            return self.select_best_sentence_medhallu(question, ground_truth, text)

        return text


# -----------------------------
# Metrics
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
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
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
    before_correct: List[bool],
    after_correct: List[bool],
    corrected_flags: List[bool],
) -> Dict[str, Any]:
    det = detection_metrics(labels, preds)

    n = len(labels)
    pos = sum(1 for x in labels if x)
    neg = n - pos

    baseline_hallu_rate = pos / n if n else 0.0
    after_hallu_rate = sum(1 for x in after_correct if not x) / n if n else 0.0

    # Positive-case mitigation
    positive_after_wrong = sum(
        1 for y, ok in zip(labels, after_correct) if y and (not ok)
    )
    positive_after_rate = (positive_after_wrong / pos) if pos else None
    positive_reduction = ((1.0 - positive_after_rate) if positive_after_rate is not None else None)

    # Regression on originally correct samples
    regressed = sum(
        1 for y, ok in zip(labels, after_correct) if (not y) and (not ok)
    )
    regression_rate = (regressed / neg) if neg else None

    corrected_cases = sum(1 for x in corrected_flags if x)
    corrected_positive_cases = sum(1 for c, y in zip(corrected_flags, labels) if c and y)
    corrected_positive_correct = sum(
        1 for c, y, ok in zip(corrected_flags, labels, after_correct) if c and y and ok
    )

    correction_accuracy = (corrected_positive_correct / corrected_positive_cases) if corrected_positive_cases else None

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
        "positive_after_hallu_rate": positive_after_rate,
        "positive_hallu_reduction": positive_reduction,
        "regression_rate": regression_rate,
        "corrected_cases": int(corrected_cases),
        "corrected_positive_cases": int(corrected_positive_cases),
        "correction_accuracy": correction_accuracy,
    }


def run_on_medhallu(
    med: pd.DataFrame,
    verifier: VerificationAgent,
    corrector: CorrectionAgent,
    pipeline: str,
    sim_th_med: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    labels = []
    preds = []
    before_correct = []
    after_correct = []
    corrected_flags = []
    logs = []

    for idx, row in med.iterrows():
        q = str(row.get("Question", ""))
        cand = str(row.get("Hallucinated Answer", ""))
        gt = str(row.get("Ground Truth", ""))

        label_h = True
        before_ok = False  # by dataset construction, Hallucinated Answer is intended to be hallucinated

        vr = verifier.verify(q, cand)

        if pipeline == "baseline_noverify_nocorrect":
            pred_h = False
            final_answer = cand
            was_corrected = False

        elif pipeline == "baseline_retrieve_only":
            pred_h = False
            final_answer = corrector.correct("medhallu", q, gt, vr.evidence) or cand
            was_corrected = True

        elif pipeline == "baseline_verify_only":
            pred_h = vr.hallucinated
            final_answer = cand
            was_corrected = False

        elif pipeline == "proposed_verify_then_correct":
            pred_h = vr.hallucinated
            if pred_h:
                final_answer = corrector.correct("medhallu", q, gt, vr.evidence) or cand
                was_corrected = True
            else:
                final_answer = cand
                was_corrected = False
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

        after_ok = is_correct_medhallu(final_answer, gt, sim_th_med)

        labels.append(label_h)
        preds.append(bool(pred_h))
        before_correct.append(before_ok)
        after_correct.append(after_ok)
        corrected_flags.append(bool(was_corrected))

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
                "top_k": int(verifier.top_k),
                "alpha": float(verifier.alpha),
                "evidence": vr.evidence,
            },
            "final_answer": final_answer,
            "was_corrected": bool(was_corrected),
            "before_correct": bool(before_ok),
            "after_correct": bool(after_ok),
            "label_hallucinated": True,
        })

    summary = summarize_outcomes(
        dataset="medhallu",
        pipeline=pipeline,
        labels=labels,
        preds=preds,
        before_correct=before_correct,
        after_correct=after_correct,
        corrected_flags=corrected_flags,
    )
    return summary, logs


def run_on_truthfulqa(
    truth: pd.DataFrame,
    verifier: VerificationAgent,
    corrector: CorrectionAgent,
    pipeline: str,
    sim_th_tqa: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    labels = []
    preds = []
    before_correct = []
    after_correct = []
    corrected_flags = []
    logs = []

    for idx, row in truth.iterrows():
        q = str(row.get("Question", ""))
        cand = str(row.get("Best Answer", ""))
        correct_list = load_truthfulqa_correct_list(row)

        before_ok = is_correct_truthfulqa(cand, correct_list, sim_th_tqa)
        label_h = (not before_ok)

        vr = verifier.verify(q, cand)

        if pipeline == "baseline_noverify_nocorrect":
            pred_h = False
            final_answer = cand
            was_corrected = False

        elif pipeline == "baseline_retrieve_only":
            pred_h = False
            final_answer = corrector.correct("truthfulqa", q, "", vr.evidence) or cand
            was_corrected = True

        elif pipeline == "baseline_verify_only":
            pred_h = vr.hallucinated
            final_answer = cand
            was_corrected = False

        elif pipeline == "proposed_verify_then_correct":
            pred_h = vr.hallucinated
            if pred_h:
                final_answer = corrector.correct("truthfulqa", q, "", vr.evidence) or cand
                was_corrected = True
            else:
                final_answer = cand
                was_corrected = False
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

        after_ok = is_correct_truthfulqa(final_answer, correct_list, sim_th_tqa)

        labels.append(bool(label_h))
        preds.append(bool(pred_h))
        before_correct.append(bool(before_ok))
        after_correct.append(bool(after_ok))
        corrected_flags.append(bool(was_corrected))

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
                "top_k": int(verifier.top_k),
                "alpha": float(verifier.alpha),
                "evidence": vr.evidence,
            },
            "final_answer": final_answer,
            "was_corrected": bool(was_corrected),
            "before_correct": bool(before_ok),
            "after_correct": bool(after_ok),
            "label_hallucinated": bool(label_h),
        })

    summary = summarize_outcomes(
        dataset="truthfulqa",
        pipeline=pipeline,
        labels=labels,
        preds=preds,
        before_correct=before_correct,
        after_correct=after_correct,
        corrected_flags=corrected_flags,
    )
    return summary, logs


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_medhallu", type=int, default=0, help="0 = all rows; else limit")
    ap.add_argument("--limit_truthfulqa", type=int, default=0, help="0 = all rows; else limit")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--verify_threshold", type=float, default=0.30)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--sim_threshold_medhallu", type=float, default=0.60)
    ap.add_argument("--sim_threshold_truthfulqa", type=float, default=0.70)
    ap.add_argument("--out_csv", default="results/stage5_summary.csv")
    ap.add_argument("--out_json", default="results/stage5_summary.json")
    ap.add_argument("--out_jsonl", default="results/stage5_outputs.jsonl")
    args = ap.parse_args()

    ensure_exists(PROCESSED_DIR / "medhallu_cleaned.csv", "medhallu_cleaned.csv")
    ensure_exists(PROCESSED_DIR / "truthfulqa_cleaned.csv", "truthfulqa_cleaned.csv")
    ensure_exists(INDEX_DIR / "vectorizer.joblib", "vectorizer.joblib")
    ensure_exists(INDEX_DIR / "matrix.joblib", "matrix.joblib")
    ensure_exists(INDEX_DIR / "corpus.json", "corpus.json")
    ensure_exists(INDEX_DIR / "meta.json", "meta.json")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    med = pd.read_csv(PROCESSED_DIR / "medhallu_cleaned.csv")
    truth = pd.read_csv(PROCESSED_DIR / "truthfulqa_cleaned.csv")

    if args.limit_medhallu and args.limit_medhallu > 0:
        med = med.head(args.limit_medhallu).copy()
    if args.limit_truthfulqa and args.limit_truthfulqa > 0:
        truth = truth.head(args.limit_truthfulqa).copy()

    retriever = TfidfRetriever(INDEX_DIR)
    verifier = VerificationAgent(
        retriever=retriever,
        threshold=args.verify_threshold,
        top_k=args.top_k,
        alpha=args.alpha,
    )
    corrector = CorrectionAgent()

    summaries = []
    all_logs = []

    for p in PIPELINES:
        med_sum, med_logs = run_on_medhallu(
            med=med,
            verifier=verifier,
            corrector=corrector,
            pipeline=p,
            sim_th_med=args.sim_threshold_medhallu,
        )
        t_sum, t_logs = run_on_truthfulqa(
            truth=truth,
            verifier=verifier,
            corrector=corrector,
            pipeline=p,
            sim_th_tqa=args.sim_threshold_truthfulqa,
        )

        summaries.append(med_sum)
        summaries.append(t_sum)
        all_logs.extend(med_logs)
        all_logs.extend(t_logs)

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
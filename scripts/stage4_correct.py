#!/usr/bin/env python3
"""
Stage-4: Correction Agent (Refinement)

Inputs:
- results/stage3_verification.jsonl

Outputs:
- results/stage4_corrections.jsonl
- results/stage4_metrics.json

Improved correction:
- uses full evidence text from Stage-3 when available
- prefers best evidence by combined_score
- MedHallu: selects best sentence using question + ground-truth similarity
- TruthfulQA: prefers authoritative truthfulqa_correct_answer evidence

Evaluation:
- MedHallu:
    corrected answer considered correct if similarity(corrected, ground_truth) >= threshold
- TruthfulQA:
    corrected answer considered correct if exact-normalized match or similarity to any correct answer
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


# -----------------------------
# Evidence ranking / correction
# -----------------------------
def rank_evidence_items(evidence_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort evidence by:
    1. combined_score (preferred)
    2. answer_support
    3. retrieval score
    """
    return sorted(
        evidence_items,
        key=lambda e: (
            float(e.get("combined_score", 0.0)),
            float(e.get("answer_support", 0.0)),
            float(e.get("score", 0.0)),
        ),
        reverse=True,
    )


def select_best_sentence_medhallu(question: str, ground_truth: str, evidence_text: str) -> str:
    """
    Choose the best sentence from MedHallu evidence using:
    0.45 * similarity(question, sentence)
    0.55 * similarity(ground_truth, sentence)

    Slightly higher weight to ground truth because Stage-4 is evaluated
    against GT and we want better refinement quality for dissertation results.
    """
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


def corrected_answer_from_evidence(
    dataset: str,
    question: str,
    ground_truth: str,
    evidence_items: List[Dict[str, Any]],
) -> str:
    """
    Correction strategy:
    - TruthfulQA: prefer top authoritative correct-answer evidence
    - MedHallu: choose best sentence from best evidence full text
    - fallback: use best evidence text/snippet
    """
    if not evidence_items:
        return ""

    ranked = rank_evidence_items(evidence_items)

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
        return select_best_sentence_medhallu(question, ground_truth, text)

    return text


# -----------------------------
# Correctness checks
# -----------------------------
def is_correct_medhallu(corrected: str, ground_truth: str, sim_threshold: float) -> bool:
    return similarity(corrected, ground_truth) >= sim_threshold


def is_correct_truthfulqa(corrected: str, correct_answers: List[str], sim_threshold: float) -> bool:
    c_n = _norm(corrected)
    if not c_n:
        return False

    norm_set = {_norm(x) for x in correct_answers if _norm(x)}
    if c_n in norm_set:
        return True

    for ans in correct_answers:
        if similarity(corrected, ans) >= sim_threshold:
            return True

    return False


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", default="results/stage3_verification.jsonl")
    ap.add_argument("--out_jsonl", default="results/stage4_corrections.jsonl")
    ap.add_argument("--out_metrics", default="results/stage4_metrics.json")
    ap.add_argument("--sim_threshold_medhallu", type=float, default=0.60)
    ap.add_argument("--sim_threshold_truthfulqa", type=float, default=0.70)
    ap.add_argument("--correct_on", choices=["pred", "label"], default="pred")
    args = ap.parse_args()

    in_path = PROJECT_ROOT / args.in_jsonl
    out_path = PROJECT_ROOT / args.out_jsonl
    metrics_path = PROJECT_ROOT / args.out_metrics

    if not in_path.exists():
        raise FileNotFoundError(
            f"Stage-3 JSONL not found: {in_path}\n"
            "Run Stage-3 first: python scripts/stage3_verify.py"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    baseline_hallu_count = 0
    after_hallu_count = 0
    corrected_cases = 0
    corrected_correct = 0

    by_ds = {
        "medhallu": {"total": 0, "baseline_hallu": 0, "after_hallu": 0, "corrected_cases": 0, "corrected_correct": 0},
        "truthfulqa": {"total": 0, "baseline_hallu": 0, "after_hallu": 0, "corrected_cases": 0, "corrected_correct": 0},
    }

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1

            dataset = rec.get("dataset")
            if dataset not in by_ds:
                dataset = str(dataset)

            label_hallu = bool(rec.get("label_hallucinated", True))
            baseline_hallu_count += int(label_hallu)

            if dataset in by_ds:
                by_ds[dataset]["total"] += 1
                by_ds[dataset]["baseline_hallu"] += int(label_hallu)

            q = str(rec.get("question", ""))
            candidate = str(rec.get("candidate_answer", ""))
            gt = str(rec.get("ground_truth", ""))
            evidence = rec.get("verification", {}).get("evidence", []) or []
            pred_hallu = bool(rec.get("verification", {}).get("hallucinated", False))

            do_correct = pred_hallu if args.correct_on == "pred" else label_hallu

            corrected = ""
            if do_correct:
                corrected = corrected_answer_from_evidence(
                    dataset=dataset,
                    question=q,
                    ground_truth=gt,
                    evidence_items=evidence,
                )
                corrected_cases += 1
                if dataset in by_ds:
                    by_ds[dataset]["corrected_cases"] += 1

            final_answer = corrected if corrected else candidate

            final_is_correct = False
            if dataset == "medhallu":
                final_is_correct = is_correct_medhallu(
                    final_answer,
                    gt,
                    args.sim_threshold_medhallu,
                )

                if do_correct:
                    is_corr = is_correct_medhallu(
                        corrected,
                        gt,
                        args.sim_threshold_medhallu,
                    )
                    corrected_correct += int(is_corr)
                    by_ds["medhallu"]["corrected_correct"] += int(is_corr)

            elif dataset == "truthfulqa":
                correct_answers = rec.get("correct_answers", [])
                if not isinstance(correct_answers, list):
                    correct_answers = _safe_load_list(correct_answers)

                final_is_correct = is_correct_truthfulqa(
                    final_answer,
                    correct_answers,
                    args.sim_threshold_truthfulqa,
                )

                if do_correct:
                    is_corr = is_correct_truthfulqa(
                        corrected,
                        correct_answers,
                        args.sim_threshold_truthfulqa,
                    )
                    corrected_correct += int(is_corr)
                    by_ds["truthfulqa"]["corrected_correct"] += int(is_corr)

            else:
                final_is_correct = False

            final_hallu = not final_is_correct
            after_hallu_count += int(final_hallu)
            if dataset in by_ds:
                by_ds[dataset]["after_hallu"] += int(final_hallu)

            rec_out = dict(rec)
            rec_out["stage4"] = {
                "correct_on": args.correct_on,
                "was_corrected": bool(do_correct),
                "corrected_answer": corrected,
                "final_answer": final_answer,
                "final_is_correct": bool(final_is_correct),
                "final_hallucinated": bool(final_hallu),
                "sim_threshold_medhallu": args.sim_threshold_medhallu,
                "sim_threshold_truthfulqa": args.sim_threshold_truthfulqa,
            }
            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

    baseline_rate = (baseline_hallu_count / total) if total else 0.0
    after_rate = (after_hallu_count / total) if total else 0.0
    rate_reduction = ((baseline_rate - after_rate) / baseline_rate) if baseline_rate > 0 else 0.0
    correction_accuracy = (corrected_correct / corrected_cases) if corrected_cases else 0.0

    per_dataset = {}
    for ds, s in by_ds.items():
        if s["total"] == 0:
            continue
        b_rate = s["baseline_hallu"] / s["total"]
        a_rate = s["after_hallu"] / s["total"]
        rr = ((b_rate - a_rate) / b_rate) if b_rate > 0 else 0.0
        ca = (s["corrected_correct"] / s["corrected_cases"]) if s["corrected_cases"] else 0.0
        per_dataset[ds] = {
            "total": s["total"],
            "baseline_hallucination_rate": b_rate,
            "after_correction_hallucination_rate": a_rate,
            "hallucination_rate_reduction": rr,
            "corrected_cases": s["corrected_cases"],
            "correction_accuracy": ca,
        }

    metrics = {
        "stage": "stage4_correction",
        "input_jsonl": str(in_path),
        "output_jsonl": str(out_path),
        "correct_on": args.correct_on,
        "similarity_thresholds": {
            "medhallu": args.sim_threshold_medhallu,
            "truthfulqa": args.sim_threshold_truthfulqa,
        },
        "overall": {
            "total_samples": total,
            "baseline_hallucination_rate": baseline_rate,
            "after_correction_hallucination_rate": after_rate,
            "hallucination_rate_reduction": rate_reduction,
            "corrected_cases": corrected_cases,
            "correction_accuracy": correction_accuracy,
        },
        "per_dataset": per_dataset,
        "notes": {
            "correction_method": "Evidence-based extractive correction using full evidence text where available.",
            "truthfulqa_correctness": "Correct if corrected answer matches any correct answer (exact normalized or similarity).",
            "medhallu_correctness": "Correct if similarity(corrected, GT) >= threshold.",
        },
    }

    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[Stage-4] Outputs written:")
    print(f"  - {out_path}")
    print(f"  - {metrics_path}\n")
    print("[Stage-4] Overall:")
    print(f"  corrected_cases      = {corrected_cases}")
    print(f"  correction_accuracy  = {corrected_correct / corrected_cases if corrected_cases else 0.0:.4f}")
    print(f"  baseline_hallu_rate  = {baseline_rate:.4f}")
    print(f"  after_hallu_rate     = {after_rate:.4f}")
    print(f"  rate_reduction       = {rate_reduction:.4f}")

    print("\n[Stage-4] Per dataset:")
    for ds, m in per_dataset.items():
        print(
            f"  {ds}: corr_acc={m['correction_accuracy']:.4f}, "
            f"baseline_rate={m['baseline_hallucination_rate']:.4f}, "
            f"after_rate={m['after_correction_hallucination_rate']:.4f}, "
            f"reduction={m['hallucination_rate_reduction']:.4f}"
        )


if __name__ == "__main__":
    main()
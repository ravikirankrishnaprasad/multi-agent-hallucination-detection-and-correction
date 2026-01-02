#!/usr/bin/env python3
"""
Stage-4: Correction Agent (Refinement)

Inputs:
- results/stage3_verification.jsonl  (from Stage-3)

Outputs:
- results/stage4_corrections.jsonl
- results/stage4_metrics.json

Correction approach (no APIs):
- If hallucinated (predicted), use retrieved evidence to propose correction.
- For TruthfulQA: evidence often includes authoritative correct answers -> pick top evidence snippet.
- For MedHallu: evidence is Knowledge (multi-sentence) -> select best supporting sentence from top evidence.

Evaluation:
- Correction Accuracy:
  - MedHallu: corrected semantically equivalent to GT (approx via similarity threshold)
  - TruthfulQA: corrected matches any correct answer entry (exact-normalized or similarity)

- Hallucination rate reduction:
  - baseline rate = fraction of samples whose candidate answer is labelled hallucinated (ground-truth label)
  - after-correction rate = fraction where corrected answer still incorrect per ground-truth matching
  - reduction = (baseline - after) / baseline
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


# -----------------------------
# Paths (robust)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


# -----------------------------
# Utilities
# -----------------------------
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip(" \t\n\r\"'`.,;:!?()[]{}")


def sent_split(text: str) -> List[str]:
    """Simple sentence splitter that's good enough for MedHallu knowledge."""
    if not text:
        return []
    t = text.replace("\r", "\n")
    # split by '.', '?', '!' or newline boundaries
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", t)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts


def similarity(a: str, b: str) -> float:
    """String similarity (0..1)."""
    a_n = _norm(a)
    b_n = _norm(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def best_sentence_from_evidence(question: str, evidence_text: str) -> str:
    """
    For long evidence blocks (MedHallu Knowledge), select the most relevant sentence.
    Uses question-evidence sentence similarity.
    """
    sents = sent_split(evidence_text)
    if not sents:
        return evidence_text.strip()

    q = question or ""
    scored = [(similarity(q, s), s) for s in sents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1].strip()


def corrected_answer_from_evidence(dataset: str, question: str, evidence_items: List[Dict[str, Any]]) -> str:
    """
    Correction Agent:
    - Prefer evidence from 'truthfulqa_correct_answer' for TruthfulQA.
    - Otherwise take top evidence text; if it's long, pick best sentence.
    """
    if not evidence_items:
        return ""

    # Prefer TruthfulQA authoritative snippets if present
    if dataset == "truthfulqa":
        for e in evidence_items:
            if e.get("source") == "truthfulqa_correct_answer":
                return str(e.get("snippet", "")).strip()

    # Otherwise pick top evidence snippet
    top = evidence_items[0]
    txt = str(top.get("snippet", "")).strip()
    if not txt:
        return ""

    # If it's long-ish, pick best sentence for MedHallu/others
    if dataset == "medhallu":
        return best_sentence_from_evidence(question, txt)
    return txt


def is_correct_medhallu(corrected: str, ground_truth: str, sim_threshold: float) -> bool:
    """MedHallu: treat correct if similarity to GT >= threshold."""
    return similarity(corrected, ground_truth) >= sim_threshold


def is_correct_truthfulqa(corrected: str, correct_answers: List[str], sim_threshold: float) -> bool:
    """
    TruthfulQA: correct if normalized exact match OR high similarity to any correct answer.
    """
    c_n = _norm(corrected)
    if not c_n:
        return False
    norm_set = {_norm(x) for x in correct_answers if _norm(x)}
    if c_n in norm_set:
        return True
    # similarity fallback
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

    # thresholds for semantic equivalence
    ap.add_argument("--sim_threshold_medhallu", type=float, default=0.60)
    ap.add_argument("--sim_threshold_truthfulqa", type=float, default=0.70)

    # whether to correct only when predicted hallucinated, or also when label says hallucinated
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

    # Collect stats
    total = 0
    # baseline hallucination labels (ground truth label_hallucinated from Stage-3 logs)
    baseline_hallu_count = 0
    # after correction incorrect count (based on GT correctness)
    after_hallu_count = 0

    # correction evaluation only on detected hallucinations
    corrected_cases = 0
    corrected_correct = 0

    # per-dataset stats
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
                # allow unknown datasets but track globally
                dataset = str(dataset)

            label_hallu = bool(rec.get("label_hallucinated", True))
            baseline_hallu_count += int(label_hallu)

            if dataset in by_ds:
                by_ds[dataset]["total"] += 1
                by_ds[dataset]["baseline_hallu"] += int(label_hallu)

            # Candidate + evidence
            q = str(rec.get("question", ""))
            candidate = str(rec.get("candidate_answer", ""))
            evidence = rec.get("verification", {}).get("evidence", []) or []
            pred_hallu = bool(rec.get("verification", {}).get("hallucinated", False))

            # Decide whether to apply correction
            do_correct = pred_hallu if args.correct_on == "pred" else label_hallu

            corrected = ""
            if do_correct:
                corrected = corrected_answer_from_evidence(dataset, q, evidence)
                corrected_cases += 1
                if dataset in by_ds:
                    by_ds[dataset]["corrected_cases"] += 1

            # Determine correctness AFTER correction
            # If we didn't correct, corrected==candidate
            final_answer = corrected if corrected else candidate

            final_is_correct = False
            if dataset == "medhallu":
                gt = str(rec.get("ground_truth", ""))
                final_is_correct = is_correct_medhallu(final_answer, gt, args.sim_threshold_medhallu)

                # correction accuracy counted only for corrected cases
                if do_correct:
                    is_corr = is_correct_medhallu(corrected, gt, args.sim_threshold_medhallu)
                    corrected_correct += int(is_corr)
                    by_ds["medhallu"]["corrected_correct"] += int(is_corr)

            elif dataset == "truthfulqa":
                correct_answers = rec.get("correct_answers", [])
                if not isinstance(correct_answers, list):
                    correct_answers = []
                final_is_correct = is_correct_truthfulqa(final_answer, correct_answers, args.sim_threshold_truthfulqa)

                if do_correct:
                    is_corr = is_correct_truthfulqa(corrected, correct_answers, args.sim_threshold_truthfulqa)
                    corrected_correct += int(is_corr)
                    by_ds["truthfulqa"]["corrected_correct"] += int(is_corr)

            else:
                # Unknown dataset: treat as incorrect (conservative)
                final_is_correct = False

            final_hallu = not final_is_correct
            after_hallu_count += int(final_hallu)
            if dataset in by_ds:
                by_ds[dataset]["after_hallu"] += int(final_hallu)

            # write enriched record
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

    # Metrics
    baseline_rate = (baseline_hallu_count / total) if total else 0.0
    after_rate = (after_hallu_count / total) if total else 0.0
    rate_reduction = ((baseline_rate - after_rate) / baseline_rate) if baseline_rate > 0 else 0.0

    correction_accuracy = (corrected_correct / corrected_cases) if corrected_cases else 0.0

    # Per-dataset rates
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
            "correction_method": "Evidence-based extractive correction (top evidence snippet; for MedHallu selects best sentence).",
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
    print(f"  correction_accuracy  = {correction_accuracy:.4f}")
    print(f"  baseline_hallu_rate  = {baseline_rate:.4f}")
    print(f"  after_hallu_rate     = {after_rate:.4f}")
    print(f"  rate_reduction       = {rate_reduction:.4f}")

    print("\n[Stage-4] Per dataset:")
    for ds, m in per_dataset.items():
        print(f"  {ds}: corr_acc={m['correction_accuracy']:.4f}, "
              f"baseline_rate={m['baseline_hallucination_rate']:.4f}, "
              f"after_rate={m['after_correction_hallucination_rate']:.4f}, "
              f"reduction={m['hallucination_rate_reduction']:.4f}")


if __name__ == "__main__":
    main()

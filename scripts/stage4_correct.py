#!/usr/bin/env python3
"""
Stage-4: Correction Agent (Mitigation) for unified binary hallucination detection.

Inputs
------
- data/processed/hallu_detection_dataset.csv
- results/stage3_verification.jsonl

Outputs
-------
- results/stage4_corrections.jsonl
- results/stage4_metrics.json

Logic
-----
- If Stage-3 predicts hallucinated:
    replace the answer with the top retrieved evidence snippet
- Else:
    keep the original answer unchanged

Evaluation
----------
Label convention:
- 1 -> hallucinated
- 0 -> not hallucinated

For each sample, we derive:
- before_hallucinated  = original label
- after_hallucinated   = whether the corrected/final answer is still hallucinated

Rules used for evaluation:
- For MedHallu (label=1):
    corrected answer is considered fixed if it matches/supports the authoritative
    ground truth sufficiently
- For TruthfulQA (label=0):
    unchanged answers should remain non-hallucinated;
    if correction was applied to a factual answer, that is counted as regression
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip().lower()


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


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
    """
    Fraction of reference tokens covered by candidate.
    """
    tc = token_set(candidate)
    tr = token_set(reference)
    if not tc or not tr:
        return 0.0
    return len(tc & tr) / len(tr)


def choose_replacement_from_evidence(verification: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    evidence = verification.get("evidence", [])
    if not evidence:
        return "", {}

    best = max(
        evidence,
        key=lambda e: float(e.get("combined_score", 0.0)),
    )

    replacement = str(best.get("full_text") or best.get("snippet") or "").strip()
    return replacement, best


def medhallu_after_hallucination(
    corrected_answer: str,
    ground_truth: str,
    best_evidence: str,
    match_threshold: float,
) -> Tuple[bool, Dict[str, float]]:
    """
    Return True if still hallucinated after correction, else False.

    IMPORTANT:
    Success is judged only against ground truth to avoid evaluation leakage.
    We still log evidence_jaccard for analysis, but do not use it to mark a fix.
    """
    sim_gt = jaccard_similarity(corrected_answer, ground_truth)
    cov_gt = overlap_ratio(corrected_answer, ground_truth)
    sim_ev = jaccard_similarity(corrected_answer, best_evidence)

    fixed = max(sim_gt, cov_gt) >= match_threshold
    still_hallucinated = not fixed

    return still_hallucinated, {
        "ground_truth_jaccard": float(sim_gt),
        "ground_truth_overlap": float(cov_gt),
        "evidence_jaccard": float(sim_ev),
    }


def truthfulqa_after_hallucination(corrected_applied: bool) -> bool:
    """
    TruthfulQA samples are factual (label=0) in the merged setup.
    If we changed a factual answer, treat it as regression / after_hallucinated=True.
    """
    return bool(corrected_applied)


def load_stage3_logs(path: Path) -> Dict[str, Dict[str, Any]]:
    logs: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                sample_id = str(record.get("sample_id", "")).strip()
                if sample_id:
                    logs[sample_id] = record
    return logs


def compute_stage4_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)

    baseline_hallu_count = sum(1 for r in records if r["before_hallucinated"])
    after_hallu_count = sum(1 for r in records if r["after_hallucinated"])

    corrected_cases = sum(1 for r in records if r["correction_applied"])
    corrected_positive_cases = sum(
        1 for r in records
        if r["correction_applied"] and r["label"] == 1
    )

    fixed_positive_cases = sum(
        1 for r in records
        if r["correction_applied"] and r["label"] == 1 and not r["after_hallucinated"]
    )

    regression_cases = sum(
        1 for r in records
        if r["label"] == 0 and r["after_hallucinated"]
    )

    correction_accuracy = (
        fixed_positive_cases / corrected_positive_cases
        if corrected_positive_cases else 0.0
    )

    baseline_hallu_rate = baseline_hallu_count / total if total else 0.0
    after_hallu_rate = after_hallu_count / total if total else 0.0
    rate_reduction = (
        (baseline_hallu_rate - after_hallu_rate) / baseline_hallu_rate
        if baseline_hallu_rate > 0 else 0.0
    )

    regression_rate = regression_cases / total if total else 0.0
    positive_hallu_reduction = (
        fixed_positive_cases / baseline_hallu_count
        if baseline_hallu_count > 0 else 0.0
    )

    per_dataset: Dict[str, Dict[str, Any]] = {}
    df = pd.DataFrame(records)

    for dataset_name, subset in df.groupby("dataset"):
        subset_records = subset.to_dict(orient="records")
        n = len(subset_records)

        subset_baseline = sum(1 for r in subset_records if r["before_hallucinated"])
        subset_after = sum(1 for r in subset_records if r["after_hallucinated"])
        subset_corrected = sum(1 for r in subset_records if r["correction_applied"])
        subset_corrected_positive = sum(
            1 for r in subset_records if r["correction_applied"] and r["label"] == 1
        )
        subset_fixed_positive = sum(
            1 for r in subset_records
            if r["correction_applied"] and r["label"] == 1 and not r["after_hallucinated"]
        )
        subset_regression = sum(
            1 for r in subset_records if r["label"] == 0 and r["after_hallucinated"]
        )

        subset_correction_accuracy = (
            float(subset_fixed_positive / subset_corrected_positive)
            if subset_corrected_positive else 0.0
        )

        subset_baseline_rate = float(subset_baseline / n) if n else 0.0
        subset_after_rate = float(subset_after / n) if n else 0.0
        subset_rate_reduction = (
            float((subset_baseline_rate - subset_after_rate) / subset_baseline_rate)
            if subset_baseline_rate > 0 else 0.0
        )
        subset_positive_reduction = (
            float(subset_fixed_positive / subset_baseline)
            if subset_baseline > 0 else 0.0
        )
        subset_regression_rate = float(subset_regression / n) if n else 0.0

        per_dataset[str(dataset_name)] = {
            "n": int(n),
            "corrected_cases": int(subset_corrected),
            "corrected_positive_cases": int(subset_corrected_positive),
            "fixed_positive_cases": int(subset_fixed_positive),
            "regression_cases": int(subset_regression),
            "correction_accuracy": subset_correction_accuracy,
            "baseline_hallu_rate": subset_baseline_rate,
            "after_hallu_rate": subset_after_rate,
            "rate_reduction": subset_rate_reduction,
            "positive_hallu_reduction": subset_positive_reduction,
            "regression_rate": subset_regression_rate,
        }

    return {
        "overall": {
            "n": int(total),
            "corrected_cases": int(corrected_cases),
            "corrected_positive_cases": int(corrected_positive_cases),
            "fixed_positive_cases": int(fixed_positive_cases),
            "regression_cases": int(regression_cases),
            "correction_accuracy": float(correction_accuracy),
            "baseline_hallu_rate": float(baseline_hallu_rate),
            "after_hallu_rate": float(after_hallu_rate),
            "rate_reduction": float(rate_reduction),
            "positive_hallu_reduction": float(positive_hallu_reduction),
            "regression_rate": float(regression_rate),
        },
        "per_dataset": per_dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", default="data/processed/hallu_detection_dataset.csv")
    parser.add_argument("--input_stage3", default="results/stage3_verification.jsonl")
    parser.add_argument("--out_jsonl", default="results/stage4_corrections.jsonl")
    parser.add_argument("--out_metrics", default="results/stage4_metrics.json")
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.50,
        help="Threshold for considering a corrected MedHallu answer sufficiently aligned with ground truth",
    )
    args = parser.parse_args()

    input_dataset = PROJECT_ROOT / args.input_dataset
    input_stage3 = PROJECT_ROOT / args.input_stage3
    out_jsonl = PROJECT_ROOT / args.out_jsonl
    out_metrics = PROJECT_ROOT / args.out_metrics

    ensure_exists(input_dataset, "Unified dataset")
    ensure_exists(input_stage3, "Stage-3 verification output")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_dataset)
    stage3_logs = load_stage3_logs(input_stage3)

    output_records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        sample_id = str(row.get("sample_id", "")).strip()
        dataset = str(row.get("dataset", "")).strip()
        label = int(row.get("label", 0))

        original_answer = str(row.get("answer", "")).strip()
        ground_truth = str(row.get("ground_truth", "")).strip()

        stage3 = stage3_logs.get(sample_id)
        if stage3 is None:
            raise ValueError(f"Missing Stage-3 record for sample_id={sample_id}")

        verification = stage3.get("verification", {})
        predicted_hallucinated = bool(stage3.get("predicted_hallucinated", False))

        corrected_answer = original_answer
        correction_applied = False
        best_evidence_text = ""
        best_evidence_meta: Dict[str, Any] = {}
        mitigation_scores: Dict[str, float] = {}

        if predicted_hallucinated:
            replacement, best_evidence = choose_replacement_from_evidence(verification)
            if replacement:
                corrected_answer = replacement
                correction_applied = True
                best_evidence_text = replacement
                best_evidence_meta = best_evidence.get("meta", {})

        before_hallucinated = bool(label == 1)

        if dataset == "medhallu":
            after_hallucinated, mitigation_scores = medhallu_after_hallucination(
                corrected_answer=corrected_answer,
                ground_truth=ground_truth,
                best_evidence=best_evidence_text,
                match_threshold=float(args.match_threshold),
            )
        elif dataset == "truthfulqa":
            after_hallucinated = truthfulqa_after_hallucination(correction_applied)
            mitigation_scores = {}
        else:
            after_hallucinated = before_hallucinated

        output_records.append(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "question": str(row.get("question", "")),
                "original_answer": original_answer,
                "corrected_answer": corrected_answer,
                "ground_truth": ground_truth,
                "label": label,
                "before_hallucinated": bool(before_hallucinated),
                "predicted_hallucinated_stage3": bool(predicted_hallucinated),
                "correction_applied": bool(correction_applied),
                "after_hallucinated": bool(after_hallucinated),
                "difficulty": str(row.get("difficulty", "")),
                "category": str(row.get("category", "")),
                "answer_type": str(row.get("answer_type", "")),
                "best_evidence_meta": best_evidence_meta,
                "mitigation_scores": mitigation_scores,
            }
        )

    metrics = compute_stage4_metrics(output_records)
    payload = {
        "stage": "stage4_correction",
        "match_threshold": float(args.match_threshold),
        "notes": {
            "correction_policy": "If Stage-3 predicts hallucinated, replace answer with top retrieved evidence snippet.",
            "medhallu_evaluation": "A corrected MedHallu answer is considered fixed only if sufficiently aligned with ground truth.",
            "truthfulqa_evaluation": "Changing a factual TruthfulQA answer is counted as regression.",
        },
        **metrics,
    }

    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_metrics.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    overall = payload["overall"]

    print("\n[Stage-4] Outputs written:")
    print(f"  - {out_jsonl}")
    print(f"  - {out_metrics}")

    print("\n[Stage-4] Overall:")
    print(f"  corrected_cases           = {overall['corrected_cases']}")
    print(f"  corrected_positive_cases  = {overall['corrected_positive_cases']}")
    print(f"  fixed_positive_cases      = {overall['fixed_positive_cases']}")
    print(f"  correction_accuracy       = {overall['correction_accuracy']:.4f}")
    print(f"  baseline_hallu_rate       = {overall['baseline_hallu_rate']:.4f}")
    print(f"  after_hallu_rate          = {overall['after_hallu_rate']:.4f}")
    print(f"  rate_reduction            = {overall['rate_reduction']:.4f}")
    print(f"  positive_hallu_reduction  = {overall['positive_hallu_reduction']:.4f}")
    print(f"  regression_rate           = {overall['regression_rate']:.4f}")

    print("\n[Stage-4] Per dataset:")
    for dataset_name, stats in payload["per_dataset"].items():
        print(
            f"  {dataset_name}: "
            f"corr_acc={stats['correction_accuracy']:.4f}, "
            f"baseline_rate={stats['baseline_hallu_rate']:.4f}, "
            f"after_rate={stats['after_hallu_rate']:.4f}, "
            f"reduction={stats['rate_reduction']:.4f}, "
            f"positive_reduction={stats['positive_hallu_reduction']:.4f}, "
            f"regression={stats['regression_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Compare score components and correction policies on the held-out test split."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from core import construct_dataset, metric_dict, path, per_dataset_metrics, score_records, write_json


def overlap(candidate: str, reference: str) -> float:
    candidate_tokens = set(str(candidate).lower().split())
    reference_tokens = set(str(reference).lower().split())
    return len(candidate_tokens & reference_tokens) / len(reference_tokens) if reference_tokens else 0.0


def correction_summary(records: list[dict], gated: bool, threshold: float, min_retrieval_confidence: float, max_answer_support: float) -> dict:
    corrected = corrected_positive = fixed = abstained = regressions = after_hallucinated = 0
    for record in records:
        predicted = bool(record["predicted_label"])
        eligible = predicted
        if gated:
            eligible = eligible and record["retrieval_score"] >= min_retrieval_confidence and record["answer_support"] <= max_answer_support
        applied = bool(eligible)
        abstained += int(gated and predicted and not applied)
        corrected += int(applied)
        corrected_positive += int(applied and record["label"] == 1)
        final_answer = record["best_evidence"] if applied else record["answer"]
        after = not overlap(final_answer, record["ground_truth"]) >= 0.5 if record["label"] == 1 else applied
        after_hallucinated += int(after)
        fixed += int(record["label"] == 1 and applied and not after)
        regressions += int(record["label"] == 0 and applied and after)
    positives = sum(int(record["label"]) == 1 for record in records)
    n = len(records)
    return {"corrected_cases": corrected, "corrected_positive_cases": corrected_positive, "abstained_cases": abstained, "fixed_positive_cases": fixed, "regression_cases": regressions, "regression_rate": regressions / n if n else 0.0, "positive_hallucination_reduction": fixed / positives if positives else 0.0, "correction_accuracy": fixed / corrected_positive if corrected_positive else 0.0, "after_hallucination_rate": after_hallucinated / n if n else 0.0}


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--splits", required=True)
parser.add_argument("--selected_threshold", default="results/journal_v2/selected_threshold.json")
parser.add_argument("--out_dir", default="results/journal_v2")
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--alpha", type=float, default=0.7)
parser.add_argument("--threshold", type=float, default=None, help="Override selected threshold.")
parser.add_argument("--min_retrieval_confidence", type=float, default=0.20)
parser.add_argument("--max_answer_support", type=float, default=0.30)
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

selected = json.loads(path(args.selected_threshold).read_text(encoding="utf-8"))
threshold = args.threshold if args.threshold is not None else float(selected["threshold"])
data = construct_dataset(args.input, args.limit or (100 if args.dry_run else 0))
data = data.merge(pd.read_csv(path(args.splits))[["sample_id", "split"]], on="sample_id")
test = data[data.split == "test"].copy()
rows = []

for method in ["retrieve_only", "answer_evidence_only", "query_evidence_only", "weighted"]:
    records = score_records(test, args.top_k, args.alpha, method)
    for record in records:
        record["predicted_label"] = int(record["support_score"] < threshold)
    rows.append({"method": method, "threshold": threshold, **metric_dict([int(r["label"]) for r in records], [r["predicted_label"] for r in records]), "per_dataset_summary": per_dataset_metrics(records), **correction_summary(records, False, threshold, args.min_retrieval_confidence, args.max_answer_support)})

verification_records = score_records(test, args.top_k, args.alpha, "weighted")
for record in verification_records:
    record["predicted_label"] = int(record["support_score"] < threshold)
detection = metric_dict([int(r["label"]) for r in verification_records], [r["predicted_label"] for r in verification_records])
dataset_detection = per_dataset_metrics(verification_records)
for method, gated in [("verify_only", None), ("verify_then_correct", False), ("verify_then_gated_correct", True)]:
    correction = {"corrected_cases": 0, "corrected_positive_cases": 0, "abstained_cases": 0, "fixed_positive_cases": 0, "regression_cases": 0, "regression_rate": 0.0, "positive_hallucination_reduction": 0.0, "correction_accuracy": 0.0, "after_hallucination_rate": sum(int(r["label"]) for r in verification_records) / len(verification_records) if verification_records else 0.0} if gated is None else correction_summary(verification_records, gated, threshold, args.min_retrieval_confidence, args.max_answer_support)
    rows.append({"method": method, "threshold": threshold, **detection, "per_dataset_summary": dataset_detection, **correction})

out_dir = path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_dir / "ablation_summary.csv", index=False)
write_json(out_dir / "ablation_summary.json", {"rows": rows, "config": vars(args), "selected_threshold": selected})
print(f"Wrote {len(rows)} ablation rows using threshold {threshold:.3f}")

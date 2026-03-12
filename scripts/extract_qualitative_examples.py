#!/usr/bin/env python3
"""
Extract balanced qualitative examples for dissertation.

It produces a mix of:
A) Corrected & correct (best successes)
B) Corrected but still incorrect (failure cases)
C) Not corrected but correct (verifier avoided unnecessary correction)
D) Not corrected and incorrect (missed hallucination / false negative)

Input (preferred):
- results/stage4_corrections.jsonl

Fallback:
- results/stage3_verification.jsonl  (if stage4 file not available)

Output:
- results/qualitative_examples.json
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

IN_STAGE4 = RESULTS / "stage4_corrections.jsonl"
IN_STAGE3 = RESULTS / "stage3_verification.jsonl"
OUT_JSON = RESULTS / "qualitative_examples.json"

# How many total examples you want in dissertation
TARGET_TOTAL = 10

# Desired mix per dataset (will try best-effort)
TARGET_PER_DATASET = {
    "medhallu": 5,
    "truthfulqa": 5,
}

# Desired mix per category within a dataset
CATEGORY_TARGETS = {
    "A_success_corrected": 2,   # corrected AND final correct
    "B_failure_corrected": 1,   # corrected BUT final incorrect
    "C_correct_nochange": 1,    # not corrected AND correct
    "D_incorrect_nochange": 1,  # not corrected AND incorrect
}

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def safe_get_stage4(rec):
    return rec.get("stage4", {}) if isinstance(rec.get("stage4", {}), dict) else {}

def classify(rec):
    """
    Returns (dataset, category_key, summary_dict)
    """
    dataset = rec.get("dataset", "unknown")
    q = rec.get("question", "")
    cand = rec.get("candidate_answer", "")
    ver = rec.get("verification", {}) or {}
    evidence = ver.get("evidence", []) or []

    stage4 = safe_get_stage4(rec)
    was_corrected = bool(stage4.get("was_corrected", False))
    final_is_correct = bool(stage4.get("final_is_correct", False))

    # If stage4 not present, approximate:
    # - treat "final_is_correct" unknown -> False
    # - was_corrected False
    if "stage4" not in rec:
        was_corrected = False
        final_is_correct = False

    if was_corrected and final_is_correct:
        cat = "A_success_corrected"
    elif was_corrected and (not final_is_correct):
        cat = "B_failure_corrected"
    elif (not was_corrected) and final_is_correct:
        cat = "C_correct_nochange"
    else:
        cat = "D_incorrect_nochange"

    corrected_answer = stage4.get("corrected_answer", "") or stage4.get("final_answer", "")
    final_answer = stage4.get("final_answer", "") if "final_answer" in stage4 else (corrected_answer or cand)

    item = {
        "dataset": dataset,
        "category": cat,
        "question": q,
        "candidate_answer": cand,
        "support_score": ver.get("support_score", None),
        "pred_hallucinated": ver.get("hallucinated", None),
        "evidence_top1": evidence[:1],
        "corrected_answer": corrected_answer,
        "final_answer": final_answer,
        "final_is_correct": final_is_correct,
    }

    # Include extra context helpful for dissertation
    if dataset == "medhallu":
        item["ground_truth"] = rec.get("ground_truth", "")
        item["hallucination_category"] = rec.get("hallucination_category", rec.get("Category of Hallucination", ""))
        item["difficulty"] = rec.get("difficulty", rec.get("Difficulty Level", ""))
    elif dataset == "truthfulqa":
        item["truth_category"] = rec.get("category", rec.get("Category", ""))
        item["type"] = rec.get("type", rec.get("Type", ""))
        item["correct_answers"] = rec.get("correct_answers", [])

    return dataset, cat, item

def pick_balanced(rows):
    # Bucket by dataset then category
    buckets = {}
    for rec in rows:
        ds, cat, item = classify(rec)
        buckets.setdefault(ds, {}).setdefault(cat, []).append(item)

    # For determinism (optional): keep original order but you can sort by support_score if needed
    selected = []

    for ds, ds_target in TARGET_PER_DATASET.items():
        if ds not in buckets:
            continue

        # For each category, pick up to CATEGORY_TARGETS
        for cat, n_cat in CATEGORY_TARGETS.items():
            items = buckets[ds].get(cat, [])
            if not items:
                continue

            # Heuristic: for "success" pick higher support_score; for failures pick lower to show challenges
            def score_key(x):
                s = x.get("support_score")
                return -1.0 if s is None else float(s)

            if cat == "A_success_corrected":
                items = sorted(items, key=score_key, reverse=True)
            elif cat == "B_failure_corrected":
                items = sorted(items, key=score_key)  # lower support = more interesting failure
            else:
                # keep as-is
                pass

            selected.extend(items[:n_cat])

        # If we still don't meet per-dataset count, fill with whatever is left (any category)
        if sum(1 for x in selected if x["dataset"] == ds) < ds_target:
            remaining = []
            for cat_items in buckets[ds].values():
                remaining.extend(cat_items)
            # remove already selected (by question+candidate_answer)
            seen = {(x["question"], x["candidate_answer"]) for x in selected if x["dataset"] == ds}
            remaining = [x for x in remaining if (x["question"], x["candidate_answer"]) not in seen]

            # Prefer corrected cases first, then others
            remaining = sorted(
                remaining,
                key=lambda x: (0 if x["category"].startswith(("A_", "B_")) else 1, -(x["support_score"] or 0)),
            )

            need = ds_target - sum(1 for x in selected if x["dataset"] == ds)
            selected.extend(remaining[:max(0, need)])

    # If still fewer than TARGET_TOTAL, fill globally from any dataset/category
    if len(selected) < TARGET_TOTAL:
        all_items = []
        for ds in buckets:
            for cat in buckets[ds]:
                all_items.extend(buckets[ds][cat])
        seen_all = {(x["dataset"], x["question"], x["candidate_answer"]) for x in selected}
        all_items = [x for x in all_items if (x["dataset"], x["question"], x["candidate_answer"]) not in seen_all]
        all_items = sorted(all_items, key=lambda x: (0 if x["category"].startswith(("A_", "B_")) else 1))
        selected.extend(all_items[: (TARGET_TOTAL - len(selected))])

    # Trim to TARGET_TOTAL
    return selected[:TARGET_TOTAL]

def main():
    if IN_STAGE4.exists():
        rows = load_jsonl(IN_STAGE4)
        source = str(IN_STAGE4)
    elif IN_STAGE3.exists():
        rows = load_jsonl(IN_STAGE3)
        source = str(IN_STAGE3)
    else:
        raise FileNotFoundError("No Stage-4 or Stage-3 JSONL found in results/. Run Stage-3/4 first.")

    selected = pick_balanced(rows)
    OUT_JSON.write_text(json.dumps({
        "source": source,
        "count": len(selected),
        "examples": selected
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Source: {source}")
    print(f"Saved {len(selected)} examples to {OUT_JSON}")

    # Quick category summary
    cat_counts = {}
    ds_counts = {}
    for ex in selected:
        cat_counts[ex["category"]] = cat_counts.get(ex["category"], 0) + 1
        ds_counts[ex["dataset"]] = ds_counts.get(ex["dataset"], 0) + 1
    print("By dataset:", ds_counts)
    print("By category:", cat_counts)

if __name__ == "__main__":
    main()

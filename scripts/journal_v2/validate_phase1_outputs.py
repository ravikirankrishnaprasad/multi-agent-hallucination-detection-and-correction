#!/usr/bin/env python3
"""
journal_v2 / Phase 1 -- Output validator.

Lightweight, dependency-minimal checks that the Phase 1 foundation is correct.
Exits with code 0 if all checks pass, 1 otherwise. Intended to be run after:
    build_balanced_dataset.py -> create_grouped_split.py -> audit_dataset.py

Checks
------
1. balanced_dataset.csv exists
2. balanced_dataset_with_splits.csv exists
3. required columns exist
4. labels are only {0, 1}
5. MedHallu has both labels
6. TruthfulQA has both labels
7. split values are only {train, dev, test}
8. no normalized_question appears in more than one split
9. no empty-answer rows
10. audit files exist
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "results" / "journal_v2" / "data"
AUDIT_DIR = PROJECT_ROOT / "results" / "journal_v2" / "audit"

BALANCED = DATA_DIR / "balanced_dataset.csv"
SPLITS = DATA_DIR / "balanced_dataset_with_splits.csv"

REQUIRED_COLUMNS = [
    "sample_id", "dataset", "original_id", "question", "normalized_question",
    "answer", "label", "answer_type", "evidence_source", "evidence_text",
    "category", "source_split",
]

AUDIT_FILES = [
    "dataset_audit.md",
    "class_distribution_by_dataset.csv",
    "class_distribution_by_split.csv",
    "answer_type_distribution.csv",
    "split_distribution.csv",
    "leakage_check.md",
]


def main() -> int:
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    # 1 & 2: files exist
    check("balanced_dataset.csv exists", BALANCED.exists(), str(BALANCED))
    check("balanced_dataset_with_splits.csv exists", SPLITS.exists(), str(SPLITS))

    df = None
    if SPLITS.exists():
        df = pd.read_csv(SPLITS, keep_default_na=False)
        df["normalized_question"] = df["normalized_question"].fillna("").astype(str)

    # 3: required columns
    if df is not None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        check("required columns exist", not missing, f"missing={missing}")
    else:
        check("required columns exist", False, "splits file missing")

    if df is not None:
        # 4: labels only 0/1
        labels = set(pd.to_numeric(df["label"], errors="coerce").dropna().astype(int).unique())
        check("labels are only {0,1}", labels <= {0, 1} and len(labels) > 0, f"labels={sorted(labels)}")

        # 5 & 6: per-dataset both labels
        for ds in ["medhallu", "truthfulqa"]:
            sub = df[df["dataset"] == ds]
            ds_labels = set(pd.to_numeric(sub["label"], errors="coerce").dropna().astype(int).unique())
            check(f"{ds} has both labels", ds_labels >= {0, 1}, f"{ds} labels={sorted(ds_labels)} rows={len(sub)}")

        # 7: split values
        split_vals = set(df["split"].dropna().astype(str).unique())
        check("split values are only {train,dev,test}", split_vals <= {"train", "dev", "test"} and len(split_vals) > 0, f"splits={sorted(split_vals)}")

        # 8: no normalized_question across splits
        q_split = df.groupby("normalized_question")["split"].nunique()
        leaked = int((q_split > 1).sum())
        check("no normalized_question across multiple splits", leaked == 0, f"leaked_questions={leaked}")

        # 9: no empty answers
        empty_ans = int(df["answer"].fillna("").astype(str).str.strip().eq("").sum())
        check("no empty-answer rows", empty_ans == 0, f"empty_answers={empty_ans}")

    # 10: audit files
    for fn in AUDIT_FILES:
        check(f"audit file {fn} exists", (AUDIT_DIR / fn).exists(), str(AUDIT_DIR / fn))

    # Report
    all_ok = all(ok for _, ok, _ in results)
    print("=" * 64)
    print("journal_v2 Phase 1 validation")
    print("=" * 64)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line)
    print("=" * 64)
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

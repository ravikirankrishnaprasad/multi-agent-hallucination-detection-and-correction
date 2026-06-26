#!/usr/bin/env python3
"""
journal_v2 / Phase 1 -- Balanced dataset construction.

Purpose
-------
Fixes the deepest structural flaw of the rejected v1 pipeline: in v1 the *dataset
identity was confounded with the label* (every MedHallu row was treated as
hallucinated, label=1; every TruthfulQA row as factual, label=0). A classifier
could therefore separate the classes by recognising provenance rather than by
detecting hallucination.

This script rebuilds a *balanced* unified dataset in which BOTH source datasets
contribute BOTH a factual (label=0) and a hallucinated (label=1) answer for the
same question, using the correct/incorrect answer fields that already exist in
each benchmark.

MedHallu
    - "Ground Truth" / factual answer  -> label = 0  (answer_type = ground_truth)
    - "Hallucinated Answer"            -> label = 1  (answer_type = hallucinated_answer)
    - evidence = "Knowledge" field      (evidence_source = medhallu_knowledge)

TruthfulQA
    - "Best Answer" and "Correct Answers"   -> label = 0  (best_answer / correct_answer)
    - "Incorrect Answers"                   -> label = 1  (incorrect_answer)
    - evidence = correct answers / source   (evidence_source = truthfulqa_reference)

Output
------
results/journal_v2/data/balanced_dataset.csv

Unified columns
---------------
sample_id, dataset, original_id, question, normalized_question, answer, label,
answer_type, evidence_source, evidence_text, category, source_split

Notes
-----
* This is Phase 1 of the journal_v2 rebuild. It does NOT implement retrieval,
  verification, correction, or any modelling. It only reconstructs the data
  foundation so that later stages can be evaluated without label/provenance
  confounding.
* The v1 scripts under scripts/ are left untouched as the historical, rejected
  baseline.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# --------------------------------------------------------------------------
# Paths (repo-relative). This file lives at scripts/journal_v2/, so the
# project root is two levels up.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEDHALLU = PROJECT_ROOT / "data" / "raw" / "medhallu_data.csv"
DEFAULT_TRUTHFULQA = PROJECT_ROOT / "data" / "raw" / "TruthfulQA.csv"
DEFAULT_OUT = PROJECT_ROOT / "results" / "journal_v2" / "data" / "balanced_dataset.csv"

UNIFIED_COLUMNS = [
    "sample_id",
    "dataset",
    "original_id",
    "question",
    "normalized_question",
    "answer",
    "label",
    "answer_type",
    "evidence_source",
    "evidence_text",
    "category",
    "source_split",
]


# --------------------------------------------------------------------------
# Text utilities
# --------------------------------------------------------------------------
_ws_re = re.compile(r"\s+")
_zero_width_re = re.compile("[\u200b-\u200d\ufeff]")
_punct_re = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(value: Any) -> str:
    """Light normalisation preserving human readability (used for displayed text)."""
    if value is None:
        return ""
    s = str(value)
    s = _zero_width_re.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _ws_re.sub(" ", s).strip()
    return s


def normalize_question(value: Any) -> str:
    """
    Aggressive normalisation used ONLY for grouped splitting and leakage checks.

    Steps: lowercase -> trim -> collapse whitespace -> remove punctuation
    (so that punctuation-only differences collapse to the same group) ->
    collapse whitespace again.
    """
    s = normalize_text(value).lower()
    s = _punct_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip()
    return s


def parse_list_like(value: Any) -> List[str]:
    """
    Parse a list-like answer field. TruthfulQA stores multiple answers as a
    single string, usually '; '-separated. Also tolerates Python/JSON list
    literals.
    """
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []

    # Python / JSON list literal
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(s)
            if isinstance(parsed, list):
                return [normalize_text(x) for x in parsed if normalize_text(x)]
        except Exception:
            pass

    # Most common TruthfulQA form: '; ' separated, also handle ';' and newlines
    if ";" in s:
        parts = s.split(";")
    elif "\n" in s:
        parts = s.split("\n")
    else:
        parts = [s]

    return [normalize_text(p) for p in parts if normalize_text(p)]


def pick_column(df: pd.DataFrame, candidates: List[str], required: bool, what: str) -> Optional[str]:
    """Return the first matching column name (case-insensitive), or None."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    if required:
        raise ValueError(
            f"Could not find a column for '{what}'. Tried {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# MedHallu
# --------------------------------------------------------------------------
def build_medhallu_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    col_q = pick_column(df, ["Question", "question"], True, "MedHallu question")
    col_know = pick_column(df, ["Knowledge", "knowledge", "Context", "context"], True, "MedHallu knowledge/evidence")
    col_gt = pick_column(df, ["Ground Truth", "ground_truth", "Ground_Truth", "Correct Answer", "answer"], True, "MedHallu ground truth")
    col_hallu = pick_column(df, ["Hallucinated Answer", "hallucinated_answer", "Hallucinated_Answer"], True, "MedHallu hallucinated answer")
    col_diff = pick_column(df, ["Difficulty Level", "difficulty", "difficulty_level"], False, "MedHallu difficulty")
    col_cat = pick_column(df, ["Category of Hallucination", "hallucination_category", "category"], False, "MedHallu category")
    col_split = pick_column(df, ["split", "source_split", "Split"], False, "MedHallu split")

    rows: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        question = normalize_text(row.get(col_q, ""))
        knowledge = normalize_text(row.get(col_know, ""))
        ground_truth = normalize_text(row.get(col_gt, ""))
        hallucinated = normalize_text(row.get(col_hallu, ""))
        category = normalize_text(row.get(col_cat, "")) if col_cat else ""
        source_split = normalize_text(row.get(col_split, "")) if col_split else ""
        nq = normalize_question(question)

        if not question:
            continue

        # label = 0 : factual ground-truth answer
        if ground_truth:
            rows.append(
                {
                    "dataset": "medhallu",
                    "original_id": f"medhallu_{int(i)}",
                    "question": question,
                    "normalized_question": nq,
                    "answer": ground_truth,
                    "label": 0,
                    "answer_type": "ground_truth",
                    "evidence_source": "medhallu_knowledge",
                    "evidence_text": knowledge,
                    "category": category,
                    "source_split": source_split,
                }
            )

        # label = 1 : hallucinated answer
        if hallucinated:
            rows.append(
                {
                    "dataset": "medhallu",
                    "original_id": f"medhallu_{int(i)}",
                    "question": question,
                    "normalized_question": nq,
                    "answer": hallucinated,
                    "label": 1,
                    "answer_type": "hallucinated_answer",
                    "evidence_source": "medhallu_knowledge",
                    "evidence_text": knowledge,
                    "category": category,
                    "source_split": source_split,
                }
            )
    return rows


# --------------------------------------------------------------------------
# TruthfulQA
# --------------------------------------------------------------------------
def build_truthfulqa_rows(df: pd.DataFrame, include_correct_answers: bool = True) -> List[Dict[str, Any]]:
    col_q = pick_column(df, ["Question", "question"], True, "TruthfulQA question")
    col_best = pick_column(df, ["Best Answer", "best_answer", "Best_Answer"], True, "TruthfulQA best answer")
    col_correct = pick_column(df, ["Correct Answers", "correct_answers", "Correct_Answers"], False, "TruthfulQA correct answers")
    col_incorrect = pick_column(df, ["Incorrect Answers", "incorrect_answers", "Incorrect_Answers"], True, "TruthfulQA incorrect answers")
    col_cat = pick_column(df, ["Category", "category"], False, "TruthfulQA category")
    col_source = pick_column(df, ["Source", "source"], False, "TruthfulQA source")
    col_split = pick_column(df, ["split", "source_split", "Split"], False, "TruthfulQA split")

    rows: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        question = normalize_text(row.get(col_q, ""))
        if not question:
            continue
        nq = normalize_question(question)
        best = normalize_text(row.get(col_best, ""))
        correct_list = parse_list_like(row.get(col_correct, "")) if col_correct else []
        incorrect_list = parse_list_like(row.get(col_incorrect, ""))
        category = normalize_text(row.get(col_cat, "")) if col_cat else ""
        source = normalize_text(row.get(col_source, "")) if col_source else ""
        source_split = normalize_text(row.get(col_split, "")) if col_split else ""

        # Reference evidence for the question = correct answers (+ best answer), joined.
        reference_pool = []
        if best:
            reference_pool.append(best)
        reference_pool.extend(correct_list)
        # de-dup keep order
        seen = set()
        reference = []
        for r in reference_pool:
            k = r.lower()
            if k not in seen:
                reference.append(r)
                seen.add(k)
        evidence_text = " ".join(reference)
        if source:
            evidence_text = (evidence_text + " " + source).strip()

        # label = 0 : best answer
        if best:
            rows.append(
                {
                    "dataset": "truthfulqa",
                    "original_id": f"truthfulqa_{int(i)}",
                    "question": question,
                    "normalized_question": nq,
                    "answer": best,
                    "label": 0,
                    "answer_type": "best_answer",
                    "evidence_source": "truthfulqa_reference",
                    "evidence_text": evidence_text,
                    "category": category,
                    "source_split": source_split,
                }
            )

        # label = 0 : additional correct answers (optional, deduped against best)
        if include_correct_answers:
            for ans in correct_list:
                if best and ans.lower() == best.lower():
                    continue
                rows.append(
                    {
                        "dataset": "truthfulqa",
                        "original_id": f"truthfulqa_{int(i)}",
                        "question": question,
                        "normalized_question": nq,
                        "answer": ans,
                        "label": 0,
                        "answer_type": "correct_answer",
                        "evidence_source": "truthfulqa_reference",
                        "evidence_text": evidence_text,
                        "category": category,
                        "source_split": source_split,
                    }
                )

        # label = 1 : incorrect answers
        for ans in incorrect_list:
            rows.append(
                {
                    "dataset": "truthfulqa",
                    "original_id": f"truthfulqa_{int(i)}",
                    "question": question,
                    "normalized_question": nq,
                    "answer": ans,
                    "label": 1,
                    "answer_type": "incorrect_answer",
                    "evidence_source": "truthfulqa_reference",
                    "evidence_text": evidence_text,
                    "category": category,
                    "source_split": source_split,
                }
            )
    return rows


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def assemble(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # Drop any rows with empty answers (defensive; loaders already skip blanks)
    df = df[df["answer"].map(lambda x: bool(normalize_text(x)))].reset_index(drop=True)
    # Stable ordering for reproducibility
    df = df.sort_values(["dataset", "original_id", "label", "answer_type"]).reset_index(drop=True)
    # Assign globally unique sample ids
    df.insert(0, "sample_id", [f"{r.dataset}_{idx:07d}" for idx, r in zip(range(len(df)), df.itertuples())])
    # Enforce column order
    df = df[UNIFIED_COLUMNS]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced unified hallucination dataset (journal_v2 Phase 1).")
    parser.add_argument("--medhallu_path", type=Path, default=DEFAULT_MEDHALLU)
    parser.add_argument("--truthfulqa_path", type=Path, default=DEFAULT_TRUTHFULQA)
    parser.add_argument("--out_path", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no_correct_answers",
        action="store_true",
        help="If set, do NOT expand TruthfulQA 'Correct Answers' into extra label=0 rows (keep only Best Answer).",
    )
    args = parser.parse_args()

    if not args.medhallu_path.exists():
        raise FileNotFoundError(f"MedHallu raw CSV not found: {args.medhallu_path}")
    if not args.truthfulqa_path.exists():
        raise FileNotFoundError(f"TruthfulQA raw CSV not found: {args.truthfulqa_path}")

    med_df = pd.read_csv(args.medhallu_path, keep_default_na=False)
    tqa_df = pd.read_csv(args.truthfulqa_path, keep_default_na=False)

    print(f"[build] MedHallu raw rows   : {len(med_df)}")
    print(f"[build] TruthfulQA raw rows : {len(tqa_df)}")

    rows: List[Dict[str, Any]] = []
    rows.extend(build_medhallu_rows(med_df))
    rows.extend(build_truthfulqa_rows(tqa_df, include_correct_answers=not args.no_correct_answers))

    df = assemble(rows)

    ensure_dir(args.out_path.parent)
    df.to_csv(args.out_path, index=False)

    # Console summary (no results invented; these are just counts of what was built)
    print(f"\n[build] Unified balanced rows : {len(df)}")
    print("[build] Label counts by dataset:")
    print(df.groupby(["dataset", "label"]).size().to_string())
    print(f"\n[build] Wrote -> {args.out_path}")


if __name__ == "__main__":
    main()

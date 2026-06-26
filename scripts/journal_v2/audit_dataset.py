#!/usr/bin/env python3
"""
journal_v2 / Phase 1 -- Dataset audit.

Produces reviewer-defensible evidence that the reconstructed dataset:
  * contains both classes in BOTH source datasets,
  * is split by question with NO normalized_question leakage across splits,
  * has documented class / answer-type / dataset distributions,
  * does not (in this construction) score a row against its own evidence at test
    time -- a flag is computed for transparency.

Input
-----
results/journal_v2/data/balanced_dataset_with_splits.csv

Outputs
-------
results/journal_v2/audit/dataset_audit.md
results/journal_v2/audit/class_distribution_by_dataset.csv
results/journal_v2/audit/class_distribution_by_split.csv
results/journal_v2/audit/answer_type_distribution.csv
results/journal_v2/audit/split_distribution.csv
results/journal_v2/audit/leakage_check.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = PROJECT_ROOT / "results" / "journal_v2" / "data" / "balanced_dataset_with_splits.csv"
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "results" / "journal_v2" / "audit"

SPLITS = ["train", "dev", "test"]


def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the balanced split dataset (journal_v2 Phase 1).")
    parser.add_argument("--in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--audit_dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()

    if not args.in_path.exists():
        raise FileNotFoundError(f"Split dataset not found: {args.in_path}. Run create_grouped_split.py first.")

    # Display the input path relative to the repo root so committed audit files
    # do not embed machine-specific absolute paths.
    try:
        disp_in = args.in_path.resolve().relative_to(PROJECT_ROOT)
    except Exception:
        disp_in = args.in_path

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.in_path, keep_default_na=False)
    for c in ["normalized_question", "dataset", "answer_type", "split"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    # ---------------- Distributions ----------------
    class_by_dataset = (
        df.groupby(["dataset", "label"]).size().rename("count").reset_index()
        .pivot(index="dataset", columns="label", values="count").fillna(0).astype(int)
    )
    class_by_dataset.columns = [f"label_{c}" for c in class_by_dataset.columns]
    class_by_dataset = class_by_dataset.reset_index()
    class_by_dataset.to_csv(args.audit_dir / "class_distribution_by_dataset.csv", index=False)

    class_by_split = (
        df.groupby(["split", "label"]).size().rename("count").reset_index()
        .pivot(index="split", columns="label", values="count").fillna(0).astype(int)
    )
    class_by_split.columns = [f"label_{c}" for c in class_by_split.columns]
    class_by_split = class_by_split.reset_index()
    class_by_split.to_csv(args.audit_dir / "class_distribution_by_split.csv", index=False)

    answer_type_dist = (
        df.groupby(["dataset", "answer_type", "label"]).size().rename("count").reset_index()
    )
    answer_type_dist.to_csv(args.audit_dir / "answer_type_distribution.csv", index=False)

    split_dist = (
        df.groupby(["split", "dataset"]).size().rename("rows").reset_index()
    )
    uq_per_split = df.groupby("split")["normalized_question"].nunique().rename("unique_questions").reset_index()
    split_dist = split_dist.merge(uq_per_split, on="split", how="left")
    split_dist.to_csv(args.audit_dir / "split_distribution.csv", index=False)

    # ---------------- Leakage checks ----------------
    # 1. normalized_question overlap across splits
    q_split = df.groupby("normalized_question")["split"].nunique()
    leaked_questions = q_split[q_split > 1]
    overlap_count = int(len(leaked_questions))

    # 2. examples per normalized question
    per_q = df.groupby("normalized_question").size()
    n_unique_q = int(df["normalized_question"].nunique())

    # 3. duplicate/near-duplicate questions spanning original ids (possible cross-source leakage)
    q_to_ids = df.groupby("normalized_question")["original_id"].nunique()
    multi_id_questions = q_to_ids[q_to_ids > 1]

    # 4. test rows used as own retrieval evidence: in this construction the stored
    #    evidence_text is paired to the row; we flag whether answer text is literally
    #    contained in its own evidence (informational; retrieval is split-aware later).
    def answer_in_own_evidence(row) -> bool:
        ans = str(row.get("answer", "")).strip().lower()
        ev = str(row.get("evidence_text", "")).strip().lower()
        if not ans or not ev:
            return False
        return ans in ev

    df_test = df[df["split"] == "test"].copy()
    if len(df_test):
        df_test["answer_in_own_evidence"] = df_test.apply(answer_in_own_evidence, axis=1)
        test_self_evidence = int(df_test["answer_in_own_evidence"].sum())
    else:
        test_self_evidence = 0

    # ---------------- leakage_check.md ----------------
    leakage_lines: List[str] = []
    leakage_lines.append("# Leakage Check (journal_v2 Phase 1)\n")
    leakage_lines.append(f"- Source file: `{disp_in}`")
    leakage_lines.append(f"- Total rows: **{len(df)}**")
    leakage_lines.append(f"- Unique normalized questions: **{n_unique_q}**")
    leakage_lines.append("")
    leakage_lines.append("## 1. normalized_question overlap across train/dev/test")
    if overlap_count == 0:
        leakage_lines.append("**PASS** — no normalized_question appears in more than one split.")
    else:
        leakage_lines.append(f"**FAIL** — {overlap_count} normalized_question(s) appear in multiple splits:")
        for q in list(leaked_questions.index)[:20]:
            leakage_lines.append(f"  - `{q[:120]}`")
    leakage_lines.append("")
    leakage_lines.append("## 2. Examples per normalized question")
    leakage_lines.append(f"- min: {int(per_q.min()) if len(per_q) else 0}, "
                         f"max: {int(per_q.max()) if len(per_q) else 0}, "
                         f"mean: {float(per_q.mean()) if len(per_q) else 0:.2f}")
    leakage_lines.append("")
    leakage_lines.append("## 3. Normalized questions spanning multiple original ids")
    leakage_lines.append(
        "These are normalized questions that map to more than one source row "
        "(potential near-duplicate / cross-source collision). Grouped splitting "
        "keeps them in a single split, so they do not cause cross-split leakage."
    )
    leakage_lines.append(f"- count: **{int(len(multi_id_questions))}**")
    for q in list(multi_id_questions.index)[:20]:
        leakage_lines.append(f"  - `{q[:120]}` -> {int(multi_id_questions[q])} ids")
    leakage_lines.append("")
    leakage_lines.append("## 4. Test rows whose answer text is contained in their own stored evidence")
    leakage_lines.append(
        "Informational only. In Phase 1 no retrieval is performed. The stored "
        "`evidence_text` is paired reference material; later retrieval indices "
        "must be split-aware so a test answer is never scored against its own "
        "originating evidence (see docs/journal_v2/retrieval_leakage_control.md)."
    )
    leakage_lines.append(f"- test rows: **{len(df_test)}**, answer-in-own-evidence: **{test_self_evidence}**")
    leakage_lines.append("")
    (args.audit_dir / "leakage_check.md").write_text("\n".join(leakage_lines), encoding="utf-8")

    # ---------------- dataset_audit.md ----------------
    datasets_present = sorted(df["dataset"].unique())
    both_classes_ok = all(
        set(df[df["dataset"] == d]["label"].unique()) >= {0, 1} for d in datasets_present
    )

    audit_lines: List[str] = []
    audit_lines.append("# Dataset Audit (journal_v2 Phase 1)\n")
    audit_lines.append(
        "> Provenance: this audit is generated by `audit_dataset.py` from the "
        "input file shown below (built from `data/raw` via the Phase 1 "
        "scripts). Numbers are descriptive dataset counts only — no "
        "experimental results are claimed. Regenerate deterministically with "
        "the Phase 1 reproduction steps.\n"
    )
    audit_lines.append(f"- Source file: `{disp_in}`")
    audit_lines.append(f"- Total rows: **{len(df)}**")
    audit_lines.append(f"- Datasets present: {', '.join(datasets_present)}")
    audit_lines.append(f"- Unique normalized questions: **{n_unique_q}**")
    audit_lines.append("")
    audit_lines.append("## Both datasets contain both labels?")
    audit_lines.append(f"**{'PASS' if both_classes_ok else 'FAIL'}**")
    audit_lines.append("")
    audit_lines.append("## Class distribution by dataset")
    audit_lines.append(md_table(class_by_dataset))
    audit_lines.append("")
    audit_lines.append("## Class distribution by split")
    audit_lines.append(md_table(class_by_split))
    audit_lines.append("")
    audit_lines.append("## Answer-type distribution")
    audit_lines.append(md_table(answer_type_dist))
    audit_lines.append("")
    audit_lines.append("## Dataset distribution by split")
    audit_lines.append(md_table(split_dist))
    audit_lines.append("")
    audit_lines.append("## Leakage summary")
    audit_lines.append(f"- normalized_question overlap across splits: "
                       f"**{'PASS (0)' if overlap_count == 0 else f'FAIL ({overlap_count})'}**")
    audit_lines.append(f"- test answer-in-own-evidence flags: **{test_self_evidence}** "
                       f"(informational; see leakage_check.md)")
    audit_lines.append("")
    audit_lines.append("See `leakage_check.md` for full leakage detail.")
    (args.audit_dir / "dataset_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")

    # ---------------- Console ----------------
    print("[audit] Wrote:")
    for fn in [
        "dataset_audit.md",
        "class_distribution_by_dataset.csv",
        "class_distribution_by_split.csv",
        "answer_type_distribution.csv",
        "split_distribution.csv",
        "leakage_check.md",
    ]:
        print(f"  - {args.audit_dir / fn}")
    print(f"[audit] both datasets contain both labels: {'PASS' if both_classes_ok else 'FAIL'}")
    print(f"[audit] normalized_question split overlap: {'PASS (0)' if overlap_count == 0 else f'FAIL ({overlap_count})'}")


if __name__ == "__main__":
    main()

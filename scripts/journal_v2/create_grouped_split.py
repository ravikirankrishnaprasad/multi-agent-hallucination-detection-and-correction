#!/usr/bin/env python3
"""
journal_v2 / Phase 1 -- Grouped train/dev/test split.

Why grouped?
------------
Splitting by individual ROW would let the factual and hallucinated answers for
the SAME question land in different splits, leaking information across the
train/dev/test boundary. We therefore split by `normalized_question`: every row
that shares a normalized question is assigned to exactly one split.

Determinism
-----------
The set of unique normalized questions is sorted, then shuffled with a fixed
seed, then partitioned 70/15/15 by question count. This is fully reproducible.

Input
-----
results/journal_v2/data/balanced_dataset.csv

Output
------
results/journal_v2/data/balanced_dataset_with_splits.csv
    (adds a `split` column with values train / dev / test)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = PROJECT_ROOT / "results" / "journal_v2" / "data" / "balanced_dataset.csv"
DEFAULT_OUT = PROJECT_ROOT / "results" / "journal_v2" / "data" / "balanced_dataset_with_splits.csv"


def assign_groups_to_splits(
    groups: List[str],
    seed: int,
    train_frac: float,
    dev_frac: float,
) -> Dict[str, str]:
    """Assign each group (normalized_question) to a single split."""
    ordered = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    n_train = int(round(train_frac * n))
    n_dev = int(round(dev_frac * n))
    # Guard against rounding overflow
    n_train = min(n_train, n)
    n_dev = min(n_dev, n - n_train)

    mapping: Dict[str, str] = {}
    for idx, g in enumerate(ordered):
        if idx < n_train:
            mapping[g] = "train"
        elif idx < n_train + n_dev:
            mapping[g] = "dev"
        else:
            mapping[g] = "test"
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Create grouped train/dev/test split (journal_v2 Phase 1).")
    parser.add_argument("--in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out_path", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--dev_frac", type=float, default=0.15)
    # test_frac is implied = 1 - train - dev
    args = parser.parse_args()

    if not args.in_path.exists():
        raise FileNotFoundError(f"Balanced dataset not found: {args.in_path}. Run build_balanced_dataset.py first.")

    if args.train_frac + args.dev_frac >= 1.0:
        raise ValueError("train_frac + dev_frac must be < 1.0 (test gets the remainder).")

    df = pd.read_csv(args.in_path, keep_default_na=False)
    if "normalized_question" not in df.columns:
        raise ValueError("Input is missing 'normalized_question'. Rebuild with build_balanced_dataset.py.")

    df["normalized_question"] = df["normalized_question"].fillna("").astype(str)

    mapping = assign_groups_to_splits(
        groups=df["normalized_question"].tolist(),
        seed=args.seed,
        train_frac=args.train_frac,
        dev_frac=args.dev_frac,
    )
    df["split"] = df["normalized_question"].map(mapping)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_path, index=False)

    n_groups = df["normalized_question"].nunique()
    print(f"[split] seed={args.seed}  train={args.train_frac}  dev={args.dev_frac}  test={1 - args.train_frac - args.dev_frac:.2f}")
    print(f"[split] unique normalized questions: {n_groups}")
    print("[split] rows per split:")
    print(df["split"].value_counts().to_string())
    print("[split] unique questions per split:")
    print(df.groupby("split")["normalized_question"].nunique().to_string())
    print(f"\n[split] Wrote -> {args.out_path}")


if __name__ == "__main__":
    main()

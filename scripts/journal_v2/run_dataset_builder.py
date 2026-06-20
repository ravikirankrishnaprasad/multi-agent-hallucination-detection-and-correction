#!/usr/bin/env python3
import argparse
from core import construct_dataset, make_splits, path, summary, write_json

p=argparse.ArgumentParser(description="Build reproducible within-dataset journal-v2 splits.")
p.add_argument("--input", required=True); p.add_argument("--out_dir", default="results/journal_v2")
p.add_argument("--random_seed", type=int, default=42); p.add_argument("--dev_size", type=float, default=.2); p.add_argument("--test_size", type=float, default=.2); p.add_argument("--stratified", action="store_true")
p.add_argument("--dataset_filter", default=""); p.add_argument("--limit", type=int, default=0); p.add_argument("--limit_medhallu", type=int, default=0); p.add_argument("--limit_truthfulqa", type=int, default=0); p.add_argument("--dry_run", action="store_true")
a=p.parse_args(); out=path(a.out_dir); df=construct_dataset(a.input, a.limit or (100 if a.dry_run else 0), a.limit_medhallu, a.limit_truthfulqa)
if a.dataset_filter: df=df[df.dataset.isin([x.strip().lower() for x in a.dataset_filter.split(",")])].copy()
splits=make_splits(df,a.random_seed,a.dev_size,a.test_size,a.stratified); out.mkdir(parents=True,exist_ok=True); splits.to_csv(out/"split_assignments.csv",index=False); df.to_csv(out/"constructed_dataset.csv",index=False)
s=summary(df,splits); s["config"]=vars(a); write_json(out/"dataset_summary.json",s); print(f"Wrote {len(df)} constructed examples to {out}")

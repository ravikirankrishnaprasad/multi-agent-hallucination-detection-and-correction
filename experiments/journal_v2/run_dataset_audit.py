#!/usr/bin/env python3
"""Audit split leakage and duplicate records for journal-v2 constructed data."""
import argparse
import json
import pandas as pd
from core import construct_dataset, path, text, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--splits", required=True)
parser.add_argument("--out_dir", default="results/journal_v2")
args = parser.parse_args()

data = construct_dataset(args.input)
splits = pd.read_csv(path(args.splits))
data = data.merge(splits[["sample_id", "split"]], on="sample_id")
data["original_sample_id"] = data.sample_id.map(lambda value: str(value).rsplit("::", 1)[0])
data["normalized_question"] = data.question.map(lambda value: text(value).lower())
data["normalized_answer"] = data.answer.map(lambda value: text(value).lower())

issues = []
for field, issue_type in [("original_sample_id", "derived_group_split_leakage"), ("normalized_question", "question_split_leakage")]:
    for key, group in data.groupby(field):
        if key and group.split.nunique() > 1:
            issues.append({"issue_type": issue_type, "key": key, "splits": ",".join(sorted(group.split.unique())), "labels": ",".join(map(str, sorted(group.label.unique()))), "count": len(group)})
for key, group in data.groupby(["normalized_question", "normalized_answer"]):
    if len(group) > 1:
        issues.append({"issue_type": "duplicate_question_answer", "key": " || ".join(key), "splits": ",".join(sorted(group.split.unique())), "labels": ",".join(map(str, sorted(group.label.unique()))), "count": len(group)})

counts = data.groupby(["dataset", "split", "label"]).size().rename("count").reset_index().to_dict("records")
payload = {"n_samples": len(data), "leakage_found": any(x["issue_type"] != "duplicate_question_answer" for x in issues), "issue_counts": pd.Series([x["issue_type"] for x in issues]).value_counts().to_dict(), "label_counts_by_dataset_split": counts}
out = path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
write_json(out / "dataset_audit.json", payload)
pd.DataFrame(issues, columns=["issue_type", "key", "splits", "labels", "count"]).to_csv(out / "dataset_audit.csv", index=False)
print(f"Audit complete: {len(issues)} findings; leakage={payload['leakage_found']}")

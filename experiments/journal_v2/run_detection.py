#!/usr/bin/env python3
import argparse, json
import pandas as pd
from core import construct_dataset, metric_dict, path, per_dataset_metrics, read_jsonl, score_records, write_json
p=argparse.ArgumentParser(description="Evaluate the selected threshold on test only."); p.add_argument("--input",required=True); p.add_argument("--splits",required=True); p.add_argument("--selected_threshold",required=True); p.add_argument("--out_dir",default="results/journal_v2"); p.add_argument("--top_k",type=int,default=5); p.add_argument("--alpha",type=float,default=.7); p.add_argument("--limit",type=int,default=0); p.add_argument("--limit_medhallu",type=int,default=0); p.add_argument("--limit_truthfulqa",type=int,default=0); p.add_argument("--dry_run",action="store_true"); a=p.parse_args()
df=construct_dataset(a.input,a.limit or (100 if a.dry_run else 0),a.limit_medhallu,a.limit_truthfulqa).merge(pd.read_csv(path(a.splits))[["sample_id","split"]],on="sample_id"); test=df[df.split=="test"].copy(); threshold=json.loads(path(a.selected_threshold).read_text())["threshold"]; records=score_records(test,a.top_k,a.alpha)
for r in records: r["predicted_label"]=int(r["support_score"]<threshold); r["threshold"]=threshold
out=path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
with (out/"predictions.jsonl").open("w",encoding="utf8") as f:
 for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
metrics={"split":"test","threshold":threshold,"overall":metric_dict([int(r["label"]) for r in records],[r["predicted_label"] for r in records]),"per_dataset":per_dataset_metrics(records),"config":vars(a)}; write_json(out/"metrics.json",metrics); print(f"Wrote {len(records)} test predictions")

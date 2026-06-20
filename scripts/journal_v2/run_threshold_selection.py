#!/usr/bin/env python3
import argparse
import pandas as pd
from core import construct_dataset, metric_dict, path, score_records, write_json
p=argparse.ArgumentParser(description="Sweep thresholds on dev only."); p.add_argument("--input",required=True); p.add_argument("--splits",required=True); p.add_argument("--out_dir",default="results/journal_v2"); p.add_argument("--top_k",type=int,default=5); p.add_argument("--alpha",type=float,default=.7); p.add_argument("--thresholds",default="0.10,0.20,0.30,0.40,0.50"); p.add_argument("--selection_metric",choices=["macro_f1","balanced_accuracy","f1"],default="macro_f1"); p.add_argument("--limit",type=int,default=0); p.add_argument("--limit_medhallu",type=int,default=0); p.add_argument("--limit_truthfulqa",type=int,default=0); p.add_argument("--dry_run",action="store_true"); a=p.parse_args()
df=construct_dataset(a.input,a.limit or (100 if a.dry_run else 0),a.limit_medhallu,a.limit_truthfulqa).merge(pd.read_csv(path(a.splits))[["sample_id","split"]],on="sample_id"); dev=df[df.split=="dev"].copy(); records=score_records(dev,a.top_k,a.alpha); rows=[]
for t in [float(x) for x in a.thresholds.split(",")]:
  pred=[int(r["support_score"]<t) for r in records]; m=metric_dict([int(r["label"]) for r in records],pred); rows.append({"threshold":t,**m})
out=path(a.out_dir); out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/"threshold_sweep.csv",index=False); best=max(rows,key=lambda r:r[a.selection_metric]); write_json(out/"selected_threshold.json",{"threshold":best["threshold"],"selection_metric":a.selection_metric,"dev_metrics":best,"config":vars(a)}); print(f"Selected {best['threshold']:.3f} on dev by {a.selection_metric}")

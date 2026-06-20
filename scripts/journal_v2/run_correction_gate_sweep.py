#!/usr/bin/env python3
"""Select correction gates on dev only; optionally report a frozen-gate test sweep."""
import argparse, itertools
import pandas as pd
from core import path, read_jsonl, text, write_json
def overlap(a,b):
    aa=set(text(a).lower().split()); bb=set(text(b).lower().split()); return len(aa&bb)/len(bb) if bb else 0.
p=argparse.ArgumentParser(); p.add_argument("--predictions",required=True); p.add_argument("--out_dir",default="results/journal_v2"); p.add_argument("--split",choices=["dev","test"],default="dev"); p.add_argument("--allow_test_selection",action="store_true"); p.add_argument("--min_retrieval_confidences",default="0.20,0.30,0.40,0.50"); p.add_argument("--max_answer_supports",default="0.10,0.20,0.30"); p.add_argument("--min_evidence_supports",default="0.20,0.30,0.40,0.50"); a=p.parse_args(); records=read_jsonl(path(a.predictions))
if any(r.get("split") != a.split for r in records): raise ValueError("Prediction records do not match --split.")
if a.split == "test" and a.allow_test_selection: print("WARNING: selecting on test is explicitly overridden and is not journal-valid.")
rows=[]
for retrieval,answer,evidence in itertools.product(*[[float(v) for v in getattr(a,n).split(",")] for n in ["min_retrieval_confidences","max_answer_supports","min_evidence_supports"]]):
    applied=[]; fixed=[]; regression=[]
    for r in records:
        gate=bool(r["predicted_label"]) and r["retrieval_score"]>=retrieval and r["answer_support"]<=answer and r["retrieval_score"]>=evidence; applied.append(gate); final=r["best_evidence"] if gate else r["answer"]; fixed.append(r["label"]==1 and gate and overlap(final,r["ground_truth"])>=.5); regression.append(r["label"]==0 and gate)
    n=len(records); positives=sum(r["label"]==1 for r in records); corrected=sum(applied); fixed_n=sum(fixed); corrected_positive=sum(g and r["label"]==1 for g,r in zip(applied,records)); regression_n=sum(regression)
    rows.append({"min_retrieval_confidence":retrieval,"max_answer_support":answer,"min_evidence_support":evidence,"corrected_cases":corrected,"abstention_rate":(sum(bool(r["predicted_label"]) for r in records)-corrected)/n if n else 0.,"regression_rate":regression_n/n if n else 0.,"fixed_positive_cases":fixed_n,"positive_hallucination_reduction":fixed_n/positives if positives else 0.,"correction_accuracy":fixed_n/corrected_positive if corrected_positive else 0.})
frame=pd.DataFrame(rows).sort_values(["regression_rate","correction_accuracy","fixed_positive_cases","positive_hallucination_reduction","abstention_rate"],ascending=[True,False,False,False,False]); out=path(a.out_dir);out.mkdir(parents=True,exist_ok=True); frame.to_csv(out/f"correction_gate_sweep_{a.split}.csv",index=False)
if a.split == "dev":
    selectable=frame[frame.corrected_cases>0]; selected=(selectable if not selectable.empty else frame).iloc[0].to_dict(); selected["selected_on_dev"]=True; selected["evaluated_on_test"]=False; write_json(out/"selected_correction_gate.json",selected); print("Selected dev gate",selected)
elif not a.allow_test_selection: print("Test sweep written for diagnostics only; no gate selected.")

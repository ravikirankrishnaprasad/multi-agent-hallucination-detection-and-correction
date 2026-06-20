#!/usr/bin/env python3
"""Choose conservative correction gates using held-out prediction records."""
import argparse
import itertools
import pandas as pd
from core import path, read_jsonl, text, write_json

def overlap(a, b):
    aa=set(text(a).lower().split()); bb=set(text(b).lower().split())
    return len(aa & bb) / len(bb) if bb else 0.0

parser=argparse.ArgumentParser()
parser.add_argument("--predictions", required=True); parser.add_argument("--out_dir", default="results/journal_v2")
parser.add_argument("--min_retrieval_confidences", default="0.20,0.30,0.40,0.50")
parser.add_argument("--max_answer_supports", default="0.10,0.20,0.30")
parser.add_argument("--min_evidence_supports", default="0.20,0.30,0.40,0.50")
args=parser.parse_args(); records=read_jsonl(path(args.predictions)); rows=[]
for retrieval, answer, evidence in itertools.product(*[[float(v) for v in getattr(args, name).split(",")] for name in ["min_retrieval_confidences", "max_answer_supports", "min_evidence_supports"]]):
    applied=[]; fixed=[]; regression=[]
    for r in records:
        gate=bool(r["predicted_label"]) and r["retrieval_score"]>=retrieval and r["answer_support"]<=answer and r["retrieval_score"]>=evidence
        applied.append(gate)
        final=r["best_evidence"] if gate else r["answer"]
        fixed.append(r["label"]==1 and gate and overlap(final,r["ground_truth"])>=.5)
        regression.append(r["label"]==0 and gate)
    positives=sum(r["label"]==1 for r in records); corrected=sum(applied); fixed_n=sum(fixed); regression_n=sum(regression); n=len(records)
    rows.append({"min_retrieval_confidence":retrieval,"max_answer_support":answer,"min_evidence_support":evidence,"corrected_cases":corrected,"regression_rate":regression_n/n if n else 0.,"positive_hallucination_reduction":fixed_n/positives if positives else 0.,"correction_accuracy":fixed_n/sum(g and r["label"]==1 for g,r in zip(applied,records)) if any(g and r["label"]==1 for g,r in zip(applied,records)) else 0.})
frame=pd.DataFrame(rows).sort_values(["regression_rate","positive_hallucination_reduction","correction_accuracy"],ascending=[True,False,False])
# Exclude abstain-on-everything configurations: they are safe but do not test a
# correction policy. The full sweep remains in CSV for transparent reporting.
selectable=frame[frame.corrected_cases > 0]
if selectable.empty: selectable=frame
out=path(args.out_dir); out.mkdir(parents=True,exist_ok=True); frame.to_csv(out/"correction_gate_sweep.csv",index=False); write_json(out/"selected_correction_gate.json",selectable.iloc[0].to_dict()); print("Selected safe gate",selectable.iloc[0].to_dict())

#!/usr/bin/env python3
"""Apply evidence-gated correction to journal-v2 test predictions."""
import argparse, json
from collections import defaultdict
import pandas as pd
from core import path, read_jsonl, text, write_json

def overlap(a,b):
    aa=set(text(a).lower().split()); bb=set(text(b).lower().split())
    return len(aa&bb)/len(bb) if bb else 0.0

def metrics(rows):
    total=len(rows); positives=sum(r["label"]==1 for r in rows)
    corrected=sum(r["correction_applied"] for r in rows)
    corrected_positives=sum(r["label"]==1 and r["correction_applied"] for r in rows)
    abstained=sum(r["correction_abstained"] for r in rows)
    fixed=sum(r["label"]==1 and r["correction_applied"] and not r["after_hallucinated"] for r in rows)
    regressions=sum(r["label"]==0 and r["correction_applied"] and r["after_hallucinated"] for r in rows)
    return {"n":total,"corrected_cases":corrected,"corrected_positive_cases":corrected_positives,"correction_abstained_cases":abstained,"fixed_positive_cases":fixed,"regression_cases":regressions,"regression_rate":regressions/total if total else 0.,"positive_hallucination_reduction":fixed/positives if positives else 0.,"after_hallucination_rate":sum(r["after_hallucinated"] for r in rows)/total if total else 0.,"correction_accuracy":fixed/corrected_positives if corrected_positives else 0.}

p=argparse.ArgumentParser(); p.add_argument("--predictions",required=True); p.add_argument("--out_dir",default="results/journal_v2"); p.add_argument("--selected_gate",default=""); p.add_argument("--correction_threshold",type=float,default=None); p.add_argument("--min_retrieval_confidence",type=float,default=.20); p.add_argument("--max_answer_support",type=float,default=.30); p.add_argument("--min_evidence_support",type=float,default=0.0); p.add_argument("--safe_mode",action="store_true"); p.add_argument("--max_regression_target",type=float,default=None); p.add_argument("--use_nli_gate",action="store_true",help="Require an NLI unsupported/contradiction result when present."); p.add_argument("--enable_abstention",action="store_true"); a=p.parse_args()
if a.selected_gate:
    gate=json.loads(path(a.selected_gate).read_text(encoding="utf-8")); a.min_retrieval_confidence=float(gate["min_retrieval_confidence"]); a.max_answer_support=float(gate["max_answer_support"]); a.min_evidence_support=float(gate["min_evidence_support"]); a.safe_mode=True
rows=[]
for r in read_jsonl(path(a.predictions)):
    threshold=a.correction_threshold if a.correction_threshold is not None else float(r["threshold"]); predicted=bool(r["predicted_label"])
    evidence_support=float(r.get("evidence_support", r["retrieval_score"]))
    eligible=predicted and r["support_score"]<threshold and r["retrieval_score"]>=a.min_retrieval_confidence and r["answer_support"]<=a.max_answer_support
    if a.safe_mode: eligible = eligible and evidence_support >= a.min_evidence_support
    if a.use_nli_gate: eligible = eligible and r.get("nli_label") in {"contradiction", "neutral"}
    reason="" if eligible or not predicted else ("weak_evidence" if r["retrieval_score"]<a.min_retrieval_confidence else "answer_support_too_high")
    if predicted and not eligible and a.safe_mode and evidence_support < a.min_evidence_support: reason="insufficient_best_evidence_support"
    if predicted and not eligible and a.use_nli_gate and not r.get("nli_label"): reason="nli_gate_unavailable"
    applied=bool(eligible); abstained=bool(predicted and not eligible and a.enable_abstention); corrected=r["best_evidence"] if applied else r["answer"]
    after=not bool(overlap(corrected,r["ground_truth"])>=.5) if r["label"]==1 else bool(applied)
    rows.append({**r,"original_answer":r["answer"],"corrected_answer":corrected,"correction_applied":applied,"correction_abstained":abstained,"abstention_reason":reason if abstained else "","before_hallucinated":bool(r["label"]),"after_hallucinated":after,"regression_flag":bool(r["label"]==0 and applied and after),"selected_on_dev":bool(a.selected_gate),"evaluated_on_test":r.get("split")=="test"})
out=path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
with (out/"corrections.jsonl").open("w",encoding="utf8") as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
by=defaultdict(list)
for r in rows:by[r["dataset"]].append(r)
payload={"overall":metrics(rows),"per_dataset":{k:metrics(v) for k,v in by.items()},"config":vars(a)}
payload["overall"]["correction_coverage"]=payload["overall"]["corrected_cases"]/payload["overall"]["n"] if payload["overall"]["n"] else 0.0
payload["overall"]["abstention_rate"]=payload["overall"]["correction_abstained_cases"]/payload["overall"]["n"] if payload["overall"]["n"] else 0.0
payload["overall"]["safe_correction_rate"]=payload["overall"]["fixed_positive_cases"]/payload["overall"]["n"] if payload["overall"]["n"] else 0.0
if a.max_regression_target is not None: payload["regression_target_met"]=payload["overall"]["regression_rate"] <= a.max_regression_target
write_json(out/"correction_metrics.json",payload)
errors=[]
for r in rows:
    typ="false_positive" if r["label"]==0 and r["predicted_label"] else "false_negative" if r["label"]==1 and not r["predicted_label"] else "correction_regression" if r["regression_flag"] else "abstained_hallucinated" if r["correction_abstained"] and r["label"]==1 else "successful_correction" if r["correction_applied"] and r["label"]==1 and not r["after_hallucinated"] else ""
    if typ: errors.append({"sample_id":r["sample_id"],"dataset":r["dataset"],"error_type":typ,"question":r["question"],"original_answer":r["original_answer"],"corrected_answer":r["corrected_answer"],"ground_truth":r["ground_truth"],"support_score":r["support_score"],"answer_support":r["answer_support"],"retrieval_score":r["retrieval_score"],"best_evidence":r["best_evidence"],"explanation":r["abstention_reason"] or "Evaluation outcome recorded by the gated correction protocol."})
error_columns=["sample_id","dataset","error_type","question","original_answer","corrected_answer","ground_truth","support_score","answer_support","retrieval_score","best_evidence","explanation"]
pd.DataFrame(errors, columns=error_columns).to_csv(out/"error_analysis_samples.csv",index=False); print(f"Wrote {len(rows)} gated corrections")

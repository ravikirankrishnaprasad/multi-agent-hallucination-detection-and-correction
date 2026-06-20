#!/usr/bin/env python3
"""Run grouped TF-IDF evaluation and frozen correction for several random seeds."""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
from core import ROOT, path, write_json
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out_dir',default='results/journal_v2');p.add_argument('--seeds',default='13,21,42,87,101');a=p.parse_args();out=path(a.out_dir);rows=[]
for seed in [int(x) for x in a.seeds.split(',')]:
 with tempfile.TemporaryDirectory(prefix=f'journal-v2-{seed}-') as temp:
  d=Path(temp); run=lambda *args: subprocess.run([sys.executable,*args],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
  run('scripts/journal_v2/run_dataset_builder.py','--input',a.input,'--out_dir',str(d),'--random_seed',str(seed),'--dev_size','.20','--test_size','.20','--stratified')
  run('scripts/journal_v2/run_threshold_selection.py','--input',a.input,'--splits',str(d/'split_assignments.csv'),'--out_dir',str(d),'--selection_metric','macro_f1')
  run('scripts/journal_v2/run_detection.py','--input',a.input,'--splits',str(d/'split_assignments.csv'),'--selected_threshold',str(d/'selected_threshold.json'),'--evaluation_split','dev','--out_dir',str(d))
  run('scripts/journal_v2/run_detection.py','--input',a.input,'--splits',str(d/'split_assignments.csv'),'--selected_threshold',str(d/'selected_threshold.json'),'--evaluation_split','test','--out_dir',str(d))
  run('scripts/journal_v2/run_correction_gate_sweep.py','--predictions',str(d/'dev_predictions.jsonl'),'--split','dev','--out_dir',str(d))
  run('scripts/journal_v2/run_correction.py','--predictions',str(d/'predictions.jsonl'),'--selected_gate',str(d/'selected_correction_gate.json'),'--enable_abstention','--out_dir',str(d))
  m=json.loads((d/'metrics.json').read_text())['overall'];c=json.loads((d/'correction_metrics.json').read_text())['overall']
  rows.append({'seed':seed,'threshold':json.loads((d/'selected_threshold.json').read_text())['threshold'],**{k:m[k] for k in ['macro_f1','balanced_accuracy','specificity','false_positive_rate']},**{k:c[k] for k in ['regression_rate','correction_accuracy','correction_coverage']}})
frame=pd.DataFrame(rows);frame.to_csv(out/'multiseed_summary.csv',index=False);stats=frame.drop(columns='seed').agg(['mean','std']).to_dict();write_json(out/'multiseed_summary.json',{'runs':rows,'mean_std':stats});print('Multiseed complete')

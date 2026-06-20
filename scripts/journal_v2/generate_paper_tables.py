#!/usr/bin/env python3
"""Export compact manuscript tables from frozen journal-v2 outputs."""
import json
import pandas as pd
from core import path
out=path('results/journal_v2'); metrics=json.loads((out/'metrics.json').read_text()); correction=json.loads((out/'correction_metrics.json').read_text())
rows=[{'method':'tfidf_weighted','scope':'overall',**metrics['overall']}]+[{'method':'tfidf_weighted','scope':k,**v} for k,v in metrics['per_dataset'].items()]
baseline=out/'retrieval_baseline_metrics.json'
if baseline.exists():
 for method, value in json.loads(baseline.read_text()).items():
  if isinstance(value,dict) and 'overall' in value: rows.append({'method':method,'scope':'overall',**value['overall']})
pd.DataFrame(rows).to_csv(out/'table_detection_results.csv',index=False)
pd.DataFrame([{'scope':'overall',**correction['overall']},*({'scope':k,**v} for k,v in correction.get('per_dataset',{}).items())]).to_csv(out/'table_correction_results.csv',index=False)
pd.read_csv(out/'ablation_summary.csv').to_csv(out/'table_ablation_results.csv',index=False)
summary=out/'multiseed_summary.csv'
pd.read_csv(summary).to_csv(out/'table_multiseed_results.csv',index=False) if summary.exists() else pd.DataFrame(columns=['status']).to_csv(out/'table_multiseed_results.csv',index=False)
print('Paper tables written')

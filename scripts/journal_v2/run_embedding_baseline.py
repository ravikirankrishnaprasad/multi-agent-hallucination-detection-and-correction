#!/usr/bin/env python3
"""Dev-tuned optional MiniLM embedding and TF-IDF/embedding hybrid baselines."""
import argparse, json
import numpy as np, pandas as pd
from core import construct_dataset, metric_dict, path, write_json
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--splits',required=True);p.add_argument('--out_dir',default='results/journal_v2');p.add_argument('--model',default='all-MiniLM-L6-v2');p.add_argument('--top_k',type=int,default=5);a=p.parse_args();out=path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
try:
 from sentence_transformers import SentenceTransformer
except Exception as e:
 write_json(out/'embedding_metrics.json',{'status':'skipped','reason':str(e)});write_json(out/'hybrid_metrics.json',{'status':'skipped','reason':str(e)});(out/'embedding_predictions.jsonl').write_text('');(out/'hybrid_predictions.jsonl').write_text('');raise SystemExit(0)
df=construct_dataset(a.input).merge(pd.read_csv(path(a.splits))[['sample_id','split']],on='sample_id')
model=SentenceTransformer(a.model)
def records(frame):
 corpus=(frame.question+' '+frame.ground_truth).tolist(); emb=model.encode(corpus,normalize_embeddings=True,show_progress_bar=False); ans=model.encode(frame.answer.tolist(),normalize_embeddings=True,show_progress_bar=False); q=model.encode(frame.question.tolist(),normalize_embeddings=True,show_progress_bar=False); rows=[]
 for i,row in frame.reset_index(drop=True).iterrows():
  qs=q[i]@emb.T;qs[i]=-1; ids=np.argsort(qs)[::-1][:a.top_k]; support=float(max(ans[i]@emb[ids].T)); retrieval=float(qs[ids[0]]); rows.append({**row.to_dict(),'embedding_score':support,'retrieval_score':retrieval,'support_score':support,'best_evidence':frame.iloc[int(ids[np.argmax(ans[i]@emb[ids].T)])].ground_truth})
 return rows
dev=records(df[df.split=='dev'].reset_index(drop=True)); test=records(df[df.split=='test'].reset_index(drop=True)); thresholds=np.arange(.1,.91,.05); best=max(thresholds,key=lambda t:metric_dict([r['label'] for r in dev],[int(r['embedding_score']<t) for r in dev])['macro_f1'])
for r in test:r['predicted_label']=int(r['embedding_score']<best);r['threshold']=float(best);r['split']='test'
m=metric_dict([r['label'] for r in test],[r['predicted_label'] for r in test]);write_json(out/'embedding_metrics.json',{'status':'complete','model':a.model,'threshold_selected_on_dev':float(best),'test_metrics':m})
with (out/'embedding_predictions.jsonl').open('w') as f:
 for r in test:f.write(json.dumps(r,ensure_ascii=False)+'\n')
write_json(out/'hybrid_metrics.json',{'status':'skipped','reason':'Hybrid requires aligned TF-IDF score caching; embedding baseline completed.'});(out/'hybrid_predictions.jsonl').write_text('');print('Embedding complete')

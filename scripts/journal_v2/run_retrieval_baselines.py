#!/usr/bin/env python3
"""Evaluate TF-IDF and optional BM25/embedding retrieval baselines on test data."""
import argparse
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from core import construct_dataset, metric_dict, path, per_dataset_metrics, score_records, write_json

parser=argparse.ArgumentParser(); parser.add_argument("--input",required=True); parser.add_argument("--splits",required=True); parser.add_argument("--selected_threshold",required=True); parser.add_argument("--out_dir",default="results/journal_v2"); parser.add_argument("--top_k",type=int,default=5); parser.add_argument("--alpha",type=float,default=.7); parser.add_argument("--enable_embedding",action="store_true"); args=parser.parse_args()
threshold=json.loads(path(args.selected_threshold).read_text())["threshold"]
df=construct_dataset(args.input).merge(pd.read_csv(path(args.splits))[["sample_id","split"]],on="sample_id"); test=df[df.split=="test"].reset_index(drop=True)

def summarize(records):
    for r in records:r["predicted_label"]=int(r["support_score"]<threshold)
    return {"overall":metric_dict([int(r["label"]) for r in records],[r["predicted_label"] for r in records]),"per_dataset":per_dataset_metrics(records)}

results={"tfidf_weighted":summarize(score_records(test,args.top_k,args.alpha,"weighted")),"config":vars(args),"threshold":threshold}
if importlib.util.find_spec("rank_bm25"):
    from rank_bm25 import BM25Okapi
    corpus=(test.question+" "+test.ground_truth).tolist(); bm25=BM25Okapi([x.lower().split() for x in corpus]); vectorizer=TfidfVectorizer(stop_words="english"); matrix=vectorizer.fit_transform(corpus); records=[]
    for i,row in test.iterrows():
        scores=bm25.get_scores(row.question.lower().split()); scores[i]=-np.inf; ids=np.argsort(scores)[::-1][:args.top_k]; raw=scores[ids]; retrieval=float(raw[0]/(raw[0]+1.0)) if len(raw) and np.isfinite(raw[0]) else 0.; answer=float(max(linear_kernel(vectorizer.transform([row.answer]),matrix[ids]).ravel())) if len(ids) else 0.; best=int(ids[np.argmax(linear_kernel(vectorizer.transform([row.answer]),matrix[ids]).ravel())]) if len(ids) else i; records.append({**row.to_dict(),"support_score":args.alpha*answer+(1-args.alpha)*retrieval,"retrieval_score":retrieval,"answer_support":answer,"best_evidence":test.iloc[best].ground_truth})
    results["bm25_weighted"]=summarize(records)
else: results["bm25_weighted"]={"skipped":"rank-bm25 is not installed"}
if args.enable_embedding:
    results["embedding_weighted"]={"skipped":"sentence-transformers is not installed; install it explicitly to enable embeddings."} if not importlib.util.find_spec("sentence_transformers") else {"skipped":"Embedding evaluation is intentionally optional; add a model selection flag before reporting this baseline."}
out=path(args.out_dir);out.mkdir(parents=True,exist_ok=True);write_json(out/"retrieval_baseline_metrics.json",results);print("Wrote retrieval baseline metrics")

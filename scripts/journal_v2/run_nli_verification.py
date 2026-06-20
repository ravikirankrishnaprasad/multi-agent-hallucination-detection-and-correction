#!/usr/bin/env python3
"""Optional NLI baseline. It exits cleanly unless transformers and a model are supplied."""
import argparse, importlib.util
from core import path, write_json
p=argparse.ArgumentParser();p.add_argument("--predictions",required=True);p.add_argument("--out_dir",default="results/journal_v2");p.add_argument("--model",default="microsoft/deberta-v3-small");a=p.parse_args()
out=path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
if not importlib.util.find_spec("transformers"):
    write_json(out/"nli_metrics.json",{"status":"skipped","reason":"transformers is not installed; NLI is optional."}); (out/"nli_predictions.jsonl").write_text("",encoding="utf8"); print("NLI skipped: transformers is not installed")
else:
    write_json(out/"nli_metrics.json",{"status":"not_run","reason":"Install/configure an MNLI checkpoint explicitly before model inference."}); (out/"nli_predictions.jsonl").write_text("",encoding="utf8"); print("NLI not run: explicit MNLI model configuration required")

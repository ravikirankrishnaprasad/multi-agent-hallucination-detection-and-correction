"""Shared utilities for the journal-v2 modular retrieval-grounded pipeline."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ["sample_id", "dataset", "question", "answer", "ground_truth", "label", "category", "difficulty", "source"]


def text(value: Any) -> str:
    return " ".join(str(value if value is not None else "").replace("\n", " ").replace("\r", " ").split())


def path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read_jsonl(filename: Path) -> list[dict[str, Any]]:
    with filename.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(filename: Path, value: Any) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_incorrect(value: Any) -> list[str]:
    s = text(value)
    try:
        found = ast.literal_eval(s)
        if isinstance(found, list):
            return [text(x) for x in found if text(x)]
    except (ValueError, SyntaxError):
        pass
    return [x.strip() for x in re.split(r"\s*;\s*", s) if x.strip()]


def construct_dataset(input_path: str, limit: int = 0, limit_medhallu: int = 0, limit_truthfulqa: int = 0) -> pd.DataFrame:
    """Create within-dataset factual/hallucinated pairs from the two retained sources.

    MedHallu contributes its provided hallucinated answer plus ground truth as a factual
    counterpart. TruthfulQA contributes Best Answer plus its supplied incorrect answer(s).
    This is explicit derived-label construction, not an assertion that the raw sources
    intrinsically contain both labels.
    """
    raw = pd.read_csv(path(input_path)).fillna("")
    if "dataset" not in raw or "question" not in raw:
        raise ValueError("Input must be the v1 unified CSV with at least dataset and question columns.")
    rows: list[dict[str, Any]] = []
    for ix, r in raw.iterrows():
        dataset = text(r.get("dataset", "")).lower()
        if dataset not in {"medhallu", "truthfulqa"}:
            continue
        base = text(r.get("sample_id", "")) or f"{dataset}-{ix}"
        common = {"dataset": dataset, "question": text(r.get("question", "")), "ground_truth": text(r.get("ground_truth", r.get("best_answer", ""))),
                  "category": text(r.get("category", r.get("hallucination_category", ""))), "difficulty": text(r.get("difficulty", "")), "source": text(r.get("source", dataset))}
        if not common["question"] or not common["ground_truth"]:
            continue
        if dataset == "medhallu":
            bad = text(r.get("answer", ""))
            if bad:
                rows.append({**common, "sample_id": base + "::hallucinated", "answer": bad, "label": 1, "derived_from": "provided_hallucinated_answer"})
            rows.append({**common, "sample_id": base + "::factual", "answer": common["ground_truth"], "label": 0, "derived_from": "ground_truth_counterpart"})
        else:
            good = text(r.get("answer", r.get("best_answer", "")))
            if good:
                rows.append({**common, "sample_id": base + "::factual", "answer": good, "label": 0, "derived_from": "best_answer"})
            for j, bad in enumerate(parse_incorrect(r.get("incorrect_answers", ""))[:1]):
                rows.append({**common, "sample_id": f"{base}::hallucinated-{j}", "answer": bad, "label": 1, "derived_from": "supplied_incorrect_answer"})
    df = pd.DataFrame(rows, columns=REQUIRED + ["derived_from"])
    for dataset, cap in (("medhallu", limit_medhallu), ("truthfulqa", limit_truthfulqa)):
        if cap > 0:
            df = pd.concat([df[df.dataset != dataset], df[df.dataset == dataset].head(cap)], ignore_index=True)
    if limit > 0:
        # A smoke-test limit should remain useful for the two-dataset protocol.
        # The v1 CSV is ordered by dataset, so head(limit) would otherwise omit
        # TruthfulQA entirely.
        datasets = list(df["dataset"].drop_duplicates())
        per_dataset = max(1, limit // len(datasets))
        sampled = [df[df["dataset"] == dataset].head(per_dataset) for dataset in datasets]
        remainder = limit - sum(len(part) for part in sampled)
        if remainder > 0:
            used = set(pd.concat(sampled)["sample_id"])
            sampled.append(df[~df["sample_id"].isin(used)].head(remainder))
        df = pd.concat(sampled, ignore_index=True).head(limit).copy()
    if df.empty:
        raise ValueError("No usable v2 examples were constructed from the input.")
    return df.reset_index(drop=True)


def make_splits(df: pd.DataFrame, seed: int, dev_size: float, test_size: float, stratified: bool) -> pd.DataFrame:
    if not 0 <= dev_size < 1 or not 0 <= test_size < 1 or dev_size + test_size >= 1:
        raise ValueError("dev_size and test_size must be >= 0 and sum to less than 1.")
    ids = df.sample_id.to_numpy()
    # Question grouping is stricter than source-id grouping: it keeps derived
    # siblings together and also prevents exact repeated questions from leaking
    # across partitions under distinct source identifiers.
    groups = df.question.map(lambda value: text(value).lower()).to_numpy()
    # Derived factual/hallucinated siblings must never appear in different splits.
    # The requested 60/20/20 protocol maps exactly to three folds of a 5-fold
    # stratified group split.
    if stratified and abs(dev_size - .2) < 1e-9 and abs(test_size - .2) < 1e-9 and len(set(groups)) >= 5:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(splitter.split(df, df.label, groups))
        test_ids = ids[folds[0][1]]
        dev_ids = ids[folds[1][1]]
        train_ids = ids[np.concatenate([folds[i][1] for i in range(2, 5)])]
        mapping = {x: "train" for x in train_ids} | {x: "dev" for x in dev_ids} | {x: "test" for x in test_ids}
        out = df[["sample_id", "dataset", "label"]].copy()
        out["original_sample_id"] = df.sample_id.map(lambda value: str(value).rsplit("::", 1)[0])
        out["split"] = out.sample_id.map(mapping)
        return out
    labels = df.label.to_numpy()
    strat = labels if stratified and len(np.unique(labels)) > 1 and min(np.bincount(labels)) >= 2 else None
    train_ids, hold_ids = train_test_split(ids, test_size=dev_size + test_size, random_state=seed, stratify=strat)
    if len(hold_ids) and dev_size and test_size:
        hold_labels = df.set_index("sample_id").loc[hold_ids, "label"].to_numpy()
        hold_strat = hold_labels if stratified and len(np.unique(hold_labels)) > 1 and min(np.bincount(hold_labels)) >= 2 else None
        dev_ids, test_ids = train_test_split(hold_ids, test_size=test_size / (dev_size + test_size), random_state=seed, stratify=hold_strat)
    elif dev_size:
        dev_ids, test_ids = hold_ids, []
    else:
        dev_ids, test_ids = [], hold_ids
    mapping = {x: "train" for x in train_ids} | {x: "dev" for x in dev_ids} | {x: "test" for x in test_ids}
    out = df[["sample_id", "dataset", "label"]].copy()
    out["original_sample_id"] = df.sample_id.map(lambda value: str(value).rsplit("::", 1)[0])
    out["split"] = out.sample_id.map(mapping)
    return out


def summary(df: pd.DataFrame, splits: pd.DataFrame) -> dict[str, Any]:
    joined = df.merge(splits[["sample_id", "split"]], on="sample_id")
    counts = joined.groupby(["dataset", "split", "label"]).size().rename("count").reset_index().to_dict("records")
    warnings = []
    for (dataset, split), group in joined.groupby(["dataset", "split"]):
        if group.label.nunique() < 2:
            warnings.append(f"{dataset}/{split} contains one class only; binary metrics are not independently interpretable.")
    return {"schema": REQUIRED, "n_samples": int(len(joined)), "construction": "Within-dataset derived factual/hallucinated examples; see derived_from in the constructed data.", "label_counts_by_dataset_split": counts, "warnings": warnings}


def metric_dict(y: list[int], p: list[int]) -> dict[str, Any]:
    if not y:
        return {k: 0.0 for k in ("precision", "recall", "f1", "macro_f1", "accuracy", "specificity", "balanced_accuracy", "false_positive_rate")} | {"confusion_matrix": [[0,0],[0,0]], "n": 0, "label_counts": {"0": 0, "1": 0}}
    precision, recall, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    macro = precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2]
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "macro_f1": float(macro), "accuracy": float((tp+tn)/len(y)), "specificity": float(specificity), "balanced_accuracy": float((recall+specificity)/2), "false_positive_rate": float(fp/(fp+tn)) if fp+tn else 0.0, "confusion_matrix": [[int(tn),int(fp)],[int(fn),int(tp)]], "n": len(y), "label_counts": {"0": int(y.count(0)), "1": int(y.count(1))}}


def per_dataset_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    if frame.empty: return {}
    return {str(d): metric_dict(g.label.astype(int).tolist(), g.predicted_label.astype(int).tolist()) for d, g in frame.groupby("dataset")}


def score_records(df: pd.DataFrame, top_k: int, alpha: float, method: str = "weighted") -> list[dict[str, Any]]:
    corpus = (df.question + " " + df.ground_truth).tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    result = []
    for i, row in df.reset_index(drop=True).iterrows():
        q_scores = linear_kernel(vectorizer.transform([row.question]), matrix).ravel()
        q_scores[i] = -1 # prevent a constructed sample retrieving itself
        ids = np.argsort(q_scores)[::-1][:max(1, min(top_k, len(df)-1))]
        answer_scores = linear_kernel(vectorizer.transform([row.answer]), matrix[ids]).ravel() if len(ids) else np.array([0.0])
        retrieval = float(max(q_scores[ids])) if len(ids) else 0.0
        answer_support = float(max(answer_scores)) if len(answer_scores) else 0.0
        if method == "retrieve_only": support = retrieval
        elif method == "answer_evidence_only": support = answer_support
        elif method == "query_evidence_only": support = retrieval
        else: support = alpha * answer_support + (1-alpha) * retrieval
        best = int(ids[np.argmax(answer_scores)]) if len(ids) else i
        result.append({**row.to_dict(), "support_score": float(support), "retrieval_score": retrieval, "answer_support": answer_support, "best_evidence": text(df.iloc[best].ground_truth), "evidence_snippets": [text(df.iloc[j].ground_truth)[:450] for j in ids]})
    return result

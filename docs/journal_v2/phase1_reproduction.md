# Phase 1 Reproduction (journal_v2)

This guide reproduces the journal_v2 **Phase 1** dataset foundation: a balanced
unified dataset (both labels in both source datasets), a grouped train/dev/test
split with no question leakage, an audit, and a validation pass.

Phase 1 implements **only** dataset reconstruction (C4) and the terminology
reframe (C3). It does not implement semantic retrieval, NLI verification,
correction gating, confidence intervals, human evaluation, or any manuscript
rewriting.

## 0. Prerequisites

- Python 3.10+
- Dependencies: only `pandas` is required for Phase 1 (the heavier
  `requirements.txt` is for later phases).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas
```

## 1. Place the raw datasets

The raw benchmark files are **not** committed to the repository. Download them
and place them under `data/raw/`:

```
data/raw/medhallu_data.csv      # MedHallu (columns include: Question, Knowledge,
                                #   Ground Truth, Hallucinated Answer,
                                #   Difficulty Level, Category of Hallucination)
data/raw/TruthfulQA.csv         # TruthfulQA v0 (columns include: Type, Category,
                                #   Question, Best Answer, Correct Answers,
                                #   Incorrect Answers, Source)
```

Sources:
- MedHallu: https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu
- TruthfulQA: https://github.com/sylinrl/TruthfulQA/blob/main/data/v0/TruthfulQA.csv

The build script matches column names case-insensitively and tolerates common
variants. If your export uses different names, see "Assumptions" at the bottom.

## 2. Build the balanced dataset

```bash
python scripts/journal_v2/build_balanced_dataset.py \
  --medhallu_path data/raw/medhallu_data.csv \
  --truthfulqa_path data/raw/TruthfulQA.csv
# -> results/journal_v2/data/balanced_dataset.csv
```

## 3. Create the grouped split

```bash
python scripts/journal_v2/create_grouped_split.py --seed 42
# -> results/journal_v2/data/balanced_dataset_with_splits.csv
```

The split is by `normalized_question` (70% train / 15% dev / 15% test), so the
factual and hallucinated answers for a question always stay in the same split.

## 4. Run the dataset audit

```bash
python scripts/journal_v2/audit_dataset.py
# -> results/journal_v2/audit/dataset_audit.md
# -> results/journal_v2/audit/class_distribution_by_dataset.csv
# -> results/journal_v2/audit/class_distribution_by_split.csv
# -> results/journal_v2/audit/answer_type_distribution.csv
# -> results/journal_v2/audit/split_distribution.csv
# -> results/journal_v2/audit/leakage_check.md
```

## 5. Validate the outputs

```bash
python scripts/journal_v2/validate_phase1_outputs.py
# prints PASS/FAIL per check; exit code 0 = all pass
```

## 6. Inspect the audit outputs

```bash
cat results/journal_v2/audit/dataset_audit.md
cat results/journal_v2/audit/leakage_check.md
column -s, -t results/journal_v2/audit/class_distribution_by_dataset.csv
column -s, -t results/journal_v2/audit/class_distribution_by_split.csv
```

Key things to confirm:
- "Both datasets contain both labels?" = **PASS**
- "normalized_question overlap across splits" = **PASS (0)**
- Class counts are present for `label_0` and `label_1` in both `medhallu` and
  `truthfulqa`.

## Assumptions about column names

The build script (`build_balanced_dataset.py`) resolves columns
case-insensitively using these candidate lists:

- MedHallu question: `Question`
- MedHallu evidence: `Knowledge`, `Context`
- MedHallu factual answer (label 0): `Ground Truth`, `Correct Answer`
- MedHallu hallucinated answer (label 1): `Hallucinated Answer`
- MedHallu category: `Category of Hallucination`, `hallucination_category`
- TruthfulQA question: `Question`
- TruthfulQA factual (label 0): `Best Answer` (+ `Correct Answers`, '; '-separated)
- TruthfulQA hallucinated (label 1): `Incorrect Answers` ('; '-separated)

If your MedHallu export (e.g. the Hugging Face parquet) uses different field
names, either rename the columns to the above or extend the candidate lists in
`pick_column(...)`. Any such change should be noted in the commit message so it
stays reviewer-defensible.

# Leakage Check (journal_v2 Phase 1)

- Source file: `results/journal_v2/data/balanced_dataset_with_splits.csv`
- Total rows: **24208**
- Unique normalized questions: **9816**

## 1. normalized_question overlap across train/dev/test
**PASS** — no normalized_question appears in more than one split.

## 2. Examples per normalized question
- min: 2, max: 24, mean: 2.47

## 3. Normalized questions spanning multiple original ids
These are normalized questions that map to more than one source row (potential near-duplicate / cross-source collision). Grouped splitting keeps them in a single split, so they do not cause cross-split leakage.
- count: **1**
  - `does molecular characterisation and antimicrobial resistance patterns of escherichia coli isolate from goats slaughtered` -> 2 ids

## 4. Test rows whose answer text is contained in their own stored evidence
Informational only. In Phase 1 no retrieval is performed. The stored `evidence_text` is paired reference material; later retrieval indices must be split-aware so a test answer is never scored against its own originating evidence (see docs/journal_v2/retrieval_leakage_control.md).
- test rows: **3593**, answer-in-own-evidence: **401**

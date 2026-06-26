# results/journal_v2

Generated artifacts from the journal_v2 Phase 1 pipeline.

```
data/   balanced_dataset.csv, balanced_dataset_with_splits.csv   (gitignored)
audit/  dataset_audit.md, leakage_check.md, *_distribution.csv   (committed)
```

## Provenance and version control

- `data/*.csv` are **derived** from `data/raw/` (MedHallu, TruthfulQA). Because
  the raw data is gitignored for size/licensing reasons, the derived dataset
  files are also gitignored (see `data/.gitignore`). Regenerate them with the
  Phase 1 scripts; they are deterministic given the same raw inputs and seed.
- `audit/*` are aggregate statistics (class/answer-type/split distributions and
  leakage checks). They contain no raw passages and are committed as the
  reviewer-facing evidence that the reconstructed dataset is balanced and split
  without question leakage.

Reproduce everything with `docs/journal_v2/phase1_reproduction.md`.

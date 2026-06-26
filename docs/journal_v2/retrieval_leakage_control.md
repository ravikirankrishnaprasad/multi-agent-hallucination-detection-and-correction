# Retrieval Leakage Control (journal_v2)

This note records the leakage problem in the rejected v1 pipeline and the design
constraints that journal_v2 retrieval must satisfy in later phases. **No
retrieval is implemented in Phase 1** — this is a design contract for Phase 2+.

## The problem in v1

In v1 the retrieval corpus was built from the *same rows* that were being
evaluated. Specifically:

- MedHallu evidence was the per-row `Knowledge` field of each evaluated item.
- TruthfulQA evidence was the per-row correct answers of each evaluated item.

As a result, when the verifier scored an answer for a given question, the
evidence pool contained that question's own authoritative passage. This makes
retrieval close to an oracle: the "evidence" is guaranteed to be on-topic and to
contain the ground truth. Detection scores measured under this setup are
optimistic and will not transfer to a realistic deployment where the supporting
passage for a specific claim is not handed to the system.

## What v2 must guarantee

1. **An answer must never be scored against its own originating evidence.**
   A test item's paired ground-truth/hallucinated passage (same `original_id`)
   must be excluded from that item's retrieval candidates.
2. **Retrieval must be split-aware.** The index available at evaluation time must
   not contain documents derived from the dev/test items being evaluated.
3. **The same `normalized_question` must not leak across the index/eval boundary.**
   Because Phase 1 already splits by `normalized_question`, the retrieval index
   can be constructed from train-split material only and remain consistent with
   the split.

## Proposed safe options (to choose from in Phase 2)

**a. Train-only evidence index.**
Build the retrieval index exclusively from train-split evidence (and any external
corpus). Dev/test answers are scored only against this index, so a test item can
never retrieve its own row. Simple and fully reproducible; the trade-off is that
evidence for some test questions may be absent, which is itself a realistic and
reportable condition.

**b. External corpus evidence index.**
Use an independent knowledge source (e.g. a domain corpus for the medical items)
rather than the benchmark's own passages. This is the most realistic setting and
most clearly separates evidence from labels, at the cost of additional data
engineering and corpus-quality controls.

**c. Leave-one-question-out retrieval within a dataset.**
Keep the full per-dataset evidence pool but, at query time, exclude every
document sharing the query item's `original_id` (and `normalized_question`). This
preserves coverage while removing self-evidence. It is more complex to implement
and audit than option (a).

## Auditing

`scripts/journal_v2/audit_dataset.py` already computes an informational flag for
how many test rows have their answer text contained in their own stored
`evidence_text`. Once a retrieval index exists, Phase 2 must add an explicit
check that no retrieved candidate for a test item shares that item's
`original_id` or `normalized_question`.

## Phase 1 status

Implemented: balanced dataset, grouped split by `normalized_question`, and the
self-evidence flag. Not implemented (deferred to Phase 2): the retrieval index
itself and the split-aware retrieval constraint above.

# README note for journal_v2

The repository's `README.md` currently describes the v1 dissertation work using
"multi-agent" terminology. Phase 1 does **not** rewrite that README wholesale and
does **not** rename the repository. Instead, add the following note near the top
of `README.md` (immediately after the title/badges) so readers arriving from the
dissertation or preprint understand the reframing.

---

> **Note on versions and framing.**
> This repository began as an LJMU MSc AI/ML dissertation and an earlier
> manuscript that used "multi-agent" terminology. That earlier work is preserved
> as the historical **v1** baseline (git tag `v1-ljmu-dissertation`; original
> scripts under `scripts/`).
>
> The **journal_v2** rebuild (`scripts/journal_v2/`, `docs/journal_v2/`,
> `results/journal_v2/`) reframes the implementation more accurately as a
> **lightweight modular retrieval-grounded hallucination detection and
> conservative correction pipeline** — a sequence of deterministic processing
> modules, not independent reasoning agents. Phase 1 of the rebuild reconstructs
> the dataset so that both MedHallu and TruthfulQA contribute both factual
> (label 0) and hallucinated (label 1) examples, removing the v1 confound in
> which dataset identity was tied to the label. See
> `docs/journal_v2/terminology_reframe.md` and
> `REVIEWER_REJECTION_ACTION_PLAN.md`.

---

This note file exists so the reframing is captured in the repository even before
the README is edited. When you edit `README.md`, paste the block above and you
may delete this file or keep it as a record.

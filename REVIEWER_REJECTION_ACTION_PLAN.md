# Reviewer Rejection Action Plan — v2 Rebuild

**Manuscript (v1, rejected):** *A Lightweight Retrieval-Grounded Framework for Hallucination Detection and Correction in Large Language Models*
**Venue (rejected):** Discover Artificial Intelligence (Springer Nature), Submission ID 8767a5e7-106e-407c-8555-c66794febe15
**Editor:** Fahim Sufi — decision: **Reject**, 8 June 2026
**Origin:** LJMU MSc AI/ML dissertation → manuscript conversion → journal submission → rejected
**Code reviewed:** `multi-agent-hallucination-detection-and-correction` (`run_project.py`, `build_retrieval_index.py`, `stage3_verify.py`, `stage4_correct.py`, `stage5_run_experiments.py`)

## Purpose and scope

This document converts the editor's rejection into a concrete, prioritised engineering and writing roadmap for a **v2 rebuild** — not a light edit of v1. It is the controlling plan for the project. Per project rules: code is improved *before* the paper is rewritten, the system is **not** described as "multi-agent" unless the implementation genuinely supports independent reasoning agents, and the work is repositioned as a *lightweight modular retrieval-grounded hallucination detection and conservative correction pipeline*. No new metrics, datasets, citations, or results are asserted here; target numbers are to be produced by the experiments listed, not assumed.

## What the code review revealed (root causes behind the editor's comments)

These findings are the factual basis for the plan and should be quoted back to reviewers as the diagnosis v2 fixes:

1. **Label is perfectly confounded with dataset identity.** In `run_project.py`, every MedHallu row uses the *Hallucinated Answer* column with `label = 1`, and every TruthfulQA row uses the *Best Answer* column with `label = 0`. The unified set therefore contains 9,000 all-positive MedHallu rows and 817 all-negative TruthfulQA rows. A classifier can separate the two classes by recognising which dataset a row came from rather than by detecting hallucination. This is the single deepest flaw and underlies the editor's comments on artificial dataset construction, weak TruthfulQA performance, and immature validation.
2. **Both datasets natively support balanced positive/negative pairs, but half of each was discarded.** MedHallu carries both *Ground Truth* and *Hallucinated Answer*; TruthfulQA carries both *Correct Answers* and *Incorrect Answers*. v1 used only the hallucinated side of MedHallu and only the correct side of TruthfulQA.
3. **Retrieval evidence is drawn from the same rows being evaluated.** In `build_retrieval_index.py`, the MedHallu evidence corpus is the per-row `knowledge` field and the TruthfulQA corpus is the per-row correct answers. Verification compares each answer against evidence that includes its own row's source passage, giving a near-oracle, circular retrieval setting that inflates detection scores and will not transfer to a realistic open-corpus deployment.
4. **TruthfulQA's "weak performance" is an artefact, not a measurement.** Because TruthfulQA contributes no positive (hallucinated) examples, positive-class precision/recall/F1 are mathematically 0/0/0 (Table 5 in v1). The benchmark was configured so it can only ever measure specificity.
5. **Correction is verbatim evidence substitution with no gating.** In `stage4_correct.py` / `stage5_run_experiments.py`, a "corrected" answer is the top retrieved evidence snippet copied verbatim whenever the verifier predicts hallucinated. Any change to a (factual) TruthfulQA answer is by definition counted as regression, which is the entire source of the reported 0.2485 regression. MedHallu "fix" success is judged by Jaccard/token-overlap ≥ 0.50 against ground truth — a weak lexical proxy, not a factuality judgement.
6. **The "agents" are deterministic processing stages.** Classes named `VerificationAgent` and `CorrectionAgent` perform fixed TF-IDF arithmetic with no independent reasoning, no model calls, and no inter-agent communication. The repository README still says "multi-agent"; the manuscript inconsistently retains agent language.
7. **The detection threshold is selected on the same data it is evaluated on.** The sweep in `stage3_verify.py` runs over the full combined set and τ = 0.55 is chosen from it; there is no train/validation/test split, no cross-validation, no confidence intervals, and no significance testing.

---

## How to read the tables

Each editor criticism below is a self-contained worksheet with eight fields: the rejection point, why it was a weakness in v1, required code changes, required new experiments, tables/figures to regenerate, manuscript sections to rewrite, the reviewer-facing explanation, and priority. Priorities are sequenced so that **dataset reconstruction (C4) and leakage removal are prerequisites** for almost everything else — do those first.

---

## C1 — Limited novelty

| Dimension | Detail |
|---|---|
| **1. Rejection point** | "The novelty of this paper is somewhat limited." |
| **2. Why this was a weakness in v1** | v1's contribution is a weighted combination of two TF-IDF cosine similarities plus a threshold — each component is decades-old and individually unremarkable. The paper claimed novelty for the *pipeline* but offered no comparison showing it beats an existing simple baseline, and no clearly articulated, defensible "what is new here" beyond "we combined retrieval, verification and correction." |
| **3. What needs to change in the code** | Do not manufacture novelty in code. Instead make the *defensible* contribution measurable: (a) implement at least one genuinely stronger verification signal (semantic embedding similarity and/or a lightweight NLI/entailment check) so the contribution becomes "a calibrated, leakage-controlled, conservative-correction pipeline benchmarked against both trivial and strong baselines," not "TF-IDF + threshold"; (b) implement a **confidence-gated conservative correction** policy (only correct when evidence support is high AND the answer is confidently unsupported) as the actual novel, defensible mechanism. |
| **4. What new experiments are required** | Benchmark the proposed pipeline against (i) a majority-class baseline, (ii) retrieval-only, (iii) TF-IDF verification (the v1 method, now the *weak* baseline), and (iv) at least one published reference method that can be run on the reconstructed data (e.g. an embedding-similarity or SelfCheckGPT-style consistency detector). Novelty is demonstrated by the *gap* between the conservative-gated pipeline and these baselines on a leakage-free split. |
| **5. What tables/figures must be regenerated** | New "positioning vs. baselines" results table (method × detection metrics) replacing the current self-comparison-only Table 6; a contribution-delta figure showing each added component's marginal effect (ablation bar chart, regenerated from `detection_f1_by_pipeline.png`). |
| **6. What manuscript sections must be rewritten** | Reframe Contributions list (Sec. 1) and Research Gap/Positioning (Sec. 2.6) to claim only what is implemented and measured: a reproducible, leakage-controlled, conservatively-gated lightweight pipeline with calibrated thresholds. Remove any implication that TF-IDF itself is the contribution. |
| **7. How to explain the improvement to a future reviewer** | "We agree the original components are individually simple. v2 reframes the contribution as an *engineering-for-reliability* result: a leakage-controlled evaluation protocol plus a confidence-gated conservative-correction policy that, unlike prior simple detectors, avoids degrading correct answers. We now quantify the contribution against trivial *and* strong baselines rather than against ourselves." |
| **8. Priority** | **Medium** (depends on C2/C4 being done first; novelty is reframed, not invented). |

---

## C2 — TF-IDF + cosine + thresholding is methodologically simple

| Dimension | Detail |
|---|---|
| **1. Rejection point** | "TF-IDF retrieval plus cosine similarity plus thresholding is methodologically simple." |
| **2. Why this was a weakness in v1** | The entire detection signal is `score = α·cos(answer, evidence) + (1−α)·cos(query, evidence)` with a hard threshold (`stage3_verify.py`). Lexical similarity cannot capture paraphrase, negation, or numerical/clinical contradiction — exactly the cases that matter for hallucination. There is no learned component, no calibration, and no semantic understanding. |
| **3. What needs to change in the code** | Add a semantic retrieval/verification path alongside TF-IDF: (a) sentence-embedding retrieval and cosine (e.g. a small open sentence-transformer) as a configurable retriever in `build_retrieval_index.py`; (b) a lightweight entailment/NLI verifier that scores whether evidence *supports* or *contradicts* the answer, replacing pure lexical overlap as the decision signal; (c) probability calibration on the support score so the threshold is principled, not hand-picked. Keep TF-IDF as an explicit, reportable *efficiency baseline*, not the headline method. |
| **4. What new experiments are required** | Controlled comparison: TF-IDF vs. embedding vs. NLI verifier on the leakage-free reconstructed split, reporting detection metrics *and* compute cost/latency (to preserve the "lightweight" claim honestly). Sensitivity analysis over α and threshold with calibration. |
| **5. What tables/figures must be regenerated** | New verifier-comparison table (signal type × metrics × latency); recompute `threshold_sweep_detection_metrics.png` per verifier; add a calibration/reliability plot. Regenerate Table 4 (config) to list the new components. |
| **6. What manuscript sections must be rewritten** | Methodology (Sec. 3.3–3.4): replace the single TF-IDF formulation with a tiered verifier description (lexical → semantic → entailment) and a calibration subsection. Be explicit that TF-IDF is retained only as a cost reference. |
| **7. How to explain the improvement to a future reviewer** | "v2 keeps a lightweight footprint but no longer relies on lexical similarity alone. Detection now uses semantic and entailment-based verification with calibrated thresholds, and we report the accuracy/cost trade-off explicitly so 'lightweight' is a measured property rather than a synonym for 'simple.'" |
| **8. Priority** | **High.** |

---

## C3 — "Multi-agent" framing is weak (modular stages, not independent agents)

| Dimension | Detail |
|---|---|
| **1. Rejection point** | "The paper presents the system as a 'multi-agent' or 'agent-inspired' framework, but the agents are essentially modular processing stages rather than genuinely independent reasoning agents." |
| **2. Why this was a weakness in v1** | Accurate criticism. `VerificationAgent`/`CorrectionAgent` are deterministic functions over a TF-IDF matrix — no autonomy, no independent reasoning, no model-mediated communication. The title's grounding repo is literally named "multi-agent…", and the manuscript keeps agent language while the abstract already half-retreats to "modular." This reads as overclaiming. |
| **3. What needs to change in the code** | Rename classes and modules to neutral, accurate names (e.g. `Retriever`, `Verifier`, `Corrector`, `PipelineStage`). Remove "agent" from docstrings, the README, and `CITATION.cff`. Optionally rename the repository or add a prominent note that v2 reframes the system as a modular pipeline. Do **not** add fake agent wrappers to justify the term. |
| **4. What new experiments are required** | None specific to terminology. (If, and only if, a genuinely independent reasoning component is later implemented — e.g. an LLM-based verifier and an LLM-based corrector that disagree and reconcile — then agent language could be reconsidered and would need its own ablation. Treat that as **future work**, clearly separated.) |
| **5. What tables/figures must be regenerated** | Regenerate `architecture.png` and the sequence diagram to show a *pipeline* of modular stages, not collaborating agents. Update Fig. 1 caption. |
| **6. What manuscript sections must be rewritten** | Title, abstract, keywords (remove "Multi-Agent Systems"), Sec. 1 and Sec. 3.2 framing. Replace every "agent" with "module/stage." Add one sentence explicitly acknowledging the system is a deterministic modular pipeline, not a multi-agent system. |
| **7. How to explain the improvement to a future reviewer** | "We accept this point fully. v2 removes all multi-agent terminology and describes the system precisely as a modular retrieval-grounded pipeline of deterministic stages. We reserve 'agent' for genuine independent-reasoning extensions, which we list only as future work." |
| **8. Priority** | **High** (cheap, fully within our control, directly answers a named objection; do early). |

---

## C4 — Artificial dataset construction needs justification or rectification

| Dimension | Detail |
|---|---|
| **1. Rejection point** | "…needs significant improvement to justify or rectify inartificial [artificial] dataset construction." |
| **2. Why this was a weakness in v1** | The unified benchmark is constructed so that **dataset = label** (all MedHallu positive, all TruthfulQA negative; see `run_project.py`). Detection metrics are therefore confounded: a model can score well by detecting provenance, not hallucination. Compounding this, retrieval evidence is taken from the same rows (`build_retrieval_index.py`), creating circular near-oracle retrieval. This is the structural root of C5 and C7. |
| **3. What needs to change in the code** | Rebuild dataset construction: (a) for MedHallu, emit **two rows per item** — the *Ground Truth* answer (`label = 0`) and the *Hallucinated Answer* (`label = 1`) — yielding a balanced positive/negative set within the same domain; (b) for TruthfulQA, emit both *Best/Correct Answers* (`label = 0`) and *Incorrect Answers* (`label = 1`); (c) decouple the retrieval corpus from the evaluation rows — build evidence from a held-out knowledge source or a separated split so an answer is never scored against its own originating passage; (d) add a documented train/validation/test split with fixed seeds. |
| **4. What new experiments are required** | Re-run the full detection and correction pipeline on the reconstructed, balanced, leakage-controlled dataset. Report per-dataset *and* combined metrics where each dataset now contains both classes. Add a leakage ablation (with vs. without same-row evidence) to quantify how much v1's scores were inflated by circular retrieval. |
| **5. What tables/figures must be regenerated** | Table 1 (dataset distribution) — now showing both classes per dataset; Table 3 (config); `class_distribution_across_datasets.png`, `positive_cases_by_dataset.png`; every downstream detection/correction table and figure must be recomputed on the new split. |
| **6. What manuscript sections must be rewritten** | Sec. 4.1 (Datasets) — full rewrite documenting balanced construction and the rationale; new "Data construction and leakage control" subsection; Sec. 4.3 (Setup) to describe the split. Add a Limitations note on residual benchmark artificiality. |
| **7. How to explain the improvement to a future reviewer** | "We agree the original split confounded dataset identity with the label. v2 reconstructs both benchmarks to contain matched correct/hallucinated answers per item, removes same-row evidence leakage, and adds a seeded train/val/test split. We additionally report a leakage ablation that isolates how much of the original performance was attributable to circular retrieval." |
| **8. Priority** | **High — do first.** Prerequisite for C1, C2, C5, C6, C7. |

---

## C5 — Weak TruthfulQA performance

| Dimension | Detail |
|---|---|
| **1. Rejection point** | "…weak TruthfulQA performance." |
| **2. Why this was a weakness in v1** | TruthfulQA detection precision/recall/F1 are 0.000 (Table 5) because TruthfulQA contributes **no positive examples** — the 0.000 is an arithmetic artefact of an all-negative subset, not a measured capability. The paper reports it as if it were a performance number and then explains it away, which signals an immature evaluation rather than addressing it. |
| **3. What needs to change in the code** | Fixed primarily by C4: once TruthfulQA includes *Incorrect Answers* as `label = 1`, the subset has both classes and real detection metrics become computable. Add per-class support counts to all metric outputs in `stage3_verify.py` so empty-class cases can never again be silently reported as 0.000. |
| **4. What new experiments are required** | Genuine TruthfulQA detection evaluation on the balanced subset (both correct and incorrect answers), reported alongside MedHallu. Optional: stratify by TruthfulQA category to show where lexical/semantic verification still struggles (adversarial/misconception items), framed honestly. |
| **5. What tables/figures must be regenerated** | Table 5 (final detection) — TruthfulQA row replaced with real two-class metrics; `before_after_hallucination_rate_truthfulqa.png`; any figure currently showing 0.000 TruthfulQA bars. |
| **6. What manuscript sections must be rewritten** | Sec. 5.1 and the TruthfulQA interpretation paragraph — remove the "not directly applicable" framing and report actual results; Error Analysis (Sec. 5.5) updated with real failure cases. |
| **7. How to explain the improvement to a future reviewer** | "The original TruthfulQA scores of zero reflected a single-class subset, not weak detection. v2 evaluates TruthfulQA with both correct and incorrect answers, so the metrics are now genuine. We report results candidly, including categories where semantic verification remains hard." |
| **8. Priority** | **High** (largely resolved by C4; ensure honest reporting). |

---

## C6 — Correction introduces regression

| Dimension | Detail |
|---|---|
| **1. Rejection point** | "…correction regression." |
| **2. Why this was a weakness in v1** | Correction copies the top evidence snippet verbatim whenever the verifier fires (`stage4_correct.py`). With factual TruthfulQA answers, any edit is counted as regression by construction (0.2485). There is no gate preventing the system from "correcting" answers that were already right, and "fix" is judged by token overlap, not factuality. The pipeline can therefore make correct answers worse. |
| **3. What needs to change in the code** | Implement **confidence-gated conservative correction**: only rewrite when (i) the answer is confidently unsupported *and* (ii) high-support evidence exists *and* (iii) the candidate correction is entailed by the evidence; otherwise abstain and keep the original. Replace verbatim snippet substitution with an evidence-constrained rewrite that preserves answer form. Add an explicit "abstain" outcome and track it. Strengthen correction evaluation beyond Jaccard (entailment/semantic check, ideally a small human-rated sample). |
| **4. What new experiments are required** | Correction ablation: ungated (v1 behaviour) vs. gated conservative correction, reporting reduction *and* regression jointly, so the headline becomes "comparable reduction at near-zero regression." Report regression on the now-two-class TruthfulQA. Add a small human/manual factuality audit of a corrected-answer sample to validate the automatic fix metric. |
| **5. What tables/figures must be regenerated** | Table 7 (correction), `regression_rate_by_pipeline.png`, `correction_accuracy_by_pipeline.png`, `positive_hallucination_reduction.png`, `before_after_hallucination_rate_*` — all recomputed with the gated policy and honest regression accounting. |
| **6. What manuscript sections must be rewritten** | Sec. 3.5 (Correction Agent → Correction module) — describe gating and abstention; Sec. 5.4 (Hallucination Reduction) — present reduction/regression trade-off; Discussion — replace the "future work" hand-wave about gating with the implemented result. |
| **7. How to explain the improvement to a future reviewer** | "v1 corrected indiscriminately and so degraded correct answers. v2 makes correction conservative and gated: it abstains unless the evidence both contradicts the answer and supports a specific fix. We report reduction and regression together and validate fixes with an entailment check and a manual audit, demonstrating mitigation without harming correct responses." |
| **8. Priority** | **High.** |

---

## C7 — Experimental validation not mature enough for publication

| Dimension | Detail |
|---|---|
| **1. Rejection point** | "…insufficiently mature experimental validation." |
| **2. Why this was a weakness in v1** | No train/val/test split; the operating threshold τ = 0.55 is chosen on the same combined data it is then evaluated on (`stage3_verify.py` sweep), which is selection-on-test. No cross-validation, no confidence intervals, no significance tests, no comparison to any *executed* external baseline, no human evaluation, and circular retrieval (C4). Collectively this is not yet publishable validation. |
| **3. What needs to change in the code** | Add: seeded train/validation/test splitting; threshold/α selection on validation only, reported on test; k-fold or bootstrap resampling to produce confidence intervals; a statistics module for significance tests between pipelines; runnable wrappers for at least one external baseline; logging of per-class support and compute cost. Make the whole run reproducible via a single seeded entry point and pinned `requirements.txt`. |
| **4. What new experiments are required** | Full re-evaluation under the new protocol: validation-selected thresholds, test-set reporting with bootstrap CIs, significance testing (proposed vs. each baseline), ablations (verifier type, leakage on/off, correction gating on/off), and a small human factuality audit. Report latency/cost to substantiate "lightweight." |
| **5. What tables/figures must be regenerated** | Every results table and figure (Tables 5–7; all `results/figures/*.png`) must be regenerated on the test split with CIs and significance annotations. Add: a protocol/split summary table, a baseline-comparison table with significance markers, and a cost/latency table. |
| **6. What manuscript sections must be rewritten** | Sec. 4 (Experiments) — add Experimental Protocol subsection (splits, selection, statistics); Sec. 5 (Results) — report with CIs and significance throughout; Discussion/Limitations — state remaining threats to validity. |
| **7. How to explain the improvement to a future reviewer** | "v2 adopts a standard validation protocol: hyperparameters are selected on a validation split and reported on a held-out test set, with bootstrap confidence intervals and significance tests against trivial and strong baselines. We removed retrieval leakage, added ablations and a human audit, and report compute cost. The evaluation now supports the claims it makes." |
| **8. Priority** | **High** (integrates the fixes from C1–C6; the validation layer that makes the rebuild credible). |

---

## Suggested execution order

1. **C4 (dataset reconstruction + leakage removal)** and **C3 (terminology/repositioning)** — foundational; everything else depends on C4, and C3 is cheap and directly answers a named objection.
2. **C2 (stronger verifier + calibration)** — builds on the clean dataset.
3. **C6 (gated conservative correction)** — depends on the verifier and clean labels.
4. **C5 (honest two-class TruthfulQA reporting)** — falls out of C4; verify reporting.
5. **C7 (mature validation protocol, CIs, significance, baselines, audit)** — wraps all of the above.
6. **C1 (novelty reframing)** — finalised last, once results quantify the defensible contribution.

## Guardrails for the rebuild (from project rules)

- Do not invent citations, metrics, datasets, or results. Target numbers come from re-run experiments, not assertion.
- Do not call the system "multi-agent." It is a modular retrieval-grounded pipeline.
- Keep v1 frozen as the rejected baseline (git tag `v1-ljmu-dissertation`); build v2 cleanly alongside it.
- Use conservative academic wording; separate implemented work from future work explicitly.
- Every change should be explainable to a reviewer in the terms given in field 7 of each table.

## Open decisions to confirm before coding

- Target resubmission venue (journal vs. conference) — affects scope, baseline expectations, and page limits.
- Whether to introduce an LLM-based verifier/corrector in v2 (raises cost, weakens the "lightweight" claim, but strengthens novelty) or keep v2 strictly lightweight and defer LLM components to future work.
- Extent of human evaluation feasible (sample size, who rates) for the correction audit in C6/C7.

*This is a planning document only. No source code has been modified.*

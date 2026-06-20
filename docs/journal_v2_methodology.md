# Journal v2 methodology

## Framing

Journal v2 is a modular retrieval-grounded verification pipeline, not a claim of autonomous multi-agent reasoning. Retrieval, support scoring, verification, and correction are reproducible processing stages.

The executable implementation is in `scripts/journal_v2/`; v1 dissertation scripts remain unchanged in `scripts/`.

## Dataset protocol

The v1 merged set largely aligned dataset identity with label: MedHallu supplied hallucinated answers and TruthfulQA supplied factual best answers. V2 retains both datasets while deriving paired examples within each source: MedHallu's ground truth is a factual counterpart to its supplied hallucinated answer; TruthfulQA's supplied incorrect answer is a hallucinated counterpart to its best answer. `derived_from` records this provenance. The builder saves assignments and warns whenever a dataset/split has one class.

## Evaluation and correction

Splits are seeded and grouped by normalized question to prevent derived examples or repeated questions leaking across partitions. Thresholds and correction gates are selected only on development data, then applied once to test data. Report precision, recall, F1, macro-F1, accuracy, specificity, balanced accuracy, false-positive rate, a confusion matrix, label counts, and separate dataset metrics. TF-IDF cosine is the required baseline; BM25 and MiniLM embedding retrieval are comparison baselines, with embedding optional. Correction requires a hallucination prediction, low support score, adequate retrieval confidence, and low answer support; otherwise it keeps the answer or records an abstention. Ground-truth overlap is an offline correction-evaluation proxy, never a gating signal.

## Limitations and extension

Derived labels do not replace independent human annotation, and retrieval evidence may be incomplete. A future third independently labelled dataset should be added through the common schema before claiming broad robustness.

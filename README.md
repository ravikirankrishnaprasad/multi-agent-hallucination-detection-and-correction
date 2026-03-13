# Multi-Agent Hallucination Detection and Correction System

This repository contains the implementation accompanying the MSc dissertation
submitted to **Liverpool John Moores University (LJMU)**.

The project investigates methods for **detecting and correcting hallucinations in Large Language Model (LLM) outputs** using a modular multi-agent verification framework.

---

# Overview

Large Language Models often generate **hallucinated responses**, where statements appear plausible but are not supported by factual evidence.

This project proposes a **multi-agent hallucination detection and correction framework** that combines:

* retrieval-based evidence grounding
* similarity-based hallucination verification
* evidence-guided response correction
* modular experiment orchestration

The goal is to improve **factual reliability of LLM-generated responses** using lightweight verification strategies.

---

# Datasets

The system is evaluated using two publicly available hallucination benchmarks:

• **MedHallu** – medical hallucination detection benchmark
• **TruthfulQA** – adversarial truthfulness evaluation dataset

Due to licensing and dataset size constraints, the raw datasets are **not included in this repository**.

Users must download the datasets separately and place them in:

```
data/raw/
```

---

# Project Structure

```
scripts/        Pipeline scripts for preprocessing, verification, correction, and experiments

data/
   raw/         Raw dataset files (not included)
   processed/   Cleaned datasets used in experiments

indices/        TF-IDF retrieval indices (excluded from version control)

results/        Experiment outputs, summary metrics, and figures
```

---

# Experimental Pipeline

The system is implemented as a **multi-stage experimental pipeline**.

Stage 1 – Dataset preprocessing
Stage 2 – Retrieval index construction
Stage 3 – Hallucination verification
Stage 4 – Response correction
Stage 5 – Experiment orchestration
Stage 6 – Result visualization

Key scripts:

```
scripts/run_project.py
scripts/build_retrieval_index.py
scripts/stage3_verify.py
scripts/stage4_correct.py
scripts/stage5_run_experiments.py
scripts/stage6_plots.py
```

---

# Running the Pipeline

Example workflow:

### 1. Preprocess datasets

```
python scripts/run_project.py \
  --medhallu_path data/raw/medhallu_data.csv \
  --truthfulqa_path data/raw/TruthfulQA.csv
```

### 2. Build retrieval index

```
python scripts/build_retrieval_index.py
```

### 3. Run hallucination verification

```
python scripts/stage3_verify.py --threshold 0.30 --top_k 5 --alpha 0.7
```

### 4. Run correction stage

```
python scripts/stage4_correct.py
```

### 5. Run experiment pipeline

```
python scripts/stage5_run_experiments.py --verify_threshold 0.30 --top_k 5 --alpha 0.7
```

### 6. Generate result plots

```
python scripts/stage6_plots.py
```

---

# Outputs

The pipeline produces several output files:

```
results/stage3_metrics.json
results/stage4_metrics.json
results/stage5_summary.csv
results/stage5_summary.json
results/stage5_outputs.jsonl
```

Visualization figures are generated in:

```
results/figures/
```

---

# Key Results

Experimental evaluation shows:

* High precision hallucination detection
* Approximately **32% hallucination recall** on MedHallu
* Approximately **5–6% hallucination reduction** after correction

The results demonstrate that **retrieval-grounded verification can provide lightweight hallucination detection without requiring additional LLM inference**.

---

# Reproducibility

The dissertation implementation is frozen under:

```
Git tag: v1.0-ljmu-dissertation
```

This ensures that the exact code used for the dissertation experiments can be reproduced.

---

# Large Artifacts

Per-sample experiment logs are excluded from version control due to size constraints.

Summary metrics and figures required to reproduce the results are included.

---

# Author

**Ravikiran VK**
MSc Artificial Intelligence & Machine Learning
Liverpool John Moores University

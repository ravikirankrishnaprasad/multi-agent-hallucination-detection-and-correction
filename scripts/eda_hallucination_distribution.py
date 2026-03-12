import pandas as pd
import matplotlib.pyplot as plt

# Load processed datasets
med = pd.read_csv("data/processed/medhallu_cleaned.csv")
truth = pd.read_csv("data/processed/truthfulqa_cleaned.csv")

# Dataset-level label assignment for EDA
# MedHallu entries represent hallucinated answers
med_hallucinated = len(med)

# TruthfulQA "Best Answer" entries represent supported/correct answers
truth_supported = len(truth)

plot_df = pd.DataFrame({
    "Dataset": ["MedHallu", "TruthfulQA"],
    "Hallucinated": [med_hallucinated, 0],
    "Supported": [0, truth_supported]
})

print(plot_df)

ax = plot_df.set_index("Dataset").plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.xlabel("Dataset")
plt.ylabel("Number of Responses")
plt.title("Distribution of Hallucinated vs Supported Responses")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("results/figures/hallucination_distribution.png")
print("Saved: results/figures/hallucination_distribution.png")
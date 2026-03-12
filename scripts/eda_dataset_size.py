import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
med = pd.read_csv("data/processed/medhallu_cleaned.csv")
truth = pd.read_csv("data/processed/truthfulqa_cleaned.csv")

# Dataset sizes
datasets = ["MedHallu", "TruthfulQA"]
sizes = [len(med), len(truth)]

plt.figure(figsize=(7, 5))
bars = plt.bar(datasets, sizes)

plt.xlabel("Dataset")
plt.ylabel("Number of Samples")
plt.title("Dataset Size Comparison")

# Add values on bars
for bar, size in zip(bars, sizes):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(size),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("results/figures/dataset_size_comparison.png")
print("Saved: results/figures/dataset_size_comparison.png")
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
med = pd.read_csv("data/processed/medhallu_cleaned.csv")
truth = pd.read_csv("data/processed/truthfulqa_cleaned.csv")

# Correct columns for response text
med["length"] = med["Hallucinated Answer"].astype(str).apply(lambda x: len(x.split()))
truth["length"] = truth["Best Answer"].astype(str).apply(lambda x: len(x.split()))

# Print averages
print("Average MedHallu response length:", round(med["length"].mean(), 2))
print("Average TruthfulQA response length:", round(truth["length"].mean(), 2))

# Plot
plt.figure(figsize=(8, 5))
plt.hist(med["length"], bins=40, alpha=0.6, label="MedHallu")
plt.hist(truth["length"], bins=40, alpha=0.6, label="TruthfulQA")

plt.xlabel("Response Length (Number of Words)")
plt.ylabel("Frequency")
plt.title("Distribution of Response Lengths Across Datasets")
plt.legend()
plt.tight_layout()

plt.savefig("results/figures/response_length_distribution.png")
print("Saved: results/figures/response_length_distribution.png")
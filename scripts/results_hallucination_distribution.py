import pandas as pd
import matplotlib.pyplot as plt

med = pd.read_csv("data/processed/medhallu_cleaned.csv")

cat_counts = med["Category of Hallucination"].value_counts()

plt.figure(figsize=(10, 5))
cat_counts.plot(kind="bar")

plt.xlabel("Hallucination Category")
plt.ylabel("Number of Samples")
plt.title("Distribution of Hallucination Categories in MedHallu")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("results/figures/medhallu_category_distribution.png")
print("Saved: results/figures/medhallu_category_distribution.png")
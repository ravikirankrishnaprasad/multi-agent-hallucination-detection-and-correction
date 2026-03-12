import matplotlib.pyplot as plt

steps = [
    "Raw Dataset",
    "Data Cleaning",
    "Text Normalization",
    "Tokenization",
    "Structured CSV/JSON",
    "Retrieval Index Ready"
]

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")

x = 5
y_positions = [11, 9, 7, 5, 3, 1]

for step, y in zip(steps, y_positions):
    ax.text(
        x, y, step,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black")
    )

for y1, y2 in zip(y_positions[:-1], y_positions[1:]):
    ax.annotate(
        "",
        xy=(x, y2 + 0.5),
        xytext=(x, y1 - 0.5),
        arrowprops=dict(arrowstyle="->", lw=1.5)
    )

plt.title("Data Preprocessing Pipeline")
plt.tight_layout()
plt.savefig("results/figures/preprocessing_pipeline.png")
print("Saved: results/figures/preprocessing_pipeline.png")
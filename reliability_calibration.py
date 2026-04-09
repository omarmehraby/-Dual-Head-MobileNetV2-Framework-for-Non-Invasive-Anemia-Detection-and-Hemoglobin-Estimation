# --- Fig. 6: Reliability Diagram + Prediction Histogram (Post Calibration Only) ---

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Plot reliability diagram and prediction histogram")
parser.add_argument("--bins_csv", required=True,
                    help="Path to calibration bins CSV (e.g., figures/calibration_bins_test.csv)")
parser.add_argument("--outdir", default=None,
                    help="Directory to save output figure (default: same directory as bins_csv)")
args = parser.parse_args()

outdir = args.outdir if args.outdir else os.path.dirname(os.path.abspath(args.bins_csv))
os.makedirs(outdir, exist_ok=True)

# === Load data ===
df = pd.read_csv(args.bins_csv)
print("Columns:", list(df.columns))

# Expected columns: confidence (mean predicted prob per bin),
# empirical (observed accuracy per bin),
# count (number of samples in that bin)
df = df.sort_values("confidence")

# === Reliability diagram ===
plt.figure(figsize=(12, 5))

# Left pane: expected vs observed accuracy
plt.subplot(1, 2, 1)
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
plt.plot(df["confidence"], df["empirical"], marker="o", color="green", label="Post Calibration")
plt.xlabel("Predicted Probability (Confidence)")
plt.ylabel("Observed Accuracy")
plt.title("Reliability Diagram (Post Calibration)")
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)

# Right pane: histogram of prediction confidences
plt.subplot(1, 2, 2)
bin_width = 0.05
plt.bar(df["confidence"], df["count"], width=bin_width, color="skyblue", edgecolor="k", alpha=0.8)
plt.xlabel("Predicted Probability (Confidence)")
plt.ylabel("Frequency (Count)")
plt.title("Prediction Probability Histogram")
plt.grid(True, linestyle="--", alpha=0.6)

plt.suptitle("Fig. 6 – Reliability Diagram & Prediction Histogram (MobileNetV2, Post Calibration)",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])

# === Save & show ===
output_path = os.path.join(outdir, "Fig6_Reliability_PostCalibration.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Figure saved at: {output_path}")

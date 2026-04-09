import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # for optional AUROC smoothing

parser = argparse.ArgumentParser(description="Plot training/validation loss and AUROC curves from history CSV")
parser.add_argument("--history_csv", required=True,
                    help="Path to training history CSV (e.g., history_mobilenetv2.csv)")
parser.add_argument("--outdir", default=None,
                    help="Directory to save output figures (default: same directory as history_csv)")
parser.add_argument("--model_name", default="MobileNetV2",
                    help="Model name label for plot titles (default: MobileNetV2)")
args = parser.parse_args()

outdir = args.outdir if args.outdir else os.path.dirname(os.path.abspath(args.history_csv))
os.makedirs(outdir, exist_ok=True)

# =====================================================
# Load histories (add more models here if available)
# =====================================================
mobilenet = pd.read_csv(args.history_csv)

# Dictionary structure lets you expand easily to 3 models
histories = {
    args.model_name: mobilenet,
    # "EfficientNet-B0": pd.read_csv("/path/to/efficientnet/history.csv"),
    # "ResNet-50": pd.read_csv("/path/to/resnet/history.csv"),
}

plt.style.use("seaborn-v0_8-whitegrid")

# =====================================================
# FIG. 1 — Training and Validation Loss per Model
# =====================================================
n_models = len(histories)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
if n_models == 1:
    axes = [axes]  # ensure iterable

fig.suptitle("Fig. 1 – Training and Validation Loss per Model", fontsize=14, fontweight="bold")

for ax, (model_name, df) in zip(axes, histories.items()):
    ax.plot(df["epoch"], df["train_loss"], label="Training Loss", color="#1f77b4", marker="o", linewidth=2)
    ax.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="#ff7f0e", marker="s", linewidth=2)
    ax.set_title(model_name, fontsize=12, fontweight="semibold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(outdir, "fig1_loss.png"), dpi=300, bbox_inches="tight")
plt.show()

# =====================================================
# FIG. 2 — Validation AUROC per Epoch (smoothed)
# =====================================================
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
if n_models == 1:
    axes = [axes]

fig.suptitle("Fig. 2 – Validation AUROC per Epoch per Model", fontsize=14, fontweight="bold")

for ax, (model_name, df) in zip(axes, histories.items()):
    if "val_auroc" in df.columns:
        smoothed = gaussian_filter1d(df["val_auroc"], sigma=1)
        ax.plot(df["epoch"], smoothed, label="Smoothed AUROC", color="#2ca02c", marker="^", linewidth=2)
        ax.plot(df["epoch"], df["val_auroc"], color="#2ca02c", alpha=0.3, linewidth=1)  # faint raw curve
    else:
        ax.text(0.5, 0.5, "No AUROC data found", ha="center", va="center", fontsize=11)
    ax.set_title(model_name, fontsize=12, fontweight="semibold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(outdir, "fig2_auroc.png"), dpi=300, bbox_inches="tight")
plt.show()

print("✅ Figures saved as:")
print("   Fig. 1 → fig1_loss.png")
print("   Fig. 2 → fig2_auroc.png")

import argparse, os, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

# ---------- utils ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def cm_to_df(cm, labels=("Non-anemic","Anemic")):
    tn, fp, fn, tp = cm.ravel()
    return pd.DataFrame(
        [[tn, fp],[fn, tp]],
        index=[f"True {labels[0]}", f"True {labels[1]}"],
        columns=[f"Pred {labels[0]}", f"Pred {labels[1]}"]
    )

def plot_confusion_matrix(cm, outpng):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-anemic","Anemic"])
    ax.set_yticklabels(["Non-anemic","Anemic"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

def reliability_bins(y_true, y_prob, n_bins=10):
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df = df.sort_values("p").reset_index(drop=True)
    bins = np.linspace(0,1,n_bins+1)
    df["bin"] = np.clip(np.digitize(df["p"], bins)-1, 0, n_bins-1)
    rows = []
    for b in range(n_bins):
        chunk = df[df["bin"]==b]
        if len(chunk)==0:
            rows.append(( (bins[b]+bins[b+1])/2, np.nan, np.nan, 0))
            continue
        conf = chunk["p"].mean()
        emp = chunk["y"].mean() if len(chunk)>0 else np.nan
        rows.append((conf, emp, len(chunk), b))
    out = pd.DataFrame(rows, columns=["confidence","empirical","count","bin_idx"])
    return out, bins

def plot_calibration(bins_df, outpng):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.plot([0,1],[0,1], linestyle="--")  # perfect
    ok = bins_df.dropna()
    ax.plot(ok["confidence"], ok["empirical"], marker="o")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed anemia rate")
    ax.set_title("Calibration (Reliability) Curve")
    plt.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_hist(probs, outpng, title="Score histogram"):
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.hist(probs, bins=20)
    ax.set_xlabel("Predicted probability (anemia)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_roc(y_true, y_prob, outpng, label=""):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"{label} AUC={roc_auc:.3f}" if label else f"AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    if label:
        ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return roc_auc

def plot_pr(y_true, y_prob, outpng, label=""):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.plot(rec, prec, label=f"{label} AUC={pr_auc:.3f}" if label else f"AUC={pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    if label:
        ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pr_auc

def mark_operating_point_on_roc(y_true, y_prob, threshold, outpng):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    # find closest threshold
    idx = np.argmin(np.abs(thr - threshold))
    op_fpr, op_tpr = fpr[idx], tpr[idx]
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1], linestyle="--")
    ax.scatter([op_fpr],[op_tpr], marker="o")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC with Operating Point (t={threshold:.3f})")
    plt.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_thr_sweep_val(y_true, y_prob, outpng):
    ths = np.linspace(0.01,0.99,99)
    precs, recs, f1s = [], [], []
    for t in ths:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_true==1)&(y_pred==1)).sum()
        fp = ((y_true==0)&(y_pred==1)).sum()
        fn = ((y_true==1)&(y_pred==0)).sum()
        prec = tp/max(1,(tp+fp))
        rec  = tp/max(1,(tp+fn))
        f1 = 2*prec*rec/max(1e-12,(prec+rec))
        precs.append(prec); recs.append(rec); f1s.append(f1)
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.plot(ths, precs, label="Precision")
    ax.plot(ths, recs, label="Recall")
    ax.plot(ths, f1s, label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold (VAL)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_hb_scatter(y_true, y_pred, outpng):
    fig = plt.figure(figsize=(4.5,4.5))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=14)
    m = max(np.max(y_true), np.max(y_pred))
    mi = min(np.min(y_true), np.min(y_pred))
    ax.plot([mi,m],[mi,m], linestyle="--")
    ax.set_xlabel("True Hb (g/dL)")
    ax.set_ylabel("Predicted Hb (g/dL)")
    ax.set_title("Hb Regression: Pred vs True")
    plt.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_residual_hist(residuals, outpng):
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.hist(residuals, bins=20)
    ax.set_xlabel("Residual (Pred - True) Hb (g/dL)")
    ax.set_ylabel("Count")
    ax.set_title("Residuals Histogram")
    plt.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_residuals_vs_true(y_true, residuals, outpng):
    fig = plt.figure(figsize=(5,3.5))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, residuals, s=14)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("True Hb (g/dL)")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("Residuals vs True Hb")
    plt.tight_layout()
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_preds", required=True)
    ap.add_argument("--test_preds", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--tuned_threshold", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_calib_bins", type=int, default=10)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Load artifacts
    val_df  = pd.read_csv(args.val_preds)
    test_df = pd.read_csv(args.test_preds)
    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(args.tuned_threshold, "r", encoding="utf-8") as f:
        tuned = json.load(f)
    thr = float(tuned.get("threshold", metrics.get("val_threshold", 0.5)))

    # ----- VAL curves -----
    yv = val_df["true_label"].to_numpy().astype(int)
    pv = val_df["prob"].to_numpy().astype(float)

    roc_auc_val = plot_roc(yv, pv, outdir/"roc_curve_val.png", label="VAL")
    pr_auc_val  = plot_pr(yv, pv, outdir/"pr_curve_val.png", label="VAL")
    plot_thr_sweep_val(yv, pv, outdir/"thr_sweep_val.png")

    # ----- TEST curves -----
    yt = test_df["true_label"].to_numpy().astype(int)
    pt = test_df["prob"].to_numpy().astype(float)
    hb_true = test_df["hb_true"].to_numpy().astype(float)
    hb_pred = test_df["hb_pred"].to_numpy().astype(float)

    roc_auc_test = plot_roc(yt, pt, outdir/"roc_curve_test.png", label="TEST")
    pr_auc_test  = plot_pr(yt, pt, outdir/"pr_curve_test.png", label="TEST")
    mark_operating_point_on_roc(yt, pt, thr, outdir/"roc_with_op_test.png")
    plot_hist(pt, outdir/"prob_hist_test.png", title="Predicted Probabilities (TEST)")

    # Confusion matrix at tuned threshold
    yhat = (pt >= thr).astype(int)
    cm = confusion_matrix(yt, yhat, labels=[0,1])
    plot_confusion_matrix(cm, outdir/"confusion_matrix_test.png")
    cm_to_df(cm).to_csv(outdir/"confusion_matrix_test.csv")

    # Calibration / reliability
    calib_df, bins = reliability_bins(yt, pt, n_bins=args.n_calib_bins)
    calib_df.to_csv(outdir/"calibration_bins_test.csv", index=False)
    plot_calibration(calib_df, outdir/"calibration_curve_test.png")

    # Regression plots
    plot_hb_scatter(hb_true, hb_pred, outdir/"hb_scatter_test.png")
    residuals = hb_pred - hb_true
    plot_residual_hist(residuals, outdir/"hb_residual_hist_test.png")
    plot_residuals_vs_true(hb_true, residuals, outdir/"hb_residuals_vs_true_test.png")

    # ----- Markdown report -----
    # Read key metrics (fall back to recomputing AUCs we already have)
    rep = []
    rep.append(f"# MobileNetV2 Evaluation Report\n")
    rep.append(f"- **Best epoch (VAL)**: {metrics.get('best_epoch','?')}")
    rep.append(f"- **Validation tuned threshold**: {metrics.get('val_threshold', thr):.3f}")
    rep.append(f"- **Validation** — F1 {metrics.get('val_f1','?'):.3f}, AUROC {metrics.get('val_auroc','?'):.3f}, AUPRC {metrics.get('val_auprc','?'):.3f}, Acc {metrics.get('val_acc','?'):.3f}, Prec {metrics.get('val_precision','?'):.3f}, Rec {metrics.get('val_recall','?'):.3f}, MAE {metrics.get('val_mae','?'):.2f} g/dL")
    rep.append(f"- **Test (tuned @ VAL)** — F1 {metrics.get('test_f1','?'):.3f}, AUROC {metrics.get('test_auroc','?'):.3f}, AUPRC {metrics.get('test_auprc','?'):.3f}, Acc {metrics.get('test_acc','?'):.3f}, Prec {metrics.get('test_precision','?'):.3f}, Rec {metrics.get('test_recall','?'):.3f}, MAE {metrics.get('test_mae','?'):.2f} g/dL")
    rep.append(f"- **Confusion (TEST)** — TP {metrics.get('test_tp','?')}, TN {metrics.get('test_tn','?')}, FP {metrics.get('test_fp','?')}, FN {metrics.get('test_fn','?')}\n")

    rep.append("## Curves and Figures\n")
    figs = [
        "roc_curve_val.png", "pr_curve_val.png", "thr_sweep_val.png",
        "roc_curve_test.png", "roc_with_op_test.png", "pr_curve_test.png",
        "confusion_matrix_test.png", "prob_hist_test.png",
        "calibration_curve_test.png",
        "hb_scatter_test.png", "hb_residual_hist_test.png", "hb_residuals_vs_true_test.png"
    ]
    for f in figs:
        rep.append(f"![{f}]({f})")

    report_md = "\n".join(rep)
    with open(outdir/"report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("Saved figures and report to:", outdir)

if __name__ == "__main__":
    main()

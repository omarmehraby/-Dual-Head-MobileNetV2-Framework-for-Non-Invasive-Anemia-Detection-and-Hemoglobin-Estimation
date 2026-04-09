# Anemia Detection from Conjunctival Images

> Screening for anemia by analyzing photos of the inner eyelid (conjunctiva) using a multitask deep learning model — no blood test required.

![Accuracy](https://img.shields.io/badge/Accuracy-87.7%25-blue)
![Recall](https://img.shields.io/badge/Recall-93.7%25-green)
![AUROC](https://img.shields.io/badge/AUROC-92.9%25-orange)
![F1](https://img.shields.io/badge/F1--Score-90.1%25-yellow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

---

## What This Does

Anemia affects ~2 billion people globally. Clinical diagnosis requires a blood test, but conjunctival pallor (paleness of the inner eyelid) is a known visual indicator. This project automates that visual check using deep learning.

A MobileNetV2 model is fine-tuned with **two output heads**:
- **Classification**: anemic vs. non-anemic (binary)
- **Regression**: predicted hemoglobin level in g/dL (auxiliary task to improve feature learning)

The threshold is clinically tuned to **maximize sensitivity** — missing an anemic patient is worse than a false positive that gets a confirmatory blood test.

---

## Results

### Test Set (n = 107 images, threshold = 0.835)

| Metric | Value |
|---|---|
| Accuracy | 87.74% |
| Precision | 86.76% |
| **Recall / Sensitivity** | **93.65%** |
| F1-Score | 90.08% |
| ROC-AUC | 92.88% |
| PR-AUC | 94.45% |
| True Positives / True Negatives | 59 / 34 |
| False Positives / False Negatives | 9 / 4 |

### ROC Curve & Operating Point

![ROC Curve](results/roc_with_op_test.png)

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix_test_mobilenetv2.png)

### Precision-Recall Curve

![PR Curve](results/pr_curve_test_mobilenetv2.png)

### Training & Validation Loss

![Training Loss](results/fig1_loss.png)

### Validation AUROC per Epoch

![AUROC](results/fig2_auroc.png)

### Hemoglobin Regression (auxiliary head)

| Metric | Value |
|---|---|
| MAE | 1.60 g/dL |
| RMSE | 2.01 g/dL |
| R² | 0.198 |
| Pearson r | 0.556 |

![Hb Scatter](results/hb_scatter_test_mobilenetv2.png)

---

## Error Analysis

Of 107 test images, **13 were misclassified** (4 FN, 9 FP). Most errors trace to image quality problems, not model limitations:

![Error Categories](results/Fig10_Error_Categories.png)

| Error Category | Count |
|---|---|
| Other / uncertain | 5 |
| Glare / reflection | 2 |
| Borderline Hb (~11.0 g/dL) | 2 |
| Shadow / poor lighting | 1 |
| Blur / out of focus | 1 |
| Partial ROI | 1 |
| Occlusion / noisy texture | 1 |

---

## Project Structure

```
anemia_project20/
├── results/                          # Key figures for reference (tracked)
│   ├── roc_with_op_test.png
│   ├── confusion_matrix_test_mobilenetv2.png
│   ├── pr_curve_test_mobilenetv2.png
│   ├── hb_scatter_test_mobilenetv2.png
│   ├── fig1_loss.png
│   ├── fig2_auroc.png
│   └── Fig10_Error_Categories.png
├── docs/
│   └── mbv2 REPORT.docx              # Full project report
├── Anemic/                           # Patient images (not tracked)
├── Non-anemic/                       # Patient images (not tracked)
├── splits/                           # Train/val/test CSVs (not tracked)
├── exp/mobilenetv2_mtl_320/          # Model weights, predictions (not tracked)
├── make_splits.py                    # Step 1: create dataset splits
├── train_multitask_mobilenetv2.py    # Step 2: train MobileNetV2 MTL model
├── make_all_figures_and_report.py    # Step 3: generate evaluation figures
├── t_v_auroc.py                      # Plot training loss and AUROC curves
├── reliability_calibration.py        # Plot reliability/calibration diagram
├── error_analysis.py                 # Categorize misclassified images
├── error_samples.py                  # Count false positives / negatives
├── requirements.txt
└── .gitignore
```

---

## Quickstart

```bash
git clone https://github.com/your-username/anemia_project20.git
cd anemia_project20
pip install -r requirements.txt
```

### Step 1 — Create dataset splits

```bash
python make_splits.py --root /path/to/anemia_project20
```

### Step 2 — Train the model

```bash
python train_multitask_mobilenetv2.py \
  --train_csv splits/train.csv \
  --val_csv   splits/val.csv \
  --test_csv  splits/test.csv \
  --outdir    exp/mobilenetv2_mtl_320
```

### Step 3 — Generate evaluation figures

```bash
python make_all_figures_and_report.py \
  --val_csv   exp/mobilenetv2_mtl_320/val_preds_mobilenetv2.csv \
  --test_csv  exp/mobilenetv2_mtl_320/test_preds_Mobilenetv2.csv \
  --metrics   exp/mobilenetv2_mtl_320/metrics.json \
  --threshold exp/mobilenetv2_mtl_320/tuned_threshold.json \
  --outdir    exp/mobilenetv2_mtl_320/figures
```

### Step 4 — Plot training curves

```bash
python t_v_auroc.py \
  --history_csv exp/mobilenetv2_mtl_320/history_mobilenetv2.csv \
  --outdir      exp/mobilenetv2_mtl_320
```

### Step 5 — Error analysis

```bash
python error_analysis.py --preds_csv exp/mobilenetv2_mtl_320/test_preds_MobilenetV2.csv
python error_samples.py  --preds_csv exp/mobilenetv2_mtl_320/test_preds_Mobilenetv2.csv
```

### Step 6 — Reliability diagram

```bash
python reliability_calibration.py \
  --bins_csv exp/mobilenetv2_mtl_320/figures/calibration_bins_test.csv \
  --outdir   exp/mobilenetv2_mtl_320/figures
```

---

## Dataset

- **731 conjunctival images** (PNG), split **70 / 15 / 15** → train / val / test
- Hemoglobin ground truth from clinical data collection sheet
- Images are **not included** in this repository (patient privacy)

---

## Model Architecture

**MTLMobileNetV2** — MobileNetV2 backbone (ImageNet pretrained) with two heads:

```
Input Image (320×320)
       │
  MobileNetV2 Backbone
       │
  Global Average Pooling
       ├──→ Classification Head → sigmoid → anemic / non-anemic
       └──→ Regression Head    → linear  → Hb (g/dL)
```

- Loss: `BCEWithLogitsLoss` (classification) + `MSELoss` (regression)
- Class imbalance handled via weighted random sampling
- Decision threshold tuned on validation set over 181 values (0.05–0.95) to maximize F1
- Best epoch: 40 / 50 | Optimal threshold: **0.835**

---

## Tech Stack

`Python` · `PyTorch` · `torchvision` · `OpenCV` · `scikit-learn` · `pandas` · `matplotlib`

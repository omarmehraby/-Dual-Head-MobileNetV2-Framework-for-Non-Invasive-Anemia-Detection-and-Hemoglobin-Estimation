import argparse, os, json, math, random, csv, time
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def read_csv_rows(csv_path: str):
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "image_path": r["image_path"],
                "class_label": int(r["class_label"]),
                "hb": float(r["hb"]) if "hb" in r and r["hb"] != "" else float("nan"),
            })
    return rows

class ConjunctivaDataset(Dataset):
    def __init__(self, rows, img_size=320, augment=False):
        self.rows = rows
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if augment:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.1)], p=0.3),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        x = self.tf(img)
        y_cls = torch.tensor(r["class_label"], dtype=torch.float32)  # binary (0/1)
        y_reg = torch.tensor(r["hb"], dtype=torch.float32)
        return x, y_cls, y_reg, r["image_path"]

# ---------------------------
# Model
# ---------------------------
class MTLMobileNetV2(nn.Module):
    def __init__(self, pretrained=True, dropout_p=0.2):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.classifier = nn.Identity()         # keep features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout_p)
        feat_dim = 1280

        # Heads
        self.cls_head = nn.Linear(feat_dim, 1)           # logits
        self.reg_head = nn.Linear(feat_dim, 1)           # Hb

    def forward(self, x):
        feats = self.backbone.features(x)                # [B,1280,H',W']
        pooled = self.pool(feats).flatten(1)             # [B,1280]
        z = self.dropout(pooled)
        logit = self.cls_head(z).squeeze(1)              # [B]
        hb = self.reg_head(z).squeeze(1)                 # [B]
        return logit, hb

# ---------------------------
# Metrics
# ---------------------------
def sigmoid(x): return 1 / (1 + np.exp(-x))

def binary_metrics_from_logits(y_true, y_logit, threshold=0.5):
    y_prob = sigmoid(y_logit)
    y_pred = (y_prob >= threshold).astype(np.int32)

    tp = int(((y_true==1) & (y_pred==1)).sum())
    tn = int(((y_true==0) & (y_pred==0)).sum())
    fp = int(((y_true==0) & (y_pred==1)).sum())
    fn = int(((y_true==1) & (y_pred==0)).sum())

    acc = (tp+tn)/max(1,(tp+tn+fp+fn))
    prec = tp/max(1,(tp+fp))
    rec = tp/max(1,(tp+fn))
    f1 = 2*prec*rec/max(1e-12,(prec+rec))

    # AUROC/AUPRC (simple fallbacks if sklearn not present)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auroc, auprc = float("nan"), float("nan")

    return dict(acc=acc, precision=prec, recall=rec, f1=f1, auroc=auroc, auprc=auprc,
                tp=tp, tn=tn, fp=fp, fn=fn)

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    # R2
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = float(1 - ss_res/max(1e-12, ss_tot))
    # Pearson r
    try:
        r = float(np.corrcoef(y_true, y_pred)[0,1])
    except Exception:
        r = float("nan")
    return dict(mae=mae, rmse=rmse, r2=r2, pearson_r=r)

def tune_threshold_for_f1(y_true, y_logit):
    y_prob = sigmoid(y_logit)
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= t).astype(np.int32)
        tp = ((y_true==1)&(y_pred==1)).sum()
        fp = ((y_true==0)&(y_pred==1)).sum()
        fn = ((y_true==1)&(y_pred==0)).sum()
        prec = tp/max(1,(tp+fp))
        rec  = tp/max(1,(tp+fn))
        f1 = 2*prec*rec/max(1e-12,(prec+rec))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)

# ---------------------------
# Train / Eval
# ---------------------------
def make_sampler(rows):
    # handle class imbalance
    labels = np.array([r["class_label"] for r in rows], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def epoch_loop(model, loader, device, criterion_cls, criterion_reg, w_cls, w_reg, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    all_logits, all_cls, all_hb_true, all_hb_pred, all_paths = [], [], [], [], []

    for x, y_cls, y_reg, pth in loader:
        x = x.to(device)
        y_cls = y_cls.to(device)
        y_reg = y_reg.to(device)

        with torch.set_grad_enabled(is_train):
            logit, hb_pred = model(x)
            loss_cls = criterion_cls(logit, y_cls)
            loss_reg = criterion_reg(hb_pred, y_reg)
            loss = w_cls*loss_cls + w_reg*loss_reg

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * x.size(0)

        all_logits.append(logit.detach().cpu().numpy())
        all_cls.append(y_cls.cpu().numpy())
        all_hb_true.append(y_reg.cpu().numpy())
        all_hb_pred.append(hb_pred.detach().cpu().numpy())
        all_paths.extend(list(pth))

    N = len(loader.dataset)
    avg_loss = running_loss / max(1,N)
    y_logit = np.concatenate(all_logits)
    y_true  = np.concatenate(all_cls).astype(np.int32)
    hb_true = np.concatenate(all_hb_true)
    hb_pred = np.concatenate(all_hb_pred)

    return avg_loss, y_true, y_logit, hb_true, hb_pred, all_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--test_csv",  default="")
    ap.add_argument("--outdir",    required=True)
    ap.add_argument("--epochs",    type=int, default=50)
    ap.add_argument("--bs",        type=int, default=16)
    ap.add_argument("--img_size",  type=int, default=320)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--dropout",   type=float, default=0.2)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--w_cls",     type=float, default=1.0)
    ap.add_argument("--w_reg",     type=float, default=0.25)
    ap.add_argument("--regression_clip", type=float, default=0.0, help=">0 to clamp Hb preds to [clip_min, clip_max] where min/max seen in train")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    train_rows = read_csv_rows(args.train_csv)
    val_rows   = read_csv_rows(args.val_csv)
    test_rows  = read_csv_rows(args.test_csv) if args.test_csv else []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTLMobileNetV2(pretrained=True, dropout_p=args.dropout).to(device)

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # Datasets & Loaders
    ds_train = ConjunctivaDataset(train_rows, img_size=args.img_size, augment=True)
    ds_val   = ConjunctivaDataset(val_rows,   img_size=args.img_size, augment=False)
    sampler  = make_sampler(train_rows)
    dl_train = DataLoader(ds_train, batch_size=args.bs, sampler=sampler, num_workers=4, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.bs*2, shuffle=False, num_workers=4, pin_memory=True)

    # Losses
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # For optional clipping of regression to realistic range seen in train
    hb_train_vals = np.array([r["hb"] for r in train_rows], dtype=np.float32)
    hb_min, hb_max = float(np.nanmin(hb_train_vals)), float(np.nanmax(hb_train_vals))

    best_key, best_val = "val_f1_at_tuned", -1
    history = []

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_y, tr_logit, tr_hb, tr_hb_pred, _ = epoch_loop(
            model, dl_train, device, criterion_cls, criterion_reg, args.w_cls, args.w_reg, optimizer
        )
        va_loss, va_y, va_logit, va_hb, va_hb_pred, va_paths = epoch_loop(
            model, dl_val, device, criterion_cls, criterion_reg, args.w_cls, args.w_reg, optimizer=None
        )

        if args.regression_clip > 0:
            tr_hb_pred = np.clip(tr_hb_pred, hb_min, hb_max)
            va_hb_pred = np.clip(va_hb_pred, hb_min, hb_max)

        # Tune threshold on VAL for best F1
        tuned_t, tuned_f1 = tune_threshold_for_f1(va_y, va_logit)

        # Metrics @0.5 and @tuned
        m_val_05 = binary_metrics_from_logits(va_y, va_logit, threshold=0.5)
        m_val_tt = binary_metrics_from_logits(va_y, va_logit, threshold=tuned_t)
        m_tr_05  = binary_metrics_from_logits(tr_y, tr_logit, threshold=0.5)

        r_tr = regression_metrics(tr_hb, tr_hb_pred)
        r_va = regression_metrics(va_hb, va_hb_pred)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "val_tuned_threshold": tuned_t,
            "val_f1_at_0.5": m_val_05["f1"],
            "val_f1_at_tuned": m_val_tt["f1"],
            "val_auroc": m_val_tt["auroc"],
            "val_auprc": m_val_tt["auprc"],
            "val_acc_at_tuned": m_val_tt["acc"],
            "val_precision_at_tuned": m_val_tt["precision"],
            "val_recall_at_tuned": m_val_tt["recall"],
            "train_auroc": m_tr_05["auroc"],
            "train_f1_at_0.5": m_tr_05["f1"],
            "train_mae": r_tr["mae"], "train_rmse": r_tr["rmse"], "train_r2": r_tr["r2"],
            "val_mae": r_va["mae"], "val_rmse": r_va["rmse"], "val_r2": r_va["r2"],
            "time_sec": round(time.time()-t0, 2)
        }
        history.append(row)
        print(f"[{epoch:03d}] val F1@tuned={row['val_f1_at_tuned']:.4f} | AUROC={row['val_auroc']:.4f} | MAE={row['val_mae']:.3f}")

        # save best
        key_val = row[best_key]
        if key_val > best_val:
            best_val = key_val
            torch.save(model.state_dict(), os.path.join(args.outdir,"model_best.pt"))

        # Save running history
        hist_path = os.path.join(args.outdir, "history.csv")
        with open(hist_path, "w", newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader(); w.writerows(history)

        # Save current val preds
        val_preds_path = os.path.join(args.outdir, "val_preds.csv")
        with open(val_preds_path, "w", newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["image_path","true_label","logit","prob","pred_at_tuned","hb_true","hb_pred"])
            probs = 1/(1+np.exp(-va_logit))
            preds = (probs >= tuned_t).astype(int)
            for p, yt, lg, pr, pd, hb_t, hb_p in zip(va_paths, va_y, va_logit, probs, preds, va_hb, va_hb_pred):
                w.writerow([p, int(yt), float(lg), float(pr), int(pd), float(hb_t), float(hb_p)])

        # Save tuned threshold
        with open(os.path.join(args.outdir, "tuned_threshold.json"), "w", encoding="utf-8") as f:
            json.dump({"threshold": tuned_t, "val_f1": tuned_f1}, f, indent=2)

    # Final evaluation on TEST (if provided)
    results = {}
    if args.test_csv:
        ds_test = ConjunctivaDataset(test_rows, img_size=args.img_size, augment=False)
        dl_test = DataLoader(ds_test, batch_size=args.bs*2, shuffle=False, num_workers=4, pin_memory=True)

        # Load best model
        model.load_state_dict(torch.load(os.path.join(args.outdir,"model_best.pt"), map_location=device))
        _, te_y, te_logit, te_hb, te_hb_pred, te_paths = epoch_loop(
            model, dl_test, device, criterion_cls, criterion_reg, args.w_cls, args.w_reg, optimizer=None
        )
        # Use tuned threshold from VAL
        with open(os.path.join(args.outdir, "tuned_threshold.json"), "r", encoding="utf-8") as f:
            tuned = json.load(f).get("threshold", 0.5)
        m_te = binary_metrics_from_logits(te_y, te_logit, threshold=tuned)
        r_te = regression_metrics(te_hb, te_hb_pred)

        # Save test preds
        with open(os.path.join(args.outdir, "test_preds.csv"), "w", newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["image_path","true_label","logit","prob","pred_at_tuned","hb_true","hb_pred"])
            probs = 1/(1+np.exp(-te_logit))
            preds = (probs >= tuned).astype(int)
            for p, yt, lg, pr, pd, hb_t, hb_p in zip(te_paths, te_y, te_logit, probs, preds, te_hb, te_hb_pred):
                w.writerow([p, int(yt), float(lg), float(pr), int(pd), float(hb_t), float(hb_p)])

        results.update({
            "test_threshold": tuned,
            "test_acc": m_te["acc"], "test_precision": m_te["precision"], "test_recall": m_te["recall"],
            "test_f1": m_te["f1"], "test_auroc": m_te["auroc"], "test_auprc": m_te["auprc"],
            "test_tp": m_te["tp"], "test_tn": m_te["tn"], "test_fp": m_te["fp"], "test_fn": m_te["fn"],
            "test_mae": r_te["mae"], "test_rmse": r_te["rmse"], "test_r2": r_te["r2"], "test_pearson_r": r_te["pearson_r"]
        })

    # Save summary metrics from best epoch (VAL)
    best_epoch_row = max(history, key=lambda r: r["val_f1_at_tuned"])
    results.update({
        "best_epoch": best_epoch_row["epoch"],
        "val_threshold": best_epoch_row["val_tuned_threshold"],
        "val_acc": best_epoch_row["val_acc_at_tuned"],
        "val_precision": best_epoch_row["val_precision_at_tuned"],
        "val_recall": best_epoch_row["val_recall_at_tuned"],
        "val_f1": best_epoch_row["val_f1_at_tuned"],
        "val_auroc": best_epoch_row["val_auroc"],
        "val_auprc": best_epoch_row["val_auprc"],
        "val_mae": best_epoch_row["val_mae"],
        "val_rmse": best_epoch_row["val_rmse"],
        "val_r2": best_epoch_row["val_r2"],
    })

    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

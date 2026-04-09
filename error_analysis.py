# --- Error Analysis: automatic, data-driven failure categorization (Updated Version) ---

import argparse
import os, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ---------- ARGS ----------
parser = argparse.ArgumentParser(description="Error analysis for anemia detection model predictions")
parser.add_argument("--preds_csv", required=True,
                    help="Path to test predictions CSV (e.g., test_preds_MobilenetV2.csv)")
parser.add_argument("--base_dir", default=None,
                    help="Base directory to resolve relative image paths in the CSV (optional)")
parser.add_argument("--hb_borderline_center", type=float, default=11.0,
                    help="Center of borderline Hb range in g/dL (default: 11.0)")
parser.add_argument("--hb_borderline_width", type=float, default=0.5,
                    help="Half-width of borderline Hb range in g/dL (default: 0.5)")
args = parser.parse_args()

preds_csv            = args.preds_csv
base_dir             = args.base_dir
hb_borderline_center = args.hb_borderline_center
hb_borderline_width  = args.hb_borderline_width

# ---------- LOAD ----------
df = pd.read_csv(preds_csv)

def full_path(p):
    if base_dir and not os.path.isabs(p):
        return os.path.join(base_dir, p)
    return p

if "image_path" in df.columns:
    df["image_path"] = df["image_path"].apply(full_path)

errs = df[(df["true_label"] != df["pred_at_tuned"])].copy()
print(f"🔍 Found {len(errs)} misclassified images.")

# ---------- FEATURE EXTRACTORS ----------
def load_rgb(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def image_metrics(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H,S,V = cv2.split(hsv)
    brightness = float(np.mean(V))
    shadow_frac = float(np.mean(V < 30))
    glare_frac  = float(np.mean((V > 240) & (S < 40)))
    R,G,B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    red_mask = (R > 120) & (R > G + 10) & (R > B + 10)
    red_area_ratio = float(np.mean(red_mask))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    return blur_score, brightness, shadow_frac, glare_frac, red_area_ratio, edge_density

metrics = []
missing = 0
for p in errs["image_path"]:
    img = load_rgb(p)
    if img is None:
        metrics.append((np.nan,)*6)
        missing += 1
        continue
    metrics.append(image_metrics(img))

(cols_blur, cols_bright, cols_shadow, cols_glare, cols_red, cols_edge) = zip(*metrics)
errs["blur_score"]      = cols_blur
errs["brightness"]      = cols_bright
errs["shadow_frac"]     = cols_shadow
errs["glare_frac"]      = cols_glare
errs["red_area_ratio"]  = cols_red
errs["edge_density"]    = cols_edge

if missing:
    print(f"⚠️ Warning: {missing} images could not be read. They will be tagged as 'other/uncertain'.")

# ---------- SAFE NUMERIC CONVERSION + QUANTILES ----------
for col in ["blur_score", "brightness", "shadow_frac", "glare_frac", "red_area_ratio", "edge_density"]:
    errs[col] = pd.to_numeric(errs[col], errors="coerce")

errs = errs.dropna(subset=["blur_score", "brightness", "shadow_frac", "glare_frac", "red_area_ratio", "edge_density"], how="all")

thr_blur_low    = errs["blur_score"].quantile(0.10)
thr_bright_low  = errs["brightness"].quantile(0.10)
thr_shadow_high = errs["shadow_frac"].quantile(0.90)
thr_glare_high  = errs["glare_frac"].quantile(0.90)
thr_red_low     = errs["red_area_ratio"].quantile(0.10)
thr_edge_high   = errs["edge_density"].quantile(0.90)

print("\n📏 Quantile thresholds (computed from your data):")
print({
    "blur_low": round(thr_blur_low, 3),
    "bright_low": round(thr_bright_low, 3),
    "shadow_high": round(thr_shadow_high, 3),
    "glare_high": round(thr_glare_high, 3),
    "red_low": round(thr_red_low, 3),
    "edge_high": round(thr_edge_high, 3)
})

# ---------- CATEGORY RULES ----------
def categorize(row):
    if "hb_true" in row and pd.notna(row["hb_true"]):
        if abs(row["hb_true"] - hb_borderline_center) <= hb_borderline_width:
            return "Borderline Hb (≈ threshold)"
    if row["glare_frac"] >= thr_glare_high:
        return "Glare / reflection"
    if (row["brightness"] <= thr_bright_low) or (row["shadow_frac"] >= thr_shadow_high):
        return "Shadow / poor lighting"
    if row["blur_score"] <= thr_blur_low:
        return "Blur / out of focus"
    if row["red_area_ratio"] <= thr_red_low:
        return "Partial ROI / low exposed area"
    if row["edge_density"] >= thr_edge_high:
        return "Occlusion / noisy texture"
    return "Other / uncertain"

errs["error_category"] = errs.apply(categorize, axis=1)

# ---------- OUTPUTS ----------
table7 = errs["error_category"].value_counts().reset_index()
table7.columns = ["Error Category", "Count"]
print("\n📊 Table 7 – Error categories and counts")
print(table7)

out_csv = os.path.join(os.path.dirname(preds_csv), "error_analysis_per_image.csv")
errs.to_csv(out_csv, index=False)
print(f"\n💾 Per-image diagnostics saved to: {out_csv}")

plt.figure(figsize=(8,4.5))
plt.barh(table7["Error Category"], table7["Count"], color="tomato")
plt.xlabel("Number of Misclassified Images")
plt.title("Fig. 10 – Error Categories and Counts")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

out_fig = os.path.join(os.path.dirname(preds_csv), "Fig10_Error_Categories.png")
plt.savefig(out_fig, dpi=300)
plt.show()
print(f"✅ Figure saved to: {out_fig}")

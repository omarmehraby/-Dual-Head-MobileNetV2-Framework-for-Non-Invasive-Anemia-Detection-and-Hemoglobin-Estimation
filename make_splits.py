import argparse
import os, glob, random, csv, re
from pathlib import Path
import pandas as pd

random.seed(42)

parser = argparse.ArgumentParser(description="Create stratified train/val/test splits for anemia dataset")
parser.add_argument("--root", required=True,
                    help="Root project directory containing Anemic/ and Non-anemic/ subdirectories")
parser.add_argument("--anemic_dir", default=None,
                    help="Path to anemic images directory (default: <root>/Anemic)")
parser.add_argument("--nonanemic_dir", default=None,
                    help="Path to non-anemic images directory (default: <root>/Non-anemic)")
parser.add_argument("--xlsx", default=None,
                    help="Path to Excel metadata file with Hb values (default: <root>/Anemia_Data_Collection_Sheet.xlsx)")
parser.add_argument("--outdir", default=None,
                    help="Output directory for split CSVs (default: <root>/splits)")
parser.add_argument("--frac_train", type=float, default=0.70, help="Training fraction (default: 0.70)")
parser.add_argument("--frac_val",   type=float, default=0.15, help="Validation fraction (default: 0.15)")
args = parser.parse_args()

ROOT         = args.root
ANEMIC_DIR   = Path(args.anemic_dir)    if args.anemic_dir    else Path(ROOT) / "Anemic"
NONANEMIC_DIR= Path(args.nonanemic_dir) if args.nonanemic_dir else Path(ROOT) / "Non-anemic"
XLSX         = Path(args.xlsx)          if args.xlsx          else Path(ROOT) / "Anemia_Data_Collection_Sheet.xlsx"

# -------- helpers
def norm_col(s: str) -> str:
    return re.sub(r'\s+', '', str(s).strip().lower())

def pick_column(cols, patterns):
    """Return first column whose normalized name matches any regex in patterns."""
    ncols = {norm_col(c): c for c in cols}
    for pat in patterns:
        rx = re.compile(pat)
        for n, orig in ncols.items():
            if rx.fullmatch(n) or rx.search(n):
                return orig
    return None

def key_from_name(name: str) -> str:
    base = os.path.basename(str(name))
    stem = os.path.splitext(base)[0]
    return norm_col(stem)

def build_hb_map(xlsx_path):
    # read first sheet
    df = pd.read_excel(xlsx_path)
    if df.empty:
        raise RuntimeError("Excel is empty.")

    # Try to find filename/image column
    filename_col = pick_column(
        df.columns,
        patterns=[
            r'(file(name)?|image(name)?|img(name)?|photo|picture|path)',
            r'^(name|id)$'
        ],
    )
    # Try to find Hb column
    hb_col = pick_column(
        df.columns,
        patterns=[
            r'(hb|hgb|hemoglobin|haemoglobin)',
            r'hemoglobin\(g/dl\)',
            r'hb_g/?dl'
        ],
    )

    if filename_col is None or hb_col is None:
        print("Available columns in Excel:", list(df.columns))
        raise RuntimeError(
            f"Could not auto-detect columns. "
            f"Detected filename_col={filename_col}, hb_col={hb_col}. "
            f"Rename your columns to something like 'filename' and 'hb' OR edit the patterns."
        )

    # build map key -> hb
    df = df[[filename_col, hb_col]].dropna()
    df[filename_col] = df[filename_col].astype(str)
    hb_map = {}
    for _, row in df.iterrows():
        k = key_from_name(row[filename_col])
        try:
            hb = float(row[hb_col])
        except Exception:
            continue
        hb_map[k] = hb
    print(f"Loaded Hb entries: {len(hb_map)} from columns "
          f"filename='{filename_col}' hb='{hb_col}'")
    return hb_map

def collect(dirpath: Path, label: int, hb_map: dict):
    rows = []
    for p in glob.glob(str(dirpath / "**" / "*.*"), recursive=True):
        if not re.search(r'\.(jpg|jpeg|png|bmp|tif|tiff)$', p, flags=re.I):
            continue
        k = key_from_name(p)
        # try exact stem; if not found, try without underscores/dashes
        hb = hb_map.get(k)
        if hb is None:
            k2 = re.sub(r'[_\-]+', '', k)
            # try relaxed match
            for kk in (k2,):
                if kk in hb_map:
                    hb = hb_map[kk]; break
        if hb is None:
            # no Hb for this image; skip to keep regression consistent
            continue
        rows.append((p, label, hb))
    return rows

def strat_split(items, frac_train=0.70, frac_val=0.15):
    byy = {0: [], 1: []}
    for it in items: byy[it[1]].append(it)
    for k in byy: random.shuffle(byy[k])

    def take(group):
        n = len(group)
        ntr = int(round(frac_train*n))
        nva = int(round(frac_val*n))
        return group[:ntr], group[ntr:ntr+nva], group[ntr+nva:]

    tr, va, te = [], [], []
    for k in byy:
        a,b,c = take(byy[k]); tr+=a; va+=b; te+=c
    random.shuffle(tr); random.shuffle(va); random.shuffle(te)
    return tr, va, te

def write_csv(rows, outpath):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path","class_label","hb"])
        w.writerows(rows)
    print("Wrote", outpath, "n=", len(rows))

# -------- main
if __name__ == "__main__":
    hb_map = build_hb_map(XLSX)

    pos = collect(ANEMIC_DIR, 1, hb_map)
    neg = collect(NONANEMIC_DIR, 0, hb_map)
    print(f"Matched images with Hb: anemic={len(pos)} non-anemic={len(neg)}")

    data = pos + neg
    if len(data) == 0:
        raise RuntimeError("No images matched with Hb values. "
                           "Check filenames in Excel vs. image basenames.")

    train, val, test = strat_split(data, frac_train=args.frac_train, frac_val=args.frac_val)

    OUT = Path(args.outdir) if args.outdir else Path(ROOT) / "splits"
    write_csv(train, OUT/"train.csv")
    write_csv(val,   OUT/"val.csv")
    write_csv(test,  OUT/"test.csv")
    print("Done.")

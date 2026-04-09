import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Count false positives and false negatives in test predictions")
parser.add_argument("--preds_csv", required=True,
                    help="Path to test predictions CSV (e.g., test_preds_Mobilenetv2.csv)")
args = parser.parse_args()

df = pd.read_csv(args.preds_csv)

# Separate misclassified samples
fp = df[(df["true_label"] == 0) & (df["pred_at_tuned"] == 1)]
fn = df[(df["true_label"] == 1) & (df["pred_at_tuned"] == 0)]

print("False Positives:", len(fp))
print("False Negatives:", len(fn))

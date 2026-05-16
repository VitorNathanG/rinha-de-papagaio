"""
Step 2.5 — Partition references into 3 box-classes based on label and k=25
neighborhood homogeneity. This is the training target for the unified
"box classifier" (router).

Classes:
    A-Legit (0)  label == legit  AND  count_k25 == 0
    A-Fraud (1)  label == fraud  AND  count_k25 == 25
    B       (2)  everything else
                 — mixed neighborhoods (count_k25 in 1..24)
                 — label outliers (label != neighborhood consensus)

Inputs:  data/labels.npy, data/fraud_counts_k25.npy
Output:  data/box_labels.npy   shape (N,) uint8, values 0/1/2
"""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

A_LEGIT = 0
A_FRAUD = 1
B = 2
K = 25  # the neighborhood width used to define the boxes


def main():
    out_path = DATA_DIR / "box_labels.npy"
    if out_path.exists() and not os.environ.get("FORCE"):
        existing = np.load(out_path)
        print(f"[partition] already computed: {out_path} (shape={existing.shape})")
        _report(existing)
        return

    labels = np.load(DATA_DIR / "labels.npy")               # bool, True=fraud
    counts = np.load(DATA_DIR / f"fraud_counts_k{K}.npy")   # uint8, 0..K

    is_a_legit = (~labels) & (counts == 0)
    is_a_fraud = labels & (counts == K)

    box = np.full(labels.shape, B, dtype=np.uint8)
    box[is_a_legit] = A_LEGIT
    box[is_a_fraud] = A_FRAUD

    np.save(out_path, box)
    print(f"[partition] wrote {out_path}")
    _report(box)


def _report(box: np.ndarray):
    n = len(box)
    counts_bin = np.bincount(box, minlength=3)
    print(f"[partition] class distribution among {n:,} references:")
    for c, name in [(A_LEGIT, "A-Legit"), (A_FRAUD, "A-Fraud"), (B, "B")]:
        bar = "#" * int(40 * counts_bin[c] / n)
        print(f"        {c} {name:>8}: {counts_bin[c]:>10,} ({100*counts_bin[c]/n:5.2f}%) {bar}")
    a_total = counts_bin[A_LEGIT] + counts_bin[A_FRAUD]
    print(f"[partition] A total: {a_total:,} ({100*a_total/n:.2f}%)")
    print(f"[partition] B total: {counts_bin[B]:,} ({100*counts_bin[B]/n:.2f}%)")


if __name__ == "__main__":
    main()

"""
Step 1/4 — Decompress references.json.gz into compact numpy arrays.

Source:  ../rinha-de-backend-2026/resources/references.json.gz
         (override with env REFERENCES_GZ)

Outputs (idempotent — skipped if already present):
    data/references.npy  — shape (N, 14) float32   (~168 MB at N=3M)
    data/labels.npy      — shape (N,)    bool       (True = fraud)

Notes:
- We keep the sentinel -1 values in indices 5 and 6 as-is. The k-NN does
  not know they are special; geometric distance handles them naturally
  because "no previous tx" vectors cluster together at (-1, -1) in those dims.
"""
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DEFAULT_REF_GZ = ROOT.parent / "rinha-de-backend-2026" / "resources" / "references.json.gz"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    src = Path(os.environ.get("REFERENCES_GZ", str(DEFAULT_REF_GZ)))
    vectors_path = DATA_DIR / "references.npy"
    labels_path = DATA_DIR / "labels.npy"

    if vectors_path.exists() and labels_path.exists():
        v = np.load(vectors_path, mmap_mode="r")
        l = np.load(labels_path, mmap_mode="r")
        print(f"[prepare] already prepared: {v.shape} vectors, {l.shape} labels "
              f"(fraud rate {l.mean()*100:.2f}%)")
        return

    if not src.exists():
        raise SystemExit(
            f"[prepare] missing source file: {src}\n"
            f"          set REFERENCES_GZ env var or place the file at that path"
        )

    print(f"[prepare] decompressing+parsing {src} ({src.stat().st_size/1e6:.0f} MB)")
    t0 = time.time()
    with gzip.open(src, "rt", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[prepare] parsed {len(data):,} entries in {time.time()-t0:.1f}s")

    n = len(data)
    vectors = np.empty((n, 14), dtype=np.float32)
    labels = np.empty(n, dtype=bool)
    t0 = time.time()
    for i, row in enumerate(data):
        vectors[i] = row["vector"]
        labels[i] = (row["label"] == "fraud")
    print(f"[prepare] converted to arrays in {time.time()-t0:.1f}s")

    np.save(vectors_path, vectors)
    np.save(labels_path, labels)
    print(f"[prepare] wrote {vectors_path} ({vectors.nbytes/1e6:.1f} MB)")
    print(f"[prepare] wrote {labels_path} ({labels.nbytes/1e6:.1f} MB)")
    print(f"[prepare] fraud rate in references: {labels.mean()*100:.2f}%")


if __name__ == "__main__":
    main()

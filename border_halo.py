"""
border_halo.py — extend Box-B with "A" refs that sit on the class boundary.

Why this step exists:
    partition.py labels a ref as A-Legit / A-Fraud when its own k=25
    neighbourhood is unanimous (an *inward* purity check). But refs that
    sit on the *outward* boundary of a pure cluster — closest neighbours
    of borderline queries that orbit the cluster's edge — get missed:
    their k=25 stays unanimous (inward) even though they face the other
    class. The slow-path k-NN over Box-B then misses them and votes the
    wrong way.

    nearest_opp.py computes, for each ref, the euclidean distance to its
    nearest opposite-label ref. This step promotes refs in A whose
    nearest-opp distance is below a threshold D into Box-B.

Pipeline position:
    prepare → label (K=25) → partition → nearest_opp → **border_halo** →
    export_box_b → (sweep_ivf | train_router | export_router)

Idempotent + re-runnable with a new D: reads from the pristine
partition snapshot, so changing D never double-promotes.

Tunables (env):
    D       threshold on nearest_opp distance (default 0.23).
            Calibrated via sim_borderhalo.py — at D=0.23 the resulting
            Box-B (with vectorize::round4 in the backend) flips the
            verdict on all 5 of our known boundary-miss mismatches.

Inputs:
    data/box_labels.before_halo.npy produced by partition.py
                                    (pristine, never overwritten by halo)
    data/nearest_opp_dist.npy       produced by nearest_opp.py

Output:
    data/box_labels.npy             A→B promotions applied; consumed by
                                    export_box_b.py and train_router.py
"""
import pipeline_log

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def main():
    D = float(os.environ.get("D", 0.23))
    in_path = DATA / "box_labels.before_halo.npy"
    out_path = DATA / "box_labels.npy"
    opp_path = DATA / "nearest_opp_dist.npy"

    if not opp_path.exists():
        raise SystemExit(
            f"[halo] {opp_path} not found — run `python nearest_opp.py` first")
    if not in_path.exists():
        raise SystemExit(
            f"[halo] {in_path} not found — run `python partition.py` first")

    box_base = np.load(in_path)
    opp = np.load(opp_path)
    if box_base.shape != opp.shape:
        raise SystemExit(
            f"[halo] shape mismatch: box_labels {box_base.shape} vs "
            f"nearest_opp {opp.shape}")

    in_A = (box_base != 2)
    halo_mask = in_A & (opp < D)
    n_add = int(halo_mask.sum())
    n_add_legit = int(((box_base == 0) & halo_mask).sum())
    n_add_fraud = int(((box_base == 1) & halo_mask).sum())

    box_new = box_base.copy()
    box_new[halo_mask] = 2

    n_B_before = int((box_base == 2).sum())
    n_B_after = int((box_new == 2).sum())

    np.save(out_path, box_new)
    print(f"[halo] D = {D}", flush=True)
    print(f"[halo] promoted {n_add:,} refs from A → B "
          f"(A-Legit: {n_add_legit:,}, A-Fraud: {n_add_fraud:,})", flush=True)
    print(f"[halo] Box-B: {n_B_before:,} → {n_B_after:,}  "
          f"(+{n_B_after - n_B_before:,})", flush=True)
    print(f"[halo] wrote {out_path}", flush=True)


if __name__ == "__main__":
    pipeline_log.setup(__file__)
    sys.exit(main() or 0)

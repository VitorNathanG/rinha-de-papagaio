"""
Diagnose the 5 backend↔ground-truth disagreements found by find_mismatches.py.

Ground truth: k=5 brute-force euclidean kNN over the full 3M references
(threshold=0.6 → count<3 = approved). See
/projects/rinha-de-backend-2026/data-generator/main.c knn_classify().

Our backend's slow path does the same kNN but over a 105 k Box-B subset
via an IVF index. When the subset is missing some of the actual top-5
neighbours, the vote can flip. This script computes both kNNs for each
mismatch and reports:
  - what the full-3M top-5 says (== expected_approved)
  - what brute over Box-B says (matches our backend's slow path)
  - what our IVF answer was (from mismatches.json)
  - which of the 5 NN are in Box-B vs not
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def vectorize(req):
    """Replicate backend/src/vectorize.rs as a Python function. Returns a
    14-element float32 array."""
    MAX_AMOUNT = 10000.0
    MAX_INSTALLMENTS = 12.0
    AMOUNT_VS_AVG_RATIO = 10.0
    MAX_MINUTES = 1440.0
    MAX_KM = 1000.0
    MAX_TX_COUNT_24H = 20.0
    MAX_MERCH_AVG = 10000.0

    def clamp01(x):
        return max(0.0, min(1.0, x))

    tx = req["transaction"]
    cust = req["customer"]
    merch = req["merchant"]
    term = req["terminal"]
    last = req.get("last_transaction")

    amount = tx["amount"]
    installments = tx["installments"]
    requested_at_str = tx["requested_at"]
    requested_at = datetime.fromisoformat(requested_at_str.replace("Z", "+00:00"))

    cust_avg = cust["avg_amount"]
    tx_24h = cust["tx_count_24h"]
    known = cust["known_merchants"]

    merchant_id = merch["id"]
    mcc = merch["mcc"]
    merch_avg = merch["avg_amount"]

    if last is not None:
        last_ts = datetime.fromisoformat(last["timestamp"].replace("Z", "+00:00"))
        mins = (requested_at - last_ts).total_seconds() / 60.0
        km_last = last["km_from_current"]
    else:
        mins = -1.0
        km_last = -1.0

    mcc_risk = {
        "5411": 0.15, "5812": 0.30, "5912": 0.20, "5944": 0.45,
        "7801": 0.80, "7802": 0.75, "7995": 0.85, "4511": 0.35,
        "5311": 0.25, "5999": 0.50,
    }.get(mcc, 0.50)

    v = np.zeros(14, dtype=np.float32)
    v[0] = clamp01(amount / MAX_AMOUNT)
    v[1] = clamp01(installments / MAX_INSTALLMENTS)
    v[2] = clamp01((amount / cust_avg) / AMOUNT_VS_AVG_RATIO)
    v[3] = requested_at.hour / 23.0
    # data-generator/main.c uses Mon=0..Sun=6, same as Python's weekday().
    v[4] = requested_at.weekday() / 6.0
    if mins < 0.0:
        v[5] = -1.0
        v[6] = -1.0
    else:
        v[5] = clamp01(mins / MAX_MINUTES)
        v[6] = clamp01(km_last / MAX_KM)
    v[7] = clamp01(term["km_from_home"] / MAX_KM)
    v[8] = clamp01(tx_24h / MAX_TX_COUNT_24H)
    v[9] = 1.0 if term["is_online"] else 0.0
    v[10] = 1.0 if term["card_present"] else 0.0
    v[11] = 0.0 if merchant_id in known else 1.0
    v[12] = mcc_risk
    v[13] = clamp01(merch_avg / MAX_MERCH_AVG)
    return v


def topk_brute(query, refs, labels, k=5):
    """Return (sorted_dists, sorted_idx, sorted_labels) of the k nearest."""
    d = np.sum((refs - query) ** 2, axis=1)
    idx = np.argpartition(d, k)[:k]
    idx = idx[np.argsort(d[idx])]
    return d[idx], idx, labels[idx]


def main():
    print("[analyze] loading references...")
    refs_full = np.load(DATA / "references.npy").astype(np.float32)  # (3M, 14)
    labels_full = np.load(DATA / "labels.npy")                       # bool
    box = np.load(DATA / "box_labels.npy")                           # u8 0/1/2
    counts_k25 = np.load(DATA / "fraud_counts_k25.npy")              # u8 0..25

    # Build Box-B subset (the same one our backend's slow path searches).
    B_idx = np.where(box == 2)[0]
    refs_b = refs_full[B_idx]
    labels_b = labels_full[B_idx]
    box_name = {0: "A-Legit", 1: "A-Fraud", 2: "B"}
    print(f"[analyze] full: {refs_full.shape[0]:,} refs   "
          f"box-B: {refs_b.shape[0]:,} refs")

    mismatches = json.loads((DATA / "mismatches.json").read_text())
    print(f"[analyze] {len(mismatches)} mismatches to diagnose\n")

    for m in mismatches:
        idx = m["index"]
        expected = m["expected_approved"]
        backend_approved = m["actual_approved"]
        backend_score = m["actual_fraud_score"]
        backend_count = int(round(backend_score * 5))

        v = vectorize(m["request"])

        # Full 3M ground truth
        d_full, idx_full, lab_full = topk_brute(v, refs_full, labels_full, k=5)
        count_full = int(lab_full.sum())
        gt_approved = count_full < 3

        # Brute over Box-B
        d_b, idx_b_local, lab_b = topk_brute(v, refs_b, labels_b, k=5)
        idx_b_global = B_idx[idx_b_local]   # global index back into full refs
        count_b = int(lab_b.sum())
        b_approved = count_b < 3

        # How many of the true top-5 are in Box-B?
        full_in_b = sum(int(i in set(idx_b_global)) for i in idx_full)

        # Are the 5 NN selected by Box-B in the *real* top-5 of full?
        b_in_full_topk = sum(int(i in set(idx_full)) for i in idx_b_global)

        # Distance of Box-B's 5th NN vs full's 5th NN
        d_full_max = d_full.max()
        d_b_max = d_b.max()

        print(f"=== index {idx}  (expected_approved={expected}) ===")
        print(f"  request id: {m['request']['id']}  "
              f"mcc={m['request']['merchant']['mcc']}  "
              f"amount={m['request']['transaction']['amount']:.2f}")
        print(f"  backend:  approved={backend_approved}  "
              f"score={backend_score}  (count={backend_count})")
        print(f"  brute over Box-B (k=5): count={count_b}  approved={b_approved}  "
              f"d_max={d_b_max:.5f}")
        print(f"     ↑ this matches our backend ✓" if b_approved == backend_approved
              else f"     ↑ ⚠ DOES NOT match backend — IVF approximation error")
        print(f"  brute over full 3M (k=5): count={count_full}  approved={gt_approved}  "
              f"d_max={d_full_max:.5f}")
        print(f"     ↑ this matches expected ✓" if gt_approved == expected
              else f"     ↑ ⚠ DOES NOT match expected — vectorize mismatch?")
        print(f"  overlap of Box-B's top-5 with full's top-5: {b_in_full_topk}/5")
        print(f"  of full's top-5, {full_in_b}/5 are in Box-B at all")

        # Per-NN breakdown: for each of the full-3M top-5, where is it and why.
        print(f"  full-3M top-5 breakdown:")
        b_set = set(idx_b_global.tolist())
        for rank, (d, gi, lab) in enumerate(zip(d_full, idx_full, lab_full)):
            in_b = gi in b_set
            b_lbl = box_name[int(box[gi])]
            ck25 = int(counts_k25[gi])
            mark = "in Box-B" if in_b else f"NOT in Box-B (assigned {b_lbl})"
            print(f"     {rank+1}. ref#{gi:>7}  d={d:.5f}  label={int(lab)}  "
                  f"k25_fraud_count={ck25:>2}/25  → {mark}")
        print()


if __name__ == "__main__":
    main()

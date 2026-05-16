"""
Step 5 — Validate that k=5 over a candidate slow-path subset gives the same
binary "deny / approve" verdict as k=5 over the full 3M references, on the
test split.

By default the subset is Box B (~105k refs). Set SUBSET_FILE=<path> to point
at a different subset (e.g. halo_indices_m5.npy from halo.py).

If agreement is high on the queries the router actually sends to slow path,
the production slow path can use just the candidate subset instead of
materializing the full 3M (168 MB, memory-bound at ~14 ms per query).

Setup:
    subset:        either Box B (default) or SUBSET_FILE
    queries:       references in router test split  (~300k held-out)
    ground truth:  fraud_counts_k5.npy   (leave-one-out k=5 over 3M)
    measured:      counts_subset         (leave-one-out k=5 over subset)

Reports:
    1. Overall binary + exact agreement.
    2. 6x6 confusion matrix of c_subset vs c_full.
    3. Per-tau agreement on queries the router would slow-route.

Outputs:
    data/validate_subset_results_<name>.json   where name is "BoxB" or the
                                               SUBSET_FILE stem
    stdout report
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from train_router import Router

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

B = 2


def main():
    refs = np.load(DATA_DIR / "references.npy")
    labels = np.load(DATA_DIR / "labels.npy")
    box = np.load(DATA_DIR / "box_labels.npy")
    counts_k5_full = np.load(DATA_DIR / "fraud_counts_k5.npy")
    test_idx = np.load(DATA_DIR / "router_test_indices.npy")
    N = len(refs)

    subset_file = os.environ.get("SUBSET_FILE")
    if subset_file:
        subset_idx = np.load(subset_file).astype(np.int64)
        subset_name = Path(subset_file).stem
    else:
        subset_idx = np.where(box == B)[0].astype(np.int64)
        subset_name = "BoxB"
    M = len(subset_idx)
    subset_fraud_labels = labels[subset_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[validate] device: {device}")
    print(f"[validate] subset: {subset_name}  size: {M:,} "
          f"({100*M/N:.2f}% of refs, {M*14*4/1e6:.2f} MB at fp32)")
    print(f"[validate] test queries: {len(test_idx):,}")
    print()

    R_subset = torch.from_numpy(refs[subset_idx]).to(device)         # (M, 14)
    L_subset = torch.from_numpy(subset_fraud_labels).to(device).to(torch.uint8)
    R_sq = (R_subset ** 2).sum(dim=1)

    pos_in_subset = np.full(N, -1, dtype=np.int64)
    pos_in_subset[subset_idx] = np.arange(M)
    pos_in_subset_t = torch.from_numpy(pos_in_subset).to(device)

    test_idx_t = torch.from_numpy(test_idx.astype(np.int64)).to(device)
    Q = torch.from_numpy(refs[test_idx]).to(device)                  # (T, 14)
    T = Q.shape[0]

    batch = int(os.environ.get("BATCH", 2048))
    counts_sub = torch.empty(T, dtype=torch.uint8, device=device)
    t0 = time.time()
    last_log = t0
    for start in range(0, T, batch):
        end = min(start + batch, T)
        Bb = end - start
        q = Q[start:end]
        q_sq = (q ** 2).sum(dim=1, keepdim=True)
        d = q_sq + R_sq.unsqueeze(0) - 2.0 * (q @ R_subset.T)         # (Bb, M)

        # Self-exclusion: if a test query is itself in the subset, mask its
        # own position so we get leave-one-out behavior (matches how
        # fraud_counts_k5.npy was produced).
        batch_global_ref = test_idx_t[start:end]
        batch_pos = pos_in_subset_t[batch_global_ref]                 # (Bb,)
        in_subset_mask = batch_pos >= 0
        if in_subset_mask.any():
            rows = torch.arange(Bb, device=device)[in_subset_mask]
            cols = batch_pos[in_subset_mask]
            d[rows, cols] = float("inf")

        _, top5 = torch.topk(d, 5, dim=1, largest=False)
        counts_sub[start:end] = L_subset[top5].sum(dim=1).to(torch.uint8)

        now = time.time()
        if now - last_log > 3.0 or end == T:
            elapsed = now - t0
            rate = end / max(elapsed, 1e-3)
            print(f"[validate] {end:>7,}/{T:,} ({100*end/T:5.1f}%)  "
                  f"{rate:>6.0f} q/s  elapsed={elapsed:.1f}s")
            last_log = now

    counts_sub_np = counts_sub.cpu().numpy()
    print()

    c_full = counts_k5_full[test_idx]                                 # uint8
    c_sub = counts_sub_np                                             # uint8
    v_full = c_full >= 3
    v_sub = c_sub >= 3

    bin_agree = float((v_full == v_sub).mean())
    exact_agree = float((c_full == c_sub).mean())
    n_dis_bin = int((v_full != v_sub).sum())

    print(f"[validate] OVERALL agreement on {T:,} test queries:")
    print(f"           binary verdict (count >= 3):  {100*bin_agree:.4f}%")
    print(f"           exact k=5 count:              {100*exact_agree:.4f}%")
    print(f"           binary disagreements:         {n_dis_bin:,} / {T:,}")
    print()

    flat_idx = c_full.astype(np.int64) * 6 + c_sub.astype(np.int64)
    cm = np.bincount(flat_idx, minlength=36).reshape(6, 6)
    print(f"[validate] 6x6 confusion matrix "
          f"(rows = c_full from 3M, cols = c_subset from Box B):")
    print(f"{'c_full \\ c_sub':>15}" + "".join(f"{i:>10}" for i in range(6))
          + f"{'row total':>12}")
    print("-" * 85)
    for i in range(6):
        row_total = int(cm[i].sum())
        if row_total == 0:
            continue
        cells = "".join(f"{int(cm[i,j]):>10,}" for j in range(6))
        print(f"{i:>15}" + cells + f"{row_total:>12,}")
    print()

    ckpt = torch.load(DATA_DIR / "router.pt", weights_only=True)
    model = Router(hidden=ckpt["hidden"], depth=ckpt["depth"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Q), dim=1).cpu().numpy()
    p_B = probs[:, B]

    print(f"[validate] agreement on queries the router would slow-route "
          f"(P(B) > tau):")
    print(f"{'tau':>5} {'slow N':>10} {'slow %':>8} {'bin agree':>11} "
          f"{'bin disagree':>14} {'exact agree':>12}")
    print("-" * 68)
    by_tau_rows = []
    for tau in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]:
        slow_mask = p_B > tau
        n_slow = int(slow_mask.sum())
        if n_slow == 0:
            by_tau_rows.append({"tau": tau, "n_slow": 0, "binary_agreement": None,
                                "n_binary_disagree": 0, "exact_agreement": None})
            continue
        a_bin = float((v_full[slow_mask] == v_sub[slow_mask]).mean())
        nd = int((v_full[slow_mask] != v_sub[slow_mask]).sum())
        a_exact = float((c_full[slow_mask] == c_sub[slow_mask]).mean())
        by_tau_rows.append({"tau": tau, "n_slow": n_slow, "slow_rate": n_slow/T,
                            "binary_agreement": a_bin, "n_binary_disagree": nd,
                            "exact_agreement": a_exact})
        print(f"{tau:>5.2f} {n_slow:>10,} {100*n_slow/T:>7.2f}% "
              f"{100*a_bin:>10.4f}% {nd:>14,} {100*a_exact:>11.4f}%")
    print()

    out = {
        "subset_name": subset_name,
        "subset_size": int(M),
        "subset_pct_of_refs": float(M / N),
        "subset_bytes_fp32": int(M * 14 * 4),
        "n_test": int(T),
        "overall_binary_agreement": bin_agree,
        "overall_exact_agreement": exact_agree,
        "overall_binary_disagree_count": n_dis_bin,
        "confusion_matrix": cm.tolist(),
        "by_tau": by_tau_rows,
        "notes": [
            f"Subset source: {subset_name}.",
            "Self-exclusion is applied for queries that are themselves in",
            "  the subset, matching how fraud_counts_k5.npy was built.",
            "c_full comes from fraud_counts_k5.npy (k=5 over full 3M, LOO).",
            "c_sub is k=5 over the candidate subset only, computed here.",
            "Binary verdict = (count >= 3), the rinha 0.6 threshold.",
        ],
    }
    out_file = DATA_DIR / f"validate_subset_results_{subset_name}.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"[validate] wrote {out_file}")


if __name__ == "__main__":
    main()

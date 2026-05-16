"""
sample_subset.py — Build the slow-path subset by FREQUENCY-WEIGHTED sampling.

Refinement over the unweighted version:
  Box-A refs that appear as 5-NN of *threshold queries* (true count in {2, 3})
  get weight 1.0; refs appearing only in deep-cluster queries (count in
  {0, 1, 4, 5}) get weight 0.1. This biases the ranking toward refs whose
  inclusion/exclusion actually changes the rinha binary verdict (count >= 3).

Algorithm:
1. Generate N random query vectors (uniform [0,1]^14, -1 sentinels in 20%).
2. Run the router; keep only queries with P(B) > TAU.
3. For each kept query, find its TRUE 5-NN over the FULL 3M references.
4. Compute the true k=5 count per query. Set w = 1.0 if count in {2,3}
   else 0.1.
5. For each 5-NN slot that is a Box-A ref, accumulate the query's weight
   into a per-ref float counter.
6. Sort Box-A refs by counter descending — most-binary-decision-critical first.
7. Sweep subset sizes; on a held-out batch, measure binary agreement
   against the 3M ground truth.
8. Print an ASCII visualization of the curve.

Env vars:
   N_SAMPLES   queries for the counter stage      (default 2_000_000)
   N_VALIDATE  held-out queries for the curve     (default 100_000)
   SEED        sample RNG seed                    (default 42)
   TAU         P(B) threshold for routing         (default 0.5)
   TARGET      target binary agreement            (default 0.99)
   W_HIGH      weight when count in {2, 3}        (default 1.0)
   W_LOW       weight otherwise                   (default 0.1)

Outputs:
   data/sampled_a_sorted.npy        Box-A ref indices sorted by weighted score (desc)
   data/sampled_counter.npy         per-ref float counter (full 3M-length)
   data/best_subset_indices.npy     smallest subset hitting TARGET (if reached)
   data/sampling_subset_curve.json  curve + meta-info
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ivf import brute_count_k5
from train_router import Router

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def brute_5nn_indices(queries, refs, k=5, batch=2048, chunk=16384):
    """Exact k-NN brute force on GPU. Returns (T, k) index tensor into refs."""
    device = refs.device
    dtype = refs.dtype
    T = queries.shape[0]
    N = refs.shape[0]
    R_sq = (refs ** 2).sum(dim=1)
    R_T = refs.T.contiguous()
    out = torch.empty((T, k), dtype=torch.long, device=device)
    INF = float("inf")
    for start in range(0, T, batch):
        end = min(start + batch, T)
        B = end - start
        q = queries[start:end]
        best_d = torch.full((B, k), INF, dtype=dtype, device=device)
        best_idx = torch.full((B, k), -1, dtype=torch.long, device=device)
        for j_start in range(0, N, chunk):
            j_end = min(j_start + chunk, N)
            d = torch.addmm(R_sq[j_start:j_end].unsqueeze(0),
                            q, R_T[:, j_start:j_end],
                            beta=1.0, alpha=-2.0)
            chunk_d, chunk_local = torch.topk(d, k, dim=1, largest=False)
            cand_d = torch.cat([best_d, chunk_d], dim=1)
            cand_idx = torch.cat([best_idx, chunk_local + j_start], dim=1)
            best_d, sel = torch.topk(cand_d, k, dim=1, largest=False)
            best_idx = torch.gather(cand_idx, 1, sel)
        out[start:end] = best_idx
    return out


def gen_uniform(N, rng):
    Q = rng.uniform(0, 1, size=(N, 14)).astype(np.float32)
    null_mask = rng.uniform(0, 1, size=N) < 0.2
    Q[null_mask, 5] = -1.0
    Q[null_mask, 6] = -1.0
    return Q


def router_pB(model, Q_gpu_fp32, chunk=65536):
    out = []
    with torch.no_grad():
        for s in range(0, Q_gpu_fp32.shape[0], chunk):
            e = min(s + chunk, Q_gpu_fp32.shape[0])
            probs = torch.softmax(model(Q_gpu_fp32[s:e]), dim=1)
            out.append(probs[:, 2])
    return torch.cat(out)


def ascii_curve(curve, target):
    """ASCII line chart of binary + exact agreement vs subset size."""
    width = 56
    lo, hi = 85.0, 100.0  # display window in %
    print()
    print(f"  ASCII curve (binary in █, exact in ▒, scale {lo:.0f}-{hi:.0f}%):")
    print(f"  {'+a_refs':>9}  {'subset':>9}  [" + " " * width + f"]  binary  exact")
    for r in curve:
        n = r["n_added_a"]
        s = r["subset_size"]
        b = 100.0 * r["binary_agreement"]
        e = 100.0 * r["exact_agreement"]
        bp = max(0, min(width, int(round(width * (b - lo) / (hi - lo)))))
        ep = max(0, min(width, int(round(width * (e - lo) / (hi - lo)))))
        bar = list("·" * width)
        # exact agreement (lighter shade) first, binary overwrites
        for i in range(ep):
            bar[i] = "▒"
        for i in range(bp):
            bar[i] = "█"
        # target threshold line
        tp = max(0, min(width - 1, int(round(width * (100*target - lo) / (hi - lo)))))
        if bar[tp] == "·":
            bar[tp] = "│"
        bar_s = "".join(bar)
        marker = " ✓" if b >= 100 * target else ""
        print(f"  {n:>9,}  {s:>9,}  [{bar_s}]  {b:>5.2f}% {e:>5.2f}%{marker}")
    # legend
    print(f"  legend: █ binary agreement | ▒ exact agreement | │ target {100*target:.1f}%")


def ascii_marginal_gain(curve):
    """ASCII bar chart of marginal gain in binary agreement per A refs added."""
    print()
    print("  Marginal gain (Δbinary % per Δ1k A refs added):")
    width = 50
    deltas = []
    for i in range(1, len(curve)):
        prev, cur = curve[i - 1], curve[i]
        dn = cur["n_added_a"] - prev["n_added_a"]
        if dn == 0:
            continue
        db = 100.0 * (cur["binary_agreement"] - prev["binary_agreement"])
        gain_per_1k = db / (dn / 1000.0)
        deltas.append((prev["n_added_a"], cur["n_added_a"], gain_per_1k))
    if not deltas:
        return
    max_gain = max(abs(d[2]) for d in deltas) or 1.0
    for a_from, a_to, gain in deltas:
        bar_len = int(round(width * abs(gain) / max_gain))
        bar = "█" * bar_len
        sign = "+" if gain >= 0 else "-"
        print(f"  {a_from:>8,} → {a_to:<8,}  {sign}{abs(gain):>6.3f} pp/1k  {bar}")


def main():
    n_samples = int(os.environ.get("N_SAMPLES", 2_000_000))
    n_validate = int(os.environ.get("N_VALIDATE", 100_000))
    seed = int(os.environ.get("SEED", 42))
    tau = float(os.environ.get("TAU", 0.5))
    target = float(os.environ.get("TARGET", 0.99))
    w_high = float(os.environ.get("W_HIGH", 1.0))
    w_low = float(os.environ.get("W_LOW", 0.1))

    refs_np = np.load(DATA / "references.npy")
    labels_np = np.load(DATA / "labels.npy")
    box_np = np.load(DATA / "box_labels.npy")
    box_b_idx = np.where(box_np == 2)[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    print(f"[sample] device: {device}")
    print(f"[sample] N_SAMPLES={n_samples:,}  N_VALIDATE={n_validate:,}  "
          f"TAU={tau}  TARGET={target}  weights=(hi={w_high}, lo={w_low})")

    refs = torch.from_numpy(refs_np).to(device=device, dtype=dtype)
    labels = torch.from_numpy(labels_np).to(device).to(torch.uint8)
    box_gpu = torch.from_numpy(box_np).to(device)

    ckpt = torch.load(DATA / "router.pt", weights_only=True)
    model = Router(hidden=ckpt["hidden"], depth=ckpt["depth"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ---------- Stage 1: weighted counter ----------
    rng = np.random.default_rng(seed)
    Q_all_np = gen_uniform(n_samples, rng)
    Q_all_fp32 = torch.from_numpy(Q_all_np).to(device=device, dtype=torch.float32)

    print(f"\n[stage 1] router-filtering {n_samples:,} random queries")
    p_B = router_pB(model, Q_all_fp32)
    is_b = (p_B > tau).cpu().numpy()
    n_b = int(is_b.sum())
    print(f"[stage 1] router → Box B: {n_b:,}/{n_samples:,} ({100*n_b/n_samples:.2f}%)")
    if n_b == 0:
        raise SystemExit("[sample] no queries routed to Box B")

    Q_b = torch.from_numpy(Q_all_np[is_b]).to(device=device, dtype=dtype)

    print(f"[stage 1] computing TRUE 5-NN over 3M for {n_b:,} queries")
    t0 = time.time()
    nn = brute_5nn_indices(Q_b, refs, k=5)              # (n_b, 5)
    elapsed = time.time() - t0
    print(f"[stage 1] done in {elapsed:.1f}s ({n_b/elapsed:.0f} q/s)")

    # True 5-NN count per query (over 3M)
    true_counts = labels[nn].sum(dim=1).to(torch.long)  # (n_b,)
    counts_dist = torch.bincount(true_counts, minlength=6).cpu().numpy()
    print(f"[stage 1] true 5-NN count distribution among routed-B queries:")
    for c in range(6):
        pct = 100 * counts_dist[c] / n_b
        bar = "█" * int(pct / 2)
        print(f"           c={c}: {counts_dist[c]:>10,} ({pct:>5.2f}%) {bar}")
    n_threshold = int(((true_counts == 2) | (true_counts == 3)).sum().item())
    print(f"[stage 1] queries near threshold (c ∈ {{2,3}}): "
          f"{n_threshold:,} ({100*n_threshold/n_b:.2f}%)")
    print(f"[stage 1] applying weights: w={w_high} for those, w={w_low} otherwise")

    # Per-query weight, then broadcast to per-slot, then keep only A slots
    weights_per_query = torch.where(
        (true_counts == 2) | (true_counts == 3),
        torch.tensor(w_high, device=device),
        torch.tensor(w_low, device=device),
    ).to(torch.float32)
    weights_per_slot = weights_per_query.repeat_interleave(5)    # (n_b * 5,)

    nn_flat = nn.flatten()
    nn_box = box_gpu[nn_flat]
    is_a = nn_box != 2
    a_indices = nn_flat[is_a]
    weights_a = weights_per_slot[is_a]
    print(f"[stage 1] Box-A neighbor slots: {len(a_indices):,} / {5*n_b:,} "
          f"({100*len(a_indices)/(5*n_b):.2f}%)")

    counter = torch.zeros(len(refs_np), dtype=torch.float32, device=device)
    counter.scatter_add_(0, a_indices, weights_a)
    counter_np = counter.cpu().numpy()
    n_unique_a = int((counter_np > 0).sum())
    max_score = float(counter_np.max()) if n_unique_a > 0 else 0.0
    print(f"[stage 1] unique Box-A refs touched: {n_unique_a:,}")
    print(f"[stage 1] max weighted score for a single A ref: {max_score:.2f}")

    a_idx_all = np.where(box_np != 2)[0]
    a_scores = counter_np[a_idx_all]
    order = np.argsort(-a_scores)
    a_sorted = a_idx_all[order]
    nonzero_mask = a_scores[order] > 0
    a_sorted_nonzero = a_sorted[nonzero_mask]
    print(f"[stage 1] A refs candidates with score > 0: {len(a_sorted_nonzero):,}")

    np.save(DATA / "sampled_a_sorted.npy", a_sorted_nonzero)
    np.save(DATA / "sampled_counter.npy", counter_np)

    # ---------- Stage 2: validation curve ----------
    print(f"\n[stage 2] generating {n_validate:,} held-out queries (seed={seed+1000})")
    rng_v = np.random.default_rng(seed + 1000)
    Q_val_np = gen_uniform(n_validate, rng_v)
    Q_val_fp32 = torch.from_numpy(Q_val_np).to(device=device, dtype=torch.float32)
    p_B_val = router_pB(model, Q_val_fp32)
    is_b_val = (p_B_val > tau).cpu().numpy()
    n_b_val = int(is_b_val.sum())
    print(f"[stage 2] held-out Box B queries: {n_b_val:,}")
    Q_val_b = torch.from_numpy(Q_val_np[is_b_val]).to(device=device, dtype=dtype)

    print(f"[stage 2] computing 3M ground-truth 5-NN counts for validation")
    truth_counts = brute_count_k5(Q_val_b, refs, labels, k=5).cpu().numpy()

    # Denser size sweep — capture both the steep climb and the asymptote
    raw_sizes = [
        0, 100, 200, 300, 500, 750,
        1_000, 1_500, 2_000, 3_000, 5_000, 7_500,
        10_000, 15_000, 20_000, 30_000, 50_000, 75_000,
        100_000, 150_000, 200_000, 300_000,
    ]
    sizes = sorted(set(min(s, len(a_sorted_nonzero)) for s in raw_sizes))
    if sizes[-1] < len(a_sorted_nonzero):
        sizes.append(len(a_sorted_nonzero))

    print(f"\n[stage 2] subset agreement curve:")
    print(f"  {'+a_refs':>10} {'subset':>10} {'MB(fp32)':>10} "
          f"{'binary':>10} {'exact':>10}")
    curve = []
    best_subset = None
    best_idx_for_save = None
    for n_add in sizes:
        subset_idx = np.concatenate([box_b_idx, a_sorted_nonzero[:n_add]])
        subset_t = torch.from_numpy(subset_idx.astype(np.int64)).to(device)
        sub_refs = refs[subset_t]
        sub_labels = labels[subset_t]
        sub_counts = brute_count_k5(Q_val_b, sub_refs, sub_labels, k=5).cpu().numpy()
        bin_agree = float(((sub_counts >= 3) == (truth_counts >= 3)).mean())
        exact_agree = float((sub_counts == truth_counts).mean())
        size_mb = len(subset_idx) * 14 * 4 / 1e6

        marker = ""
        if bin_agree >= target and best_subset is None:
            best_subset = len(subset_idx)
            best_idx_for_save = subset_idx
            marker = "  <-- first hit target"

        print(f"  {n_add:>10,} {len(subset_idx):>10,} {size_mb:>9.2f}  "
              f"{100*bin_agree:>9.4f}% {100*exact_agree:>9.4f}%{marker}")
        curve.append({
            "n_added_a": int(n_add),
            "subset_size": int(len(subset_idx)),
            "subset_size_mb_fp32": float(size_mb),
            "binary_agreement": bin_agree,
            "exact_agreement": exact_agree,
        })

    if best_idx_for_save is not None:
        np.save(DATA / "best_subset_indices.npy", best_idx_for_save)
        print(f"\n[stage 2] saved data/best_subset_indices.npy "
              f"({len(best_idx_for_save):,} entries)")
    else:
        last = curve[-1]
        print(f"\n[stage 2] target {100*target}% NOT reached; "
              f"max bin agreement {100*last['binary_agreement']:.4f}%")

    # ASCII curve visualization
    ascii_curve(curve, target)
    ascii_marginal_gain(curve)

    out = {
        "n_samples": n_samples,
        "n_validate": n_validate,
        "tau": tau,
        "target_agreement": target,
        "weights": {"high": w_high, "low": w_low},
        "n_box_b_routed_train": n_b,
        "n_box_b_routed_validate": n_b_val,
        "n_unique_a_touched": n_unique_a,
        "max_single_a_weighted_score": max_score,
        "true_count_distribution_train": counts_dist.tolist(),
        "curve": curve,
        "best_subset_size": best_subset,
    }
    (DATA / "sampling_subset_curve.json").write_text(json.dumps(out, indent=2))
    print(f"\n[stage 2] wrote data/sampling_subset_curve.json")


if __name__ == "__main__":
    main()

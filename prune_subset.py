"""
prune_subset.py — Inverse instance reduction. Build the slow-path subset by
PRUNING from the full 3M, rather than growing it from Box B.

Same sampling methodology as sample_subset.py but tracks ALL refs (Box A AND
Box B). No forced Box B inclusion — let the data decide what to keep.

The interesting question: can we get a SMALLER subset than the 105k Box B
itself by dropping rarely-used BORDERLINE refs? If yes, slow-path working
set shrinks below 6 MB and may fit in Mac Mini's 3-4 MB L3.

Algorithm:
1. Generate N random queries; filter via router (P(B) > TAU).
2. For each kept query, find TRUE 5-NN over 3M.
3. Per-query weight: 1.0 if true k=5 count ∈ {2, 3} else 0.1.
4. For each 5-NN slot, accumulate weight into a per-ref counter (ALL refs).
5. Sort all 3M refs by counter descending.
6. Sweep top-N subset sizes; on a held-out batch measure binary agreement
   against the 3M ground truth.

Env vars:
   N_SAMPLES   queries for the counter stage     (default 2_000_000)
   N_VALIDATE  held-out queries for the curve    (default 100_000)
   SEED        sample RNG seed                   (default 42)
   TAU         P(B) threshold                    (default 0.5)
   TARGET      target binary agreement           (default 0.99)
   W_HIGH      weight when count ∈ {2, 3}        (default 1.0)
   W_LOW       otherwise                         (default 0.1)

Outputs:
   data/pruned_sorted.npy        ref indices sorted by weighted usage (desc)
   data/pruned_counter.npy       per-ref float counter (full 3M-length)
   data/pruned_curve.json        curve + meta-info
   data/best_pruned_indices.npy  smallest subset hitting TARGET (if reached)
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ivf import brute_count_k5
from sample_subset import brute_5nn_indices, gen_uniform, router_pB
from train_router import Router

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def ascii_curve(curve, target):
    """ASCII line chart of binary + exact agreement vs subset size."""
    width = 56
    lo, hi = 50.0, 100.0  # wider window since pruning starts from small subsets
    print()
    print(f"  ASCII curve (binary in █, exact in ▒, scale {lo:.0f}-{hi:.0f}%):")
    print(f"  {'top-N':>10}  {'MB':>7}  [" + " " * width + f"]  binary  exact")
    for r in curve:
        n = r["subset_size"]
        mb = r["subset_size_mb_fp32"]
        b = 100.0 * r["binary_agreement"]
        e = 100.0 * r["exact_agreement"]
        bp = max(0, min(width, int(round(width * (b - lo) / (hi - lo)))))
        ep = max(0, min(width, int(round(width * (e - lo) / (hi - lo)))))
        bar = list("·" * width)
        for i in range(ep):
            bar[i] = "▒"
        for i in range(bp):
            bar[i] = "█"
        tp = max(0, min(width - 1, int(round(width * (100 * target - lo) / (hi - lo)))))
        if bar[tp] == "·":
            bar[tp] = "│"
        marker = " ✓" if b >= 100 * target else ""
        print(f"  {n:>10,}  {mb:>6.2f}M  [{''.join(bar)}]  {b:>5.2f}% {e:>5.2f}%{marker}")
    print(f"  legend: █ binary | ▒ exact | │ target {100*target:.1f}%")


def ascii_marginal_gain(curve):
    print()
    print("  Marginal gain (Δbinary % per Δ1k refs added):")
    width = 50
    deltas = []
    for i in range(1, len(curve)):
        prev, cur = curve[i - 1], curve[i]
        dn = cur["subset_size"] - prev["subset_size"]
        if dn == 0:
            continue
        db = 100.0 * (cur["binary_agreement"] - prev["binary_agreement"])
        gain_per_1k = db / (dn / 1000.0)
        deltas.append((prev["subset_size"], cur["subset_size"], gain_per_1k))
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
    N_total = len(refs_np)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    print(f"[prune] device: {device}")
    print(f"[prune] N_SAMPLES={n_samples:,}  N_VALIDATE={n_validate:,}  "
          f"TAU={tau}  TARGET={target}  weights=(hi={w_high}, lo={w_low})")

    refs = torch.from_numpy(refs_np).to(device=device, dtype=dtype)
    labels = torch.from_numpy(labels_np).to(device).to(torch.uint8)

    ckpt = torch.load(DATA / "router.pt", weights_only=True)
    model = Router(hidden=ckpt["hidden"], depth=ckpt["depth"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ---------- Stage 1: per-ref usage counter (ALL refs) ----------
    rng = np.random.default_rng(seed)
    Q_all_np = gen_uniform(n_samples, rng)
    Q_all_fp32 = torch.from_numpy(Q_all_np).to(device=device, dtype=torch.float32)

    print(f"\n[stage 1] router-filtering {n_samples:,} random queries")
    p_B = router_pB(model, Q_all_fp32)
    is_b = (p_B > tau).cpu().numpy()
    n_b = int(is_b.sum())
    print(f"[stage 1] router → Box B: {n_b:,}/{n_samples:,} ({100*n_b/n_samples:.2f}%)")

    Q_b = torch.from_numpy(Q_all_np[is_b]).to(device=device, dtype=dtype)
    print(f"[stage 1] computing TRUE 5-NN over 3M for {n_b:,} queries")
    t0 = time.time()
    nn = brute_5nn_indices(Q_b, refs, k=5)
    elapsed = time.time() - t0
    print(f"[stage 1] done in {elapsed:.1f}s ({n_b/elapsed:.0f} q/s)")

    # Per-query weights from true count
    true_counts = labels[nn].sum(dim=1).to(torch.long)
    counts_dist = torch.bincount(true_counts, minlength=6).cpu().numpy()
    print(f"[stage 1] true count distribution among routed-B queries:")
    for c in range(6):
        pct = 100 * counts_dist[c] / n_b
        print(f"           c={c}: {counts_dist[c]:>10,} ({pct:>5.2f}%)")
    weights_per_query = torch.where(
        (true_counts == 2) | (true_counts == 3),
        torch.tensor(w_high, device=device),
        torch.tensor(w_low, device=device),
    ).to(torch.float32)
    weights_per_slot = weights_per_query.repeat_interleave(5)

    # Accumulate for ALL refs (no A-only filter)
    nn_flat = nn.flatten()
    counter = torch.zeros(N_total, dtype=torch.float32, device=device)
    counter.scatter_add_(0, nn_flat, weights_per_slot)
    counter_np = counter.cpu().numpy()
    n_touched = int((counter_np > 0).sum())
    print(f"[stage 1] unique refs touched as 5-NN: {n_touched:,} / {N_total:,} "
          f"({100*n_touched/N_total:.2f}%)")

    # Breakdown of touched refs by box class
    touched_mask = counter_np > 0
    touched_box = box_np[touched_mask]
    touched_per_box = np.bincount(touched_box, minlength=3)
    print(f"[stage 1] touched ref breakdown:")
    print(f"           Box A-Legit: {touched_per_box[0]:>10,}")
    print(f"           Box A-Fraud: {touched_per_box[1]:>10,}")
    print(f"           Box B:       {touched_per_box[2]:>10,}")

    untouched_b = int(np.sum((box_np == 2) & (counter_np == 0)))
    print(f"[stage 1] Box B refs NEVER touched: {untouched_b:,} / "
          f"{int((box_np == 2).sum()):,} "
          f"({100*untouched_b/int((box_np == 2).sum()):.2f}%)  ← prunable")

    # Sort all refs by usage descending
    order = np.argsort(-counter_np)
    sorted_indices = order[counter_np[order] > 0]
    print(f"[stage 1] refs candidates (count > 0): {len(sorted_indices):,}")
    np.save(DATA / "pruned_sorted.npy", sorted_indices)
    np.save(DATA / "pruned_counter.npy", counter_np)

    # ---------- Stage 2: top-N agreement curve ----------
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

    raw_sizes = [
        10_000, 20_000, 30_000, 50_000, 75_000,
        100_000, 105_504,  # match Box B alone for direct comparison
        110_000, 120_000, 150_000, 200_000, 250_000, 300_000,
    ]
    sizes = sorted(set(min(s, len(sorted_indices)) for s in raw_sizes))
    if sizes[-1] < len(sorted_indices):
        sizes.append(len(sorted_indices))

    print(f"\n[stage 2] top-N agreement curve:")
    print(f"  {'top_N':>10} {'MB':>9} {'binary':>10} {'exact':>10}")
    curve = []
    best_subset_size = None
    best_idx_for_save = None
    for n_keep in sizes:
        subset_idx = sorted_indices[:n_keep]
        subset_t = torch.from_numpy(subset_idx.astype(np.int64)).to(device)
        sub_refs = refs[subset_t]
        sub_labels = labels[subset_t]
        sub_counts = brute_count_k5(Q_val_b, sub_refs, sub_labels, k=5).cpu().numpy()
        bin_agree = float(((sub_counts >= 3) == (truth_counts >= 3)).mean())
        exact_agree = float((sub_counts == truth_counts).mean())
        size_mb = len(subset_idx) * 14 * 4 / 1e6

        marker = ""
        if bin_agree >= target and best_subset_size is None:
            best_subset_size = len(subset_idx)
            best_idx_for_save = subset_idx
            marker = "  <-- first hit"

        print(f"  {n_keep:>10,} {size_mb:>8.2f}M  "
              f"{100*bin_agree:>9.4f}% {100*exact_agree:>9.4f}%{marker}")
        curve.append({
            "subset_size": int(len(subset_idx)),
            "subset_size_mb_fp32": float(size_mb),
            "binary_agreement": bin_agree,
            "exact_agreement": exact_agree,
        })

    if best_idx_for_save is not None:
        np.save(DATA / "best_pruned_indices.npy", best_idx_for_save)
        print(f"\n[stage 2] saved data/best_pruned_indices.npy "
              f"({len(best_idx_for_save):,} entries, "
              f"{len(best_idx_for_save)*14*4/1e6:.2f} MB)")
    else:
        last = curve[-1]
        print(f"\n[stage 2] target {100*target}% NOT reached; "
              f"max bin agreement {100*last['binary_agreement']:.4f}%")

    ascii_curve(curve, target)
    ascii_marginal_gain(curve)

    # Compare: at subset_size == 105_504 (Box B size), how does pure top-N
    # compare to Box B itself?
    box_b_alone_idx = np.where(box_np == 2)[0]
    box_b_alone_t = torch.from_numpy(box_b_alone_idx.astype(np.int64)).to(device)
    sub_refs = refs[box_b_alone_t]
    sub_labels = labels[box_b_alone_t]
    box_b_counts = brute_count_k5(Q_val_b, sub_refs, sub_labels, k=5).cpu().numpy()
    box_b_bin = float(((box_b_counts >= 3) == (truth_counts >= 3)).mean())

    near_105k = next((r for r in curve if r["subset_size"] >= 105_000 and r["subset_size"] <= 110_000), None)
    print()
    print(f"  Direct comparison at ~105k refs:")
    print(f"    Box B (forced):          {100*box_b_bin:.4f}% binary")
    if near_105k:
        print(f"    Pure top-N (sampled):    {100*near_105k['binary_agreement']:.4f}% binary  "
              f"(size {near_105k['subset_size']:,})")

    out = {
        "n_samples": n_samples,
        "n_validate": n_validate,
        "tau": tau,
        "target_agreement": target,
        "weights": {"high": w_high, "low": w_low},
        "n_box_b_routed_train": n_b,
        "n_box_b_routed_validate": n_b_val,
        "n_refs_touched": n_touched,
        "n_box_b_untouched": int(untouched_b),
        "touched_by_box": touched_per_box.tolist(),
        "true_count_distribution_train": counts_dist.tolist(),
        "curve": curve,
        "box_b_baseline_binary": box_b_bin,
        "best_pruned_size": best_subset_size,
    }
    (DATA / "pruned_curve.json").write_text(json.dumps(out, indent=2))
    print(f"\n[stage 2] wrote data/pruned_curve.json")


if __name__ == "__main__":
    main()

"""
For every one of the 3M references, compute the distance to its nearest
*opposite-label* reference (legit→fraud or fraud→legit). This is the
proper "is this ref on the class boundary?" metric.

Why we want it: partition.py classifies a ref as A-Legit / A-Fraud when its
own 25-NN are unanimous. That is an inward-facing purity check. But for the
slow-path k-NN over Box-B to give the right answer on borderline queries we
need refs that are on the *outward* boundary of A clusters (refs whose 25-NN
are pure but who themselves sit on the edge facing the other class). The
nearest-opp distance picks those up directly.

GPU brute-force with chunking on both axes so the (queries × others) matmul
stays under VRAM. fp16 for the matmul (same precision policy as ivf.py).

Output:
    data/nearest_opp_dist.npy   float32, shape (3_000_000,)  in raw euclidean

Then sweeps several thresholds D and reports:
  - how many A refs satisfy nearest_opp < D (i.e., would be added to Box-B)
  - new Box-B total size
  - how many of the 11 ground-truth-missing refs are recovered

Usage:
    python -u nearest_opp.py
    # or to skip recomputing if data/nearest_opp_dist.npy already exists:
    SKIP_COMPUTE=1 python -u nearest_opp.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT_PATH = DATA / "nearest_opp_dist.npy"

# Outer batch: how many query refs we process per kernel launch.
# Inner chunk: how many "other-class" refs we materialize the (B × m) distance
# matrix against. The matmul intermediate is B * m * 4 bytes (fp32), so we
# target ~1 GB peak: 2048 × 131072 × 4 = 1 GB. Fits comfortably in 16 GB
# alongside the 5-class fp16 ref arrays and q_sq buffers.
BATCH = int(os.environ.get("BATCH", 2048))
INNER = int(os.environ.get("INNER", 131_072))

# The 11 refs that show up as the "missing top-5 NN" in our 5 confirmed
# mismatches. Used to evaluate threshold quality.
MISSING_REFS = [
    1030342, 758391, 466941, 2610013,  # case 8854 (fraud-pure, in A-Fraud)
    2442925, 716251, 2506490,           # case 15414 (legit-pure)
    892400,                              # case 36711 (legit-pure)
    2908288, 137370, 817280,             # case 53480 (legit-pure)
]


def fmt_eta(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    m, s = divmod(int(secs), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def nearest_opp_one_side(Q_fp32, others_fp32_T, others_sq_fp32, *,
                          device, batch, inner):
    """For each query in `Q_fp32` (Nq, 14) fp32, compute the min euclidean
    distance to any other ref. Pre-transposed and pre-squared "others" are
    passed in so the inner loop does pure matmul + add + min.

    Returns a numpy fp32 array of shape (Nq,)."""
    Nq = Q_fp32.shape[0]
    M = others_fp32_T.shape[1]                          # (14, M)
    out = torch.full((Nq,), float("inf"), dtype=torch.float32, device=device)

    # Precompute ||q||² once for all queries.
    q_sq = (Q_fp32 ** 2).sum(dim=1, keepdim=True)       # (Nq, 1)

    t0 = time.time()
    last_log = t0
    for i in range(0, Nq, batch):
        end = min(i + batch, Nq)
        qb = Q_fp32[i:end]                              # (b, 14)
        qsq_b = q_sq[i:end]                             # (b, 1)
        best = torch.full((end - i,), float("inf"),
                          dtype=torch.float32, device=device)
        for j in range(0, M, inner):
            oj_T = others_fp32_T[:, j:j+inner]          # (14, m)
            oj_sq = others_sq_fp32[j:j+inner]           # (m,)
            # d² = ||q||² + ||o||² - 2 q.o
            d2 = qsq_b + oj_sq[None, :] - 2.0 * (qb @ oj_T)
            best = torch.minimum(best, d2.min(dim=1).values)
            del d2
        out[i:end] = best
        del best, qb, qsq_b

        now = time.time()
        if now - last_log > 3.0 or end == Nq:
            elapsed = now - t0
            rate = end / max(elapsed, 1e-3)
            eta = (Nq - end) / max(rate, 1e-3)
            print(f"      {end:>8,}/{Nq:,}  ({rate:>6.0f} refs/s, "
                  f"ETA {fmt_eta(eta)})", flush=True)
            last_log = now

    return torch.sqrt(torch.clamp(out, min=0)).cpu().numpy()


def main():
    if OUT_PATH.exists() and os.environ.get("SKIP_COMPUTE"):
        print(f"[opp] reusing existing {OUT_PATH} (SKIP_COMPUTE=1)", flush=True)
        opp = np.load(OUT_PATH)
    else:
        print(f"[opp] loading refs / labels / box ...", flush=True)
        refs = np.load(DATA / "references.npy").astype(np.float32)
        labels = np.load(DATA / "labels.npy")            # bool
        box = np.load(DATA / "box_labels.npy")           # u8

        N = len(refs)
        n_fraud = int(labels.sum())
        n_legit = N - n_fraud
        print(f"[opp] N={N:,}  legits={n_legit:,}  frauds={n_fraud:,}", flush=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[opp] device={device}  BATCH={BATCH:,}  INNER={INNER:,}",
              flush=True)
        if device.type != "cuda":
            print(f"[opp] WARNING: GPU not available, this will take hours",
                  flush=True)

        # Keep everything in fp32 on device — fp16 matmul on AMD wasn't worth
        # the precision risk and we have plenty of VRAM headroom with the
        # tight BATCH/INNER tuning below.
        R = torch.from_numpy(refs).to(device=device, dtype=torch.float32)
        L = torch.from_numpy(labels).to(device).to(torch.bool)
        legit_idx_np = np.where(~labels)[0]
        fraud_idx_np = np.where(labels)[0]
        Q_legit = R[~L].contiguous()                          # (n_legit, 14)
        Q_fraud = R[L].contiguous()                           # (n_fraud, 14)
        # Pre-transpose and pre-square the "others" once so the inner loop is
        # one matmul + one broadcast add + one row-min.
        legit_T = Q_legit.T.contiguous()                      # (14, n_legit)
        fraud_T = Q_fraud.T.contiguous()                      # (14, n_fraud)
        legit_sq = (Q_legit ** 2).sum(dim=1)                  # (n_legit,)
        fraud_sq = (Q_fraud ** 2).sum(dim=1)                  # (n_fraud,)

        total_ops = 2.0 * n_legit * n_fraud * 14 * 2  # mul + add inside matmul
        print(f"[opp] approx total ops = {total_ops:.2e} "
              f"(matmul dominated; expect ~minutes on a 16 GB AMD GPU)",
              flush=True)

        out = np.zeros(N, dtype=np.float32)
        t_total = time.time()

        print(f"[opp] pass 1/2: legit → nearest fraud "
              f"({n_legit:,} queries × {n_fraud:,} others)", flush=True)
        out[legit_idx_np] = nearest_opp_one_side(
            Q_legit, fraud_T, fraud_sq,
            device=device, batch=BATCH, inner=INNER,
        )

        print(f"[opp] pass 2/2: fraud → nearest legit "
              f"({n_fraud:,} queries × {n_legit:,} others)", flush=True)
        out[fraud_idx_np] = nearest_opp_one_side(
            Q_fraud, legit_T, legit_sq,
            device=device, batch=BATCH, inner=INNER,
        )

        elapsed_total = time.time() - t_total
        print(f"[opp] total compute: {fmt_eta(elapsed_total)}", flush=True)

        np.save(OUT_PATH, out)
        print(f"[opp] wrote {OUT_PATH}  ({out.nbytes / 1e6:.1f} MB)",
              flush=True)
        opp = out

    # ---- threshold sweep -------------------------------------------------
    print(f"\n[opp] threshold sweep:", flush=True)
    box = np.load(DATA / "box_labels.npy")
    in_A = box != 2
    print(f"      Box-B (current): {(box == 2).sum():,}", flush=True)
    print(f"      A refs (current): {in_A.sum():,}\n", flush=True)

    header = f"{'D':>6}  {'A-refs added':>13}  {'new Box-B':>10}  {'recovers':>9}"
    print(f"      {header}", flush=True)
    print(f"      {'-' * len(header)}", flush=True)

    for thresh in [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25,
                   0.30, 0.40, 0.50, 0.75, 1.00]:
        add = in_A & (opp < thresh)
        new_box = (box == 2) | add
        rec = sum(int(new_box[r]) for r in MISSING_REFS)
        print(f"      {thresh:>.3f}  {add.sum():>13,}  {new_box.sum():>10,}  "
              f"{rec:>2}/11", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)

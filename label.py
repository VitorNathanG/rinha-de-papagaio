"""
Step 2/4 — Leave-one-out 5-NN labeling on GPU.

For every reference vector q in the dataset, find its 5 nearest neighbors in the
full dataset *excluding q itself* and count how many of those 5 are labeled as
fraud. That count (0..5) is the destilation target — it is exactly the value the
rinha grader uses to decide approved vs denied.

Algorithm (chunked top-5 sweep):
    The naive approach materializes the full (BATCH, N) distance matrix per batch
    and runs a single topk on it. That is bandwidth-bound on the topk pass and
    forces a small BATCH because the matrix is N * BATCH * 4 bytes.

    Instead we keep a per-query running top-5 (BATCH, 5) and sweep references in
    chunks of CHUNK rows:
        for each batch of BATCH queries:
            best = (+inf, -1) of shape (BATCH, 5)
            for each chunk of CHUNK references:
                d = R_sq_chunk - 2 * Q @ R_chunk.T          # (BATCH, CHUNK)
                mask self-distance with +inf where applicable
                topk(d, 5) -> chunk top-5
                merge with best via topk on the concatenated (BATCH, 10)
            counts[batch] = labels[best_idx].sum(dim=1)

    The "+Q_sq" term is omitted: it is constant across a row, so it does not
    change the ranking. The resulting "distance" values are negative for nearby
    points, but argmin/topk are preserved.

Precision:
    Default is fp16 — it is ~2x faster than fp32 (matmul throughput + half the
    bandwidth on the distance matrix), and disagrees with fp32 on only ~0.5% of
    queries (~0.2% binary-verdict flips). All disagreements are off-by-one in
    the count, which is well below the noise floor of the destilled MLP target.
    Set FP32=1 to force the exact fp32 path.

Memory tuning:
    BATCH    queries per outer chunk (default 2048).
    CHUNK    references per inner chunk (default 16384). Peak distance matrix
             is BATCH * CHUNK * dtype_size bytes (~64 MB fp16, ~128 MB fp32).
    SUBSET   process only the first SUBSET queries (default: all).
    K        number of neighbors to track per query (default 5). Larger K gives
             a "deepness" signal useful for prototype selection / cherry-picking
             a slow-path subset. Note: the rinha task itself is fixed at K=5.

Inputs:  data/references.npy, data/labels.npy
Output:  data/fraud_counts_k{K}.npy — shape (SUBSET or N,) uint8, values 0..K
"""
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def main():
    K = int(os.environ.get("K", 5))
    if not (1 <= K <= 255):
        raise SystemExit(f"[label] K must be in [1, 255], got {K}")
    threshold = math.ceil(0.6 * K)  # rinha-style "denied" cutoff (only meaningful at K=5)

    out_path = DATA_DIR / f"fraud_counts_k{K}.npy"
    if out_path.exists() and not os.environ.get("FORCE"):
        existing = np.load(out_path)
        print(f"[label] already computed: {out_path} (shape={existing.shape})")
        return

    refs = np.load(DATA_DIR / "references.npy")          # (N, 14) float32
    labels = np.load(DATA_DIR / "labels.npy")            # (N,) bool

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(f"[label] WARNING: running on CPU. Full N=3M will take hours.")
        print(f"[label] Set SUBSET=100000 for a quick CPU smoke test.")
    else:
        try:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[label] device: {device} ({name}, {mem:.1f} GB)")
        except Exception:
            print(f"[label] device: {device}")

    dtype = torch.float32 if os.environ.get("FP32") else torch.float16

    R = torch.from_numpy(refs).to(device=device, dtype=dtype)   # (N, 14)
    L = torch.from_numpy(labels).to(device).to(torch.uint8)
    R_sq = (R ** 2).sum(dim=1)                                  # (N,)
    R_T = R.T.contiguous()                                      # (14, N)
    N = R.shape[0]

    batch = int(os.environ.get("BATCH", 2048))
    chunk = int(os.environ.get("CHUNK", 16384))
    subset = int(os.environ.get("SUBSET", 0)) or N
    subset = min(subset, N)
    dtype_label = "fp16" if dtype == torch.float16 else "fp32"
    print(f"[label] K={K}  N={N:,}  batch={batch}  chunk={chunk:,}  "
          f"subset={subset:,}  dtype={dtype_label}")

    counts = torch.empty(subset, dtype=torch.uint8, device=device)
    INF = float("inf")
    t0 = time.time()
    last_log = t0
    for start in range(0, subset, batch):
        end = min(start + batch, subset)
        B = end - start
        Q = R[start:end]                                 # (B, 14)

        best_d = torch.full((B, K), INF, dtype=dtype, device=device)
        best_idx = torch.full((B, K), -1, dtype=torch.long, device=device)

        for j_start in range(0, N, chunk):
            j_end = min(j_start + chunk, N)
            R_chunk_T = R_T[:, j_start:j_end]            # (14, M) view
            R_sq_chunk = R_sq[j_start:j_end]             # (M,)

            # Partial distance (Q_sq omitted; constant per row, doesn't change ranking).
            # d = R_sq - 2 * (Q @ R.T)   shape (B, M)
            d = torch.addmm(R_sq_chunk.unsqueeze(0), Q, R_chunk_T,
                            beta=1.0, alpha=-2.0)

            # Mask self-distance where query range overlaps reference chunk.
            lo = max(start, j_start)
            hi = min(end, j_end)
            if lo < hi:
                rows = torch.arange(lo - start, hi - start, device=device)
                cols = torch.arange(lo - j_start, hi - j_start, device=device)
                d[rows, cols] = INF

            # Top-K within this chunk, then merge with running best on (B, 2K).
            chunk_d, chunk_local_idx = torch.topk(d, K, dim=1, largest=False)
            chunk_global_idx = chunk_local_idx + j_start

            cand_d = torch.cat([best_d, chunk_d], dim=1)            # (B, 2K)
            cand_idx = torch.cat([best_idx, chunk_global_idx], dim=1)
            best_d, sel = torch.topk(cand_d, K, dim=1, largest=False)
            best_idx = torch.gather(cand_idx, 1, sel)

        counts[start:end] = L[best_idx].sum(dim=1).to(torch.uint8)

        now = time.time()
        if now - last_log > 5.0 or end == subset:
            elapsed = now - t0
            rate = end / max(elapsed, 1e-3)
            eta = (subset - end) / max(rate, 1e-3)
            print(f"[label] {end:>10,}/{subset:,} ({100*end/subset:5.1f}%)  "
                  f"{rate:>7.0f} q/s  elapsed={elapsed:>5.0f}s  eta={eta:>5.0f}s")
            last_log = now

    counts_np = counts.cpu().numpy()
    np.save(out_path, counts_np)
    dist = np.bincount(counts_np, minlength=K + 1)
    pure_legit = int(dist[0])
    pure_fraud = int(dist[K])
    pure = pure_legit + pure_fraud
    non_pure = subset - pure
    denied = int(np.sum(counts_np >= threshold))
    print(f"[label] wrote {out_path}")
    print(f"[label] fraud-count distribution among {K} nearest neighbors:")
    for i, c in enumerate(dist):
        bar = "#" * int(40 * c / subset)
        marker = ""
        if i == 0:
            marker = "  (pure legit)"
        elif i == K:
            marker = "  (pure fraud)"
        elif i == threshold:
            marker = "  (= rinha denied threshold)" if K == 5 else ""
        print(f"        {i}/{K}: {c:>10,} ({100*c/subset:5.2f}%) {bar}{marker}")
    print(f"[label] pure clusters (count==0 or count=={K}): {pure:>10,} ({100*pure/subset:.2f}%)")
    print(f"[label] non-pure (anything between):           {non_pure:>10,} ({100*non_pure/subset:.2f}%)")
    print(f"[label] would-be denied (count>={threshold}): {100*denied/subset:.2f}%"
          + ("  <-- this is the rinha decision" if K == 5 else "  (rinha-style threshold; only canonical at K=5)"))


if __name__ == "__main__":
    main()

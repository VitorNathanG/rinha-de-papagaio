"""
Step 5.5 — Build the halo: for each Box-B reference, find its m closest
spatial neighbors in the full 3M references (excluding itself), then union
the result with Box B to form the halo subset.

Motivation: validate_subset showed that k=5 over Box B alone disagrees with
k=5 over the full 3M on ~6% of slow-routed queries. The disagreements
concentrate at the boundary (c_full in {2, 3}), where the query's true 5-NN
includes near-edge Box A points that Box B-only misses. The halo adds those
near-edge points back.

GPU brute-force, same chunked top-m pattern as label.py.

Tunables:
    M       neighbors per Box-B point (default 5)
    BATCH   queries per outer chunk (default 2048)
    CHUNK   references per inner chunk (default 16384)
    FP32    set to use fp32 instead of fp16 (default fp16)
    FORCE   recompute even if output file exists

Inputs:  data/references.npy, data/box_labels.npy
Output:  data/halo_indices_m{M}.npy   sorted unique int64 indices into refs
"""
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

B = 2


def main():
    m = int(os.environ.get("M", 5))
    out_path = DATA_DIR / f"halo_indices_m{m}.npy"
    if out_path.exists() and not os.environ.get("FORCE"):
        existing = np.load(out_path)
        print(f"[halo] already computed: {out_path} (size={len(existing):,})")
        return

    refs = np.load(DATA_DIR / "references.npy")
    box = np.load(DATA_DIR / "box_labels.npy")
    N = len(refs)
    B_idx = np.where(box == B)[0]
    num_B = len(B_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[halo] WARNING: CPU mode will be slow")
    else:
        try:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[halo] device: {device} ({name}, {mem:.1f} GB)")
        except Exception:
            print(f"[halo] device: {device}")

    dtype = torch.float32 if os.environ.get("FP32") else torch.float16
    R = torch.from_numpy(refs).to(device=device, dtype=dtype)
    R_sq = (R ** 2).sum(dim=1)
    R_T = R.T.contiguous()

    batch = int(os.environ.get("BATCH", 2048))
    chunk = int(os.environ.get("CHUNK", 16384))
    dtype_label = "fp16" if dtype == torch.float16 else "fp32"
    print(f"[halo] m={m}  Box B size={num_B:,}  batch={batch}  "
          f"chunk={chunk:,}  dtype={dtype_label}")

    neighbors = torch.empty((num_B, m), dtype=torch.long, device=device)
    INF = float("inf")
    t0 = time.time()
    last_log = t0
    for start in range(0, num_B, batch):
        end = min(start + batch, num_B)
        Bb = end - start
        global_q_idx = torch.from_numpy(B_idx[start:end].astype(np.int64)).to(device)
        Q = R[global_q_idx]                                  # (Bb, 14)

        best_d = torch.full((Bb, m), INF, dtype=dtype, device=device)
        best_idx = torch.full((Bb, m), -1, dtype=torch.long, device=device)

        for j_start in range(0, N, chunk):
            j_end = min(j_start + chunk, N)
            R_chunk_T = R_T[:, j_start:j_end]
            R_sq_chunk = R_sq[j_start:j_end]
            d = torch.addmm(R_sq_chunk.unsqueeze(0), Q, R_chunk_T,
                            beta=1.0, alpha=-2.0)

            # Self-exclusion: where the query's global index falls in this chunk
            in_chunk = (global_q_idx >= j_start) & (global_q_idx < j_end)
            if in_chunk.any():
                rows = torch.arange(Bb, device=device)[in_chunk]
                cols = global_q_idx[in_chunk] - j_start
                d[rows, cols] = INF

            chunk_d, chunk_local = torch.topk(d, m, dim=1, largest=False)
            chunk_global = chunk_local + j_start

            cand_d = torch.cat([best_d, chunk_d], dim=1)
            cand_idx = torch.cat([best_idx, chunk_global], dim=1)
            best_d, sel = torch.topk(cand_d, m, dim=1, largest=False)
            best_idx = torch.gather(cand_idx, 1, sel)

        neighbors[start:end] = best_idx

        now = time.time()
        if now - last_log > 5.0 or end == num_B:
            rate = end / max(now - t0, 1e-3)
            eta = (num_B - end) / max(rate, 1e-3)
            print(f"[halo] {end:>7,}/{num_B:,} ({100*end/num_B:5.1f}%)  "
                  f"{rate:>6.0f} q/s  elapsed={now-t0:.0f}s  eta={eta:.0f}s")
            last_log = now

    neighbors_flat = neighbors.cpu().numpy().ravel()
    union = np.concatenate([B_idx.astype(np.int64), neighbors_flat])
    halo = np.unique(union)
    np.save(out_path, halo)

    # Breakdown: how many of the halo points are from each box class
    halo_box = box[halo]
    box_counts = np.bincount(halo_box, minlength=3)

    print()
    print(f"[halo] wrote {out_path}")
    print(f"[halo] halo size:                 {len(halo):>10,}")
    print(f"[halo]   from Box A-Legit:        {box_counts[0]:>10,}")
    print(f"[halo]   from Box A-Fraud:        {box_counts[1]:>10,}")
    print(f"[halo]   from Box B:              {box_counts[2]:>10,}  (all of it)")
    print(f"[halo] added by halo (non-B):     {len(halo) - num_B:>10,}  "
          f"({100*(len(halo)-num_B)/num_B:.1f}% of Box B)")
    print(f"[halo] total fraction of refs:    {100*len(halo)/N:>9.2f}%")
    print(f"[halo] fp32 memory of subset:     {len(halo)*14*4/1e6:>9.2f} MB")


if __name__ == "__main__":
    main()

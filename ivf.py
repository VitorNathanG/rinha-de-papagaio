"""
GPU IVF (Inverted File Index) for fast approximate k-NN over the 3M references.

Used as the k-NN tool for the stress test: brute force on the GPU works at this
scale but scales linearly with N. IVF buys a 10-30x query speedup at high
nprobe, while staying entirely on the GPU.

Build (run-once, ~1-2 min on a modern GPU):
    .venv/bin/python ivf.py                  # default nlist=4096, kmeans iter=15
    NLIST=8192 ITER=20 .venv/bin/python ivf.py

Verify against brute-force ground truth on random query batch:
    CMD=verify .venv/bin/python ivf.py
    CMD=verify NPROBE=64 N_TEST=5000 .venv/bin/python ivf.py

Saved index file: data/ivf_nlist{NLIST}.pt with these tensors:
    centroids        (nlist, 14)      k-means centroids
    sort_order       (N,)             original ref indices sorted by cluster
    cluster_offsets  (nlist+1,)       CSR offsets into sort_order
    cluster_counts   (nlist,)         count per cluster

Module API (for stress_test.py to import):
    from ivf import load_index, search_ivf, brute_count_k5
"""

import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

DEFAULT_NLIST = 4096
DEFAULT_NPROBE = 16
DEFAULT_KMEANS_ITERS = 15
DEFAULT_CHUNK = 32768


# --- Brute force (used as ground truth in verify and as fallback) ---

def brute_count_k5(queries, refs, ref_labels, k=5, batch=2048, chunk=DEFAULT_CHUNK):
    """Exact k-NN via GPU brute force. Returns fraud counts (uint8). Tensors must
    be on the same device; refs and queries should share dtype."""
    device = refs.device
    dtype = refs.dtype
    T = queries.shape[0]
    N = refs.shape[0]
    R_sq = (refs ** 2).sum(dim=1)
    R_T = refs.T.contiguous()
    counts = torch.empty(T, dtype=torch.uint8, device=device)
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
        counts[start:end] = ref_labels[best_idx].sum(dim=1).to(torch.uint8)
    return counts


# --- IVF: k-means clustering -----------------------------------------------

def _kmeans(refs, nlist, n_iter, device, dtype, seed=42, chunk=DEFAULT_CHUNK):
    """Plain Lloyd's k-means with random-sample initialization.
    Accumulates centroid sums in fp32 even when refs are fp16."""
    N, D = refs.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    init_idx = torch.randperm(N, generator=g)[:nlist].to(device)
    centroids = refs[init_idx].clone()
    refs_fp32 = refs if dtype == torch.float32 else refs.to(torch.float32)

    for it in range(n_iter):
        t_iter = time.time()
        c_sq = (centroids ** 2).sum(dim=1)
        c_T = centroids.T.contiguous()

        # Assign each ref to nearest centroid (chunked argmin).
        # We omit q_sq from the distance because it's constant per row and
        # doesn't change argmin: argmin_j (c_sq[j] - 2*r.c_j) == argmin ||r-c_j||
        assignments = torch.empty(N, dtype=torch.long, device=device)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            d = torch.addmm(c_sq.unsqueeze(0), refs[s:e], c_T,
                            beta=1.0, alpha=-2.0)
            assignments[s:e] = d.argmin(dim=1)

        # Update centroids = mean of assigned refs (fp32 accumulation).
        new_centroids = torch.zeros(nlist, D, dtype=torch.float32, device=device)
        new_centroids.index_add_(0, assignments, refs_fp32)
        cluster_counts = torch.bincount(assignments, minlength=nlist).to(torch.float32)
        nonempty = cluster_counts > 0
        new_centroids[nonempty] /= cluster_counts[nonempty].unsqueeze(1)
        # Empty clusters keep their previous position (they re-attract points later).
        new_centroids[~nonempty] = centroids[~nonempty].to(torch.float32)
        new_centroids = new_centroids.to(dtype)

        shift = (new_centroids - centroids).norm().item()
        n_empty = int((~nonempty).sum().item())
        centroids = new_centroids
        elapsed = time.time() - t_iter
        print(f"[ivf] kmeans iter {it+1:>2}/{n_iter}  shift={shift:.4f}  "
              f"empty={n_empty}  elapsed={elapsed:.1f}s")

    # Final assignment with the last centroids.
    c_sq = (centroids ** 2).sum(dim=1)
    c_T = centroids.T.contiguous()
    final_assignments = torch.empty(N, dtype=torch.long, device=device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d = torch.addmm(c_sq.unsqueeze(0), refs[s:e], c_T,
                        beta=1.0, alpha=-2.0)
        final_assignments[s:e] = d.argmin(dim=1)
    return centroids, final_assignments


def _build_csr(assignments, nlist, device):
    """Build a CSR posting-list structure from per-ref assignments."""
    sort_order = assignments.argsort()
    sorted_assignments = assignments[sort_order]
    cluster_counts = torch.bincount(sorted_assignments, minlength=nlist)
    cluster_offsets = torch.zeros(nlist + 1, dtype=torch.long, device=device)
    cluster_offsets[1:] = cluster_counts.cumsum(dim=0)
    return sort_order, cluster_offsets, cluster_counts


# --- IVF: query -------------------------------------------------------------

def search_ivf(queries, refs, ref_labels, centroids, sort_order, cluster_offsets,
               cluster_counts, k=5, nprobe=16, batch=256, max_per_cluster=None):
    """Approximate k-NN via IVF on GPU. Returns fraud counts (uint8).

    For each query: find the nprobe nearest centroids, gather all refs in their
    posting lists into a padded scratch tensor (using +inf distance for the
    padding), then take the k smallest distances.
    """
    device = refs.device
    dtype = refs.dtype
    T, D = queries.shape
    N = refs.shape[0]
    if max_per_cluster is None:
        max_per_cluster = int(cluster_counts.max().item())

    counts = torch.empty(T, dtype=torch.uint8, device=device)
    INF = float("inf")
    c_sq = (centroids ** 2).sum(dim=1)
    c_T = centroids.T.contiguous()
    j_range = torch.arange(max_per_cluster, device=device)

    for s in range(0, T, batch):
        e = min(s + batch, T)
        B = e - s
        q = queries[s:e]
        q_sq = (q ** 2).sum(dim=1, keepdim=True)

        # 1. distances to all centroids → top-nprobe
        d_c = torch.addmm(c_sq.unsqueeze(0), q, c_T, beta=1.0, alpha=-2.0)
        _, top_clusters = torch.topk(d_c, nprobe, dim=1, largest=False)  # (B, nprobe)

        # 2. gather candidate ref indices via CSR offsets
        starts = cluster_offsets[top_clusters]                            # (B, nprobe)
        lens = cluster_counts[top_clusters]                               # (B, nprobe)
        valid = j_range.view(1, 1, -1) < lens.unsqueeze(-1)               # (B, nprobe, M)
        fetch = (starts.unsqueeze(-1) + j_range.view(1, 1, -1)).clamp(max=N - 1)
        cand_idx = sort_order[fetch].view(B, -1)                          # (B, nprobe*M)
        valid_flat = valid.view(B, -1)

        # 3. exact distance to candidate refs
        cand_vec = refs[cand_idx]                                         # (B, nprobe*M, D)
        cand_sq = (cand_vec ** 2).sum(dim=2)                              # (B, nprobe*M)
        dot = torch.bmm(q.unsqueeze(1), cand_vec.transpose(1, 2)).squeeze(1)
        d = q_sq + cand_sq - 2.0 * dot
        d = d.masked_fill(~valid_flat, INF)

        # 4. top-k → fraud count
        _, top_local = torch.topk(d, k, dim=1, largest=False)
        top_global = torch.gather(cand_idx, 1, top_local)
        counts[s:e] = ref_labels[top_global].sum(dim=1).to(torch.uint8)

    return counts


# --- Persistence ------------------------------------------------------------

def load_index(nlist=DEFAULT_NLIST, device=None, dtype=None):
    """Load a saved IVF index and move it onto `device` with `dtype`. Returns a dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32 if os.environ.get("FP32") else torch.float16
    state = torch.load(DATA_DIR / f"ivf_nlist{nlist}.pt", weights_only=False)
    return {
        "centroids":       state["centroids"].to(device=device, dtype=dtype),
        "sort_order":      state["sort_order"].to(device),
        "cluster_offsets": state["cluster_offsets"].to(device),
        "cluster_counts":  state["cluster_counts"].to(device),
        "nlist":           state["nlist"],
    }


# --- Driver functions -------------------------------------------------------

def build_index(nlist, n_iter):
    refs_np = np.load(DATA_DIR / "references.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if os.environ.get("FP32") else torch.float16
    refs = torch.from_numpy(refs_np).to(device=device, dtype=dtype)

    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[ivf] device: {device} ({name}, {mem:.1f} GB)")
        except Exception:
            print(f"[ivf] device: {device}")
    dtype_label = "fp16" if dtype == torch.float16 else "fp32"
    print(f"[ivf] N={refs.shape[0]:,}  D={refs.shape[1]}  nlist={nlist}  "
          f"kmeans_iter={n_iter}  dtype={dtype_label}")

    t0 = time.time()
    centroids, assignments = _kmeans(refs, nlist, n_iter, device, dtype)
    sort_order, cluster_offsets, cluster_counts = _build_csr(assignments, nlist, device)
    elapsed = time.time() - t0

    state = {
        "centroids":       centroids.cpu(),
        "sort_order":      sort_order.cpu(),
        "cluster_offsets": cluster_offsets.cpu(),
        "cluster_counts":  cluster_counts.cpu(),
        "nlist":           nlist,
        "n_iter":          n_iter,
        "dtype":           dtype_label,
    }
    out_path = DATA_DIR / f"ivf_nlist{nlist}.pt"
    torch.save(state, out_path)

    cc = cluster_counts.cpu().numpy()
    print(f"[ivf] wrote {out_path}")
    print(f"[ivf] cluster sizes: min={cc.min()}  max={cc.max()}  "
          f"mean={cc.mean():.1f}  std={cc.std():.1f}")
    print(f"[ivf] empty clusters: {int((cc == 0).sum())}/{nlist}")
    print(f"[ivf] build time: {elapsed:.1f}s")


def verify(nlist, nprobe, n_test, batch=256):
    refs_np = np.load(DATA_DIR / "references.npy")
    labels_np = np.load(DATA_DIR / "labels.npy")

    rng = np.random.default_rng(123)
    Q_np = rng.uniform(0, 1, size=(n_test, 14)).astype(np.float32)
    # 20% of queries get the "no previous tx" sentinel at dims 5 and 6
    null_mask = rng.uniform(0, 1, size=n_test) < 0.2
    Q_np[null_mask, 5] = -1.0
    Q_np[null_mask, 6] = -1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if os.environ.get("FP32") else torch.float16
    idx = load_index(nlist=nlist, device=device, dtype=dtype)

    refs = torch.from_numpy(refs_np).to(device=device, dtype=dtype)
    L = torch.from_numpy(labels_np).to(device).to(torch.uint8)
    Q = torch.from_numpy(Q_np).to(device=device, dtype=dtype)

    print(f"[ivf] verify: nlist={nlist}  nprobe={nprobe}  n_test={n_test:,}")
    print(f"[ivf] queries are uniform random in [0,1]^14 (20% with -1 sentinels)")

    t0 = time.time()
    truth = brute_count_k5(Q, refs, L, k=5).cpu().numpy()
    bt = time.time() - t0
    print(f"[ivf] brute: {bt:.2f}s ({n_test/bt:.0f} q/s)")

    t0 = time.time()
    pred = search_ivf(Q, refs, L,
                      idx["centroids"], idx["sort_order"],
                      idx["cluster_offsets"], idx["cluster_counts"],
                      k=5, nprobe=nprobe, batch=batch).cpu().numpy()
    it = time.time() - t0
    print(f"[ivf] IVF:   {it:.2f}s ({n_test/it:.0f} q/s, speedup {bt/it:.2f}x)")

    exact = float((pred == truth).mean())
    binary = float(((pred >= 3) == (truth >= 3)).mean())
    n_dis_exact = int((pred != truth).sum())
    n_dis_bin = int(((pred >= 3) != (truth >= 3)).sum())
    print(f"[ivf] exact count agreement:    {100*exact:.4f}%  ({n_dis_exact:,} mismatches)")
    print(f"[ivf] binary verdict agreement: {100*binary:.4f}%  ({n_dis_bin:,} mismatches)")

    if exact < 1.0:
        print(f"[ivf] per-count exact match breakdown:")
        for c_truth in range(6):
            mask = (truth == c_truth)
            n_m = int(mask.sum())
            if n_m == 0:
                continue
            agree = int((pred[mask] == truth[mask]).sum())
            print(f"        truth={c_truth}: {agree:,}/{n_m:,}  ({100*agree/n_m:.2f}%)")


if __name__ == "__main__":
    cmd = os.environ.get("CMD", "build")
    nlist = int(os.environ.get("NLIST", DEFAULT_NLIST))
    if cmd == "build":
        n_iter = int(os.environ.get("ITER", DEFAULT_KMEANS_ITERS))
        build_index(nlist=nlist, n_iter=n_iter)
    elif cmd == "verify":
        verify(nlist=nlist,
               nprobe=int(os.environ.get("NPROBE", DEFAULT_NPROBE)),
               n_test=int(os.environ.get("N_TEST", 1000)),
               batch=int(os.environ.get("BATCH", 256)))
    else:
        raise SystemExit(f"unknown CMD={cmd!r}; use 'build' or 'verify'")

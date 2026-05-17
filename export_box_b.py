"""
Export the Box-B subset + an IVF (Inverted File) index to Rust-friendly raw
binary files.

Refs and labels are re-ordered so all refs in the same IVF cluster live
contiguously. The Rust backend reads each probed cluster as a flat slice
during search — no scatter, no extra indirection.

Layout:
    data/box_b_refs.bin             f32, shape (N_B, 16). Sorted by cluster.
                                    Last 2 floats per row are zero padding so
                                    each row is one 64-byte cache line.
    data/box_b_refs.i16.bin         int16, shape (N_B, 16). Same ordering as
                                    box_b_refs.bin; values are round(x*10000).
                                    The vectorize step already rounds queries
                                    to k/10000 with k ∈ [-10000, 10000], so
                                    this is bit-exact on the round4 grid.
                                    Row size = 32 B (half cache line).
    data/box_b_labels.bin           u8, shape (N_B,). Sorted by cluster.
    data/box_b_ivf_centroids.bin    f32, shape (nlist, 16). Padded same way.
    data/box_b_ivf_centroids.i16.bin int16, shape (nlist, 16). round(c*10000).
                                    Centroids are k-means means so they're NOT
                                    on the round4 grid; quantization adds
                                    ±5e-5 noise per dim per centroid — small
                                    vs cluster radius, doesn't affect ranking.
    data/box_b_ivf_offsets.bin      u32, shape (nlist + 1,). CSR offsets into
                                    box_b_refs.bin in *rows* (not bytes).

Tunables (env vars):
    NLIST=256   number of IVF clusters (default 256, ≈ √N for N≈105k)
    ITER=20     Lloyd's k-means iterations
    SEED=42     k-means random init seed

Why nlist=256: keeps the centroid scan tiny (16 KB, fits L1) while giving
~410 refs per cluster on average — small enough that nprobe=16 touches only
~6.5k refs (≈ 420 KB) per query instead of all 105k (6.75 MB).
"""
import pipeline_log

import os
import time

import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"

B = 2
NLIST = int(os.environ.get("NLIST", 256))
# Hard cap on Lloyd iterations. Default is generous; convergence (no point
# changes cluster between iters) usually fires far earlier.
KMEANS_MAX_ITER = int(os.environ.get("ITER", 1000))
SEED = int(os.environ.get("SEED", 42))


def kmeans(X, k, max_iter, seed):
    """Plain Lloyd's k-means in numpy. X is (N, D) fp32. Returns
    centroids (k, D) fp32 and per-row assignments (N,) int32.

    Distance comparison uses ||X||² + ||c||² − 2 X·cᵀ, but ||X||² is
    constant per row so we omit it (doesn't change argmin).

    Stops when assignments fully stabilise (no point changes cluster) — that
    is, when the Voronoi partition is a fixed point of Lloyd's iteration.
    `max_iter` is a safety cap, not the expected stop.
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    init = rng.permutation(N)[:k]
    centroids = X[init].astype(np.float32, copy=True)
    prev_assignments = np.full(N, -1, dtype=np.int32)
    for it in range(max_iter):
        t0 = time.time()
        c_sq = (centroids ** 2).sum(axis=1)
        # (N, k) distances modulo the constant X² term.
        d = c_sq[None, :] - 2.0 * (X @ centroids.T)
        assignments = d.argmin(axis=1).astype(np.int32)
        n_changed = int((assignments != prev_assignments).sum())
        new = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        np.add.at(new, assignments, X)
        np.add.at(counts, assignments, 1)
        nonempty = counts > 0
        new[nonempty] /= counts[nonempty, None]
        # Empty clusters keep their previous position.
        new[~nonempty] = centroids[~nonempty]
        shift = float(np.linalg.norm(new - centroids))
        n_empty = int((~nonempty).sum())
        centroids = new
        print(f"[kmeans] iter {it+1:>3}/{max_iter}  shift={shift:.5f}  "
              f"changed={n_changed:>7}  empty={n_empty}  "
              f"elapsed={time.time()-t0:.2f}s")
        if n_changed == 0:
            print(f"[kmeans] converged after {it+1} iters "
                  f"(0 assignment changes)")
            return centroids, assignments
        prev_assignments = assignments
    print(f"[kmeans] hit max_iter={max_iter} without full convergence "
          f"(last changed={n_changed})")
    return centroids, assignments


def main():
    refs = np.load(DATA / "references.npy")           # (3M, 14) f32
    labels = np.load(DATA / "labels.npy")             # bool
    box = np.load(DATA / "box_labels.npy")            # u8 0/1/2

    B_idx = np.where(box == B)[0]
    refs_b = refs[B_idx].astype(np.float32, copy=False)
    labels_b = labels[B_idx].astype(np.uint8)
    n_b, d = refs_b.shape
    print(f"[export] N_B = {n_b:,}  D = {d}  "
          f"fraud rate = {labels_b.mean()*100:.2f}%")

    print(f"[export] running k-means (nlist={NLIST}, max_iter={KMEANS_MAX_ITER})...")
    centroids, assignments = kmeans(refs_b, NLIST, KMEANS_MAX_ITER, SEED)

    # Sort refs/labels by cluster so each cluster is a contiguous slice.
    sort_order = np.argsort(assignments, kind="stable")
    sorted_assignments = assignments[sort_order]
    refs_sorted = refs_b[sort_order]
    labels_sorted = labels_b[sort_order]

    counts = np.bincount(sorted_assignments, minlength=NLIST).astype(np.int64)
    offsets = np.zeros(NLIST + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(counts).astype(np.uint32)

    # Pad to 16 floats per row (one 64-byte cache line). Same padding the
    # AVX2 inner loop expects.
    refs_padded = np.zeros((n_b, 16), dtype=np.float32)
    refs_padded[:, :14] = refs_sorted
    cent_padded = np.zeros((NLIST, 16), dtype=np.float32)
    cent_padded[:, :14] = centroids

    refs_padded.tofile(DATA / "box_b_refs.bin")
    labels_sorted.tofile(DATA / "box_b_labels.bin")
    cent_padded.tofile(DATA / "box_b_ivf_centroids.bin")
    offsets.tofile(DATA / "box_b_ivf_offsets.bin")

    # Quantized int16 mirrors (scale = 10000). Refs are exact on the round4
    # grid; centroids are means and get ±0.5-step rounding. np.rint matches
    # IEEE round-half-to-even — same tie-break the Rust query path uses.
    refs_i16 = np.rint(refs_padded * 10000.0).astype(np.int16)
    cent_i16 = np.rint(cent_padded * 10000.0).astype(np.int16)
    refs_i16.tofile(DATA / "box_b_refs.i16.bin")
    cent_i16.tofile(DATA / "box_b_ivf_centroids.i16.bin")

    print(f"[export] wrote box_b_refs.bin            "
          f"{refs_padded.nbytes/1e6:.2f} MB")
    print(f"[export] wrote box_b_refs.i16.bin        "
          f"{refs_i16.nbytes/1e6:.2f} MB  (−{100*(1-refs_i16.nbytes/refs_padded.nbytes):.0f}%)")
    print(f"[export] wrote box_b_labels.bin          "
          f"{labels_sorted.nbytes/1e3:.2f} KB")
    print(f"[export] wrote box_b_ivf_centroids.bin   "
          f"{cent_padded.nbytes/1e3:.2f} KB")
    print(f"[export] wrote box_b_ivf_centroids.i16.bin {cent_i16.nbytes/1e3:.2f} KB")
    print(f"[export] wrote box_b_ivf_offsets.bin     "
          f"{offsets.nbytes} B")
    print(f"[export] cluster sizes: min={int(counts.min())}  "
          f"max={int(counts.max())}  mean={counts.mean():.1f}  "
          f"std={counts.std():.1f}")
    print(f"[export] empty clusters: {int((counts == 0).sum())}/{NLIST}")


if __name__ == "__main__":
    pipeline_log.setup(__file__)
    main()

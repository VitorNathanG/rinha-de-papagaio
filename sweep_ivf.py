"""
Build a sweep of IVF index variants for the Box-B slow path.

For each NLIST in the sweep, run Lloyd's k-means until convergence (centroid
shift < EPS) or MAX_ITER iterations — whichever comes first — and then write
the IVF artifacts to a per-NLIST subdirectory under data/ivf_sweep/. The Rust
bench harness then loads each variant in turn and measures kernel latency
for every NPROBE we care about.

Tunables (env):
    NLISTS=64,128,256,512,1024,2048   sweep grid (comma-separated)
    EPS=1e-3                          convergence threshold on centroid shift
    MAX_ITER=500                      hard cap on k-means iterations
    SEED=42                           initialization seed
"""
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = DATA / "ivf_sweep"

B = 2
EPS = float(os.environ.get("EPS", "1e-3"))
MAX_ITER = int(os.environ.get("MAX_ITER", 500))
SEED = int(os.environ.get("SEED", 42))
NLISTS = [int(x) for x in os.environ.get(
    "NLISTS", "64,128,256,512,1024,2048").split(",")]


def kmeans_to_convergence(X, k, eps, max_iter, seed):
    """Lloyd's k-means with convergence on centroid shift. Returns
    (centroids fp32, assignments int32, list-of-iter-stats)."""
    rng = np.random.default_rng(seed)
    N, _ = X.shape
    init = rng.permutation(N)[:k]
    centroids = X[init].astype(np.float32, copy=True)
    log = []
    shift = float("inf")
    for it in range(max_iter):
        t0 = time.time()
        c_sq = (centroids ** 2).sum(axis=1)
        # Argmin distance ignoring the per-row ||X||² term (constant per row).
        d = c_sq[None, :] - 2.0 * (X @ centroids.T)
        assignments = d.argmin(axis=1).astype(np.int32)
        new = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        np.add.at(new, assignments, X)
        np.add.at(counts, assignments, 1)
        nonempty = counts > 0
        new[nonempty] /= counts[nonempty, None]
        # Empty clusters stay put (will re-attract on later iters).
        new[~nonempty] = centroids[~nonempty]
        shift = float(np.linalg.norm(new - centroids))
        n_empty = int((~nonempty).sum())
        centroids = new
        log.append({"iter": it + 1, "shift": shift, "empty": n_empty,
                    "elapsed": time.time() - t0})
        if shift < eps:
            break

    # Final reassignment with the converged centroids.
    c_sq = (centroids ** 2).sum(axis=1)
    d = c_sq[None, :] - 2.0 * (X @ centroids.T)
    assignments = d.argmin(axis=1).astype(np.int32)
    return centroids, assignments, log


def build_variant(refs_b, labels_b, nlist, eps, max_iter, seed, out_dir):
    t0 = time.time()
    centroids, assignments, log = kmeans_to_convergence(
        refs_b, nlist, eps, max_iter, seed)

    sort_order = np.argsort(assignments, kind="stable")
    sorted_assignments = assignments[sort_order]
    refs_sorted = refs_b[sort_order]
    labels_sorted = labels_b[sort_order]
    counts = np.bincount(sorted_assignments, minlength=nlist).astype(np.int64)
    offsets = np.zeros(nlist + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(counts).astype(np.uint32)

    refs_padded = np.zeros((refs_b.shape[0], 16), dtype=np.float32)
    refs_padded[:, :14] = refs_sorted
    cent_padded = np.zeros((nlist, 16), dtype=np.float32)
    cent_padded[:, :14] = centroids

    out_dir.mkdir(parents=True, exist_ok=True)
    refs_padded.tofile(out_dir / "refs.bin")
    labels_sorted.tofile(out_dir / "labels.bin")
    cent_padded.tofile(out_dir / "centroids.bin")
    offsets.tofile(out_dir / "offsets.bin")

    converged = log[-1]["shift"] < eps
    meta = {
        "nlist": nlist,
        "n_refs": int(refs_b.shape[0]),
        "iters": len(log),
        "converged": converged,
        "final_shift": log[-1]["shift"],
        "build_elapsed": time.time() - t0,
        "cluster_min": int(counts.min()),
        "cluster_max": int(counts.max()),
        "cluster_mean": float(counts.mean()),
        "cluster_std": float(counts.std()),
        "empty": int((counts == 0).sum()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    refs = np.load(DATA / "references.npy")
    labels = np.load(DATA / "labels.npy")
    box = np.load(DATA / "box_labels.npy")

    B_idx = np.where(box == B)[0]
    refs_b = refs[B_idx].astype(np.float32, copy=False)
    labels_b = labels[B_idx].astype(np.uint8)
    n_b, d = refs_b.shape
    print(f"[sweep] N_B={n_b:,}  D={d}  NLISTS={NLISTS}  "
          f"EPS={EPS}  MAX_ITER={MAX_ITER}")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for nlist in NLISTS:
        out_dir = OUT / f"nlist{nlist}"
        meta = build_variant(refs_b, labels_b, nlist, EPS, MAX_ITER, SEED, out_dir)
        summary.append(meta)
        print(f"[sweep] nlist={nlist:>5}  iters={meta['iters']:>3}  "
              f"converged={str(meta['converged']):>5}  "
              f"shift={meta['final_shift']:.3g}  "
              f"size min/mean/max={meta['cluster_min']:>4}/"
              f"{meta['cluster_mean']:>5.0f}/{meta['cluster_max']:>4}  "
              f"empty={meta['empty']:>3}  build={meta['build_elapsed']:>5.1f}s")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[sweep] wrote {OUT}/")


if __name__ == "__main__":
    main()

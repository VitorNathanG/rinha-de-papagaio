"""
Stress test: compare three k=5 implementations across random query batches.

For each distribution:
    1. exact_truth  = GPU brute force over the full 3M references (ground truth).
    2. ivf_3m       = GPU IVF over the full 3M (nprobe=64).
    3. rust_boxb    = Rust AVX2 brute force over the ~105k Box-B subset.

We compare the binary verdict (fraud >= 3 of 5) and the exact count.

Distributions:
    uniform[0,1]        — worst-case OOD: uniform 14-D in [0,1] (with -1 sentinels
                          at dims 5 and 6 in 20% of queries to match the
                          "last_transaction is null" case).
    gaussian(0.5, 0.3)  — softer OOD: Gaussian centered at 0.5, clipped to [0,1].
    perturbed_refs      — in-distribution: real references plus N(0, 0.05) noise.

Outputs:
    data/stress_results.json
    stdout report
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ivf import load_index, search_ivf, brute_count_k5

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def gen_uniform(N, rng):
    Q = rng.uniform(0, 1, size=(N, 14)).astype(np.float32)
    null_mask = rng.uniform(0, 1, size=N) < 0.2
    Q[null_mask, 5] = -1.0
    Q[null_mask, 6] = -1.0
    return Q


def gen_gaussian(N, rng):
    Q = np.clip(rng.normal(0.5, 0.3, size=(N, 14)).astype(np.float32), 0, 1)
    null_mask = rng.uniform(0, 1, size=N) < 0.2
    Q[null_mask, 5] = -1.0
    Q[null_mask, 6] = -1.0
    return Q


def gen_perturbed(refs, N, rng):
    idx = rng.integers(0, len(refs), size=N)
    Q = refs[idx].copy().astype(np.float32)
    was_null_5 = Q[:, 5] == -1
    was_null_6 = Q[:, 6] == -1
    Q += rng.normal(0, 0.05, size=Q.shape).astype(np.float32)
    Q = np.clip(Q, 0, 1)
    Q[was_null_5, 5] = -1.0
    Q[was_null_6, 6] = -1.0
    return Q


def write_queries_padded(Q, path):
    """Write queries as float32 padded to 16-per-row for Rust consumption."""
    n = len(Q)
    pad = np.zeros((n, 16), dtype=np.float32)
    pad[:, :14] = Q
    pad.tofile(path)


def ensure_rust_binary():
    bench_dir = ROOT / "bench"
    binary = bench_dir / "target" / "release" / "papagaio-bench"
    if not binary.exists():
        print("[stress] building Rust binary (release)...")
        subprocess.run(["cargo", "build", "--release"], cwd=str(bench_dir),
                       check=True, stdout=sys.stdout, stderr=sys.stderr)
    return binary


def run_rust(binary, n_queries):
    t0 = time.time()
    result = subprocess.run(
        [str(binary)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"rust binary failed (exit {result.returncode})")
    # Pass through stderr (timing report) to our stdout.
    print(result.stderr, end="")
    counts = np.fromfile(DATA / "rust_results.bin", dtype=np.uint8)
    assert len(counts) == n_queries, f"rust returned {len(counts)} != {n_queries}"
    return counts, elapsed


def main():
    N = int(os.environ.get("N", 100_000))
    seed = int(os.environ.get("SEED", 7))
    ivf_nlist = int(os.environ.get("NLIST", 4096))
    ivf_nprobe = int(os.environ.get("NPROBE", 64))

    rust_bin = ensure_rust_binary()

    refs_np = np.load(DATA / "references.npy")
    labels_np = np.load(DATA / "labels.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    refs_gpu = torch.from_numpy(refs_np).to(device=device, dtype=dtype)
    labels_gpu = torch.from_numpy(labels_np).to(device).to(torch.uint8)
    ivf_idx = load_index(nlist=ivf_nlist, device=device, dtype=dtype)

    rng = np.random.default_rng(seed)
    distributions = {
        "uniform[0,1]":              gen_uniform(N, rng),
        "gaussian(0.5,0.3)":         gen_gaussian(N, rng),
        "perturbed_refs(σ=0.05)":    gen_perturbed(refs_np, N, rng),
    }

    print(f"[stress] N={N:,}  seed={seed}  device={device}  "
          f"IVF nlist={ivf_nlist} nprobe={ivf_nprobe}")
    print()

    report = {}
    for name, Q in distributions.items():
        print(f"=== distribution: {name} ===")

        # Export queries for Rust (padded to 16).
        write_queries_padded(Q, DATA / "stress_queries.bin")

        # Queries on GPU (fp16 for the matmul path).
        Q_gpu = torch.from_numpy(Q).to(device=device, dtype=dtype)

        # 1. Exact ground truth — GPU brute over 3M.
        t0 = time.time()
        truth = brute_count_k5(Q_gpu, refs_gpu, labels_gpu, k=5).cpu().numpy()
        t_brute = time.time() - t0

        # 2. GPU IVF over 3M.
        t0 = time.time()
        ivf_count = search_ivf(
            Q_gpu, refs_gpu, labels_gpu,
            ivf_idx["centroids"], ivf_idx["sort_order"],
            ivf_idx["cluster_offsets"], ivf_idx["cluster_counts"],
            k=5, nprobe=ivf_nprobe,
        ).cpu().numpy()
        t_ivf = time.time() - t0

        # 3. Rust brute over Box B (CPU).
        rust_count, t_rust = run_rust(rust_bin, N)

        v_truth = truth >= 3
        v_ivf = ivf_count >= 3
        v_rust = rust_count >= 3

        rust_vs_truth_bin = float((v_rust == v_truth).mean())
        ivf_vs_truth_bin = float((v_ivf == v_truth).mean())
        rust_vs_ivf_bin = float((v_rust == v_ivf).mean())

        rust_vs_truth_exact = float((rust_count == truth).mean())
        ivf_vs_truth_exact = float((ivf_count == truth).mean())

        print()
        print(f"  timings (N = {N:,} queries):")
        print(f"    GPU brute 3M:           {t_brute*1000:7.0f} ms  ({N/t_brute:8.0f} q/s)")
        print(f"    GPU IVF 3M (np={ivf_nprobe}): {t_ivf*1000:7.0f} ms  ({N/t_ivf:8.0f} q/s)")
        print(f"    Rust brute Box B:       {t_rust*1000:7.0f} ms  ({N/t_rust:8.0f} q/s)"
              f"  mean latency {1e6/(N/t_rust):.2f} µs")
        print()
        print(f"  binary verdict agreement (count >= 3 means deny):")
        print(f"    IVF (3M)    vs truth: {100*ivf_vs_truth_bin:7.4f}%  "
              f"({int(np.sum(v_ivf != v_truth)):,} disagreements)")
        print(f"    Rust (Box B) vs truth: {100*rust_vs_truth_bin:7.4f}%  "
              f"({int(np.sum(v_rust != v_truth)):,} disagreements)")
        print(f"    Rust (Box B) vs IVF:   {100*rust_vs_ivf_bin:7.4f}%  "
              f"({int(np.sum(v_rust != v_ivf)):,} disagreements)")
        print(f"  exact count agreement:")
        print(f"    IVF (3M)    vs truth: {100*ivf_vs_truth_exact:7.4f}%")
        print(f"    Rust (Box B) vs truth: {100*rust_vs_truth_exact:7.4f}%")
        print()

        report[name] = {
            "n": N,
            "timing_s": {
                "brute_3m_gpu": t_brute,
                "ivf_3m_gpu":   t_ivf,
                "rust_boxb_cpu": t_rust,
            },
            "binary_agreement": {
                "rust_vs_truth": rust_vs_truth_bin,
                "ivf_vs_truth":  ivf_vs_truth_bin,
                "rust_vs_ivf":   rust_vs_ivf_bin,
            },
            "exact_count_agreement": {
                "rust_vs_truth": rust_vs_truth_exact,
                "ivf_vs_truth":  ivf_vs_truth_exact,
            },
            "ivf_nprobe": ivf_nprobe,
            "ivf_nlist": ivf_nlist,
        }

    out_path = DATA / "stress_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[stress] wrote {out_path}")


if __name__ == "__main__":
    main()

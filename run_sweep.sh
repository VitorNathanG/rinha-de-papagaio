#!/usr/bin/env bash
# Sequential sweep over IVF (NLIST, NPROBE) combinations.
#
#   1. Build all NLIST variants under data/ivf_sweep/ via sweep_ivf.py
#      (skipped if data/ivf_sweep/summary.json already exists; set REBUILD=1
#      to force a rebuild).
#   2. Build the ivf_bench harness in release mode.
#   3. For every (NLIST, NPROBE) combo, launch the bench pinned to one core
#      so there's no cross-thread interference. Each combo is a fresh
#      process — cold caches per combo, no carry-over.
#
# Env knobs:
#   NLISTS="64 128 256 512 1024 2048"     space-separated
#   NPROBES="1 2 4 8 16 32 64"            space-separated
#   N_QUERIES=20000                       samples per combo (caps stress set)
#   WARMUP=1000                           untimed priming queries
#   PIN_CPU=0                             core id for taskset (use "" to skip)
#   REBUILD=1                             force re-running sweep_ivf.py
set -euo pipefail
cd "$(dirname "$0")"

NLISTS="${NLISTS:-64 128 256 512 1024 2048}"
NPROBES="${NPROBES:-1 2 4 8 16 32 64}"
N_QUERIES="${N_QUERIES:-20000}"
WARMUP="${WARMUP:-1000}"
PIN_CPU="${PIN_CPU:-0}"

# 1. Build IVF variants (Python).
if [[ -n "${REBUILD:-}" || ! -f data/ivf_sweep/summary.json ]]; then
    echo "[sweep] building IVF variants (NLISTS=${NLISTS// /,})..." >&2
    NLISTS="${NLISTS// /,}" .venv/bin/python sweep_ivf.py
else
    echo "[sweep] reusing data/ivf_sweep/ (set REBUILD=1 to rebuild)" >&2
fi

# 2. Build the bench binary.
echo "[sweep] building ivf-bench..." >&2
(cd ivf_bench && cargo build --release --quiet)
BENCH=ivf_bench/target/release/ivf-bench

# 3. Pin to a single core if available.
PREFIX=""
if [[ -n "$PIN_CPU" ]] && command -v taskset >/dev/null 2>&1; then
    PREFIX="taskset -c $PIN_CPU"
    echo "[sweep] pinning bench to CPU $PIN_CPU" >&2
fi

# Header
printf "%-6s  %-7s  %-8s  %-9s  %-9s  %-9s  %-10s  %-9s\n" \
    nlist nprobe samples p50_us p90_us p99_us rps recall_%
printf "%-6s  %-7s  %-8s  %-9s  %-9s  %-9s  %-10s  %-9s\n" \
    "------" "-------" "--------" "---------" "---------" "---------" "----------" "---------"

for NLIST in $NLISTS; do
    DIR="data/ivf_sweep/nlist${NLIST}"
    if [[ ! -d "$DIR" ]]; then
        echo "[skip] no $DIR" >&2; continue
    fi
    for NPROBE in $NPROBES; do
        if (( NPROBE > NLIST )); then continue; fi
        OUT=$(NLIST_LABEL="$NLIST" \
              REFS="$DIR/refs.bin" \
              LABELS="$DIR/labels.bin" \
              CENTROIDS="$DIR/centroids.bin" \
              OFFSETS="$DIR/offsets.bin" \
              NPROBE="$NPROBE" \
              N_QUERIES="$N_QUERIES" \
              WARMUP="$WARMUP" \
              $PREFIX "$BENCH")
        IFS=$'\t' read -r nl np ns p50 p90 p99 rps rec <<< "$OUT"
        printf "%-6s  %-7s  %-8s  %-9s  %-9s  %-9s  %-10s  %-9s\n" \
            "$nl" "$np" "$ns" "$p50" "$p90" "$p99" "$rps" "$rec"
    done
done

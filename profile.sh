#!/usr/bin/env bash
# profile.sh — sustained-load profiling of the Rust backend.
#
# Builds the binary with line-table debug symbols, runs it natively (no Docker —
# we want clean access to the PID and zero container overhead in the profile),
# then runs the parallel k6 test (test/profile.js) while perf samples the
# process. Generates a flamegraph SVG and a top-symbols report.
#
# Outputs in profile-out/:
#   backend.log       Rust backend stdout/stderr
#   k6.log            k6 sustained run summary
#   perf.data         raw perf samples
#   perf-report.txt   top-symbol text report
#   flame.svg         interactive flamegraph (open in any browser)

set -euo pipefail

cd "$(dirname "$0")"

K6_BIN=${K6_BIN:-/tmp/k6}
# NOTE: avoid env vars prefixed with K6_ — they collide with k6's built-in
# config (K6_DURATION, K6_VUS, etc.) and silently disable our scenarios block.
RATE=${RATE:-10000}
DURATION=${DURATION:-30s}
# In production each container runs the binary with current_thread (1 worker).
# For profiling at >5k RPS we need real parallelism on the dev box; default
# enough workers to saturate without being silly.
TOKIO_WORKERS=${TOKIO_WORKERS:-6}
SAMPLE_FREQ=${SAMPLE_FREQ:-999}
SAMPLE_DURATION=${SAMPLE_DURATION:-20}
WARMUP=${WARMUP:-5}

mkdir -p profile-out

echo "[profile] building backend with profiling profile..."
(cd backend && cargo build --profile profiling 2>&1 | tail -5)

BACKEND=./backend/target/profiling/papagaio-api
PERF_DATA=profile-out/perf.data
PERF_REPORT=profile-out/perf-report.txt
FLAME_SVG=profile-out/flame.svg

if ! command -v perf >/dev/null; then
    echo "[profile] perf not installed; abort." >&2
    exit 1
fi
if ! command -v inferno-flamegraph >/dev/null; then
    echo "[profile] inferno-flamegraph not installed (run: cargo install inferno)" >&2
    exit 1
fi
if [ ! -x "$K6_BIN" ]; then
    echo "[profile] k6 not found at $K6_BIN — set K6_BIN env var" >&2
    exit 1
fi

# Make sure nothing else is on 9999.
if curl -sS http://localhost:9999/ready >/dev/null 2>&1; then
    echo "[profile] port 9999 already in use — bring down compose or other backend first." >&2
    exit 1
fi

echo "[profile] starting backend (TOKIO_WORKERS=$TOKIO_WORKERS)..."
TOKIO_WORKERS=$TOKIO_WORKERS $BACKEND > profile-out/backend.log 2>&1 &
APP_PID=$!
trap 'kill $APP_PID 2>/dev/null || true' EXIT

# Wait for /ready
for i in $(seq 1 20); do
    if curl -sS http://localhost:9999/ready >/dev/null 2>&1; then break; fi
    sleep 0.25
done
if ! curl -sS http://localhost:9999/ready >/dev/null 2>&1; then
    echo "[profile] backend never became ready" >&2
    cat profile-out/backend.log >&2
    exit 1
fi
echo "[profile] backend ready (pid=$APP_PID)"

echo "[profile] launching k6 sustained run in background (rate=$RATE duration=$DURATION)..."
# Pass via -e so we don't leak through process env (avoids K6_DURATION clash).
$K6_BIN run -e RATE=$RATE -e DURATION=$DURATION test/profile.js > profile-out/k6.log 2>&1 &
K6_PID=$!

echo "[profile] warming up for ${WARMUP}s before sampling..."
sleep $WARMUP

echo "[profile] perf record @ ${SAMPLE_FREQ} Hz for ${SAMPLE_DURATION}s..."
perf record -F $SAMPLE_FREQ -p $APP_PID -g -o $PERF_DATA -- sleep $SAMPLE_DURATION 2>&1 | tail -5

echo "[profile] waiting for k6 to finish..."
wait $K6_PID || true

echo "[profile] stopping backend..."
kill $APP_PID 2>/dev/null || true
wait $APP_PID 2>/dev/null || true

echo "[profile] generating perf report..."
perf report -i $PERF_DATA --stdio --no-children --percent-limit 0.5 > $PERF_REPORT 2>/dev/null || true

echo "[profile] generating flamegraph..."
perf script -i $PERF_DATA 2>/dev/null \
    | inferno-collapse-perf \
    | inferno-flamegraph --hash --title "papagaio-api sustained $RATE RPS" \
    > $FLAME_SVG

echo
echo "===== k6 summary ====="
tail -40 profile-out/k6.log
echo
echo "===== perf top symbols (>= 0.5%) ====="
head -60 $PERF_REPORT
echo
echo "[profile] flamegraph: $FLAME_SVG"
echo "[profile]    open with:   xdg-open $FLAME_SVG"

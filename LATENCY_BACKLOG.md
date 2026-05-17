# Latency Backlog

Open optimizations ordered by **ROI for lower p99 / p99.9 / max latency**. Score
(currently 5700/6000) is **explicitly out of scope** here — that's bounded by
the BORDERLINE noise floor in the distillation training set, not by anything
in the runtime. Items that would change the score (kNN exato over 3M refs) are
documented in a separate section at the bottom for completeness, but not
prioritised.

Reference baseline at the time of writing (commit `04ca1a3`):

```
Stack:    nginx stream (UDS upstream) → 2× papagaio-api (current_thread tokio)
Test:     k6 peak.js @ 900 RPS, 5 s × 3 runs
p99:      127 µs (±1 µs across runs)
p99.9:    338 µs (±90 µs across runs)
max:      578 µs (±80 µs across runs)
Profile:  10 k RPS / 30 s — handle::closure 8.5 %, tokio runtime ~7.5 %,
          kernel/network ~7 %, hyper HTTP ~4 %, memchr/vectorize ~3 %,
          slow_path::ivf_k5 = 0 % (invisible).
```

## Tier 0 — Free wins (do these next)

### 1. `MADV_HUGEPAGE` on the mmap'd index files
- **What**: `mmap.advise(memmap2::Advice::HugePage)` on `box_b_refs.bin`,
  `box_b_ivf_centroids.bin`, `box_b_ivf_offsets.bin`, `box_b_labels.bin`.
  Tells the kernel to back the mapping with 2 MB pages where possible.
- **Why it matters for us**: Mac Mini (test target) has 4 MB L3 + tiny TLB.
  At 6.75 MB our refs already overflow L3; with 4 KB pages we also blow the
  data TLB across the scan. 2 MB hugepages collapse 1 700 TLB entries to 4.
- **Reference**: `rinha-rust` does this (`MADV_HUGEPAGE` on vectors + labels).
- **Expected impact**: ~5-15 µs cut from slow-path latency on Mac Mini.
  No measurable effect on this dev box where the working set already fits.
- **Effort**: ~5 lines in `mmap_static` (after the existing `Advice::WillNeed`).
- **Risk**: zero. Kernel falls back to 4 KB if hugepages aren't available.

### 2. Inline Padé `expf` in `router::infer` softmax
- **What**: Replace the three `(logits[i] - m).exp()` calls in the softmax
  with a polynomial / Padé approximation, same recipe as the tanh swap.
- **Why**: profile shows `__expf_fma` at ~0.5 % of cycles, the only
  remaining transcendental on the hot path.
- **Expected impact**: -0.5 % CPU, ~1 µs in p50.
- **Effort**: ~15 lines (Padé fit + signature swap). Verify against rinha
  test that fp/fn don't move (softmax cares more about ordering than exact
  magnitude, so drift tolerance is high).
- **Reference**: same shape as `tanh_approx` in `backend/src/router.rs`.

### 3. Pre-rendered full HTTP response bytes (`HTTP/1.1 200…`)
- **What**: Bake the entire `HTTP/1.1 200 OK\r\nContent-Length: …\r\n\r\n{…}`
  string per fraud count into `&'static [u8]`, drop hyper's response build.
  Currently we hand hyper a `Response<Full<Bytes>>` and it serialises headers
  per request.
- **Expected impact**: 1-2 µs in p50, kills part of the alloc traffic
  (`malloc/_int_free` = 1.5 % combined in profile).
- **Effort**: low (~30 lines) but requires moving to a raw-write path —
  no longer hyper-managed serialisation. Either implement on top of hyper
  via `Response::from_parts` with pre-computed headers, OR jump straight
  to item 6 below (write raw bytes ourselves).
- **Reference**: `rinha-rust` does exactly this with hardcoded byte responses.

## Tier 1 — Bigger jobs, real latency wins

### 4. `mimalloc` or `jemalloc` global allocator
- **What**: Add `#[global_allocator]` pointing at mimalloc or jemalloc in
  `main.rs`.
- **Why**: glibc malloc shows up at ~1.5 % combined (`malloc` 0.70 %,
  `_int_free` 0.76 %) at 50 k RPS. mimalloc is famously ~2× faster on
  small allocations.
- **Expected impact**: -0.5 to -1 % CPU at 50 k RPS, basically invisible at
  900 RPS but cheap insurance.
- **Effort**: 2 lines + 1 dependency. Zero risk.

### 5. Replace `memchr::memmem::find` with hand-rolled state-machine parser
- **What**: Our `vectorize.rs` uses `memchr::memmem` to locate JSON field
  delimiters. Profile shows ~2.6 % (and ~3 % combined with `from_utf8`).
  A purpose-built parser that walks the bytes once with explicit state
  for "in-object/in-string/expecting-comma" would amortize the cost.
- **Expected impact**: -1 to -2 % CPU, ~2-3 µs in p50.
- **Effort**: medium (~150 lines, rewrites the parser). Risk: easy to
  introduce subtle bugs on edge-case inputs. Mitigated by the rinha test
  catching mismatches.

### 6. `hyper` → `httparse` + manual response writer
- **What**: Drop hyper entirely from the request path. Use `httparse` to
  parse the request head (it's incremental and zero-copy), write the
  response bytes directly to the socket.
- **Why**: profile at 10 k RPS shows hyper costing ~4-5 % combined
  (Http1Transaction::parse 1.83 %, write_head 1.28 %, HeaderName::from_bytes
  0.80 %, HeaderMap drop 0.51 %, plus the buffered IO machinery).
- **Expected impact**: -4 to -5 % CPU, ~5-10 µs in p99.
- **Effort**: high (~1-2 days). Loses hyper's HTTP/1.1 edge-case handling
  (chunked transfer, weird Connection headers, etc.). For rinha (controlled
  k6 client) the spec subset we actually receive is tiny — totally feasible
  to hand-roll. Risk: subtle correctness regressions for non-rinha clients.

### 7. `io_uring` runtime (`tokio-uring` or `glommio`)
- **What**: Move from epoll-based tokio to io_uring. Submit batches of
  read/write/accept; reap completions in bulk; reduce per-syscall overhead.
- **Why**: profile at 50 k RPS shows ~5-7 % in kernel network stack +
  vdso (clock_gettime). io_uring drops these via batched submission.
- **Expected impact**: -5 % CPU at 50 k RPS; ~negligible at 900 RPS where
  syscall load is light. Real win only on high-load scenarios outside
  rinha's spec.
- **Effort**: very high — full runtime swap, framework differs from tokio
  enough to matter (`glommio` is single-threaded per core, no work-stealing).
  Not recommended unless we're chasing a much higher load target.

## Tier 2 — Pointless for rinha's load

### 8. `SO_REUSEPORT` multi-worker in the backend
- Each worker thread gets its own listener; kernel distributes connections.
  Useful at ≥10 k RPS where one tokio worker saturates. At 900 RPS we use
  ~3 % of one CPU; multi-worker would only add scheduler overhead.

### 9. Custom Rust LB
- Already explored vs `nginx stream` mode. Stream mode is `splice()`-based
  pure TCP forwarding; a Rust equivalent saves maybe 1-3 µs at most. Owning
  a custom LB costs ~200 lines + Dockerfile + tests, payback ≤ 5 µs. **Skip.**

### 10. CPU pinning, disable C-states, performance governor
- Worth doing on our dev box to get cleaner p99.9 measurements, but we
  can't control the rinha Mac Mini's BIOS/kernel. Local-only optimisation.

## Out of scope here — score moves (not latency)

### 11. Exact kNN over the full 3M references
- Replaces the 14→32→32→3 router + 105 k Box-B brute slow path with an
  IVF (or HNSW) index over the *full* 3M reference set. Eliminates the
  distillation noise floor (3 fp + 2 fn out of 54 060 at 5 700/6000) and
  pushes the score to 6000.
- The IVF infrastructure built in this repo (`sweep_ivf.py`, `ivf_bench`,
  the Rust `ivf_k5` kernel) already supports any N. The Python side has
  `ivf.py` that builds IVF over the 3M set on GPU. Scaling slow-path
  memory from 6.75 MB → ~192 MB (3M × 16 × 4 B) breaks the L3 fit and
  reintroduces a DRAM-bandwidth bottleneck.
- This is the only meaningful score-mover left. Latency-wise it would
  *increase* p99 (slow-path queries become DRAM-bound again), so it's a
  trade against the goal of this backlog.

---

**TL;DR for "lowest possible latency":** do items 1–3 (all cheap, all
additive), then evaluate whether items 4–6 are worth the complexity given
how far below the 1 ms SLA we already are.

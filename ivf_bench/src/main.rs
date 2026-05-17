//! Microbenchmark for the IVF slow path. Isolates the `ivf_k5` kernel from
//! the HTTP server, request parsing, and MLP router so we can sweep
//! (NLIST, NPROBE) parameters and read pure kernel latency.
//!
//! Inputs are mmap'd raw binaries — same layout the backend uses in prod.
//! Recall is computed against a brute-force ground-truth file (optional;
//! defaults to data/rust_results.bin from the bench/ crate).
//!
//! Vectors are int16 with scale=10000. The query file (`stress_queries.bin`)
//! is still f32 — we quantize it once at startup (it stays on the round4
//! grid, so the cast is exact).
//!
//! Inputs (env vars):
//!   REFS         path to refs.i16.bin   (i16 × N × 16, padded, sorted by cluster)
//!   LABELS       path to labels.bin     (u8  × N,      sorted same as refs)
//!   CENTROIDS    path to centroids.i16.bin (i16 × nlist × 16, padded)
//!   OFFSETS      path to offsets.bin    (u32 × nlist+1, CSR row offsets)
//!   QUERIES      path to query vectors  (f32 × Q × 16, padded; default
//!                                       data/stress_queries.bin = 100k)
//!   GROUND_TRUTH path to brute counts   (u8 × Q, optional; default
//!                                       data/rust_results.bin)
//!   NPROBE       16
//!   N_QUERIES    20000   (cap; sample budget controls p99 resolution)
//!   WARMUP       1000    (queries timed but discarded; primes caches)
//!   NLIST_LABEL  ""      (cosmetic; echoed back in the first column)
//!
//! Output: one tab-separated line to stdout:
//!   nlist_label  nprobe  samples  p50_us  p90_us  p99_us  rps  recall_pct

use std::arch::x86_64::*;
use std::env;
use std::fs::File;
use std::time::Instant;

use memmap2::Mmap;

const D_PADDED: usize = 16;
const K: usize = 5;
const MAX_NPROBE: usize = 64;
const PREFETCH_AHEAD: usize = 8;

// === Kernel — kept in sync with backend/src/slow_path.rs ===========

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn distance_i16(q: __m256i, r_ptr: *const i16) -> f32 {
    let r = _mm256_loadu_si256(r_ptr as *const __m256i);
    let diff = _mm256_sub_epi16(q, r);
    let dot = _mm256_madd_epi16(diff, diff);
    let dot_f = _mm256_cvtepi32_ps(dot);
    let hi = _mm256_extractf128_ps::<1>(dot_f);
    let lo = _mm256_castps256_ps128(dot_f);
    let s128 = _mm_add_ps(hi, lo);
    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(s128, s128);
    let s1 = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(s1, s1);
    _mm_cvtss_f32(_mm_add_ss(s1, shuf2))
}

#[inline(always)]
fn insert_top_k_refs(best_d: &mut [f32; K], best_idx: &mut [u32; K], d: f32, i: u32) {
    if d < best_d[K - 1] {
        let mut pos = K - 1;
        while pos > 0 && best_d[pos - 1] > d {
            best_d[pos] = best_d[pos - 1];
            best_idx[pos] = best_idx[pos - 1];
            pos -= 1;
        }
        best_d[pos] = d;
        best_idx[pos] = i;
    }
}

#[inline(always)]
fn insert_top_centroid(top_d: &mut [f32], top_idx: &mut [u32], d: f32, c: u32) {
    let n = top_d.len();
    if d < top_d[n - 1] {
        let mut pos = n - 1;
        while pos > 0 && top_d[pos - 1] > d {
            top_d[pos] = top_d[pos - 1];
            top_idx[pos] = top_idx[pos - 1];
            pos -= 1;
        }
        top_d[pos] = d;
        top_idx[pos] = c;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn ivf_k5(
    query: &[i16; D_PADDED],
    centroids: &[i16],
    offsets: &[u32],
    refs: &[i16],
    labels: &[u8],
    nprobe: usize,
) -> u8 {
    let nlist = offsets.len() - 1;
    let q = _mm256_loadu_si256(query.as_ptr() as *const __m256i);

    let mut top_c_d_buf = [f32::INFINITY; MAX_NPROBE];
    let mut top_c_idx_buf = [0u32; MAX_NPROBE];
    let top_c_d = &mut top_c_d_buf[..nprobe];
    let top_c_idx = &mut top_c_idx_buf[..nprobe];
    let cptr = centroids.as_ptr();
    for c in 0..nlist {
        let d = distance_i16(q, cptr.add(c * D_PADDED));
        insert_top_centroid(top_c_d, top_c_idx, d, c as u32);
    }

    let mut best_d = [f32::INFINITY; K];
    let mut best_idx = [0u32; K];
    let rptr = refs.as_ptr();
    for p in 0..nprobe {
        let cluster = top_c_idx[p] as usize;
        let start = offsets[cluster] as usize;
        let end = offsets[cluster + 1] as usize;
        if start == end {
            continue;
        }
        let count = end - start;
        let n_pf = count.saturating_sub(PREFETCH_AHEAD);
        for i in 0..n_pf {
            let row = start + i;
            let pf_ptr = rptr.add((row + PREFETCH_AHEAD) * D_PADDED) as *const i8;
            _mm_prefetch::<{ _MM_HINT_T0 }>(pf_ptr);
            let d = distance_i16(q, rptr.add(row * D_PADDED));
            insert_top_k_refs(&mut best_d, &mut best_idx, d, row as u32);
        }
        for i in n_pf..count {
            let row = start + i;
            let d = distance_i16(q, rptr.add(row * D_PADDED));
            insert_top_k_refs(&mut best_d, &mut best_idx, d, row as u32);
        }
    }

    let mut count: u16 = 0;
    for i in 0..K {
        count += labels[best_idx[i] as usize] as u16;
    }
    count as u8
}

// === Bench harness =================================================

fn mmap_static(path: &str) -> &'static [u8] {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mmap = unsafe {
        Mmap::map(&file).unwrap_or_else(|e| panic!("mmap {path}: {e}"))
    };
    let _ = mmap.advise(memmap2::Advice::WillNeed);
    let leaked: &'static Mmap = Box::leak(Box::new(mmap));
    &leaked[..]
}

fn touch_pages(bytes: &[u8]) {
    let mut acc: u64 = 0;
    let mut i = 0;
    while i < bytes.len() {
        acc = acc.wrapping_add(bytes[i] as u64);
        i += 4096;
    }
    std::hint::black_box(acc);
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_or_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    if !is_x86_feature_detected!("avx2") {
        panic!("AVX2 required");
    }

    let nlist_label = env_or("NLIST_LABEL", "?");
    let refs_path = env_or("REFS", "data/box_b_refs.i16.bin");
    let labels_path = env_or("LABELS", "data/box_b_labels.bin");
    let centroids_path = env_or("CENTROIDS", "data/box_b_ivf_centroids.i16.bin");
    let offsets_path = env_or("OFFSETS", "data/box_b_ivf_offsets.bin");
    let queries_path = env_or("QUERIES", "data/stress_queries.bin");
    let gt_path = env_or("GROUND_TRUTH", "data/rust_results.bin");
    let nprobe: usize = env_or_parse("NPROBE", 16usize);
    let n_queries_cap: usize = env_or_parse("N_QUERIES", 20_000usize);
    let warmup: usize = env_or_parse("WARMUP", 1_000usize);

    let refs_bytes = mmap_static(&refs_path);
    let labels_bytes = mmap_static(&labels_path);
    let cent_bytes = mmap_static(&centroids_path);
    let offs_bytes = mmap_static(&offsets_path);
    let q_bytes = mmap_static(&queries_path);
    touch_pages(refs_bytes);
    touch_pages(labels_bytes);
    touch_pages(cent_bytes);
    touch_pages(offs_bytes);
    touch_pages(q_bytes);

    let refs: &'static [i16] = bytemuck::cast_slice(refs_bytes);
    let labels: &'static [u8] = labels_bytes;
    let centroids: &'static [i16] = bytemuck::cast_slice(cent_bytes);
    let offsets: &'static [u32] = bytemuck::cast_slice(offs_bytes);
    let queries_f32: &'static [f32] = bytemuck::cast_slice(q_bytes);

    let n_q_full = queries_f32.len() / D_PADDED;
    let n_q = n_q_full.min(n_queries_cap.max(warmup + 1));
    let nlist = offsets.len() - 1;
    assert!(nprobe >= 1 && nprobe <= nlist.min(MAX_NPROBE), "bad nprobe");
    assert_eq!(refs.len(), labels.len() * D_PADDED, "refs/labels mismatch");
    assert_eq!(centroids.len(), nlist * D_PADDED, "centroids/offsets mismatch");

    // Pre-quantize the timed slice of queries once. The f32 file is on the
    // round4 grid (rounded by the data generator that produced it), so the
    // cast is exact. Leaks the buffer so the timed loop can use &'static.
    let total_timed = n_q; // includes warmup
    let mut q_i16_buf: Vec<i16> = Vec::with_capacity(total_timed * D_PADDED);
    for qi in 0..total_timed {
        for d in 0..D_PADDED {
            let v = queries_f32[qi * D_PADDED + d];
            q_i16_buf.push((v * 10000.0).round() as i16);
        }
    }
    let queries_i16: &'static [i16] = Box::leak(q_i16_buf.into_boxed_slice());

    // Optional ground-truth file (brute-force fraud counts from the bench
    // crate). We compare against it on the post-warmup samples to compute
    // approximation recall — verdict agreement, not just exact-count match.
    let gt: Option<&'static [u8]> = if std::path::Path::new(&gt_path).exists() {
        let b = mmap_static(&gt_path);
        touch_pages(b);
        Some(b)
    } else {
        None
    };

    // Warm-up: untimed pass over the first `warmup` queries. Primes the
    // instruction cache, branch predictors, and steady-state TLB.
    {
        let mut sink: u32 = 0;
        for qi in 0..warmup.min(n_q) {
            let q: &[i16; D_PADDED] = (&queries_i16[qi * D_PADDED..qi * D_PADDED + D_PADDED])
                .try_into()
                .unwrap();
            let v = unsafe { ivf_k5(q, centroids, offsets, refs, labels, nprobe) };
            sink = sink.wrapping_add(v as u32);
        }
        std::hint::black_box(sink);
    }

    // Timed loop. Per-query Instant::now is ~25 ns of overhead on Haswell,
    // small vs. even the fastest IVF query, and gets us per-sample
    // percentiles instead of batched averages.
    let mut samples: Vec<u64> = Vec::with_capacity(n_q.saturating_sub(warmup));
    let mut answers: Vec<u8> = Vec::with_capacity(n_q.saturating_sub(warmup));
    let total_t0 = Instant::now();
    for qi in warmup..n_q {
        let q: &[i16; D_PADDED] = (&queries_i16[qi * D_PADDED..qi * D_PADDED + D_PADDED])
            .try_into()
            .unwrap();
        let t0 = Instant::now();
        let v = unsafe { ivf_k5(q, centroids, offsets, refs, labels, nprobe) };
        let dt = t0.elapsed().as_nanos() as u64;
        samples.push(dt);
        answers.push(v);
    }
    let total_elapsed = total_t0.elapsed();
    assert!(!samples.is_empty(), "no timed samples — bump N_QUERIES");

    samples.sort_unstable();
    let p50 = samples[samples.len() * 50 / 100];
    let p90 = samples[samples.len() * 90 / 100];
    let p99 = samples[samples.len() * 99 / 100];
    let rps = samples.len() as f64 / total_elapsed.as_secs_f64();

    let recall_pct = match gt {
        Some(g) => {
            let mut agree = 0usize;
            for (offset, &a) in answers.iter().enumerate() {
                let qi = warmup + offset;
                if qi < g.len() && a == g[qi] {
                    agree += 1;
                }
            }
            100.0 * agree as f64 / answers.len() as f64
        }
        None => f64::NAN,
    };

    println!(
        "{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.0}\t{:.4}",
        nlist_label,
        nprobe,
        samples.len(),
        p50 as f64 / 1e3,
        p90 as f64 / 1e3,
        p99 as f64 / 1e3,
        rps,
        recall_pct,
    );
}

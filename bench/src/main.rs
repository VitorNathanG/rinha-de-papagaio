//! Rust brute-force k=5 over the Box-B subset.
//!
//! - AVX2 + FMA inner loop.
//! - Row-major refs padded to 16 floats per row (one 64-byte cache line each).
//! - Software prefetch 8 refs ahead.
//! - Single-threaded — production p99 cares about single-query latency.
//!
//! Input files (raw binary, little-endian, no header):
//!   REFS         : f32, shape (N_refs, 16), padded with 0 at positions 14,15
//!   LABELS       : u8,  shape (N_refs,), 0 = legit, 1 = fraud
//!   QUERIES      : f32, shape (N_queries, 16), padded the same way
//! Output:
//!   OUT          : u8,  shape (N_queries,), fraud count among 5 nearest (0..5)
//!
//! Defaults:
//!   REFS=data/box_b_refs.bin
//!   LABELS=data/box_b_labels.bin
//!   QUERIES=data/stress_queries.bin
//!   OUT=data/rust_results.bin

#![allow(clippy::needless_range_loop)]

use std::arch::x86_64::*;
use std::time::Instant;

const D_PADDED: usize = 16;
const K: usize = 5;
const PREFETCH_AHEAD: usize = 8;

/// Compute ||q - r||^2 where q is preloaded in two ymm registers and r is in
/// memory at r_ptr (16 floats = 1 cache line).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn distance(q0: __m256, q1: __m256, r_ptr: *const f32) -> f32 {
    let r0 = _mm256_loadu_ps(r_ptr);
    let r1 = _mm256_loadu_ps(r_ptr.add(8));
    let diff0 = _mm256_sub_ps(q0, r0);
    let diff1 = _mm256_sub_ps(q1, r1);
    // sq0 = diff0^2 ; sum = diff1^2 + sq0
    let sq0 = _mm256_mul_ps(diff0, diff0);
    let sum = _mm256_fmadd_ps(diff1, diff1, sq0);

    // Horizontal sum of 8 floats in `sum` -> scalar.
    let hi = _mm256_extractf128_ps::<1>(sum);
    let lo = _mm256_castps256_ps128(sum);
    let s128 = _mm_add_ps(hi, lo);                  // [a,b,c,d]
    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(s128, s128);
    let s1 = _mm_add_ps(s128, shuf);                // [a+b, b+a, c+d, d+c]
    let shuf2 = _mm_movehl_ps(s1, s1);
    let final_sum = _mm_add_ss(s1, shuf2);
    _mm_cvtss_f32(final_sum)
}

/// Insertion sort with K=5: keep best_d/best_idx sorted ascending by distance.
/// Most calls fail the outer `if`, so the hot path is one compare + branch.
#[inline(always)]
fn insert_top_k(best_d: &mut [f32; K], best_idx: &mut [usize; K], d: f32, i: usize) {
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

#[target_feature(enable = "avx2,fma")]
unsafe fn brute_k5(query: &[f32; D_PADDED], refs: &[f32], labels: &[u8]) -> u8 {
    let n = labels.len();
    debug_assert_eq!(refs.len(), n * D_PADDED);

    // Load query into two ymm registers (held across the whole loop).
    let q0 = _mm256_loadu_ps(query.as_ptr());
    let q1 = _mm256_loadu_ps(query.as_ptr().add(8));

    let mut best_d = [f32::INFINITY; K];
    let mut best_idx = [0usize; K];

    let refs_ptr = refs.as_ptr();
    let n_pf = n.saturating_sub(PREFETCH_AHEAD);

    // Prefetch-aware main loop.
    for i in 0..n_pf {
        let pf_ptr = refs_ptr.add((i + PREFETCH_AHEAD) * D_PADDED) as *const i8;
        _mm_prefetch::<{ _MM_HINT_T0 }>(pf_ptr);

        let r_ptr = refs_ptr.add(i * D_PADDED);
        let d = distance(q0, q1, r_ptr);
        insert_top_k(&mut best_d, &mut best_idx, d, i);
    }
    // Tail (no prefetch needed).
    for i in n_pf..n {
        let r_ptr = refs_ptr.add(i * D_PADDED);
        let d = distance(q0, q1, r_ptr);
        insert_top_k(&mut best_d, &mut best_idx, d, i);
    }

    let mut count: u16 = 0;
    for i in 0..K {
        count += labels[best_idx[i]] as u16;
    }
    count as u8
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn read_file(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn main() {
    let refs_path = env_or("REFS", "data/box_b_refs.bin");
    let labels_path = env_or("LABELS", "data/box_b_labels.bin");
    let queries_path = env_or("QUERIES", "data/stress_queries.bin");
    let out_path = env_or("OUT", "data/rust_results.bin");

    let refs_bytes = read_file(&refs_path);
    let labels: Vec<u8> = read_file(&labels_path);
    let queries_bytes = read_file(&queries_path);

    let refs: &[f32] = bytemuck::cast_slice(&refs_bytes);
    let queries: &[f32] = bytemuck::cast_slice(&queries_bytes);

    assert_eq!(refs.len() % D_PADDED, 0, "refs not multiple of {D_PADDED}");
    assert_eq!(queries.len() % D_PADDED, 0, "queries not multiple of {D_PADDED}");
    let n_refs = refs.len() / D_PADDED;
    let n_queries = queries.len() / D_PADDED;
    assert_eq!(n_refs, labels.len(), "refs/labels length mismatch");

    eprintln!("[rust] refs:    {} × {} ({:.2} MB)", n_refs, D_PADDED, refs_bytes.len() as f64 / 1e6);
    eprintln!("[rust] queries: {}", n_queries);
    eprintln!("[rust] features: AVX2+FMA, prefetch ahead = {} refs", PREFETCH_AHEAD);

    // Warm-up so the timed run sees a hot L3.
    let mut warmup = 0u32;
    for _ in 0..2 {
        for q in 0..16.min(n_queries) {
            let q_arr: &[f32; D_PADDED] = queries[q * D_PADDED..(q + 1) * D_PADDED]
                .try_into().unwrap();
            warmup += unsafe { brute_k5(q_arr, refs, &labels) } as u32;
        }
    }
    std::hint::black_box(warmup);

    let mut results = vec![0u8; n_queries];
    let start = Instant::now();
    for q in 0..n_queries {
        let q_arr: &[f32; D_PADDED] = queries[q * D_PADDED..(q + 1) * D_PADDED]
            .try_into().unwrap();
        results[q] = unsafe { brute_k5(q_arr, refs, &labels) };
    }
    let elapsed = start.elapsed();

    let qps = n_queries as f64 / elapsed.as_secs_f64();
    let us_per_query = 1e6 / qps;
    eprintln!("[rust] processed {} queries in {:.3}s", n_queries, elapsed.as_secs_f64());
    eprintln!("[rust] throughput: {:.0} q/s", qps);
    eprintln!("[rust] latency:    {:.2} µs/query  (mean; check N_REFS={n_refs} for fairness)", us_per_query);

    std::fs::write(&out_path, &results).unwrap_or_else(|e| panic!("write {out_path}: {e}"));
    eprintln!("[rust] wrote {}", out_path);
}

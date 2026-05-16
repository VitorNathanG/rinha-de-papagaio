//! Brute-force k=5 over the Box-B subset. AVX2+FMA inner loop, row-major refs
//! padded to 16 floats per row (one 64-byte cache line). Same kernel as
//! /bench/src/main.rs — kept duplicated here to keep crates independent.

use std::arch::x86_64::*;

const D_PADDED: usize = 16;
const K: usize = 5;
const PREFETCH_AHEAD: usize = 8;

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn distance(q0: __m256, q1: __m256, r_ptr: *const f32) -> f32 {
    let r0 = _mm256_loadu_ps(r_ptr);
    let r1 = _mm256_loadu_ps(r_ptr.add(8));
    let diff0 = _mm256_sub_ps(q0, r0);
    let diff1 = _mm256_sub_ps(q1, r1);
    let sq0 = _mm256_mul_ps(diff0, diff0);
    let sum = _mm256_fmadd_ps(diff1, diff1, sq0);

    let hi = _mm256_extractf128_ps::<1>(sum);
    let lo = _mm256_castps256_ps128(sum);
    let s128 = _mm_add_ps(hi, lo);
    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(s128, s128);
    let s1 = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(s1, s1);
    let final_sum = _mm_add_ss(s1, shuf2);
    _mm_cvtss_f32(final_sum)
}

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

/// SAFETY: caller must ensure AVX2 + FMA are available. Refs must be
/// `labels.len() * 16` floats, row-major.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn brute_k5(query: &[f32; D_PADDED], refs: &[f32], labels: &[u8]) -> u8 {
    let n = labels.len();
    debug_assert_eq!(refs.len(), n * D_PADDED);

    let q0 = _mm256_loadu_ps(query.as_ptr());
    let q1 = _mm256_loadu_ps(query.as_ptr().add(8));

    let mut best_d = [f32::INFINITY; K];
    let mut best_idx = [0usize; K];

    let refs_ptr = refs.as_ptr();
    let n_pf = n.saturating_sub(PREFETCH_AHEAD);

    for i in 0..n_pf {
        let pf_ptr = refs_ptr.add((i + PREFETCH_AHEAD) * D_PADDED) as *const i8;
        _mm_prefetch::<{ _MM_HINT_T0 }>(pf_ptr);
        let r_ptr = refs_ptr.add(i * D_PADDED);
        let d = distance(q0, q1, r_ptr);
        insert_top_k(&mut best_d, &mut best_idx, d, i);
    }
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

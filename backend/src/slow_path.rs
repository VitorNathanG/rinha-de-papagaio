//! Approximate k=5 over the Box-B subset via an IVF (Inverted File) index.
//!
//! Per query:
//!   1. AVX2+FMA distance to every centroid (small N, typically 256).
//!   2. Top-nprobe centroids picked via fixed-size insertion sort.
//!   3. For each probed cluster, exact distance to its refs (contiguous slice
//!      from CSR offsets). Top-5 tracked by the same insertion sort pattern.
//!   4. Sum labels of the 5 nearest → fraud count 0..5.
//!
//! Refs and labels are pre-sorted by cluster in the binary so the inner scan
//! is sequential — no scatter, no per-cluster indirection. With nlist=256 +
//! nprobe=16, working-set per query is ~16 KB (centroids, L1) + ~420 KB
//! (probed refs, L2-warm) instead of the 6.75 MB brute force scan, fitting
//! the Mac Mini's 4 MB L3 with room to spare.

use std::arch::x86_64::*;

const D_PADDED: usize = 16;
const K: usize = 5;
const MAX_NPROBE: usize = 64;
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

/// SAFETY: caller must ensure AVX2 + FMA are available. Layouts must match
/// the Python export: refs and labels are sorted by cluster; `offsets` is
/// CSR over *row count* (not bytes); centroids are padded to 16 floats per
/// row. `nprobe` must be in `1..=MAX_NPROBE`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ivf_k5(
    query: &[f32; D_PADDED],
    centroids: &[f32],
    offsets: &[u32],
    refs: &[f32],
    labels: &[u8],
    nprobe: usize,
) -> u8 {
    debug_assert!((1..=MAX_NPROBE).contains(&nprobe));
    let nlist = offsets.len() - 1;
    debug_assert_eq!(centroids.len(), nlist * D_PADDED);

    let q0 = _mm256_loadu_ps(query.as_ptr());
    let q1 = _mm256_loadu_ps(query.as_ptr().add(8));

    // 1. Top-nprobe centroids via fixed-size insertion sort on the stack.
    let mut top_c_d_buf = [f32::INFINITY; MAX_NPROBE];
    let mut top_c_idx_buf = [0u32; MAX_NPROBE];
    let top_c_d = &mut top_c_d_buf[..nprobe];
    let top_c_idx = &mut top_c_idx_buf[..nprobe];
    let cptr = centroids.as_ptr();
    for c in 0..nlist {
        let d = distance(q0, q1, cptr.add(c * D_PADDED));
        insert_top_centroid(top_c_d, top_c_idx, d, c as u32);
    }

    // 2. Scan refs in those clusters. Track row indices globally (refs and
    //    labels are sorted in the same order, so global row == label index).
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
            let d = distance(q0, q1, rptr.add(row * D_PADDED));
            insert_top_k_refs(&mut best_d, &mut best_idx, d, row as u32);
        }
        for i in n_pf..count {
            let row = start + i;
            let d = distance(q0, q1, rptr.add(row * D_PADDED));
            insert_top_k_refs(&mut best_d, &mut best_idx, d, row as u32);
        }
    }

    let mut count: u16 = 0;
    for i in 0..K {
        count += labels[best_idx[i] as usize] as u16;
    }
    count as u8
}

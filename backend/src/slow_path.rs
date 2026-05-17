//! Approximate k=5 over the Box-B subset via an IVF (Inverted File) index.
//!
//! Per query:
//!   1. AVX2 distance to every centroid (small N, typically 256-2048).
//!   2. Top-nprobe centroids picked via fixed-size insertion sort.
//!   3. For each probed cluster, exact distance to its refs (contiguous slice
//!      from CSR offsets). Top-5 tracked by the same insertion sort pattern.
//!   4. Sum labels of the 5 nearest → fraud count 0..5.
//!
//! Vectors are stored as int16 with scale=10000 (the vectorize step rounds
//! every dim to k/10000, so this is bit-exact on the round4 grid for refs;
//! centroids are k-means means and pick up sub-cluster-radius rounding).
//!
//! The squared distance per ref is one `_mm256_madd_epi16(diff, diff)`,
//! which reduces 16 int16 diffs into 8 int32 pair-sums in a single op.
//! Those 8 int32 lanes are converted to f32 before horizontal sum: the
//! true int32 total can reach 14·(2·10000)² ≈ 5.6e9 (overflows i32), but
//! each pair-sum lane fits comfortably and the f32 reduction loses at most
//! a few ULPs — well under the rank-5/6 gap we ever care about.
//!
//! Refs and labels are pre-sorted by cluster in the binary so the inner scan
//! is sequential — no scatter, no per-cluster indirection. Each ref row is
//! 32 bytes (16 × int16), so two refs share a 64-byte cache line.

use std::arch::x86_64::*;

const D_PADDED: usize = 16;
const K: usize = 5;
const MAX_NPROBE: usize = 64;
const PREFETCH_AHEAD: usize = 8;

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn distance_i16(q: __m256i, r_ptr: *const i16) -> f32 {
    let r = _mm256_loadu_si256(r_ptr as *const __m256i);
    let diff = _mm256_sub_epi16(q, r);
    // madd_epi16: (a0,a1,...,a15) ⊙ (b0,b1,...,b15) → 8 × i32 pair-sums
    //             [a0·b0 + a1·b1, a2·b2 + a3·b3, ..., a14·b14 + a15·b15]
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

/// SAFETY: caller must ensure AVX2 is available. Layouts must match the
/// Python export: refs and labels are sorted by cluster; `offsets` is CSR
/// over *row count* (not bytes); centroids are padded to 16 int16 per row.
/// `nprobe` must be in `1..=MAX_NPROBE`.
#[target_feature(enable = "avx2")]
pub unsafe fn ivf_k5(
    query: &[i16; D_PADDED],
    centroids: &[i16],
    offsets: &[u32],
    refs: &[i16],
    labels: &[u8],
    nprobe: usize,
) -> u8 {
    debug_assert!((1..=MAX_NPROBE).contains(&nprobe));
    let nlist = offsets.len() - 1;
    debug_assert_eq!(centroids.len(), nlist * D_PADDED);

    let q = _mm256_loadu_si256(query.as_ptr() as *const __m256i);

    // 1. Top-nprobe centroids via fixed-size insertion sort on the stack.
    let mut top_c_d_buf = [f32::INFINITY; MAX_NPROBE];
    let mut top_c_idx_buf = [0u32; MAX_NPROBE];
    let top_c_d = &mut top_c_d_buf[..nprobe];
    let top_c_idx = &mut top_c_idx_buf[..nprobe];
    let cptr = centroids.as_ptr();
    for c in 0..nlist {
        let d = distance_i16(q, cptr.add(c * D_PADDED));
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

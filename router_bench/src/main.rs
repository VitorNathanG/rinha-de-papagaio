//! Microbenchmark do MLP router (14 → 64 → 64 → 3 com GELU(tanh) Padé inline).
//! Isola `router::infer` do HTTP server, parser de request e do slow path pra
//! ler latência pura do kernel.
//!
//! O kernel é duplicado de `backend/src/router.rs` deliberadamente — mesma
//! razão de `slow_path` duplicado em `ivf_bench`: o crate do bench fica
//! independente do crate do backend. Se mexer no kernel (Padé, layout dos
//! pesos, shape do MLP), atualize os dois lados.
//!
//! Inputs (env vars):
//!   WEIGHTS    path to router_weights.bin   (default data/router_weights.bin)
//!   QUERIES    path to query vectors        (default data/stress_queries.bin)
//!                                           f32 × Q × 16, padded. Só lê as 14
//!                                           primeiras dims por query. Se o
//!                                           arquivo não existir, geramos
//!                                           queries random com seed fixa —
//!                                           bom o suficiente pra medir
//!                                           latência (router::infer não tem
//!                                           branch data-dependent fora da
//!                                           saturação rara do tanh).
//!   N_QUERIES  cap em queries a cronometrar (default 100_000)
//!   WARMUP     queries descartadas pra primar caches (default 5_000)
//!   BATCH      queries por Instant::now     (default 1)
//!                                           1 = per-query timing (alto noise
//!                                             de overhead do clock ~25 ns).
//!                                           >1 = chunked timing (menos
//!                                             granularidade, p99 perde
//!                                             resolução, mas overhead some).
//!   SEED       seed para QUERIES random     (default 42)
//!
//! Output: linha tab-separada em stdout
//!   samples  batch  p50_ns  p90_ns  p99_ns  mean_ns  rps

use std::env;
use std::fs::File;
use std::time::Instant;

use memmap2::Mmap;

// === Kernel — kept in sync with backend/src/router.rs ===============

const D_IN: usize = 14;
const H: usize = 64;
const D_OUT: usize = 3;

const N_FLOATS: usize = D_IN * H + H + H * H + H + H * D_OUT + D_OUT; // 5315

struct Weights {
    w1: [[f32; D_IN]; H],
    b1: [f32; H],
    w2: [[f32; H]; H],
    b2: [f32; H],
    w3: [[f32; H]; D_OUT],
    b3: [f32; D_OUT],
}

fn load_weights(bytes: &[u8]) -> Weights {
    let expected = N_FLOATS * 4;
    assert_eq!(
        bytes.len(),
        expected,
        "router_weights.bin should be {expected} bytes, got {}",
        bytes.len()
    );
    let f: &[f32] = bytemuck::cast_slice(bytes);
    let mut o = 0;

    let mut w1 = [[0f32; D_IN]; H];
    for i in 0..H {
        for j in 0..D_IN {
            w1[i][j] = f[o];
            o += 1;
        }
    }
    let mut b1 = [0f32; H];
    for i in 0..H {
        b1[i] = f[o];
        o += 1;
    }

    let mut w2 = [[0f32; H]; H];
    for i in 0..H {
        for j in 0..H {
            w2[i][j] = f[o];
            o += 1;
        }
    }
    let mut b2 = [0f32; H];
    for i in 0..H {
        b2[i] = f[o];
        o += 1;
    }

    let mut w3 = [[0f32; H]; D_OUT];
    for i in 0..D_OUT {
        for j in 0..H {
            w3[i][j] = f[o];
            o += 1;
        }
    }
    let mut b3 = [0f32; D_OUT];
    for i in 0..D_OUT {
        b3[i] = f[o];
        o += 1;
    }
    debug_assert_eq!(o, N_FLOATS);
    Weights { w1, b1, w2, b2, w3, b3 }
}

#[inline]
fn tanh_approx(x: f32) -> f32 {
    if x.abs() >= 4.97 {
        return x.signum();
    }
    let x2 = x * x;
    let num = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let den = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    num / den
}

#[inline]
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEFF: f32 = 0.044715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + tanh_approx(inner))
}

#[inline]
fn infer(w: &Weights, x: &[f32; 16]) -> [f32; D_OUT] {
    let mut h1 = [0f32; H];
    for i in 0..H {
        let mut s = w.b1[i];
        for j in 0..D_IN {
            s += w.w1[i][j] * x[j];
        }
        h1[i] = gelu(s);
    }

    let mut h2 = [0f32; H];
    for i in 0..H {
        let mut s = w.b2[i];
        for j in 0..H {
            s += w.w2[i][j] * h1[j];
        }
        h2[i] = gelu(s);
    }

    let mut logits = [0f32; D_OUT];
    for i in 0..D_OUT {
        let mut s = w.b3[i];
        for j in 0..H {
            s += w.w3[i][j] * h2[j];
        }
        logits[i] = s;
    }

    let m = logits[0].max(logits[1]).max(logits[2]);
    let e0 = (logits[0] - m).exp();
    let e1 = (logits[1] - m).exp();
    let e2 = (logits[2] - m).exp();
    let s = e0 + e1 + e2;
    [e0 / s, e1 / s, e2 / s]
}

// === Bench harness =================================================

fn mmap_static(path: &str) -> &'static [u8] {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mmap = unsafe { Mmap::map(&file).unwrap_or_else(|e| panic!("mmap {path}: {e}")) };
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
    let weights_path = env_or("WEIGHTS", "data/router_weights.bin");
    let queries_path = env_or("QUERIES", "data/stress_queries.bin");
    let n_queries_cap: usize = env_or_parse("N_QUERIES", 100_000usize);
    let warmup: usize = env_or_parse("WARMUP", 5_000usize);
    let batch: usize = env_or_parse("BATCH", 1usize);
    let seed: u64 = env_or_parse("SEED", 42u64);
    assert!(batch >= 1, "BATCH must be >= 1");

    let w_bytes = mmap_static(&weights_path);
    touch_pages(w_bytes);
    let w = load_weights(w_bytes);

    // Queries: prefere o arquivo se existir, senão gera N_QUERIES random
    // reproduzíveis com SEED. Para o kernel router::infer, conteúdo importa
    // pouco — o único branch data-dependent é a saturação do tanh em |x|≥4.97,
    // raríssimo pra inputs realistas (round4 grid, ~0..1).
    let queries_owned: Vec<f32> = if std::path::Path::new(&queries_path).exists() {
        let q_bytes = mmap_static(&queries_path);
        touch_pages(q_bytes);
        let slice: &[f32] = bytemuck::cast_slice(q_bytes);
        slice.to_vec()
    } else {
        eprintln!(
            "[router-bench] {} não existe — gerando {} queries random (SEED={})",
            queries_path, n_queries_cap, seed,
        );
        // xorshift64 reproduzível, mapeado pra f32 em [0,1) na 14 dims úteis.
        let mut state = seed.max(1);
        let mut v = Vec::with_capacity(n_queries_cap * 16);
        for _ in 0..n_queries_cap {
            for d in 0..16 {
                if d < 14 {
                    // xorshift64
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state >> 32) as u32;
                    v.push((u as f32) / (u32::MAX as f32));
                } else {
                    v.push(0.0);
                }
            }
        }
        v
    };
    let queries: &[f32] = &queries_owned;
    let n_q_full = queries.len() / 16;
    let n_q = n_q_full.min(n_queries_cap.max(warmup + batch));
    assert!(n_q > warmup, "not enough queries after warmup");

    // Warmup — primes I-cache, branch predictor, page TLB.
    {
        let mut sink: f32 = 0.0;
        for qi in 0..warmup.min(n_q) {
            let q: &[f32; 16] = (&queries[qi * 16..qi * 16 + 16]).try_into().unwrap();
            let p = infer(&w, q);
            sink += p[0] + p[1] + p[2];
        }
        std::hint::black_box(sink);
    }

    // Timed loop. Per-query Instant::now adds ~25 ns of overhead on Haswell,
    // which is in the same order as router::infer itself. Use BATCH>1 to
    // amortize that overhead at the cost of losing per-query percentile
    // resolution (you get one sample per BATCH queries).
    let n_timed = n_q - warmup;
    let n_batches = n_timed / batch;
    let n_used = n_batches * batch;
    let mut per_batch_ns: Vec<u64> = Vec::with_capacity(n_batches);
    let mut sink: f32 = 0.0;
    let total_t0 = Instant::now();
    for b in 0..n_batches {
        let base = warmup + b * batch;
        let t0 = Instant::now();
        for k in 0..batch {
            let qi = base + k;
            let q: &[f32; 16] = (&queries[qi * 16..qi * 16 + 16]).try_into().unwrap();
            let p = std::hint::black_box(infer(&w, q));
            sink += p[0] + p[1] + p[2];
        }
        per_batch_ns.push(t0.elapsed().as_nanos() as u64);
    }
    let total_elapsed = total_t0.elapsed();
    std::hint::black_box(sink);
    assert!(!per_batch_ns.is_empty(), "no timed samples — bump N_QUERIES");

    // Normalize to per-query latency (ns) so percentiles are comparable
    // across BATCH settings.
    let mut per_query_ns: Vec<u64> = per_batch_ns
        .iter()
        .map(|&v| v / batch as u64)
        .collect();
    per_query_ns.sort_unstable();
    let p50 = per_query_ns[per_query_ns.len() * 50 / 100];
    let p90 = per_query_ns[per_query_ns.len() * 90 / 100];
    let p99 = per_query_ns[per_query_ns.len() * 99 / 100];
    let mean: u64 = (per_query_ns.iter().sum::<u64>()) / per_query_ns.len() as u64;
    let rps = n_used as f64 / total_elapsed.as_secs_f64();

    // Header (printed once, so a downstream tool can parse repeated runs).
    eprintln!("samples\tbatch\tp50_ns\tp90_ns\tp99_ns\tmean_ns\trps");
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{:.0}",
        n_used, batch, p50, p90, p99, mean, rps,
    );
}

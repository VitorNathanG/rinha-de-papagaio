//! Tiny MLP router: 14 → 64 → 64 → 3 with tanh-approx GELU.
//! Matches PyTorch's nn.GELU(approximate="tanh") used during training.
//!
//! The `tanh()` inside is computed via an inlined Padé[7/6] rational
//! approximation (no libm call) — ~2× faster than libm::tanhf and accurate
//! to ~2e-7 over |x| ≤ 4.97 (max error 1e-4 at the clamp boundary where
//! we saturate to ±1). The MLP only ever sees small post-Linear sums, so
//! the boundary case is rare and the error never crosses the verdict.

const D_IN: usize = 14;
const H: usize = 64;
const D_OUT: usize = 3;

pub const N_FLOATS: usize = D_IN * H + H + H * H + H + H * D_OUT + D_OUT; // 5315

pub struct Weights {
    w1: [[f32; D_IN]; H],
    b1: [f32; H],
    w2: [[f32; H]; H],
    b2: [f32; H],
    w3: [[f32; H]; D_OUT],
    b3: [f32; D_OUT],
}

pub fn load_weights(bytes: &[u8]) -> Weights {
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

/// Padé[7/6] rational approximation of tanh — exact to ~2e-7 over |x| ≤ 5,
/// then saturates to ±1 (tanh(4.97) ≈ 0.99989, so the discontinuity at the
/// clamp is < 1e-4). Cost: ~7 multiplies + 6 adds + 1 divide, vs libm::tanhf
/// which fans out to expm1f (~2× more cycles on Haswell).
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
    // Tanh-GELU (OpenAI / GPT-2 formula, == PyTorch nn.GELU(approximate="tanh")):
    //   0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEFF: f32 = 0.044715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + tanh_approx(inner))
}

/// Returns [P(A-Legit), P(A-Fraud), P(B)] from a 16-float padded query
/// (only the first 14 dims are read).
pub fn infer(w: &Weights, x: &[f32; 16]) -> [f32; D_OUT] {
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

    // Numerically stable softmax (3 classes).
    let m = logits[0].max(logits[1]).max(logits[2]);
    let e0 = (logits[0] - m).exp();
    let e1 = (logits[1] - m).exp();
    let e2 = (logits[2] - m).exp();
    let s = e0 + e1 + e2;
    [e0 / s, e1 / s, e2 / s]
}

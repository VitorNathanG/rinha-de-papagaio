//! Zero-allocation byte parser + 14-dim vectorizer.
//!
//! The rinha generator emits the JSON fields in a deterministic order. Instead
//! of running a generic JSON parser we scan the body forward, snapping to each
//! field by its key as a byte substring (`memmem`). Returns &[u8] slices into
//! the original buffer — no allocation other than the output [f32; 16].
//!
//! Field order follows data-generator/main.c.

use memchr::{memchr, memmem};

const MAX_AMOUNT: f32 = 10000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1440.0;
const MAX_KM: f32 = 1000.0;
const MAX_TX_COUNT_24H: f32 = 20.0;
const MAX_MERCHANT_AVG_AMOUNT: f32 = 10000.0;

#[inline(always)]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Returns `Err(())` on any malformed input. Production rinha payloads don't
/// trip this; failure is treated as approve at the call site.
pub fn vectorize(body: &[u8], out: &mut [f32; 16]) -> Result<(), ()> {
    let mut cur = 0usize;

    // ---- order matches data-generator/main.c ----
    let amount = scan_f64(body, &mut cur, b"\"amount\":")?;
    let installments = scan_u32(body, &mut cur, b"\"installments\":")?;
    let requested_at = scan_string(body, &mut cur, b"\"requested_at\":\"")?;

    let cust_avg = scan_f64(body, &mut cur, b"\"avg_amount\":")?;
    let tx_count_24h = scan_u32(body, &mut cur, b"\"tx_count_24h\":")?;

    cur = find_after(body, cur, b"\"known_merchants\":[")?;
    let known_start = cur;
    let known_end = find_byte(body, cur, b']')?;
    let known = &body[known_start..known_end];
    cur = known_end + 1;

    let merchant_id = scan_string(body, &mut cur, b"\"id\":\"")?;
    let mcc = scan_string(body, &mut cur, b"\"mcc\":\"")?;
    let merch_avg = scan_f64(body, &mut cur, b"\"avg_amount\":")?;

    let is_online = scan_bool(body, &mut cur, b"\"is_online\":")?;
    let card_present = scan_bool(body, &mut cur, b"\"card_present\":")?;
    let km_home = scan_f64(body, &mut cur, b"\"km_from_home\":")?;

    cur = find_after(body, cur, b"\"last_transaction\":")?;
    let (mins, km_last) = if body.get(cur).copied() == Some(b'n') {
        (-1.0f32, -1.0f32)
    } else {
        let ts = scan_string(body, &mut cur, b"\"timestamp\":\"")?;
        let km = scan_f64(body, &mut cur, b"\"km_from_current\":")?;
        let m = minutes_between(ts, requested_at);
        (m, km as f32)
    };

    // Date parts of the current timestamp
    if requested_at.len() < 19 {
        return Err(());
    }
    let year = parse_u32_n(&requested_at[0..4]);
    let month = parse_u32_n(&requested_at[5..7]);
    let day = parse_u32_n(&requested_at[8..10]);
    let hour = parse_u32_n(&requested_at[11..13]);

    out[0] = clamp01(amount as f32 / MAX_AMOUNT);
    out[1] = clamp01(installments as f32 / MAX_INSTALLMENTS);
    out[2] = clamp01((amount as f32 / cust_avg as f32) / AMOUNT_VS_AVG_RATIO);
    out[3] = hour as f32 / 23.0;
    out[4] = day_of_week(year, month, day) as f32 / 6.0;

    if mins < 0.0 {
        out[5] = -1.0;
        out[6] = -1.0;
    } else {
        out[5] = clamp01(mins / MAX_MINUTES);
        out[6] = clamp01(km_last / MAX_KM);
    }

    out[7] = clamp01(km_home as f32 / MAX_KM);
    out[8] = clamp01(tx_count_24h as f32 / MAX_TX_COUNT_24H);
    out[9] = if is_online { 1.0 } else { 0.0 };
    out[10] = if card_present { 1.0 } else { 0.0 };
    out[11] = if known_contains(known, merchant_id) { 0.0 } else { 1.0 };
    out[12] = mcc_risk(mcc);
    out[13] = clamp01(merch_avg as f32 / MAX_MERCHANT_AVG_AMOUNT);
    out[14] = 0.0;
    out[15] = 0.0;

    Ok(())
}

// ---- low-level byte scanners --------------------------------------------

#[inline(always)]
fn find_after(body: &[u8], from: usize, needle: &[u8]) -> Result<usize, ()> {
    memmem::find(&body[from..], needle)
        .map(|p| from + p + needle.len())
        .ok_or(())
}

#[inline(always)]
fn find_byte(body: &[u8], from: usize, b: u8) -> Result<usize, ()> {
    memchr(b, &body[from..]).map(|p| from + p).ok_or(())
}

#[inline]
fn scan_f64(body: &[u8], cur: &mut usize, key: &[u8]) -> Result<f64, ()> {
    *cur = find_after(body, *cur, key)?;
    let start = *cur;
    while *cur < body.len() {
        let c = body[*cur];
        if c == b',' || c == b'}' {
            break;
        }
        *cur += 1;
    }
    std::str::from_utf8(&body[start..*cur])
        .map_err(|_| ())?
        .parse::<f64>()
        .map_err(|_| ())
}

#[inline]
fn scan_u32(body: &[u8], cur: &mut usize, key: &[u8]) -> Result<u32, ()> {
    *cur = find_after(body, *cur, key)?;
    let start = *cur;
    while *cur < body.len() && body[*cur].is_ascii_digit() {
        *cur += 1;
    }
    std::str::from_utf8(&body[start..*cur])
        .map_err(|_| ())?
        .parse::<u32>()
        .map_err(|_| ())
}

#[inline]
fn scan_string<'a>(body: &'a [u8], cur: &mut usize, key: &[u8]) -> Result<&'a [u8], ()> {
    *cur = find_after(body, *cur, key)?;
    let start = *cur;
    *cur = find_byte(body, *cur, b'"')?;
    let s = &body[start..*cur];
    *cur += 1; // skip closing quote
    Ok(s)
}

#[inline]
fn scan_bool(body: &[u8], cur: &mut usize, key: &[u8]) -> Result<bool, ()> {
    *cur = find_after(body, *cur, key)?;
    let b = body.get(*cur).copied() == Some(b't');
    *cur += if b { 4 } else { 5 };
    Ok(b)
}

/// Returns true iff `target` (a quoted-token interior) appears as one of the
/// strings inside the JSON array body `known` (which is everything between
/// `[` and `]`).
#[inline]
fn known_contains(known: &[u8], target: &[u8]) -> bool {
    let mut i = 0;
    while i < known.len() {
        // Advance to next opening quote.
        match memchr(b'"', &known[i..]) {
            Some(p) => i += p + 1,
            None => return false,
        }
        let start = i;
        // Find closing quote.
        match memchr(b'"', &known[i..]) {
            Some(p) => i += p,
            None => return false,
        }
        if &known[start..i] == target {
            return true;
        }
        i += 1;
    }
    false
}

#[inline]
fn mcc_risk(mcc: &[u8]) -> f32 {
    match mcc {
        b"5411" => 0.15,
        b"5812" => 0.30,
        b"5912" => 0.20,
        b"5944" => 0.45,
        b"7801" => 0.80,
        b"7802" => 0.75,
        b"7995" => 0.85,
        b"4511" => 0.35,
        b"5311" => 0.25,
        b"5999" => 0.50,
        _ => 0.5,
    }
}

// ---- date helpers --------------------------------------------------------

#[inline(always)]
fn parse_u32_n(b: &[u8]) -> u32 {
    let mut n: u32 = 0;
    for &c in b {
        n = n * 10 + (c - b'0') as u32;
    }
    n
}

#[inline]
fn is_leap(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

const DAYS_IN_MONTH: [u32; 13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

#[inline]
fn days_in_month_y(y: u32, m: u32) -> u32 {
    let d = DAYS_IN_MONTH[m as usize];
    if m == 2 && is_leap(y) { 29 } else { d }
}

#[inline]
fn days_since_epoch(y: u32, m: u32, d: u32) -> i64 {
    let mut days = 0i64;
    for year in 1970..y {
        days += if is_leap(year) { 366 } else { 365 };
    }
    for month in 1..m {
        days += days_in_month_y(y, month) as i64;
    }
    days + d as i64 - 1
}

#[inline]
fn epoch_seconds_from(s: &[u8]) -> i64 {
    let y = parse_u32_n(&s[0..4]);
    let mo = parse_u32_n(&s[5..7]);
    let d = parse_u32_n(&s[8..10]);
    let h = parse_u32_n(&s[11..13]);
    let mn = parse_u32_n(&s[14..16]);
    let sc = parse_u32_n(&s[17..19]);
    days_since_epoch(y, mo, d) * 86_400 + h as i64 * 3600 + mn as i64 * 60 + sc as i64
}

#[inline]
fn minutes_between(t1: &[u8], t2: &[u8]) -> f32 {
    let s1 = epoch_seconds_from(t1);
    let s2 = epoch_seconds_from(t2);
    ((s2 - s1) / 60) as f32
}

/// Zeller's congruence, normalised to Mon=0 .. Sun=6 (rinha convention).
#[inline]
fn day_of_week(y: u32, m: u32, d: u32) -> u32 {
    let (y, m) = if m < 3 { (y - 1, m + 12) } else { (y, m) };
    let k = (y % 100) as i64;
    let j = (y / 100) as i64;
    let h =
        (d as i64 + 13 * (m as i64 + 1) / 5 + k + k / 4 + j / 4 + 5 * j).rem_euclid(7);
    ((h + 5) % 7) as u32
}

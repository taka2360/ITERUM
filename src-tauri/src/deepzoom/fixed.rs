//! Signed fixed-point arithmetic for reference orbits.
//!
//! Every value a Mandelbrot reference orbit visits before escaping is bounded
//! (|Z| < bailout), so fixed point is sufficient and far simpler than
//! arbitrary-precision floating point. A value with `L` limbs is
//! `(-1)^neg * mag / 2^(64*(L-1))`; the top limb is the integer part.

use super::fexp::Fe;
use std::cmp::Ordering;

const TWO64: f64 = 18446744073709551616.0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fixed {
    pub neg: bool,
    /// Little-endian 64-bit limbs.
    pub mag: Vec<u64>,
}

impl Fixed {
    pub fn zero(limbs: usize) -> Self {
        Fixed { neg: false, mag: vec![0; limbs] }
    }

    pub fn is_zero(&self) -> bool {
        self.mag.iter().all(|&l| l == 0)
    }

    /// Parses a signed hexadecimal integer `s` that represents `s / 2^frac_bits`.
    pub fn from_hex(s: &str, frac_bits: u32, limbs: usize) -> Result<Self, String> {
        let (neg, digits) = match s.strip_prefix('-') {
            Some(d) => (true, d),
            None => (false, s),
        };
        let digits = digits.trim_start_matches("0x");
        if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("invalid hex coordinate: {s:?}"));
        }
        let bytes = digits.as_bytes();
        let mut raw: Vec<u64> = Vec::new();
        let mut end = bytes.len();
        while end > 0 {
            let start = end.saturating_sub(16);
            let chunk = std::str::from_utf8(&bytes[start..end]).unwrap();
            raw.push(u64::from_str_radix(chunk, 16).map_err(|e| e.to_string())?);
            end = start;
        }
        let target_frac = 64 * (limbs as u32 - 1);
        let mut out = vec![0u64; limbs];
        if target_frac >= frac_bits {
            shl_into(&raw, target_frac - frac_bits, &mut out);
        } else {
            shr_into(&raw, frac_bits - target_frac, &mut out);
        }
        Ok(Fixed { neg, mag: out }.normalized())
    }

    /// Parses a decimal string such as `-0.7436438870371587` (no exponent notation).
    pub fn from_decimal(s: &str, limbs: usize) -> Result<Self, String> {
        let (neg, body) = match s.trim().strip_prefix('-') {
            Some(b) => (true, b),
            None => (false, s.trim()),
        };
        let (ip, fp) = body.split_once('.').unwrap_or((body, ""));
        if ip.is_empty() && fp.is_empty()
            || !ip.bytes().chain(fp.bytes()).all(|c| c.is_ascii_digit())
        {
            return Err(format!("invalid decimal coordinate: {s:?}"));
        }
        let int: u64 = if ip.is_empty() { 0 } else { ip.parse().map_err(|e| format!("{e}"))? };
        // Horner from the last fractional digit: f = (f + d) / 10. Each digit
        // carries one extra guard limb to keep truncation error below 1 ulp.
        let mut mag = vec![0u64; limbs + 1];
        for d in fp.bytes().rev() {
            mag[limbs] += (d - b'0') as u64;
            div_small(&mut mag, 10);
        }
        let mut out = mag[1..].to_vec();
        out[limbs - 1] = int;
        Ok(Fixed { neg, mag: out }.normalized())
    }

    pub fn from_f64(x: f64, limbs: usize) -> Self {
        let a = x.abs();
        let mut mag = vec![0u64; limbs];
        let int = a.trunc();
        mag[limbs - 1] = int as u64;
        let mut frac = a - int;
        for i in (0..limbs - 1).rev() {
            frac *= TWO64;
            let w = frac.trunc();
            mag[i] = w as u64;
            frac -= w;
        }
        Fixed { neg: x < 0.0, mag }.normalized()
    }

    /// Nearest fixed-point value to `m * 2^e`.
    pub fn from_fe(v: Fe, limbs: usize) -> Self {
        if v.m == 0.0 {
            return Fixed::zero(limbs);
        }
        // |m| * 2^53 is an exact integer; place it at bit (e - 53 + frac).
        let mi = (v.m.abs() * 9007199254740992.0) as u64;
        let frac = 64 * (limbs as i64 - 1);
        let shift = v.e as i64 - 53 + frac;
        let mut out = vec![0u64; limbs];
        if shift >= 0 {
            shl_into(&[mi], shift as u32, &mut out);
        } else if shift > -64 {
            out[0] = mi >> (-shift) as u32;
        }
        Fixed { neg: v.m < 0.0, mag: out }.normalized()
    }

    /// Same value with a different limb count (fractional bits change accordingly).
    pub fn resized(&self, limbs: usize) -> Self {
        let mut out = vec![0u64; limbs];
        let l = self.mag.len();
        if limbs >= l {
            out[limbs - l..].copy_from_slice(&self.mag);
        } else {
            out.copy_from_slice(&self.mag[l - limbs..]);
        }
        Fixed { neg: self.neg, mag: out }.normalized()
    }

    /// Decimal representation with `digits` fractional digits (truncated).
    pub fn to_decimal(&self, digits: usize) -> String {
        let l = self.mag.len();
        let mut frac = self.mag.clone();
        let int = frac[l - 1];
        frac[l - 1] = 0;
        let mut s = String::new();
        if self.neg {
            s.push('-');
        }
        s.push_str(&int.to_string());
        s.push('.');
        for _ in 0..digits {
            // frac *= 10; the integer limb receives the next digit.
            let mut carry: u128 = 0;
            for w in frac.iter_mut() {
                let t = *w as u128 * 10 + carry;
                *w = t as u64;
                carry = t >> 64;
            }
            let d = frac[l - 1];
            frac[l - 1] = 0;
            s.push((b'0' + d as u8) as char);
        }
        s
    }

    fn normalized(mut self) -> Self {
        if self.is_zero() {
            self.neg = false;
        }
        self
    }

    pub fn add(&self, o: &Fixed) -> Fixed {
        let r = if self.neg == o.neg {
            Fixed { neg: self.neg, mag: add_mag(&self.mag, &o.mag) }
        } else if cmp_mag(&self.mag, &o.mag) != Ordering::Less {
            Fixed { neg: self.neg, mag: sub_mag(&self.mag, &o.mag) }
        } else {
            Fixed { neg: o.neg, mag: sub_mag(&o.mag, &self.mag) }
        };
        r.normalized()
    }

    pub fn negated(&self) -> Fixed {
        Fixed { neg: !self.neg, mag: self.mag.clone() }.normalized()
    }

    pub fn sub(&self, o: &Fixed) -> Fixed {
        self.add(&o.negated())
    }

    pub fn mul(&self, o: &Fixed) -> Fixed {
        Fixed { neg: self.neg != o.neg, mag: mul_mag(&self.mag, &o.mag) }.normalized()
    }

    pub fn sqr(&self) -> Fixed {
        Fixed { neg: false, mag: mul_mag(&self.mag, &self.mag) }
    }

    /// Converts to an extended-exponent float, keeping the top 64 significant bits.
    pub fn to_fe(&self) -> Fe {
        kernel::to_fe(self.neg, &self.mag)
    }

    pub fn to_f64(&self) -> f64 {
        self.to_fe().to_f64()
    }
}

fn add_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; a.len()];
    let mut carry = false;
    for i in 0..a.len() {
        let (s1, c1) = a[i].overflowing_add(b[i]);
        let (s2, c2) = s1.overflowing_add(carry as u64);
        out[i] = s2;
        carry = c1 || c2;
    }
    out
}

/// a - b; requires a >= b.
fn sub_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; a.len()];
    let mut borrow = false;
    for i in 0..a.len() {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow as u64);
        out[i] = d2;
        borrow = b1 || b2;
    }
    out
}

fn cmp_mag(a: &[u64], b: &[u64]) -> Ordering {
    for i in (0..a.len()).rev() {
        match a[i].cmp(&b[i]) {
            Ordering::Equal => continue,
            o => return o,
        }
    }
    Ordering::Equal
}

/// Fixed-point product `(a*b) >> 64*(L-1)`. Partial products that only land
/// below limb L-2 are skipped; together they contribute less than one unit of
/// the lowest kept limb.
fn mul_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    let l = a.len();
    let mut acc = vec![0u64; 2 * l + 1];
    let lowest = l.saturating_sub(2);
    for i in 0..l {
        if a[i] == 0 {
            continue;
        }
        let mut carry: u128 = 0;
        for j in lowest.saturating_sub(i)..l {
            let k = i + j;
            let t = (a[i] as u128) * (b[j] as u128) + acc[k] as u128 + carry;
            acc[k] = t as u64;
            carry = t >> 64;
        }
        let mut k = i + l;
        while carry != 0 {
            let t = acc[k] as u128 + carry;
            acc[k] = t as u64;
            carry = t >> 64;
            k += 1;
        }
    }
    acc[l - 1..2 * l - 1].to_vec()
}

/// Allocation-free kernels for the reference-orbit loop. Each computes exactly
/// (bit for bit) what the corresponding `Fixed` method computes, into
/// caller-provided buffers of the same limb count.
pub mod kernel {
    use super::{Fe, TWO64};
    use std::cmp::Ordering;

    /// Value `(-1)^neg * mag` as a floatexp (top 64 significant bits).
    pub fn to_fe(neg: bool, mag: &[u64]) -> Fe {
        let l = mag.len();
        let Some(top) = (0..l).rev().find(|&i| mag[i] != 0) else {
            return Fe::ZERO;
        };
        let lz = mag[top].leading_zeros();
        let mut word = mag[top] << lz;
        if lz > 0 && top > 0 {
            word |= mag[top - 1] >> (64 - lz);
        }
        // The top set bit sits at global position p = 64*top + 63 - lz, so the
        // value is (word / 2^64) * 2^(p + 1 - 64*(L-1)).
        let exp = 64 * top as i32 + 64 - lz as i32 - 64 * (l as i32 - 1);
        let m = word as f64 / TWO64;
        Fe::new(if neg { -m } else { m }, exp)
    }

    /// Signed sum `out = a + b`; returns the sign of `out` (false for zero).
    pub fn add_into(an: bool, a: &[u64], bn: bool, b: &[u64], out: &mut [u64]) -> bool {
        let neg = if an == bn {
            let mut carry = false;
            for i in 0..a.len() {
                let (s1, c1) = a[i].overflowing_add(b[i]);
                let (s2, c2) = s1.overflowing_add(carry as u64);
                out[i] = s2;
                carry = c1 || c2;
            }
            an
        } else {
            let (hi, lo, neg) = if cmp(a, b) != Ordering::Less { (a, b, an) } else { (b, a, bn) };
            let mut borrow = false;
            for i in 0..a.len() {
                let (d1, b1) = hi[i].overflowing_sub(lo[i]);
                let (d2, b2) = d1.overflowing_sub(borrow as u64);
                out[i] = d2;
                borrow = b1 || b2;
            }
            neg
        };
        neg && out.iter().any(|&w| w != 0)
    }

    fn cmp(a: &[u64], b: &[u64]) -> Ordering {
        for i in (0..a.len()).rev() {
            match a[i].cmp(&b[i]) {
                Ordering::Equal => continue,
                o => return o,
            }
        }
        Ordering::Equal
    }

    /// `out = a² >> 64*(L-1)` with the same truncation as `Fixed::sqr`.
    /// `acc` is scratch of at least 2L+1 limbs.
    pub fn sqr_into(a: &[u64], acc: &mut [u64], out: &mut [u64]) {
        let l = a.len();
        let acc = &mut acc[..2 * l + 1];
        acc.fill(0);
        let lowest = l.saturating_sub(2);
        for i in 0..l {
            if a[i] == 0 {
                continue;
            }
            let mut carry: u128 = 0;
            for j in lowest.saturating_sub(i)..l {
                let k = i + j;
                let t = (a[i] as u128) * (a[j] as u128) + acc[k] as u128 + carry;
                acc[k] = t as u64;
                carry = t >> 64;
            }
            let mut k = i + l;
            while carry != 0 {
                let t = acc[k] as u128 + carry;
                acc[k] = t as u64;
                carry = t >> 64;
                k += 1;
            }
        }
        out.copy_from_slice(&acc[l - 1..2 * l - 1]);
    }
}

fn div_small(mag: &mut [u64], d: u64) {
    let mut rem: u128 = 0;
    for w in mag.iter_mut().rev() {
        let cur = (rem << 64) | *w as u128;
        *w = (cur / d as u128) as u64;
        rem = cur % d as u128;
    }
}

fn shl_into(src: &[u64], bits: u32, out: &mut [u64]) {
    let limb = (bits / 64) as usize;
    let bit = bits % 64;
    for (i, &w) in src.iter().enumerate() {
        let k = i + limb;
        if k < out.len() {
            out[k] |= w << bit;
        }
        if bit > 0 && k + 1 < out.len() {
            out[k + 1] |= w >> (64 - bit);
        }
    }
}

fn shr_into(src: &[u64], bits: u32, out: &mut [u64]) {
    let limb = (bits / 64) as usize;
    let bit = bits % 64;
    for (k, o) in out.iter_mut().enumerate() {
        let i = k + limb;
        if i >= src.len() {
            break;
        }
        let mut w = src[i] >> bit;
        if bit > 0 && i + 1 < src.len() {
            w |= src[i + 1] << (64 - bit);
        }
        *o = w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_f64() {
        for &x in &[0.0, 1.0, -1.5, 0.3, -0.7499999, 1.0 / 3.0, -2.0e-10, 3.75] {
            assert_eq!(Fixed::from_f64(x, 4).to_f64(), x, "x={x}");
        }
    }

    #[test]
    fn arithmetic_matches_f64() {
        let xs = [0.3, -0.71, 1.25, -1.9, 0.000123];
        for &a in &xs {
            for &b in &xs {
                let fa = Fixed::from_f64(a, 3);
                let fb = Fixed::from_f64(b, 3);
                assert!((fa.add(&fb).to_f64() - (a + b)).abs() < 1e-15);
                assert!((fa.sub(&fb).to_f64() - (a - b)).abs() < 1e-15);
                assert!((fa.mul(&fb).to_f64() - (a * b)).abs() < 1e-15);
                assert!((fa.sqr().to_f64() - a * a).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn hex_parsing() {
        assert_eq!(Fixed::from_hex("18", 4, 2).unwrap().to_f64(), 1.5);
        assert_eq!(Fixed::from_hex("-c", 4, 3).unwrap().to_f64(), -0.75);
        // More fractional bits than the target holds: truncated to zero.
        assert!(Fixed::from_hex("1", 200, 2).unwrap().is_zero());
        assert_eq!(Fixed::from_hex("1", 100, 3).unwrap().to_fe().log2(), -100.0);
        // Long values spanning several hex chunks.
        let s = format!("3{}", "0".repeat(40)); // 3 * 16^40 = 3 * 2^160
        assert_eq!(Fixed::from_hex(&s, 160, 4).unwrap().to_f64(), 3.0);
    }

    #[test]
    fn decimal_parsing() {
        assert_eq!(Fixed::from_decimal("-0.75", 3).unwrap().to_f64(), -0.75);
        assert_eq!(Fixed::from_decimal("1.5", 3).unwrap().to_f64(), 1.5);
        assert_eq!(Fixed::from_decimal("0.1", 3).unwrap().to_f64(), 0.1);
        let x = "-0.743643887037158704752191506114774";
        assert_eq!(Fixed::from_decimal(x, 4).unwrap().to_f64(), x.parse::<f64>().unwrap());
        // 1e-300 written out: exponent survives.
        let s = format!("0.{}1", "0".repeat(299));
        let l = Fixed::from_decimal(&s, 18).unwrap().to_fe().log2();
        assert!((l - (-300.0 * 10f64.log2())).abs() < 1e-9);
        assert!(Fixed::from_decimal("abc", 3).is_err());
    }

    #[test]
    fn fe_conversion_and_decimal_output() {
        let f = Fixed::from_fe(Fe::new(-0.75, 0), 3);
        assert_eq!(f.to_f64(), -0.75);
        let g = Fixed::from_fe(Fe::new(0.5, -300), 6);
        assert_eq!(g.to_fe().log2(), -301.0);
        assert_eq!(Fixed::from_decimal("-1.25", 3).unwrap().to_decimal(4), "-1.2500");
        let x = "0.0000000000000000000000000000000000000012345678901234567890123";
        assert_eq!(Fixed::from_decimal(x, 6).unwrap().to_decimal(60), &x[..62]);
        let r = Fixed::from_decimal("0.3", 3).unwrap();
        assert_eq!(r.resized(5).resized(3), r);
    }

    /// Deterministic pseudo-random fixed-point values in (-4, 4), including
    /// zero limbs, all-ones runs and zero (carry and sign edge cases).
    fn rand_fixed(seed: &mut u64, limbs: usize) -> Fixed {
        let mut next = || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed
        };
        let neg = next() & 1 == 1;
        let mut mag: Vec<u64> = (0..limbs).map(|_| next()).collect();
        mag[limbs - 1] &= 3;
        match next() % 6 {
            0 => mag[limbs - 1] = 0,
            1 => mag[0] = u64::MAX,
            2 => mag.iter_mut().take(limbs - 1).for_each(|w| *w = u64::MAX),
            3 => mag.iter_mut().skip(1).for_each(|w| *w = 0),
            4 => mag.iter_mut().for_each(|w| *w = 0),
            _ => {}
        }
        Fixed { neg, mag }.normalized()
    }

    #[test]
    fn kernels_match_fixed_bit_for_bit() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        for limbs in [2usize, 3, 4, 17, 54, 105] {
            let mut acc = vec![0u64; 2 * limbs + 1];
            let mut out = vec![0u64; limbs];
            for _ in 0..300 {
                let a = rand_fixed(&mut seed, limbs);
                let b = rand_fixed(&mut seed, limbs);
                kernel::sqr_into(&a.mag, &mut acc, &mut out);
                assert_eq!(out, a.sqr().mag, "sqr, {limbs} limbs");
                let neg = kernel::add_into(a.neg, &a.mag, b.neg, &b.mag, &mut out);
                assert_eq!(Fixed { neg, mag: out.clone() }, a.add(&b), "add, {limbs} limbs");
                let neg = kernel::add_into(a.neg, &a.mag, !b.neg, &b.mag, &mut out);
                assert_eq!(Fixed { neg, mag: out.clone() }, a.sub(&b), "sub, {limbs} limbs");
                assert_eq!(kernel::to_fe(a.neg, &a.mag), a.to_fe());
                // Exact cancellation gives +0.
                assert!(!kernel::add_into(a.neg, &a.mag, !a.neg, &a.mag, &mut out));
            }
        }
    }

    #[test]
    fn tiny_values_keep_exponent() {
        let f = Fixed::from_hex("1", 1000, 18).unwrap();
        assert_eq!(f.to_fe().log2(), -1000.0);
        // 2^-1000 squared lies below 18 limbs of precision.
        assert!(f.sqr().is_zero());
        let g = Fixed::from_hex("1", 500, 18).unwrap();
        assert_eq!(g.sqr().to_fe().log2(), -1000.0);
    }
}

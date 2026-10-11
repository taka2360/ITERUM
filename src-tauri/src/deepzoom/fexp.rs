//! Extended-exponent floating point ("floatexp").
//!
//! A value is `m * 2^e` with `|m|` in `[0.5, 1)` (the `frexp` convention, which
//! WGSL's `frexp` also uses) or `m == 0, e == 0`. On the CPU the mantissa is an
//! f64; values sent to the GPU are rounded to f32 mantissas (see `format.rs`).

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fe {
    pub m: f64,
    pub e: i32,
}

/// Splits a finite f64 into mantissa in [0.5, 1) and exponent.
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    if exp == 0 {
        // Subnormal: scale into the normal range first.
        let (m, e) = frexp(x * 2f64.powi(64));
        return (m, e - 64);
    }
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52));
    (m, exp - 1022)
}

/// `m * 2^e`, safe for large |e| (saturates to 0 or infinity).
pub fn ldexp(m: f64, e: i32) -> f64 {
    let mut m = m;
    let mut e = e;
    while e > 1000 {
        m *= 2f64.powi(1000);
        e -= 1000;
        if m.is_infinite() {
            return m;
        }
    }
    while e < -1000 {
        m *= 2f64.powi(-1000);
        e += 1000;
        if m == 0.0 {
            return m;
        }
    }
    m * 2f64.powi(e)
}

impl Fe {
    pub const ZERO: Fe = Fe { m: 0.0, e: 0 };

    pub fn new(m: f64, e: i32) -> Fe {
        if m == 0.0 {
            return Fe::ZERO;
        }
        let (mm, k) = frexp(m);
        Fe { m: mm, e: e + k }
    }

    pub fn from_f64(x: f64) -> Fe {
        Fe::new(x, 0)
    }

    /// 2^x for a (possibly huge) real exponent x.
    pub fn from_log2(x: f64) -> Fe {
        let f = x.floor();
        Fe::new(2f64.powf(x - f), f as i32)
    }

    pub fn to_f64(self) -> f64 {
        ldexp(self.m, self.e)
    }

    pub fn is_zero(self) -> bool {
        self.m == 0.0
    }

    /// log2(|x|); -inf for zero.
    pub fn log2(self) -> f64 {
        if self.m == 0.0 {
            return f64::NEG_INFINITY;
        }
        self.m.abs().log2() + self.e as f64
    }

    pub fn mul(self, o: Fe) -> Fe {
        Fe::new(self.m * o.m, self.e + o.e)
    }

    pub fn div(self, o: Fe) -> Fe {
        Fe::new(self.m / o.m, self.e - o.e)
    }

    pub fn add(self, o: Fe) -> Fe {
        if self.m == 0.0 {
            return o;
        }
        if o.m == 0.0 {
            return self;
        }
        let d = self.e - o.e;
        if d > 64 {
            return self;
        }
        if d < -64 {
            return o;
        }
        if d >= 0 {
            Fe::new(self.m + ldexp(o.m, -d), self.e)
        } else {
            Fe::new(ldexp(self.m, d) + o.m, o.e)
        }
    }

    pub fn sub(self, o: Fe) -> Fe {
        self.add(Fe { m: -o.m, e: o.e })
    }

    pub fn sqrt(self) -> Fe {
        if self.m <= 0.0 {
            return Fe::ZERO;
        }
        if self.e % 2 == 0 {
            Fe::new(self.m.sqrt(), self.e / 2)
        } else {
            Fe::new((self.m * 2.0).sqrt(), (self.e - 1) / 2)
        }
    }

    pub fn max0(self) -> Fe {
        if self.m < 0.0 { Fe::ZERO } else { self }
    }

    pub fn lt(self, o: Fe) -> bool {
        self.sub(o).m < 0.0
    }

    pub fn min(self, o: Fe) -> Fe {
        if self.lt(o) { self } else { o }
    }
}

/// Complex floatexp with a shared exponent: `(re + i*im) * 2^e`, where
/// `max(|re|, |im|)` is in `[0.5, 1)` or both are zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CFe {
    pub re: f64,
    pub im: f64,
    pub e: i32,
}

impl CFe {
    pub const ZERO: CFe = CFe { re: 0.0, im: 0.0, e: 0 };
    pub const ONE: CFe = CFe { re: 0.5, im: 0.0, e: 1 };

    pub fn new(re: f64, im: f64, e: i32) -> CFe {
        let m = re.abs().max(im.abs());
        if m == 0.0 {
            return CFe::ZERO;
        }
        let (_, k) = frexp(m);
        CFe { re: ldexp(re, -k), im: ldexp(im, -k), e: e + k }
    }

    pub fn from_parts(re: Fe, im: Fe) -> CFe {
        if re.is_zero() && im.is_zero() {
            return CFe::ZERO;
        }
        let e = if re.is_zero() {
            im.e
        } else if im.is_zero() {
            re.e
        } else {
            re.e.max(im.e)
        };
        CFe::new(ldexp(re.m, re.e - e), ldexp(im.m, im.e - e), e)
    }

    pub fn mul(self, o: CFe) -> CFe {
        CFe::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
            self.e + o.e,
        )
    }

    pub fn scale(self, s: f64) -> CFe {
        CFe::new(self.re * s, self.im * s, self.e)
    }

    pub fn add(self, o: CFe) -> CFe {
        if self.re == 0.0 && self.im == 0.0 {
            return o;
        }
        if o.re == 0.0 && o.im == 0.0 {
            return self;
        }
        let d = self.e - o.e;
        if d > 64 {
            return self;
        }
        if d < -64 {
            return o;
        }
        if d >= 0 {
            CFe::new(self.re + ldexp(o.re, -d), self.im + ldexp(o.im, -d), self.e)
        } else {
            CFe::new(ldexp(self.re, d) + o.re, ldexp(self.im, d) + o.im, o.e)
        }
    }

    pub fn sub(self, o: CFe) -> CFe {
        self.add(CFe { re: -o.re, im: -o.im, e: o.e })
    }

    pub fn inv(self) -> CFe {
        let d = self.re * self.re + self.im * self.im;
        CFe::new(self.re / d, -self.im / d, -self.e)
    }

    pub fn div(self, o: CFe) -> CFe {
        self.mul(o.inv())
    }

    pub fn norm2(self) -> Fe {
        Fe::new(self.re * self.re + self.im * self.im, 2 * self.e)
    }

    pub fn abs(self) -> Fe {
        Fe::new((self.re * self.re + self.im * self.im).sqrt(), self.e)
    }

    pub fn to_c64(self) -> (f64, f64) {
        (ldexp(self.re, self.e), ldexp(self.im, self.e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frexp_ldexp_roundtrip() {
        for &x in &[1.0, -3.5, 1e-300, 4.9e-324, 1e300, 0.75] {
            let (m, e) = frexp(x);
            assert!(m.abs() >= 0.5 && m.abs() < 1.0);
            assert_eq!(ldexp(m, e), x);
        }
    }

    #[test]
    fn fe_survives_far_beyond_f64_range() {
        let tiny = Fe::new(1.0, -5000);
        let sq = tiny.mul(tiny);
        assert_eq!(sq.log2(), -10000.0);
        assert_eq!(sq.sqrt().log2(), -5000.0);
        let big = Fe::new(1.0, 4000);
        assert!((tiny.mul(big).to_f64() - 2f64.powi(-1000)).abs() < 1e-310);
    }

    #[test]
    fn fe_add_aligns_exponents_at_the_boundary() {
        // Exponent differences right at and beyond the 64-bit alignment cutoff.
        let a = Fe::new(1.0, 0);
        assert_eq!(a.add(Fe::new(1.0, -64)).to_f64(), 1.0 + 2f64.powi(-64));
        assert_eq!(a.add(Fe::new(1.0, -65)), a);
        assert_eq!(a.sub(a), Fe::ZERO);
        assert!(Fe::new(1.0, -10).lt(Fe::new(1.0, -9)));
        assert!(Fe::new(-1.0, 100).lt(Fe::new(1.0, -100)));
    }

    #[test]
    fn complex_ops() {
        let a = CFe::new(0.3, -0.4, 0);
        let b = CFe::new(-1.2, 0.7, 0);
        let (re, im) = a.mul(b).to_c64();
        assert!((re - (0.3 * -1.2 - -0.4 * 0.7)).abs() < 1e-15);
        assert!((im - (0.3 * 0.7 + -0.4 * -1.2)).abs() < 1e-15);
        let (re, im) = a.add(b).to_c64();
        assert!((re + 0.9).abs() < 1e-15 && (im - 0.3).abs() < 1e-15);
        assert!((a.abs().to_f64() - 0.5).abs() < 1e-15);
        // Shared exponent keeps the tiny component relative to the large one.
        let c = CFe::new(1.0, 1e-30, -2000);
        assert_eq!(c.norm2().log2(), -4000.0);
    }
}

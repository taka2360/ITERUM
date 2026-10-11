//! High-precision Mandelbrot reference orbits.

use super::fexp::CFe;
use super::fixed::Fixed;

/// Escape radius used for both reference orbits and pixels (large for smooth colouring).
pub const BAILOUT: f64 = 256.0;

pub struct Orbit {
    /// Z_0 = 0, Z_1 = c, ... up to and including the first escaped value (if any).
    pub z: Vec<CFe>,
    pub escaped: bool,
    /// The orbit returned to zero (within precision): the reference is a
    /// nucleus and one period is all the GPU needs, since rebasing at the end
    /// of the orbit continues it exactly.
    pub periodic: bool,
    /// Smallest log2|Z_n| for n >= 1 (how close the orbit comes to zero).
    pub min_log2_abs: f64,
}

/// Fixed-point limb count that holds at least `bits` fractional bits.
pub fn limbs_for_bits(bits: u32) -> usize {
    (bits as usize).div_ceil(64) + 1
}

/// Iterates Z_{n+1} = Z_n^2 + c in fixed point. `poll` is called every 4096
/// iterations with the iteration count (progress reporting) and returns true
/// to cancel, in which case the result is `None`.
pub fn compute(
    cr: &Fixed,
    ci: &Fixed,
    max_iter: u32,
    poll: &dyn Fn(u32) -> bool,
) -> Option<Orbit> {
    let limbs = cr.mag.len();
    let mut zr = Fixed::zero(limbs);
    let mut zi = Fixed::zero(limbs);
    let mut z = Vec::with_capacity((max_iter as usize + 1).min(1 << 24));
    z.push(CFe::ZERO);
    let mut escaped = false;
    let mut periodic = false;
    let mut min_log2_abs = f64::INFINITY;
    let bail2 = BAILOUT * BAILOUT;
    // |Z| below this is zero at the working precision.
    let zero_log2 = -(64.0 * (limbs as f64 - 1.0)) + 48.0;
    for n in 0..max_iter {
        if n & 0xfff == 0 && poll(n) {
            return None;
        }
        let zr2 = zr.sqr();
        let zi2 = zi.sqr();
        let s = zr.add(&zi).sqr();
        zi = s.sub(&zr2).sub(&zi2).add(ci);
        zr = zr2.sub(&zi2).add(cr);
        let v = CFe::from_parts(zr.to_fe(), zi.to_fe());
        z.push(v);
        let n2 = v.norm2();
        min_log2_abs = min_log2_abs.min(n2.log2() * 0.5);
        if n2.to_f64() > bail2 {
            escaped = true;
            break;
        }
        if n2.log2() * 0.5 < zero_log2 {
            periodic = true;
            break;
        }
    }
    Some(Orbit { z, escaped, periodic, min_log2_abs })
}

/// Result of iterating a single point directly at full precision (verification oracle).
pub struct DirectResult {
    pub iterations: u32,
    pub escaped: bool,
    pub final_re: f64,
    pub final_im: f64,
}

pub fn iterate_direct(cr: &Fixed, ci: &Fixed, max_iter: u32) -> DirectResult {
    let orbit = compute(cr, ci, max_iter, &|_| false).expect("never cancelled");
    let last = *orbit.z.last().unwrap();
    let (final_re, final_im) = last.to_c64();
    DirectResult {
        iterations: orbit.z.len() as u32 - 1,
        escaped: orbit.escaped,
        final_re,
        final_im,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use astro_float::{BigFloat, Consts, RoundingMode};

    const RM: RoundingMode = RoundingMode::ToEven;

    /// Independent oracle: the same recurrence evaluated with astro-float.
    fn oracle(cr: &str, ci: &str, bits: usize, max_iter: u32) -> (u32, bool, Vec<(f64, f64)>) {
        let mut cc = Consts::new().unwrap();
        let cr = BigFloat::parse(cr, astro_float::Radix::Dec, bits, RM, &mut cc);
        let ci = BigFloat::parse(ci, astro_float::Radix::Dec, bits, RM, &mut cc);
        let mut zr = BigFloat::from_f64(0.0, bits);
        let mut zi = BigFloat::from_f64(0.0, bits);
        let mut pts = vec![(0.0, 0.0)];
        let bail = BigFloat::from_f64(BAILOUT * BAILOUT, bits);
        for n in 0..max_iter {
            let zr2 = zr.mul(&zr, bits, RM);
            let zi2 = zi.mul(&zi, bits, RM);
            let zri = zr.mul(&zi, bits, RM);
            zi = zri.add(&zri, bits, RM).add(&ci, bits, RM);
            zr = zr2.sub(&zi2, bits, RM).add(&cr, bits, RM);
            pts.push((to_f64(&zr), to_f64(&zi)));
            let m = zr.mul(&zr, bits, RM).add(&zi.mul(&zi, bits, RM), bits, RM);
            if m > bail {
                return (n + 1, true, pts);
            }
        }
        (max_iter, false, pts)
    }

    fn to_f64(x: &BigFloat) -> f64 {
        // Mantissa words are little-endian and normalized: value = 0.m * 2^e.
        match x.as_raw_parts() {
            Some((m, _, s, e, _)) if !x.is_zero() => {
                let v = super::super::fexp::ldexp(*m.last().unwrap() as f64 / 18446744073709551616.0, e);
                if s == astro_float::Sign::Neg { -v } else { v }
            }
            _ => 0.0,
        }
    }

    fn dec_to_fixed(s: &str, bits: u32) -> Fixed {
        Fixed::from_decimal(s, limbs_for_bits(bits)).unwrap()
    }

    #[test]
    fn matches_astro_float_oracle_on_boundary_points() {
        // Points near the boundary where orbits are long and chaotic.
        let cases = [
            ("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", 2000),
            ("-1.25066", "0.02012", 5000),
            ("0.25000001", "0.0", 30000),
            ("-0.75", "0.1", 2000),
        ];
        for (re, im, iters) in cases {
            let bits = 256;
            let cr = dec_to_fixed(re, bits);
            let ci = dec_to_fixed(im, bits);
            let ours = compute(&cr, &ci, iters, &|_| false).unwrap();
            let (n, esc, pts) = oracle(re, im, bits as usize + 64, iters);
            assert_eq!(ours.escaped, esc, "escape flag for {re}+{im}i");
            assert_eq!(ours.z.len() as u32 - 1, n, "escape iteration for {re}+{im}i");
            for (k, (a, b)) in ours.z.iter().zip(&pts).enumerate().step_by(97) {
                let (x, y) = a.to_c64();
                let tol = 1e-12 * (1.0 + b.0.abs() + b.1.abs());
                assert!((x - b.0).abs() < tol && (y - b.1).abs() < tol, "orbit diverged at {k}: {x},{y} vs {b:?}");
            }
        }
    }

    #[test]
    fn deep_coordinate_precision_is_sufficient() {
        // A coordinate specified to ~1000 bits: compare bits vs bits+64.
        let re = "-1.7499597227903660498914828869567818698263764211926598339406394817478609312213425939651574735513283225818232838960697399040006957697467124389306810227519519014254893497961216924876049386069290213289498";
        let im = "0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001";
        for bits in [700u32, 1100] {
            let a = compute(&dec_to_fixed(re, bits), &dec_to_fixed(im, bits), 20000, &|_| false).unwrap();
            let b = compute(&dec_to_fixed(re, bits + 64), &dec_to_fixed(im, bits + 64), 20000, &|_| false).unwrap();
            assert_eq!(a.z.len(), b.z.len(), "escape count differs at {bits} bits");
            assert_eq!(a.escaped, b.escaped);
        }
    }

    #[test]
    fn nucleus_orbit_stops_after_one_period() {
        // Nucleus of the period-3 "airplane" component, to full precision.
        use crate::deepzoom::nucleus::{newton_nucleus, Point};
        let l = limbs_for_bits(256);
        let c = Point { re: Fixed::from_decimal("-1.75", l).unwrap(), im: Fixed::zero(l) };
        let n = newton_nucleus(&c, 3, 100).unwrap();
        let o = compute(&n.re, &n.im, 100000, &|_| false).unwrap();
        assert!(o.periodic && !o.escaped);
        assert_eq!(o.z.len(), 4);
    }

    #[test]
    fn cancellation() {
        let c = Fixed::from_f64(-0.1, 4);
        assert!(compute(&c, &c, 100000, &|_| true).is_none());
    }

    #[test]
    fn interior_point_reports_min_abs() {
        // c = -1 has the period-2 orbit 0, -1, 0, -1 ... so |Z| reaches exactly 0.
        let c = Fixed::from_f64(-1.0, 3);
        let o = compute(&c, &Fixed::zero(3), 10, &|_| false).unwrap();
        assert!(!o.escaped && o.periodic);
        assert_eq!(o.min_log2_abs, f64::NEG_INFINITY);
    }
}

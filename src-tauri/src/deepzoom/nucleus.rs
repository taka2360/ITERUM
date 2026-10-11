//! Locating minibrots (hyperbolic components) at arbitrary depth.
//!
//! Used offline to generate Voyage destinations and benchmark locations:
//!  * `ball_period` finds the lowest period of a component near a disk;
//!  * `newton_nucleus` refines its nucleus (z_p(c) = 0) to full precision;
//!  * `atom_size` estimates the component's size and orientation.

use super::fexp::{CFe, Fe};
use super::fixed::Fixed;

#[derive(Clone, Debug)]
pub struct Point {
    pub re: Fixed,
    pub im: Fixed,
}

impl Point {
    pub fn resized(&self, limbs: usize) -> Point {
        Point { re: self.re.resized(limbs), im: self.im.resized(limbs) }
    }

    pub fn offset(&self, d: CFe) -> Point {
        let l = self.re.mag.len();
        Point {
            re: self.re.add(&Fixed::from_fe(Fe::new(d.re, d.e), l)),
            im: self.im.add(&Fixed::from_fe(Fe::new(d.im, d.e), l)),
        }
    }
}

fn to_cfe(re: &Fixed, im: &Fixed) -> CFe {
    CFe::from_parts(re.to_fe(), im.to_fe())
}

/// One step z ← z² + c in fixed point.
fn step(zr: &mut Fixed, zi: &mut Fixed, c: &Point) {
    let zr2 = zr.sqr();
    let zi2 = zi.sqr();
    let s = zr.add(zi).sqr();
    *zi = s.sub(&zr2).sub(&zi2).add(&c.im);
    *zr = zr2.sub(&zi2).add(&c.re);
}

/// Lowest period p such that the image of the disk |c − centre| < r under
/// z_p contains 0 (first-order estimate), or None if not found below `max_p`.
pub fn ball_period(center: &Point, radius: Fe, max_p: u32) -> Option<u32> {
    let l = center.re.mag.len();
    let (mut zr, mut zi) = (Fixed::zero(l), Fixed::zero(l));
    let mut dz = CFe::ZERO;
    for p in 1..=max_p {
        let z = to_cfe(&zr, &zi);
        dz = z.scale(2.0).mul(dz).add(CFe::ONE);
        step(&mut zr, &mut zi, center);
        let z = to_cfe(&zr, &zi);
        if z.norm2().to_f64() > 1e8 {
            return None;
        }
        if z.abs().lt(radius.mul(dz.abs())) {
            return Some(p);
        }
    }
    None
}

/// Newton's method on z_p(c) = 0 starting at `c0`. Returns the nucleus, or
/// None if the iteration did not converge to full precision.
pub fn newton_nucleus(c0: &Point, period: u32, max_steps: u32) -> Option<Point> {
    let l = c0.re.mag.len();
    let tiny = Fe::new(1.0, -(64 * (l as i32 - 1)) + 8);
    let mut c = c0.clone();
    for _ in 0..max_steps {
        let (mut zr, mut zi) = (Fixed::zero(l), Fixed::zero(l));
        let mut dz = CFe::ZERO;
        for _ in 0..period {
            let z = to_cfe(&zr, &zi);
            dz = z.scale(2.0).mul(dz).add(CFe::ONE);
            step(&mut zr, &mut zi, &c);
        }
        let delta = to_cfe(&zr, &zi).div(dz);
        if !delta.re.is_finite() || !delta.im.is_finite() {
            return None;
        }
        c = c.offset(CFe { re: -delta.re, im: -delta.im, e: delta.e });
        if delta.abs().lt(tiny) {
            return Some(c);
        }
    }
    None
}

/// Newton's method for a Misiurewicz point: z_{k+q}(c) = z_k(c), with
/// preperiod `k` and period `q`. Returns None if it does not converge.
pub fn misiurewicz(c0: &Point, k: u32, q: u32, max_steps: u32) -> Option<Point> {
    let l = c0.re.mag.len();
    let tiny = Fe::new(1.0, -(64 * (l as i32 - 1)) + 16);
    let mut c = c0.clone();
    for _ in 0..max_steps {
        let (mut zr, mut zi) = (Fixed::zero(l), Fixed::zero(l));
        let mut dz = CFe::ZERO;
        let mut zk = (Fixed::zero(l), Fixed::zero(l));
        let mut dzk = CFe::ZERO;
        for i in 0..k + q {
            if i == k {
                zk = (zr.clone(), zi.clone());
                dzk = dz;
            }
            let z = to_cfe(&zr, &zi);
            dz = z.scale(2.0).mul(dz).add(CFe::ONE);
            step(&mut zr, &mut zi, &c);
        }
        let f = to_cfe(&zr.sub(&zk.0), &zi.sub(&zk.1));
        let df = dz.sub(dzk);
        if df.abs().is_zero() {
            return None;
        }
        let delta = f.div(df);
        if !delta.re.is_finite() || delta.abs().to_f64() > 0.1 {
            return None;
        }
        c = c.offset(CFe { re: -delta.re, im: -delta.im, e: delta.e });
        if delta.abs().lt(tiny) {
            return Some(c);
        }
    }
    None
}

/// Size estimate of the component with nucleus `c` and period `p`:
/// a complex number whose modulus is the radius and whose argument is the
/// orientation of the minibrot relative to the whole set.
pub fn atom_size(c: &Point, period: u32) -> CFe {
    let l = c.re.mag.len();
    let (mut zr, mut zi) = (Fixed::zero(l), Fixed::zero(l));
    let mut lam = CFe::ONE;
    let mut b = CFe::ONE;
    for _ in 1..period {
        step(&mut zr, &mut zi, c);
        lam = to_cfe(&zr, &zi).scale(2.0).mul(lam);
        b = b.add(lam.inv());
    }
    b.mul(lam).mul(lam).inv()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(re: &str, im: &str, limbs: usize) -> Point {
        Point { re: Fixed::from_decimal(re, limbs).unwrap(), im: Fixed::from_decimal(im, limbs).unwrap() }
    }

    #[test]
    fn known_nuclei() {
        // Period 2 bulb at -1, period 3 "airplane" on the real axis and the "rabbit".
        let c = newton_nucleus(&pt("-0.9", "0", 4), 2, 50).unwrap();
        assert!((c.re.to_f64() + 1.0).abs() < 1e-15 && c.im.to_f64().abs() < 1e-15);
        let c = newton_nucleus(&pt("-1.75", "0", 4), 3, 50).unwrap();
        assert!((c.re.to_f64() + 1.754877666246693).abs() < 1e-14);
        let c = newton_nucleus(&pt("-0.12", "0.74", 4), 3, 50).unwrap();
        assert!((c.re.to_f64() + 0.12256116687665362).abs() < 1e-14);
        assert!((c.im.to_f64() - 0.7448617666197442).abs() < 1e-14);
    }

    #[test]
    fn period_detection() {
        assert_eq!(ball_period(&pt("-1.754", "0", 4), Fe::from_f64(0.01), 100), Some(3));
        // A disk around the main cardioid that contains its nucleus 0.
        assert_eq!(ball_period(&pt("-0.1", "0.1", 4), Fe::from_f64(0.2), 100), Some(1));
    }

    #[test]
    fn misiurewicz_points() {
        // c = i: 0 → i → −1+i → −i → −1+i … (preperiod 2, period 2).
        let m = misiurewicz(&pt("0.01", "0.99", 4), 2, 2, 60).unwrap();
        assert!(m.re.to_f64().abs() < 1e-15 && (m.im.to_f64() - 1.0).abs() < 1e-15);
        // c = −2: 0 → −2 → 2 → 2 … (preperiod 2, period 1).
        let m = misiurewicz(&pt("-1.99", "0", 4), 2, 1, 60).unwrap();
        assert!((m.re.to_f64() + 2.0).abs() < 1e-15);
    }

    #[test]
    fn atom_size_matches_known_component() {
        // The period-3 minibrot on the real axis has radius about 0.0188 (≈1/53.2)
        // and is oriented like the whole set (argument ≈ 0).
        let c = newton_nucleus(&pt("-1.75", "0", 4), 3, 50).unwrap();
        let s = atom_size(&c, 3);
        let (re, im) = s.to_c64();
        assert!((re - 0.0188).abs() < 0.002, "size {re}");
        assert!(im.abs() < 1e-12);
    }
}

//! Bivariate linear approximation (BLA) tables for skipping perturbation iterations.
//!
//! Around reference index m, one perturbation step is
//!   δ' = 2 Z_m δ + δ² + δc  ≈  A δ + B δc   with A = 2 Z_m, B = 1,
//! valid while |δ| < R so that δ² is negligible. Two consecutive approximations
//! x then y merge into
//!   A = A_y A_x,  B = A_y B_x + B_y,
//!   R = min(R_x, max(0, (R_y − |B_x| |δc|max) / |A_x|)).
//! Coefficients over long spans overflow any fixed-exponent float, so every
//! quantity here is a floatexp.

use super::fexp::{CFe, Fe};

/// Relative size of the neglected δ² term for a single step (≈ f32 precision).
pub const EPSILON_LOG2: i32 = -24;

#[derive(Clone, Copy, Debug)]
pub struct BlaEntry {
    pub a: CFe,
    pub b: CFe,
    /// Validity radius: the approximation applies while |δ| < r.
    pub r: Fe,
}

/// Level `l` (0-based) holds spans of length 2^l starting at reference
/// indices m = 1 + i·2^l, each ending at or before the last orbit index.
pub struct BlaTable {
    pub levels: Vec<Vec<BlaEntry>>,
}

pub fn single_step(z: CFe) -> BlaEntry {
    BlaEntry {
        a: z.scale(2.0),
        b: CFe::ONE,
        r: z.abs().mul(Fe::new(1.0, EPSILON_LOG2 + 1)),
    }
}

pub fn merge(x: &BlaEntry, y: &BlaEntry, dc_max: Fe) -> BlaEntry {
    let a = y.a.mul(x.a);
    let b = y.a.mul(x.b).add(y.b);
    let ax = x.a.abs();
    let r = if ax.is_zero() {
        x.r
    } else {
        x.r.min(y.r.sub(x.b.abs().mul(dc_max)).max0().div(ax))
    };
    BlaEntry { a, b, r }
}

/// Builds every level for an orbit `z` (z[0] = 0) and a bound on |δc| over
/// the region that will use the table.
pub fn build(z: &[CFe], dc_max: Fe) -> BlaTable {
    let m_len = z.len();
    if m_len < 3 {
        return BlaTable { levels: vec![] };
    }
    // Steps start at m = 1..=M-2 so that every span ends on a valid index.
    let level0: Vec<BlaEntry> = (1..m_len - 1).map(|m| single_step(z[m])).collect();
    let mut levels = vec![level0];
    loop {
        let prev = levels.last().unwrap();
        if prev.len() < 2 {
            break;
        }
        let next: Vec<BlaEntry> = prev.chunks_exact(2).map(|p| merge(&p[0], &p[1], dc_max)).collect();
        levels.push(next);
    }
    BlaTable { levels }
}

/// What appending to a table needs to know about the entries built so far:
/// the entry count of every level, and each level's last entry (the only one
/// a later merge can still pair with a new entry).
#[derive(Clone)]
pub struct Frontier {
    pub counts: Vec<usize>,
    pub last: Vec<Option<BlaEntry>>,
}

impl Frontier {
    pub const EMPTY: Frontier = Frontier { counts: Vec::new(), last: Vec::new() };
}

/// The entries that exist for an orbit of `len` points beyond those already
/// built (`from`), level by level, and the new frontier. Entries depend only
/// on their span of the orbit and on `dc_max`, so appending gives exactly the
/// table `build` makes for the whole orbit (same `dc_max` assumed).
pub fn build_tail(z: &[CFe], dc_max: Fe, from: &Frontier) -> (Vec<Vec<BlaEntry>>, Frontier) {
    let m_len = z.len();
    let mut levels: Vec<Vec<BlaEntry>> = Vec::new();
    let mut counts = Vec::new();
    let mut last = Vec::new();
    // Level 0: steps starting at m = 1..=M-2.
    let mut count = m_len.saturating_sub(2);
    let old0 = from.counts.first().copied().unwrap_or(0);
    let mut new: Vec<BlaEntry> = (old0 + 1..=count).map(|m| single_step(z[m])).collect();
    let mut level = 0;
    loop {
        let old = from.counts.get(level).copied().unwrap_or(0);
        let prev_last = from.last.get(level).copied().flatten();
        counts.push(count);
        last.push(new.last().copied().or(prev_last));
        let done = count < 2;
        let entries = std::mem::take(&mut new);
        if !done {
            // Level + 1 pairs entries (2j, 2j + 1) of this level.
            let next_count = count / 2;
            let next_old = from.counts.get(level + 1).copied().unwrap_or(0);
            let at = |i: usize| if i >= old { entries[i - old] } else { prev_last.expect("an unpaired entry is the last one") };
            new = (next_old..next_count).map(|j| merge(&at(2 * j), &at(2 * j + 1), dc_max)).collect();
            levels.push(entries);
            count = next_count;
            level += 1;
        } else {
            levels.push(entries);
            break;
        }
    }
    (levels, Frontier { counts, last })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepzoom::fixed::Fixed;
    use crate::deepzoom::reference::{compute, limbs_for_bits};

    fn orbit(re: &str, im: &str, bits: u32, iters: u32) -> Vec<CFe> {
        let l = limbs_for_bits(bits);
        compute(&Fixed::from_decimal(re, l).unwrap(), &Fixed::from_decimal(im, l).unwrap(), iters, &|_| false)
            .unwrap()
            .z
    }

    /// Plain perturbation iteration in floatexp (the GPU's Deep-mode arithmetic, on the CPU).
    fn perturb(z: &[CFe], m0: usize, steps: usize, d0: CFe, dc: CFe) -> CFe {
        let mut d = d0;
        for m in m0..m0 + steps {
            d = z[m].scale(2.0).mul(d).add(d.mul(d)).add(dc);
        }
        d
    }

    #[test]
    fn coefficients_beyond_f32_and_f64_range_stay_finite() {
        // c = i is a Misiurewicz point: its orbit lands on a repelling cycle and never
        // escapes, so |A| (a product of |2Z|) grows without bound.
        let z = orbit("0", "1", 128, 1 << 16);
        let t = build(&z, Fe::new(1.0, -300));
        let top = t.levels.last().unwrap()[0];
        assert!(top.a.re.is_finite() && top.a.im.is_finite());
        assert!(top.b.re.is_finite() && top.b.im.is_finite());
        let la = top.a.abs().log2();
        assert!(la.is_finite());
        // Some level must have coefficients outside the f32 exponent range.
        let max_log2 = t.levels.iter().flatten().map(|e| e.a.abs().log2()).fold(f64::MIN, f64::max);
        assert!(max_log2 > 128.0, "expected coefficients beyond f32 range, got 2^{max_log2}");
    }

    fn same(a: &BlaEntry, b: &BlaEntry) -> bool {
        let c = |x: CFe, y: CFe| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits() && x.e == y.e;
        c(a.a, b.a) && c(a.b, b.b) && a.r.m.to_bits() == b.r.m.to_bits() && a.r.e == b.r.e
    }

    #[test]
    fn appended_table_equals_full_build() {
        let z = orbit("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", 256, 5000);
        let dc = Fe::new(1.0, -40);
        for cuts in [vec![3usize, 4, 5, 6, 7, 1000, 1001, 4097], vec![2, 2049], vec![1, 3000, 3001, 3002], vec![4097]] {
            let mut frontier = Frontier::EMPTY;
            let mut acc: Vec<Vec<BlaEntry>> = Vec::new();
            for &len in cuts.iter() {
                let (tail, f) = build_tail(&z[..len], dc, &frontier);
                for (l, entries) in tail.into_iter().enumerate() {
                    if acc.len() <= l {
                        acc.push(Vec::new());
                    }
                    acc[l].extend(entries);
                }
                frontier = f;
                let full = build(&z[..len], dc);
                let full_levels: Vec<_> = full.levels.iter().filter(|l| !l.is_empty()).collect();
                let got: Vec<_> = acc.iter().filter(|l| !l.is_empty()).collect();
                assert_eq!(got.len(), full_levels.len(), "levels at len {len}");
                for (a, b) in got.iter().zip(&full_levels) {
                    assert_eq!(a.len(), b.len(), "len {len}");
                    assert!(a.iter().zip(b.iter()).all(|(x, y)| same(x, y)), "entries differ at len {len}");
                }
            }
        }
    }

    #[test]
    fn approximation_error_is_small_where_valid() {
        let z = orbit("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", 256, 4000);
        let spacing = Fe::new(1.0, -60);
        let dc_max = spacing.mul(Fe::from_f64(2000.0));
        let t = build(&z, dc_max);
        let mut checked = 0;
        for (l, level) in t.levels.iter().enumerate().take(10) {
            for (i, e) in level.iter().enumerate().step_by(7) {
                let m0 = 1 + i * (1 << l);
                // A δ inside the validity radius and a δc at the bound.
                let r = e.r;
                if r.is_zero() {
                    continue;
                }
                let d0 = CFe::from_parts(r.mul(Fe::from_f64(0.5)), r.mul(Fe::from_f64(0.3)));
                let dc = CFe::from_parts(dc_max.mul(Fe::from_f64(0.6)), dc_max.mul(Fe::from_f64(-0.7)));
                let exact = perturb(&z, m0, 1 << l, d0, dc);
                let approx = e.a.mul(d0).add(e.b.mul(dc));
                let err = exact.sub(approx).abs();
                let scale = exact.abs();
                let rel = err.div(scale).to_f64();
                assert!(rel < 1e-4, "level {l} m0 {m0}: relative error {rel}");
                checked += 1;
            }
        }
        assert!(checked > 50, "only {checked} entries had a nonzero radius");
    }
}

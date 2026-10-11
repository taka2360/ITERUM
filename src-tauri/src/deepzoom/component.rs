//! Proving pixels interior to a hyperbolic component of any period
//! (src/phenomena/escape/PROOF.md, section 2).
//!
//! For a patch of parameters c = c0 + ρu, |u| ≤ 1, a Taylor model (a
//! polynomial in u plus a remainder disc, computed with outward rounding)
//! encloses everything. A Krawczyk test proves that for every c in the patch
//! f_c^p has exactly one fixed point in a disc around an approximate cycle
//! point, and encloses the multiplier λ(c) of that cycle. Where |λ(c)| < 1 the
//! cycle attracts and c is interior. The GPU receives |λ(u)|² − 1 as a real
//! polynomial in (Re u, Im u) of degree ≤ 4 with a bound on everything left
//! out, and evaluates it per pixel like the closed-form tests.

/// Degree of the Taylor models.
const N: usize = 8;
/// Relative rounding error of one f64 operation, doubled for safety.
const EPS: f64 = 2.0 * f64::EPSILON;
/// Absolute error per operation for underflow.
const TINY: f64 = 1e-300;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };
    pub fn new(re: f64, im: f64) -> C64 {
        C64 { re, im }
    }
    /// |re| + |im| (an upper bound of the modulus).
    fn l1(self) -> f64 {
        self.re.abs() + self.im.abs()
    }
    fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: C64) -> C64 {
        C64::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: C64) -> C64 {
        C64::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn scale(self, s: f64) -> C64 {
        C64::new(self.re * s, self.im * s)
    }
    fn inv(self) -> C64 {
        let d = self.re * self.re + self.im * self.im;
        C64::new(self.re / d, -self.im / d)
    }
}

/// Rounds an error bound up (it was computed with rounding to nearest).
fn up(x: f64) -> f64 {
    x * (1.0 + 4.0 * f64::EPSILON) + TINY
}

/// Σ a_k u^k + B(0, r) for |u| ≤ 1.
#[derive(Clone, Debug)]
pub struct Tm {
    pub a: [C64; N + 1],
    pub r: f64,
}

impl Tm {
    fn constant(c: C64) -> Tm {
        let mut a = [C64::ZERO; N + 1];
        a[0] = c;
        Tm { a, r: 0.0 }
    }

    /// Upper bound of |value| over |u| ≤ 1.
    fn sup(&self) -> f64 {
        up(self.a.iter().map(|c| c.l1()).sum::<f64>() + self.r)
    }

    fn add(&self, o: &Tm) -> Tm {
        let mut a = [C64::ZERO; N + 1];
        let mut err = 0.0;
        for k in 0..=N {
            a[k] = self.a[k].add(o.a[k]);
            err += EPS * a[k].l1() + TINY;
        }
        Tm { a, r: up(self.r + o.r + err) }
    }

    fn sub(&self, o: &Tm) -> Tm {
        let mut neg = o.clone();
        for c in neg.a.iter_mut() {
            *c = c.scale(-1.0);
        }
        self.add(&neg)
    }

    fn add_c(&self, c: C64) -> Tm {
        let mut t = self.clone();
        t.a[0] = t.a[0].add(c);
        t.r = up(t.r + EPS * t.a[0].l1() + TINY);
        t
    }

    fn scale(&self, s: C64) -> Tm {
        let mut a = [C64::ZERO; N + 1];
        let mut err = 0.0;
        for k in 0..=N {
            a[k] = self.a[k].mul(s);
            err += 2.0 * EPS * self.a[k].l1() * s.l1() + TINY;
        }
        Tm { a, r: up(self.r * s.l1() + err) }
    }

    fn mul(&self, o: &Tm) -> Tm {
        let mut a = [C64::ZERO; N + 1];
        let mut abs_sum = [0.0f64; N + 1];
        let mut err = 0.0;
        let mut high = 0.0;
        for i in 0..=N {
            for j in 0..=N {
                let p = self.a[i].mul(o.a[j]);
                // The complex product: ≤ 3 roundings of magnitude ≤ |a_i|₁ |b_j|₁.
                err += 2.0 * EPS * self.a[i].l1() * o.a[j].l1() + TINY;
                if i + j <= N {
                    a[i + j] = a[i + j].add(p);
                    abs_sum[i + j] += p.l1();
                } else {
                    high += p.l1();
                }
            }
        }
        for k in 0..=N {
            // Summing up to N + 1 products into a[k]: each addition errs by at
            // most EPS times the magnitude of a partial sum ≤ Σ |products|.
            err += (N as f64 + 1.0) * EPS * abs_sum[k];
        }
        let sa = self.sup_poly();
        let sb = o.sup_poly();
        let r = high + err + sa * o.r + sb * self.r + self.r * o.r;
        Tm { a, r: up(r) }
    }

    fn sup_poly(&self) -> f64 {
        up(self.a.iter().map(|c| c.l1()).sum::<f64>())
    }

    /// Value of the polynomial part at u = 0 (approximate).
    fn at0(&self) -> C64 {
        self.a[0]
    }
}

/// One patch: c = c0 + ρu with c0 = (re, im) given as an f64 point plus an
/// enclosure radius for what f64 cannot hold.
pub struct Patch {
    pub c0: C64,
    pub c0_err: f64,
    pub rho: f64,
}

/// |λ(u)|² − 1 as a real polynomial in (x, y) = (Re u, Im u): coefficient of
/// x^i y^j for i + j ≤ 4 (15 terms, ordered as interior.ts orders them), and a
/// bound on everything else (truncation, remainders, rounding).
#[derive(Clone, Debug)]
pub struct Proof {
    pub period: u32,
    pub terms: Vec<(f64, u32, u32)>,
    pub bound: f64,
}

fn c_of(p: &Patch) -> Tm {
    let mut a = [C64::ZERO; N + 1];
    a[0] = p.c0;
    a[1] = C64::new(p.rho, 0.0);
    Tm { a, r: up(p.c0_err) }
}

/// f^p and its derivative along the orbit of Z under c.
fn iterate(z: &Tm, c: &Tm, p: u32) -> (Tm, Tm) {
    let mut z = z.clone();
    let mut d = Tm::constant(C64::ONE);
    for _ in 0..p {
        d = d.mul(&z).scale(C64::new(2.0, 0.0));
        z = z.mul(&z).add(c);
    }
    (z, d)
}

/// Non-rigorous: the cycle point near `z_start` as a polynomial in u
/// (Newton's method on polynomials, remainders ignored).
fn approximate_cycle(c: &Tm, p: u32, z_start: C64) -> Option<Tm> {
    let mut c_exact = c.clone();
    c_exact.r = 0.0;
    let mut z = Tm::constant(z_start);
    for _ in 0..60 {
        let (fz, d) = iterate(&z, &c_exact, p);
        let f = fz.sub(&z);
        let fd = d.add_c(C64::new(-1.0, 0.0));
        // Newton step on polynomials: z ← z − f / f', with f'⁻¹ as a series.
        let inv = series_inverse(&fd)?;
        let step = f.mul(&inv);
        let mut next = z.sub(&step);
        next.r = 0.0;
        let delta: f64 = step.a.iter().map(|c| c.l1()).sum();
        z = next;
        if !delta.is_finite() {
            return None;
        }
        if delta < 1e-15 * (1.0 + z.a[0].l1()) {
            return Some(z);
        }
    }
    Some(z)
}

/// Power series inverse of a (polynomial part only).
fn series_inverse(t: &Tm) -> Option<Tm> {
    let a0 = t.a[0];
    if a0.l1() == 0.0 {
        return None;
    }
    let inv0 = a0.inv();
    let mut b = [C64::ZERO; N + 1];
    b[0] = inv0;
    for k in 1..=N {
        let mut s = C64::ZERO;
        for j in 1..=k {
            s = s.add(t.a[j].mul(b[k - j]));
        }
        b[k] = s.mul(inv0).scale(-1.0);
    }
    Some(Tm { a: b, r: 0.0 })
}

/// Tries to prove, for every c in the patch, an attracting-or-not cycle of
/// period p near z_start, and returns |λ|² − 1 as a polynomial with bound.
/// None if the Krawczyk test fails.
pub fn prove(patch: &Patch, p: u32, z_start: C64) -> Option<Proof> {
    let c = c_of(patch);
    let zt = approximate_cycle(&c, p, z_start)?;
    // F(z̃) with the true c (remainder included).
    let (fz, _) = iterate(&zt, &c, p);
    let f = fz.sub(&zt);
    let (_, d0) = iterate(&Tm::constant(zt.at0()), &Tm::constant(patch.c0), p);
    let df0 = d0.at0().sub(C64::ONE);
    if df0.l1() == 0.0 {
        return None;
    }
    let y = df0.inv();
    let yf = f.scale(y).sup();
    // Krawczyk: K = z̃ − Y F(z̃) + (1 − Y F'(Z))(Z − z̃) ⊂ int Z, Z = z̃ + B(0, r).
    let mut r = (4.0 * yf).max(1e-300);
    for _ in 0..40 {
        let mut zb = zt.clone();
        zb.r = r;
        let (_, d) = iterate(&zb, &c, p);
        let jac = d.add_c(C64::new(-1.0, 0.0));
        let m = Tm::constant(C64::ONE).sub(&jac.scale(y));
        let k = up(yf + m.sup() * r);
        if !k.is_finite() {
            return None;
        }
        if k < r {
            return Some(multiplier_poly(p, &zb, &c));
        }
        if m.sup() >= 1.0 {
            // No contraction anywhere near: give up.
            return None;
        }
        r = k * 1.5;
    }
    None
}

/// Λ(u) = (f^p)'(z*) over the cycle enclosure Z, then |Λ|² − 1.
fn multiplier_poly(p: u32, z: &Tm, c: &Tm) -> Proof {
    let (_, lambda) = iterate(z, c, p);
    // Real and imaginary parts as polynomials in (x, y): u^k = (x + iy)^k.
    let mut re = [[0.0f64; 2 * N + 1]; 2 * N + 1];
    let mut im = [[0.0f64; 2 * N + 1]; 2 * N + 1];
    let mut err = 0.0;
    let binom = |n: usize, k: usize| -> f64 { (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64) };
    for k in 0..=N {
        let b = lambda.a[k];
        // (x + iy)^k = Σ_j C(k, j) x^(k−j) (iy)^j
        for j in 0..=k {
            let w = binom(k, j);
            let (s_re, s_im) = match j % 4 {
                0 => (1.0, 0.0),
                1 => (0.0, 1.0),
                2 => (-1.0, 0.0),
                _ => (0.0, -1.0),
            };
            // b · w · i^j
            let t = C64::new(b.re * s_re - b.im * s_im, b.re * s_im + b.im * s_re).scale(w);
            re[k - j][j] += t.re;
            im[k - j][j] += t.im;
            err += 4.0 * EPS * w * b.l1();
        }
    }
    // |Λ_poly|² = re² + im² as a polynomial in (x, y).
    let mut sq = vec![vec![0.0f64; 4 * N + 1]; 4 * N + 1];
    let mut abs_re = 0.0;
    let mut abs_im = 0.0;
    for a in 0..=2 * N {
        for b in 0..=2 * N {
            abs_re += re[a][b].abs();
            abs_im += im[a][b].abs();
            for c2 in 0..=2 * N {
                for d2 in 0..=2 * N {
                    let v = re[a][b] * re[c2][d2] + im[a][b] * im[c2][d2];
                    if v != 0.0 {
                        sq[a + c2][b + d2] += v;
                        err += 3.0 * EPS * (re[a][b] * re[c2][d2]).abs().max((im[a][b] * im[c2][d2]).abs()) * 2.0;
                    }
                }
            }
        }
    }
    // Rounding of the polynomial evaluation into re/im above: err (scaled by sup of the other factor).
    let sup_poly = up(abs_re + abs_im);
    let mut bound = err * (1.0 + sup_poly) * 2.0;
    // Remainder of Λ: |Λ|² ≤ (|Λ_poly| + r)² = |Λ_poly|² + 2|Λ_poly| r + r².
    bound += 2.0 * sup_poly * lambda.r + lambda.r * lambda.r;
    let mut terms = Vec::new();
    for s in 0..=4usize {
        for i in (0..=s).rev() {
            let j = s - i;
            let mut v = sq[i][j];
            if i == 0 && j == 0 {
                bound += EPS * v.abs();
                v -= 1.0;
            }
            terms.push((v, i as u32, j as u32));
        }
    }
    for i in 0..=4 * N {
        for j in 0..=4 * N {
            if i + j > 4 {
                bound += sq[i][j].abs();
            }
        }
    }
    Proof { period: p, terms, bound: up(bound) }
}

/// Bytes per patch in the GPU layout (escape.wgsl, `Patch`): header
/// (d0.x, d0.y: f32, ρ exponent: i32, period: u32), bound (m: f32, e: i32,
/// 0, 0), then 15 terms (m: f32, e: i32, i: u32, j: u32).
pub const PATCH_BYTES: usize = 32 + 15 * 16;

/// Splits a finite f64 into an f32 mantissa in [0.5, 1) (rounded up in
/// magnitude when `up` is set) and an exponent.
fn to_fe32(v: f64, round_up: bool) -> (f32, i32) {
    if v == 0.0 {
        return (0.0, 0);
    }
    let (m, e) = super::fexp::frexp(v);
    let mut f = m as f32;
    if round_up && (f as f64).abs() < m.abs() {
        f = f32::from_bits(f.to_bits() + 1);
    }
    if f.abs() >= 1.0 {
        return (f / 2.0, e + 1);
    }
    (f, e)
}

/// A tile centre: an f64 point (held exactly) near the requested one, and its
/// offset from the reference point.
struct Centre {
    c0: C64,
    d0x: f64,
    d0y: f64,
}

fn centre(cr: &super::fixed::Fixed, ci: &super::fixed::Fixed, dx: f64, dy: f64) -> Option<Centre> {
    use super::fexp::Fe;
    use super::fixed::Fixed;
    let l = cr.mag.len();
    let tx = cr.add(&Fixed::from_fe(Fe::from_f64(dx), l));
    let ty = ci.add(&Fixed::from_fe(Fe::from_f64(dy), l));
    let c0 = C64::new(tx.to_f64(), ty.to_f64());
    if !c0.re.is_finite() || !c0.im.is_finite() {
        return None;
    }
    let d0x = Fixed::from_f64(c0.re, l).sub(cr).to_f64();
    let d0y = Fixed::from_f64(c0.im, l).sub(ci).to_f64();
    Some(Centre { c0, d0x, d0y })
}

fn f_iter(z: C64, c: C64, p: u32) -> (C64, C64) {
    let (mut w, mut d) = (z, C64::ONE);
    for _ in 0..p {
        d = d.mul(w).scale(2.0);
        w = w.mul(w).add(c);
    }
    (w, d)
}

/// Plain Newton's method for a point of a cycle of exact period p, from
/// `start`: the point and the cycle's multiplier, if it converged.
fn scalar_cycle(c: C64, p: u32, start: C64) -> Option<(C64, C64)> {
    let mut z = start;
    for _ in 0..50 {
        let (w, d) = f_iter(z, c, p);
        let g = d.sub(C64::ONE);
        if g.l1() == 0.0 {
            return None;
        }
        let step = w.sub(z).mul(g.inv());
        z = z.sub(step);
        if !z.re.is_finite() || !z.im.is_finite() || z.l1() > 4.0 {
            return None;
        }
        if step.l1() < 1e-14 * (1.0 + z.l1()) {
            // Not a point of a shorter cycle (its period would divide p).
            for q in 1..p {
                if p % q == 0 && f_iter(z, c, q).0.sub(z).l1() < 1e-9 * (1.0 + z.l1()) {
                    return None;
                }
            }
            return Some((z, f_iter(z, c, p).1));
        }
    }
    None
}

/// (period, cycle point) of an attracting cycle the orbit of c0 approaches.
fn seed_from_orbit(c0: C64, max_period: u32) -> Option<(u32, C64)> {
    // Candidate periods: the iterations at which |z_n| reaches a new minimum.
    let mut z = C64::ZERO;
    let mut best = f64::INFINITY;
    let mut candidates = Vec::new();
    // Long enough for slowly converging orbits near a component's boundary
    // to come close to their cycle (Newton then finds it, not a neighbour).
    for n in 1..=(1u32 << 21) {
        z = z.mul(z).add(c0);
        let a = z.re * z.re + z.im * z.im;
        if a > 4.0 {
            return None;
        }
        if a < best {
            best = a;
            if n <= max_period {
                candidates.push(n);
            }
        }
    }
    for &p in candidates.iter().take(12) {
        // Newton from points of the orbit over one period: one of them is
        // nearest to the attracting cycle (a repelling cycle of a divisor of
        // p can lie close by, near a bifurcation).
        let mut start = z;
        for _ in 0..p.min(16) {
            if let Some((zc, l)) = scalar_cycle(c0, p, start) {
                if l.re * l.re + l.im * l.im < 1.0 {
                    return Some((p, zc));
                }
            }
            start = start.mul(start).add(c0);
        }
    }
    None
}

/// The proof of a patch around `ce` from a cycle point seed, if it can prove
/// anything there (|λ|² − 1 + bound < 0 somewhere in the disc).
fn prove_patch(ce: &Centre, rho_log2: i32, p: u32, z: C64) -> Option<(Proof, C64)> {
    let (zc, _) = scalar_cycle(ce.c0, p, z)?;
    let patch = Patch { c0: ce.c0, c0_err: 0.0, rho: super::fexp::ldexp(1.0, rho_log2) };
    let proof = prove(&patch, p, zc)?;
    let rest: f64 = proof.terms[1..].iter().map(|t| t.0.abs()).sum();
    if proof.terms[0].0 - rest + proof.bound >= 0.0 {
        return None;
    }
    Some((proof, zc))
}

/// The GPU record of a proof (layout: PATCH_BYTES).
fn record(ce: &Centre, rho_log2: i32, proof: &Proof) -> Vec<u8> {
    let rho = super::fexp::ldexp(1.0, rho_log2);
    // The GPU computes u = (δc − d0) / ρ in f32, with d0 rounded to f32:
    // |u − u_true| ≤ δ. Add sup|∇P| · δ for |u| ≤ 1, and 2^-18 of the bound
    // for the shader's own addition of it.
    let delta = 2f64.powi(-22) * (1.0 + (ce.d0x.abs() + ce.d0y.abs()) / rho);
    let grad: f64 = proof.terms.iter().map(|&(v, i, j)| v.abs() * (i + j) as f64).sum();
    let bound = up((proof.bound + grad * delta * 2.0) * (1.0 + 2f64.powi(-18)));
    let mut out = Vec::with_capacity(PATCH_BYTES);
    let push = |out: &mut Vec<u8>, w: u32| out.extend_from_slice(&w.to_le_bytes());
    push(&mut out, (ce.d0x as f32).to_bits());
    push(&mut out, (ce.d0y as f32).to_bits());
    push(&mut out, rho_log2 as u32);
    push(&mut out, proof.period);
    let (bm, be) = to_fe32(bound, true);
    push(&mut out, bm.to_bits());
    push(&mut out, be as u32);
    push(&mut out, 0);
    push(&mut out, 0);
    // Coefficients rounded to f32 mantissas: the shader's error bound covers
    // a relative 2^-24 per coefficient (PROOF.md, section 1).
    for &(v, i, j) in &proof.terms {
        let (m, e) = to_fe32(v, false);
        push(&mut out, m.to_bits());
        push(&mut out, e as u32);
        push(&mut out, i);
        push(&mut out, j);
    }
    out
}

/// Runs `work` for each index in `which` on all cores.
fn parallel<T: Send>(which: &[usize], work: &(dyn Fn(usize) -> T + Sync)) -> Vec<(usize, T)> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk = which.len().div_ceil(threads).max(1);
    let mut out = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = which.chunks(chunk).map(|ks| s.spawn(move || ks.iter().map(|&k| (k, work(k))).collect::<Vec<_>>())).collect();
        for h in handles {
            out.extend(h.join().unwrap());
        }
    });
    out
}

/// Proves the tiles (centres as offsets from the reference, patch radius
/// 2^rho_log2, `grid_w` tiles per row) interior to hyperbolic components of
/// period ≤ max_period: first from the orbit of each centre, then, for the
/// rest, from the cycles found in neighbouring tiles (a tile whose centre lies
/// outside a component may still reach into it). One record per tile.
pub fn prove_tiles(cr: &super::fixed::Fixed, ci: &super::fixed::Fixed, tiles: &[(f64, f64)], grid_w: usize, rho_log2: i32, max_period: u32) -> Vec<Option<Vec<u8>>> {
    let centres: Vec<Option<Centre>> = tiles.iter().map(|&(dx, dy)| centre(cr, ci, dx, dy)).collect();
    let mut found: Vec<Option<(Proof, C64)>> = (0..tiles.len()).map(|_| None).collect();
    let all: Vec<usize> = (0..tiles.len()).collect();
    let first = |k: usize| {
        let ce = centres[k].as_ref()?;
        let (p, z) = seed_from_orbit(ce.c0, max_period)?;
        prove_patch(ce, rho_log2, p, z)
    };
    for (k, r) in parallel(&all, &first) {
        found[k] = r;
    }
    // Second pass: seeds from proven neighbours.
    let seeds: Vec<Vec<(u32, C64)>> = (0..tiles.len())
        .map(|k| {
            let (x, y) = ((k % grid_w) as i64, (k / grid_w) as i64);
            let mut v: Vec<(u32, C64)> = Vec::new();
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= grid_w as i64 {
                    continue;
                }
                let n = (ny * grid_w as i64 + nx) as usize;
                if let Some(Some((pr, z))) = found.get(n) {
                    if !v.iter().any(|&(p, _)| p == pr.period) {
                        v.push((pr.period, *z));
                    }
                }
            }
            v
        })
        .collect();
    let rest: Vec<usize> = (0..tiles.len()).filter(|&k| found[k].is_none() && !seeds[k].is_empty()).collect();
    let second = |k: usize| {
        let ce = centres[k].as_ref()?;
        seeds[k].iter().find_map(|&(p, z)| prove_patch(ce, rho_log2, p, z))
    };
    for (k, r) in parallel(&rest, &second) {
        found[k] = r;
    }
    found
        .iter()
        .enumerate()
        .map(|(k, f)| f.as_ref().map(|(pr, _)| record(centres[k].as_ref().unwrap(), rho_log2, pr)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Evaluates |λ|² − 1 from the proof at u.
    fn eval(p: &Proof, x: f64, y: f64) -> f64 {
        p.terms.iter().map(|&(v, i, j)| v * x.powi(i as i32) * y.powi(j as i32)).sum()
    }

    /// Multiplier of the attracting p-cycle of c, by iteration and Newton.
    fn lambda_direct(c: C64, p: u32) -> f64 {
        let mut z = C64::ZERO;
        for _ in 0..200000 {
            z = z.mul(z).add(c);
        }
        for _ in 0..50 {
            let (mut w, mut d) = (z, C64::ONE);
            for _ in 0..p {
                d = d.mul(w).scale(2.0);
                w = w.mul(w).add(c);
            }
            z = z.sub(w.sub(z).mul(d.sub(C64::ONE).inv()));
        }
        let (mut w, mut d) = (z, C64::ONE);
        for _ in 0..p {
            d = d.mul(w).scale(2.0);
            w = w.mul(w).add(c);
        }
        (d.re * d.re + d.im * d.im).sqrt()
    }

    #[test]
    fn period_three_bulb_matches_direct_multiplier() {
        // Inside the upper period-3 bulb (nucleus ≈ −0.1226 + 0.7449i).
        let patch = Patch { c0: C64::new(-0.12, 0.74), c0_err: 0.0, rho: 1e-3 };
        let mut z = C64::ZERO;
        for _ in 0..1000 {
            z = z.mul(z).add(patch.c0);
        }
        let proof = prove(&patch, 3, z).expect("proof");
        for (x, y) in [(0.0, 0.0), (0.5, 0.0), (-0.3, 0.7), (0.6, -0.6)] {
            let c = C64::new(-0.12 + 1e-3 * x, 0.74 + 1e-3 * y);
            let l = lambda_direct(c, 3);
            let got = eval(&proof, x, y);
            assert!((got - (l * l - 1.0)).abs() <= proof.bound + 1e-12, "at ({x},{y}): {got} vs {}", l * l - 1.0);
        }
        assert!(proof.bound < 1e-10, "bound {}", proof.bound);
        assert!(eval(&proof, 0.0, 0.0) + proof.bound < 0.0);
    }

    #[test]
    fn reported_view_period_six_component() {
        // The period-6 component just below the reported boundary view's centre.
        let c0 = C64::new(-0.115435619547424377, -0.837972464768108822 - 3e-9);
        let patch = Patch { c0, c0_err: 0.0, rho: 1e-9 };
        let mut z = C64::ZERO;
        for _ in 0..2_000_000 {
            z = z.mul(z).add(c0);
        }
        let proof = prove(&patch, 6, z).expect("proof");
        let l = lambda_direct(c0, 6);
        let got = eval(&proof, 0.0, 0.0);
        assert!((got - (l * l - 1.0)).abs() <= proof.bound + 1e-12, "{got} vs {}", l * l - 1.0);
        // Near the root shared with the period-3 component, 1/(λ − 1) is large
        // and amplifies f64 rounding; most pixels there still have |λ|² − 1 < −1e-8.
        assert!(proof.bound < 5e-9, "bound {}", proof.bound);
    }

    #[test]
    fn tile_record_for_the_reported_view() {
        use crate::deepzoom::fixed::Fixed;
        let l = 3;
        let cr = Fixed::from_decimal("-0.115435619547424377", l).unwrap();
        let ci = Fixed::from_decimal("-0.837972464768108822", l).unwrap();
        // A tile 3e-9 below the centre (inside the period-6 component,
        // |λ|² − 1 ≈ −2.3e-7) and one 6e-9 above, far enough out that no
        // point of its disc is in the component.
        let recs = prove_tiles(&cr, &ci, &[(0.0, -3e-9), (0.0, 6e-9)], 1, -30, 4096);
        let rec = recs[0].as_ref().expect("proved");
        assert_eq!(rec.len(), PATCH_BYTES);
        assert_eq!(u32::from_le_bytes(rec[12..16].try_into().unwrap()), 6);
        assert!(recs[1].is_none());
    }

    #[test]
    fn no_proof_for_an_exterior_patch() {
        // A patch around c = 0.3 (outside the set): there is a 1-cycle, but
        // it repels, so |λ|² − 1 must not come out negative.
        let patch = Patch { c0: C64::new(0.3, 0.0), c0_err: 0.0, rho: 1e-3 };
        if let Some(proof) = prove(&patch, 1, C64::new(0.5, 0.5)) {
            assert!(eval(&proof, 0.0, 0.0) - proof.bound > 0.0);
        }
    }
}


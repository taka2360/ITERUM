# Proven interior: what is proved, and the error bounds

A pixel gets status 6 ("proven") only when the shader has **proved** that its
parameter c lies in the interior of the Mandelbrot set. Every other pixel that
stops without escaping is "unresolved" (status 2, shown as interior because the
iteration limit ran out) or "cycling" (status 4: the computed f32 iteration
repeated bit for bit — a statement about the computed dynamics, not about c).

Tests in `gputest.ts` and the Rust test suite look for mistakes in this
derivation and its implementation. They are not part of the proof.

## Which c belongs to a pixel

The pixel's parameter is defined as the exact value of what the shader
iterates with:

- direct mode (MODE 0): c = the f32 value `P.center + rel_pos(pix) · spacing`;
- perturbation modes (MODE 2, 3): c = C + δc, where C is the reference point
  (an exact binary fraction held as a BigInt) and δc is the f32 (MODE 2) or
  floatexp (MODE 3) value used by the iteration. C is the point the reference
  orbit was computed for, i.e. the requested point truncated to the orbit's
  fixed-point precision (`orbitReference` in orbit.ts mirrors fixed.rs).

Any c inside the pixel's footprint is an equally valid sample, so this
definition loses nothing. The test and the iteration obtain these values from
the same functions (`direct_c`, `perturb_dc`, `deep_dc`).

The test converts the components of c (or δc) to floatexp with `frexp`, which
may flush a subnormal input to zero. Such pixels are not tested at all, so
every value entering the polynomial below is exact.

## Arithmetic model

WGSL requires f32 addition, subtraction and multiplication to be *correctly
rounded*, defined as returning one of the two representable values adjacent to
the exact result (the rounding direction is unspecified). Hence for a result r
of exact value x, |r − x| ≤ 2⁻²³ |x| while x is in the normal range. Subnormal
results may be flushed to zero: an absolute error of at most 2⁻¹²⁶. We use

- ε = 2⁻²² relative per operation (twice the bound; also covers a fused
  multiply-add, which rounds once instead of twice),
- τ = 2⁻¹²⁰ absolute per operation (flushing).

floatexp values (f32 mantissa m with |m| ∈ [0.5, 1), i32 exponent) multiply
mantissas (results ≥ 0.25 in magnitude: never subnormal) and add exponents
exactly; normalisation (`frexp`, `ldexp` by the exponent of the mantissa) is
exact. `cf_add` aligns the smaller operand with `ldexp` (exact while the result
stays normal, which normalised mantissas guarantee for shifts ≤ 30) and drops
it entirely when its exponent is more than 30 below: the dropped value is
smaller than 2⁻²⁹ of the result, below ε. So every floatexp add or multiply
has relative error ≤ ε and no absolute error term.

Magnitudes on the bound side use |z|₁ = |Re z| + |Im z| ≥ |z|, so no square
root (whose accuracy WGSL does not guarantee) enters a bound.

## 1. Main cardioid and period-2 disk (closed form)

c is interior to the period-2 component iff g(c) = |c + 1|² − 1/16 < 0, and to
the main cardioid iff h(c) = q (q + x − 1/4) − y²/4 < 0 with q = (x − 1/4)² + y²
(c = x + iy). Both components are open sets; boundary points are never
accepted because the inequalities are strict.

Write c = C + d, d = (dx, dy). g and h are polynomials in (dx, dy) of degree 2
and 4. Their coefficients a_ij are computed **exactly** on the CPU from the
BigInt C (rational arithmetic on binary fractions) and then rounded to
floatexp, each with relative error ≤ 2⁻²⁴. g has 5 nonzero coefficients, h at
most 15.

The shader evaluates v = Σ a_ij dx^i dy^j in floatexp, term by term (each term
at most 4 multiplications), and sums the terms in a fixed order (at most 14
additions). Let T = Σ |a_ij| |dx|^i |dy|^j (true values). Then

- each computed term differs from the exact term by at most
  ((1 + 2⁻²⁴)(1 + ε)⁴ − 1) |term| ≤ 5ε |term|;
- each addition adds at most ε times the magnitude of its result, which is at
  most (1 + 5ε)T; 14 additions add at most 14ε(1 + 5ε)T.

So |v_computed − v_exact| ≤ (5ε + 15ε) T ≤ 20ε T < 2⁻¹⁷ T. The shader computes
T̃ (the same sum of absolute values, itself within a factor (1 + 20ε) of T)
and uses E = 2⁻¹⁶ T̃ ≥ 2⁻¹⁷ T. The pixel is proven interior when

    v_computed + E < 0,

which implies v_exact < 0. In direct mode C = 0 and d = c, with the same
evaluation.

## 2. Any hyperbolic component: Krawczyk on Taylor models

**Statement.** If f_c has a periodic point z of period dividing p with
|(f_c^p)'(z)| < 1, the cycle of z attracts, c lies in a hyperbolic component
and therefore in the interior of M. As in section 1, this is a statement
about c alone; the pixel's own orbit plays no part.

**Why not a disc mapped into itself, evaluated per pixel in f32.** The
reported slow views sit next to a component's boundary, where
1 − |λ| ≈ 1e-8 … 1e-7. A disc on which f_c^p contracts then has a radius of
order 1 − |λ| relative to |z|, far below what f32 resolves, so such a test
could not prove these pixels. The proof is therefore done on the CPU, once
per tile of the image, in f64 with outward rounding, and the GPU only
evaluates its result.

**Patches.** The image is split into an 8 × 8 grid of tiles. For each tile
the CPU (`component.rs`) picks a centre c0 that f64 holds exactly and a radius
ρ (a power of two) covering the tile with a pixel to spare, and considers
every c = c0 + ρu, |u| ≤ 1.

**Taylor models.** A Taylor model is a polynomial in u of degree 8 with f64
complex coefficients plus a remainder radius r: it encloses a function g if
|g(u) − poly(u)| ≤ r for all |u| ≤ 1. Sums and products are formed with every
rounding error (≤ 2^-51 relative per f64 operation, counted generously, plus
an absolute 10^-300 for underflow) and every term of degree > 8 (bounded by
the sum of its coefficients' moduli, since |u| ≤ 1) moved into the remainder.
The remainder of a product is |A|·r_B + |B|·r_A + r_A·r_B with |A| bounded by
the sum of the moduli of A's coefficients.

**Cycle.** Newton's method on polynomials (not rigorous, only a guess) gives
z̃(u) ≈ a periodic point of f_{c(u)}^p. With Z = z̃ + B(0, r) and
F(z) = f^p(z) − z, the Krawczyk operator

    K = z̃ − Y F(z̃) + (1 − Y F'(Z)) (Z − z̃),   Y ≈ 1 / F'(z̃(0)),

is enclosed as a Taylor model. If sup |Y F(z̃)| + sup |1 − Y F'(Z)| · r < r,
then K(Z) ⊂ int Z for every u, so (the complex Krawczyk theorem) f_{c(u)}^p
has exactly one fixed point z*(u) in Z(u).

**Multiplier.** Λ(u) = (f^p)'(z) over the enclosure Z encloses λ(c(u)) of
that cycle. Writing u = x + iy, the real and imaginary parts of Λ's
polynomial are expanded into polynomials in (x, y) and |Λ|² − 1 is formed;
terms of total degree > 4 are bounded (|x|, |y| ≤ 1) into the bound, and so
are 2 |Λ| r_Λ + r_Λ² and the rounding of these expansions. The constant term
loses 1 with an f64 rounding of at most 2^-52.

**What the GPU checks.** The record holds d0 = c0 − C (C the reference
point; rounded to f32), the exponent of ρ, the 15 coefficients (rounded to
f32 mantissas) and the bound B. For a pixel of the tile the shader forms
u = (δc − d0) · 2^-ρe in f32. Its error is at most δ = 2^-22 (1 + |d0|/ρ)
(rounding of d0 and of the difference), so the CPU adds sup |∇P| · 2δ to B,
and the shader only uses pixels with |u|² ≤ 0.9999 (the exact u then lies in
the unit disc). The polynomial is evaluated as in section 1 (error ≤ 2^-16 T)
and the pixel is proven when value + 2^-16 T + B < 0. The CPU sends B with
2^-18 of itself to spare, more than the shader's addition can lose.

**Choosing p.** Candidates are the iterations at which |z_n| of c0's orbit
(2^21 steps in f64) reaches a new minimum. For each, plain Newton's method
from points of the orbit looks for a point of a cycle of exact period p with
|λ| < 1; only then is the Taylor-model proof attempted. Tiles whose centre
yields nothing (it escapes, or lies outside the component) try the cycles
found in neighbouring tiles: their disc may still reach into the component.
None of this choice affects correctness — a wrong guess only fails to prove.

**Limits.** f64 holds c0 and the pixels' offsets only to about 10^-16
relative, so patches need a radius well above that: perturbation views of
moderate depth. Near the root where a component meets its parent, F' → 0 and
the Krawczyk test fails on tiles that contain it; those pixels stay
unresolved.

## 3. Shading of proven pixels (a heuristic, checked against running to the limit)

The interior shade uses the smallest |z|² over the orbit, which an early
decision cannot know. Near a component's boundary (multiplier |λ| = 1 − η)
the orbit keeps passing close to 0 for a while before it settles on the
cycle; at the period-2 boundary view the last such pass came within 5/√η
iterations for every sampled point. A proven pixel therefore iterates
16/√η steps (at least 1024) and then finishes; η is estimated from the
polynomial value (period 2: η ≈ −8 g; cardioid: η ≈ −h |λ'(c)| / |∇h| with
λ'(c) = 2/√(1 − 4c), no shortcut near the cusp c = 1/4; patches:
η ≈ −(|λ|² − 1)/2). If the iteration
limit comes first, the pixel pauses like any other and resumes when the limit
is raised, so its shading is exactly what running to the limit gives
whenever the shading steps exceed the limit.

This rule affects only the shading, never which pixels are proven. It is
accepted by measurement: on the period-2 boundary view (limit 2^22) the
samples of every proven pixel are bit-identical to running to the limit.

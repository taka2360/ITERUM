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

## 2. Any hyperbolic component: a disc mapped into itself

**Theorem (Earle–Hamilton, one-variable case).** Let F be holomorphic on a
neighbourhood of a closed disc B and F(B) ⊂ int B. Then F has exactly one fixed
point in B, it is attracting, and every point of B converges to it.

With F = f_c^p this gives an attracting periodic orbit of f_c, which places c
in a hyperbolic component, i.e. in the interior of M. The statement is about c
alone: nothing depends on how accurately the pixel's own orbit was computed.

The shader proves F(B) ⊂ int B by propagating a disc enclosure (centre m,
radius r) through p perturbation steps δ' = (2 Z_k + δ) δ + δc.

**One step.** With δ = m + η, |η| ≤ r,

    δ' = (2 Z_k + m) m + δc + (2 Z_k + 2m) η + η².

The first-order coefficient is the derivative at the centre, 2 Z_k + 2m. The
stored reference value Ẑ_k differs from the true Z_k by at most
e_k = 2⁻²⁴ |Ẑ_k|₁ + 2^(8 − bits) (rounding to an f32 mantissa, plus the
fixed-point truncation of the reference computation), which changes δ' by at
most 2 e_k |δ| ≤ 2 e_k (|m| + r). The new centre m' = fl((2Ẑ_k + m) m + δc)
has rounding error at most ε_arith = ε (3 |2Ẑ_k + m|₁ |m|₁ + 2 |δc|₁) + 8τ (four
real multiplications and three additions per component, bounded generously).
Therefore the image disc is enclosed by (m', r') with

    r' ≥ |2Ẑ_k + 2m|₁ r + r² + 2 e_k (|m|₁ + r) + ε_arith,

evaluated with every operation rounded up (computed value × (1 + 2⁻²⁰)).

**Rebase** (δ ← Z_k + δ): m' = fl(Ẑ_k + m), r' ≥ r + e_k + ε (|Ẑ_k|₁ + |m|₁) + τ.

**Comparison.** The disc is formed and compared in absolute coordinates (the
state right after a rebase, where δ = z). After p steps and a final rebase,
the pixel is proven interior when

    upper(|m_p − m_0|₁) + r_p < lower(r_0),

where upper(x) = x (1 + 2⁻²⁰) + τ, lower(x) = x (1 − 2⁻²⁰).

BLA is never used inside a proof: its validity radius is an f32-precision
estimate, not a rigorous remainder bound.

## 3. Shading of proven pixels (a heuristic, checked against running to the limit)

The interior shade uses the smallest |z|² over the orbit, which an early
decision cannot know. Near a component's boundary (multiplier |λ| = 1 − η)
the orbit keeps passing close to 0 for a while before it settles on the
cycle; at the period-2 boundary view the last such pass came within 5/√η
iterations for every sampled point. A proven pixel therefore iterates
16/√η steps (at least 1024) and then finishes; η is estimated from the
polynomial value (period 2: η ≈ −8 g; cardioid: η ≈ −h |λ'(c)| / |∇h| with
λ'(c) = 2/√(1 − 4c); no shortcut near the cusp c = 1/4). If the iteration
limit comes first, the pixel pauses like any other and resumes when the limit
is raised, so its shading is exactly what running to the limit gives
whenever the shading steps exceed the limit.

This rule affects only the shading, never which pixels are proven. It is
accepted by measurement: on the period-2 boundary view (limit 2^22) the
samples of every proven pixel are bit-identical to running to the limit.

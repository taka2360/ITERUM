// Closed-form interior tests for the two largest components of the Mandelbrot
// set (see PROOF.md, section 1):
//   period-2 disk   g(c) = |c + 1|² − 1/16 < 0
//   main cardioid   h(c) = q (q + x − 1/4) − y²/4 < 0,  q = (x − 1/4)² + y²
// Around a reference point C (an exact binary fraction), g and h are
// polynomials in d = c − C. Their coefficients are computed here exactly with
// BigInt arithmetic and rounded once to floatexp; the shader evaluates the
// polynomials with a rigorous error bound.

/** A dyadic rational num / 2^k, held exactly. */
type Dyadic = [num: bigint, k: number];
/** Polynomial in (dx, dy): key i * 8 + j → coefficient of dx^i dy^j. */
type Poly = Map<number, Dyadic>;

const dyadic = (n: number | bigint, k = 0): Dyadic => [BigInt(n), k];

function dadd(a: Dyadic, b: Dyadic): Dyadic {
  const k = Math.max(a[1], b[1]);
  return [(a[0] << BigInt(k - a[1])) + (b[0] << BigInt(k - b[1])), k];
}

const dmul = (a: Dyadic, b: Dyadic): Dyadic => [a[0] * b[0], a[1] + b[1]];

function padd(p: Poly, q: Poly): Poly {
  const r = new Map(p);
  for (const [key, v] of q) r.set(key, r.has(key) ? dadd(r.get(key)!, v) : v);
  return r;
}

function pmul(p: Poly, q: Poly): Poly {
  const r: Poly = new Map();
  for (const [ka, a] of p) {
    for (const [kb, b] of q) {
      const key = ka + kb;
      const v = dmul(a, b);
      r.set(key, r.has(key) ? dadd(r.get(key)!, v) : v);
    }
  }
  return r;
}

const constant = (v: Dyadic): Poly => new Map([[0, v]]);

/** Nearest floatexp (24-bit mantissa, |m| in [0.5, 1)): relative error ≤ 2^-24. */
export function dyadicToFe([num, k]: Dyadic): [m: number, e: number] {
  if (num === 0n) return [0, 0];
  const neg = num < 0n;
  const a = neg ? -num : num;
  const bits = a.toString(2).length;
  const shift = bits - 24;
  const mant = shift > 0 ? (a + (1n << BigInt(shift - 1))) >> BigInt(shift) : a << BigInt(-shift);
  let m = Number(mant) / 2 ** 24;
  let e = bits - k;
  if (m >= 1) {
    m /= 2;
    e += 1;
  }
  return [neg ? -m : m, e];
}

/** Terms per polynomial in the packed buffer: g (degree 2), then h (degree 4). */
export const G_TERMS = 6;
export const H_TERMS = 15;
const TERM_KEYS = (deg: number) => {
  const keys: number[] = [];
  for (let s = 0; s <= deg; s++) for (let i = s; i >= 0; i--) keys.push(i * 8 + (s - i));
  return keys;
};

/** Exact g and h around C = (re + i·im) / 2^frac. */
export function boundaryPolys(re: bigint, im: bigint, frac: number): { g: Poly; h: Poly } {
  const x: Poly = new Map([[0, dyadic(re, frac)], [8, dyadic(1)]]);
  const y: Poly = new Map([[0, dyadic(im, frac)], [1, dyadic(1)]]);
  const x1 = padd(x, constant(dyadic(1)));
  const y2 = pmul(y, y);
  const g = padd(padd(pmul(x1, x1), y2), constant(dyadic(-1, 4)));
  const u = padd(x, constant(dyadic(-1, 2)));
  const q = padd(pmul(u, u), y2);
  const h = padd(padd(pmul(q, q), pmul(q, u)), pmul(y2, constant(dyadic(-1, 2))));
  return { g, h };
}

/**
 * Packed terms for the shader: G_TERMS + H_TERMS entries of
 * (mantissa f32, exponent i32, i u32, j u32), coefficient of dx^i dy^j.
 */
export function interiorTerms(re: bigint, im: bigint, frac: number): ArrayBuffer {
  const { g, h } = boundaryPolys(re, im, frac);
  const buf = new ArrayBuffer((G_TERMS + H_TERMS) * 16);
  const f = new Float32Array(buf);
  const i32 = new Int32Array(buf);
  const u = new Uint32Array(buf);
  let slot = 0;
  for (const [poly, deg] of [[g, 2], [h, 4]] as const) {
    for (const key of TERM_KEYS(deg)) {
      const [m, e] = dyadicToFe(poly.get(key) ?? [0n, 0]);
      f[slot * 4] = m;
      i32[slot * 4 + 1] = e;
      u[slot * 4 + 2] = key >> 3;
      u[slot * 4 + 3] = key & 7;
      slot++;
    }
  }
  return buf;
}

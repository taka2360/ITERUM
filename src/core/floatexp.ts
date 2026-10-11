// CPU reference of the GPU floatexp arithmetic in `src/shaders/floatexp.wgsl`.
// Values are m * 2^e with |m| in [0.5, 1) (frexp convention) or m = 0, e = 0.
// Mantissas are rounded to f32 after every operation, mirroring the shader, so
// tests can compare GPU output against this bit-for-bit-ish model.

const f32 = Math.fround;
const buf = new DataView(new ArrayBuffer(8));

/** Splits a finite number into mantissa in [0.5, 1) and integer exponent. */
export function frexp(x: number): [number, number] {
  if (x === 0 || !Number.isFinite(x)) return [x, 0];
  buf.setFloat64(0, x);
  const hi = buf.getUint32(0);
  let exp = (hi >>> 20) & 0x7ff;
  if (exp === 0) {
    const [m, e] = frexp(x * 2 ** 64);
    return [m, e - 64];
  }
  buf.setUint32(0, (hi & 0x800fffff) | (1022 << 20));
  return [buf.getFloat64(0), exp - 1022];
}

/** m * 2^e without intermediate overflow for large |e|. */
export function ldexp(m: number, e: number): number {
  while (e > 1000) { m *= 2 ** 1000; e -= 1000; if (!Number.isFinite(m)) return m; }
  while (e < -1000) { m *= 2 ** -1000; e += 1000; if (m === 0) return m; }
  return m * 2 ** e;
}

export interface Fe { m: number; e: number }
export interface CFe { re: number; im: number; e: number }

export const FE_ZERO: Fe = { m: 0, e: 0 };
export const CFE_ZERO: CFe = { re: 0, im: 0, e: 0 };

export function fe(m: number, e = 0): Fe {
  if (m === 0) return FE_ZERO;
  const [mm, k] = frexp(m);
  return { m: f32(mm), e: e + k };
}

export function feFromLog2(x: number): Fe {
  const f = Math.floor(x);
  return fe(2 ** (x - f), f);
}

export function feLog2(a: Fe): number {
  return a.m === 0 ? -Infinity : Math.log2(Math.abs(a.m)) + a.e;
}

export function feToNumber(a: Fe): number {
  return ldexp(a.m, a.e);
}

export function feMul(a: Fe, b: Fe): Fe {
  return fe(f32(a.m * b.m), a.e + b.e);
}

export function feAdd(a: Fe, b: Fe): Fe {
  if (a.m === 0) return b;
  if (b.m === 0) return a;
  const d = a.e - b.e;
  if (d > 30) return a;
  if (d < -30) return b;
  return d >= 0 ? fe(f32(a.m + f32(ldexp(b.m, -d))), a.e) : fe(f32(f32(ldexp(a.m, d)) + b.m), b.e);
}

export function feLess(a: Fe, b: Fe): boolean {
  return feAdd(a, { m: -b.m, e: b.e }).m < 0;
}

export function cfe(re: number, im: number, e = 0): CFe {
  const m = Math.max(Math.abs(re), Math.abs(im));
  if (m === 0) return CFE_ZERO;
  const k = frexp(m)[1];
  return { re: f32(ldexp(re, -k)), im: f32(ldexp(im, -k)), e: e + k };
}

export function cfeMul(a: CFe, b: CFe): CFe {
  return cfe(f32(f32(a.re * b.re) - f32(a.im * b.im)), f32(f32(a.re * b.im) + f32(a.im * b.re)), a.e + b.e);
}

export function cfeAdd(a: CFe, b: CFe): CFe {
  if (a.re === 0 && a.im === 0) return b;
  if (b.re === 0 && b.im === 0) return a;
  const d = a.e - b.e;
  if (d > 30) return a;
  if (d < -30) return b;
  if (d >= 0) return cfe(f32(a.re + f32(ldexp(b.re, -d))), f32(a.im + f32(ldexp(b.im, -d))), a.e);
  return cfe(f32(f32(ldexp(a.re, d)) + b.re), f32(f32(ldexp(a.im, d)) + b.im), b.e);
}

export function cfeNorm2(a: CFe): Fe {
  return fe(f32(f32(a.re * a.re) + f32(a.im * a.im)), 2 * a.e);
}

export function cfeToComplex(a: CFe): [number, number] {
  return [ldexp(a.re, a.e), ldexp(a.im, a.e)];
}

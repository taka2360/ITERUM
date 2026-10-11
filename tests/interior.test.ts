import { describe, expect, it } from "vitest";
import { boundaryPolys, dyadicToFe, interiorTerms, G_TERMS, H_TERMS } from "../src/phenomena/escape/interior";

// Exact evaluation of a polynomial with BigInt: returns num / 2^k.
function evalExact(p: Map<number, [bigint, number]>, dx: [bigint, number], dy: [bigint, number]): [bigint, number] {
  let acc: [bigint, number] = [0n, 0];
  const pow = (v: [bigint, number], n: number): [bigint, number] => [v[0] ** BigInt(n), v[1] * n];
  for (const [key, c] of p) {
    const a = pow(dx, key >> 3), b = pow(dy, key & 7);
    const t: [bigint, number] = [c[0] * a[0] * b[0], c[1] + a[1] + b[1]];
    const k = Math.max(acc[1], t[1]);
    acc = [(acc[0] << BigInt(k - acc[1])) + (t[0] << BigInt(k - t[1])), k];
  }
  return acc;
}

// g and h evaluated directly at c = (cx, cy) / 2^k, exactly.
function direct(cx: bigint, cy: bigint, k: number): { g: [bigint, number]; h: [bigint, number] } {
  const K = BigInt(k);
  const one = 1n << K;
  // g = (x+1)² + y² − 1/16 at scale 2^(2k+4)
  const g: [bigint, number] = [16n * ((cx + one) ** 2n + cy * cy) - one * one, 2 * k + 4];
  // u = x − 1/4 = U / 2^(k+2), y = Y / 2^(k+2), q = Q / 2^(2k+4);
  // h = q² + q u − y²/4 = (Q² + Q U 2^(k+2) − Y² 2^(2k+2)) / 2^(4k+8)
  const U = 4n * cx - one;
  const Y = 4n * cy;
  const Q = U * U + Y * Y;
  const h: [bigint, number] = [Q * Q + Q * U * (1n << (K + 2n)) - Y * Y * (1n << (2n * K + 2n)), 4 * k + 8];
  return { g, h };
}

const same = (a: [bigint, number], b: [bigint, number]) => {
  const k = Math.max(a[1], b[1]);
  return (a[0] << BigInt(k - a[1])) === (b[0] << BigInt(k - b[1]));
};

describe("closed-form interior tests", () => {
  it("expanded polynomials equal g and h exactly", () => {
    const F = 70;
    const refs: [bigint, bigint][] = [
      [-(1n << 70n) * 5n / 4n + 123456789n, -(1n << 58n) * 3n],
      [-(1n << 67n) + 99n, (1n << 69n) + 7n],
      [0n, 0n],
      [(1n << 68n) + 5n, (1n << 67n) - 3n],
    ];
    const offsets: [bigint, bigint][] = [[0n, 0n], [1n << 40n, -(1n << 41n)], [-12345n, 678n], [3n << 66n, 5n << 65n]];
    for (const [re, im] of refs) {
      const { g, h } = boundaryPolys(re, im, F);
      for (const [ox, oy] of offsets) {
        const d = direct(re + ox, im + oy, F);
        expect(same(evalExact(g, [ox, F], [oy, F]), d.g)).toBe(true);
        expect(same(evalExact(h, [ox, F], [oy, F]), d.h)).toBe(true);
      }
    }
  });

  it("rounds coefficients to the nearest 24-bit mantissa", () => {
    for (const [num, k] of [[1n, 0], [3n, 2], [-(2n ** 80n) - 1n, 10], [(2n ** 24n) - 1n, 24], [(2n ** 25n) - 1n, 0], [12345678901234567890n, 100]] as [bigint, number][]) {
      const [m, e] = dyadicToFe([num, k]);
      expect(Math.abs(m) >= 0.5 && Math.abs(m) < 1).toBe(true);
      expect(Math.fround(m)).toBe(m);
      const exact = Number(num) / 2 ** k;
      expect(Math.abs(m * 2 ** e - exact)).toBeLessThanOrEqual(Math.abs(exact) * 2 ** -24 * (1 + 1e-12));
    }
    expect(dyadicToFe([0n, 5])).toEqual([0, 0]);
  });

  it("packs every term of degree ≤ 2 and ≤ 4", () => {
    const u = new Uint32Array(interiorTerms(0n, 0n, 0));
    const keys = (start: number, n: number) => Array.from({ length: n }, (_, t) => `${u[(start + t) * 4 + 2]},${u[(start + t) * 4 + 3]}`);
    expect(new Set(keys(0, G_TERMS)).size).toBe(G_TERMS);
    expect(new Set(keys(G_TERMS, H_TERMS)).size).toBe(H_TERMS);
    expect(keys(G_TERMS, H_TERMS).every((k) => k.split(",").map(Number).reduce((a, b) => a + b) <= 4)).toBe(true);
  });
});

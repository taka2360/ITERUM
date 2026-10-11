import { describe, expect, it } from "vitest";
import { cfe, cfeAdd, cfeMul, cfeNorm2, cfeToComplex, fe, feAdd, feFromLog2, feLess, feLog2, feMul, frexp, ldexp } from "../src/core/floatexp";

describe("frexp / ldexp", () => {
  it("normalizes into [0.5, 1)", () => {
    for (const x of [1, -3.5, 1e-300, 5e-324, 1e300, 0.75]) {
      const [m, e] = frexp(x);
      expect(Math.abs(m)).toBeGreaterThanOrEqual(0.5);
      expect(Math.abs(m)).toBeLessThan(1);
      expect(ldexp(m, e)).toBe(x);
    }
  });
});

describe("floatexp (CPU model of floatexp.wgsl)", () => {
  it("represents values far outside the f32 and f64 ranges", () => {
    const a = fe(1, -5000);
    expect(feLog2(feMul(a, a))).toBe(-10000);
    expect(feLog2(feFromLog2(-1234.25))).toBeCloseTo(-1234.25, 5);
  });

  it("aligns exponents at the cut-off boundary", () => {
    // Differences up to 30 bits are added; beyond, the smaller operand vanishes (it is below f32 precision anyway).
    const one = fe(1);
    expect(feAdd(one, fe(1, -23)).m * 2).toBeCloseTo(1 + 2 ** -23, 7);
    expect(feAdd(one, fe(1, -31))).toEqual(one);
    expect(feAdd(fe(1, -31), one)).toEqual(one);
    expect(feAdd(one, fe(-1))).toEqual({ m: 0, e: 0 });
  });

  it("normalizes right after an exponent carry", () => {
    // 0.75 + 0.75 = 1.5 → mantissa 0.75, exponent +1.
    const r = feAdd(fe(0.75), fe(0.75));
    expect(r).toEqual({ m: 0.75, e: 1 });
  });

  it("compares magnitudes across exponents", () => {
    expect(feLess(fe(1, -10), fe(1, -9))).toBe(true);
    expect(feLess(fe(0.9, -100), fe(0.6, -99))).toBe(true);
    expect(feLess(fe(1, 50), fe(1, -50))).toBe(false);
  });

  it("complex multiply/add agree with doubles in range", () => {
    const a = cfe(0.3, -0.4);
    const b = cfe(-1.2, 0.7);
    const [re, im] = cfeToComplex(cfeMul(a, b));
    expect(re).toBeCloseTo(0.3 * -1.2 - -0.4 * 0.7, 6);
    expect(im).toBeCloseTo(0.3 * 0.7 + -0.4 * -1.2, 6);
    const [sr, si] = cfeToComplex(cfeAdd(a, b));
    expect(sr).toBeCloseTo(-0.9, 6);
    expect(si).toBeCloseTo(0.3, 6);
  });

  it("keeps relative precision for tiny complex values (rebase comparisons)", () => {
    // |Z + δ| < |δ| must be decidable even when both are ~2^-700.
    const z = cfe(1, 0.5, -700);
    const d = cfe(-0.9, -0.45, -700);
    const s = cfeAdd(z, d);
    expect(feLess(cfeNorm2(s), cfeNorm2(d))).toBe(true);
    expect(feLog2(cfeNorm2(s))).toBeCloseTo(2 * (-700 + Math.log2(Math.hypot(0.1, 0.05))), 4);
  });
});

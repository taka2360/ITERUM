import { describe, expect, it } from "vitest";
import { DeepCamera, bigScaled, bigToDecimal, bitLength, decimalToBig, scaledToBig, Camera2D } from "../src/core/camera2d";

describe("BigInt helpers", () => {
  it("round-trips numbers through scaled BigInts", () => {
    for (const x of [1, -1.5, 0.3, 1e-10, -123.456]) {
      for (const e of [64, 300, 2000]) {
        expect(bigScaled(scaledToBig(x, e), -e)).toBeCloseTo(x, 12);
      }
    }
  });

  it("handles fractional exponents", () => {
    const b = scaledToBig(3, 60.5);
    expect(Number(b) / 2 ** 60.5).toBeCloseTo(3, 12);
    expect(scaledToBig(-1.5, 0)).toBe(-2n); // rounds half away from zero
  });

  it("parses and prints decimals", () => {
    const b = decimalToBig("-0.743643887037158704752191506114774", 200);
    expect(bigToDecimal(b, 200, 33)).toBe("-0.743643887037158704752191506114773");
    expect(bitLength(255n)).toBe(8);
    expect(bitLength(-256n)).toBe(9);
  });
});

describe("DeepCamera", () => {
  it("keeps the point under the cursor fixed while zooming, at every depth", () => {
    const cam = new DeepCamera("-0.75", "0.1", Math.log2(4 / 1000));
    const [ox, oy] = [123.5, -77.25];
    for (let i = 0; i < 400; i++) {
      // Each step is checked against the point under the cursor right before it;
      // an anchor chosen at a coarse scale would carry that scale's rounding.
      const before = cam.pointAt(ox, oy);
      cam.zoomAt(ox, oy, 2.5);
      const [dx, dy] = cam.offsetPx(before);
      expect(dx).toBeCloseTo(ox, 6);
      expect(-dy).toBeCloseTo(oy, 6); // screen y is down, plane y is up
    }
    expect(cam.log2Scale).toBeCloseTo(Math.log2(4 / 1000) - 1000, 6);
  });

  it("stays exact at depths far beyond double precision", () => {
    const cam = new DeepCamera("-1.75", "0", -1200);
    const p = cam.pointAt(3, -4);
    cam.pan(10, 20);
    const [dx, dy] = cam.offsetPx(p);
    expect(dx).toBeCloseTo(13, 6);
    expect(dy).toBeCloseTo(-16, 6); // plane y is up: +4 before, then the view moved up 20
  });

  it("pan moves content with the drag", () => {
    const cam = new DeepCamera("0", "0", 0);
    cam.pan(5, 7);
    const [x, y] = cam.centerNumber();
    expect(x).toBe(-5);
    expect(y).toBe(7);
  });
});

describe("Camera2D", () => {
  it("zooms about the anchor", () => {
    const c = new Camera2D(0.3, -0.2, -8);
    const s0 = 2 ** c.log2Scale;
    const px = 0.3 + 50 * s0;
    c.zoomAt(50, 0, 3);
    expect(c.cx + 50 * 2 ** c.log2Scale).toBeCloseTo(px, 12);
  });
});

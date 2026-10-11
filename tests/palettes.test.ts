import { describe, expect, it } from "vitest";
import { PALETTES, hexToRgb, oklabToSrgb, paletteLut, samplePalette, srgbToOklab, LUT_SIZE } from "../src/core/palettes";

describe("palettes", () => {
  it("OKLab round-trips", () => {
    for (const hex of ["#03050b", "#9cc0d8", "#f2ead8", "#c8742f"]) {
      const c = hexToRgb(hex);
      const back = oklabToSrgb(srgbToOklab(c));
      for (let i = 0; i < 3; i++) expect(back[i]).toBeCloseTo(c[i], 5);
    }
  });

  it("are cyclic and smooth", () => {
    for (const p of PALETTES) {
      const a = samplePalette(p, 0);
      const b = samplePalette(p, 0.99999);
      for (let i = 0; i < 3; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(0.01);
      // No jumps between neighbouring LUT entries.
      const lut = paletteLut(p);
      let maxStep = 0;
      for (let k = 0; k < LUT_SIZE; k++) {
        const j = ((k + 1) % LUT_SIZE) * 4;
        for (let c = 0; c < 3; c++) maxStep = Math.max(maxStep, Math.abs(lut[k * 4 + c] - lut[j + c]));
      }
      expect(maxStep).toBeLessThanOrEqual(6);
    }
  });
});

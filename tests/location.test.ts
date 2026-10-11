import { describe, expect, it } from "vitest";
import {
  formatZoom, locationText, log2ScaleFromZoom, log2ZoomFromScale, normalizeDecimal, numberToDecimal, parseLocationText, parseZoom,
} from "../src/ui/location";

describe("zoom text", () => {
  it("parses the accepted notations", () => {
    expect(parseZoom("1")).toBeCloseTo(0, 12);
    expect(parseZoom("1e100")).toBeCloseTo(100 * Math.log2(10), 9);
    expect(parseZoom("1.5e300")).toBeCloseTo(Math.log2(1.5) + 300 * Math.log2(10), 9);
    expect(parseZoom("1.5 × 10^300")).toBeCloseTo(parseZoom("1.5e300"), 9);
    expect(parseZoom("10¹⁰⁰")).toBeCloseTo(parseZoom("1e100"), 9);
    expect(parseZoom("2^1000")).toBeCloseTo(1000, 12);
    expect(parseZoom("1e5000")).toBeCloseTo(5000 * Math.log2(10), 6); // beyond f64
    for (const bad of ["", "abc", "0", "-3", "1e"]) expect(() => parseZoom(bad)).toThrow();
  });

  it("round-trips through formatZoom", () => {
    for (const l of [0, 1.5, 10, 40, 333.3, 10000]) {
      expect(parseZoom(formatZoom(l))).toBeCloseTo(l, 2);
    }
    expect(formatZoom(parseZoom("1e100"))).toBe("1.000e100");
    expect(formatZoom(parseZoom("9.9999e99"))).toBe("1.000e100");
    expect(formatZoom(Math.log2(12.5))).toBe("12.5");
  });

  it("converts to and from the camera scale", () => {
    const s = Math.log2(4 / 1000);
    expect(log2ZoomFromScale(s, 1600, 1000)).toBeCloseTo(0, 12);
    expect(log2ScaleFromZoom(log2ZoomFromScale(-1234.5, 800, 600), 800, 600)).toBeCloseTo(-1234.5, 9);
  });
});

describe("decimal text", () => {
  it("normalises scientific notation to plain decimals", () => {
    expect(normalizeDecimal("-0.75")).toBe("-0.75");
    expect(normalizeDecimal(" +1.25e-3 ")).toBe("0.00125");
    expect(normalizeDecimal("−1.5E2")).toBe("-150");
    expect(normalizeDecimal(".5")).toBe("0.5");
    expect(normalizeDecimal("007.10")).toBe("7.10");
    for (const bad of ["", "-", "1.2.3", "abc", "1e"]) expect(() => normalizeDecimal(bad)).toThrow();
  });

  it("prints doubles with digits matching the pixel size", () => {
    expect(numberToDecimal(-0.6, Math.log2(0.003))).toBe("-0.6000");
    expect(numberToDecimal(1e-12, -60)).not.toContain("e");
  });
});

describe("pasted locations", () => {
  it("reads its own copy format back", () => {
    const text = locationText({ re: "-0.743643887037158704752", im: "0.131825904205311970493", log2Zoom: 100 });
    const p = parseLocationText(text);
    expect(p.re).toBe("-0.743643887037158704752");
    expect(p.im).toBe("0.131825904205311970493");
    expect(parseZoom(p.zoom!)).toBeCloseTo(100, 2);
  });

  it("reads bare values and other key spellings", () => {
    expect(parseLocationText("-0.75, 0.1, 1e10")).toEqual({ re: "-0.75", im: "0.1", zoom: "1e10" });
    expect(parseLocationText("-0.75 0.1")).toEqual({ re: "-0.75", im: "0.1", zoom: undefined });
    expect(parseLocationText("実部=-0.5 虚部=0.25 拡大率=100")).toEqual({ re: "-0.5", im: "0.25", zoom: "100" });
    expect(parseLocationText("hello")).toEqual({});
  });
});

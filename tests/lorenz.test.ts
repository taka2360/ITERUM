import { describe, expect, it } from "vitest";
import { integrate } from "../src/phenomena/lorenz/reference";

describe("Lorenz RK4 reference", () => {
  it("is fourth-order accurate over a short horizon", () => {
    const p0: [number, number, number] = [1, 1, 20];
    const exact = integrate(p0, 0.0001, 5000); // t = 0.5
    const coarse = integrate(p0, 0.004, 125);
    const fine = integrate(p0, 0.002, 250);
    const err = (a: number[]) => Math.hypot(a[0] - exact[0], a[1] - exact[1], a[2] - exact[2]);
    const ratio = err(coarse) / err(fine);
    expect(ratio).toBeGreaterThan(12); // 2^4 = 16 for a 4th-order method
    expect(ratio).toBeLessThan(20);
  });

  it("stays on the bounded attractor", () => {
    let p: [number, number, number] = [0.1, 0, 0];
    for (let k = 0; k < 50; k++) {
      p = integrate(p, 0.004, 200);
      expect(Math.abs(p[0])).toBeLessThan(25);
      expect(Math.abs(p[1])).toBeLessThan(30);
      expect(p[2]).toBeGreaterThan(-1);
      expect(p[2]).toBeLessThan(55);
    }
  });
});

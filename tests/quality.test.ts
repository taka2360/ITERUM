import { describe, expect, it } from "vitest";
import { QualityGovernor } from "../src/gpu/quality";

function feed(g: QualityGovernor, t0: number, seconds: number, s: { rafMs: number; gpuMs?: number; cpuMs: number }): number {
  let t = t0;
  let changes = 0;
  for (; t < t0 + seconds * 1000; t += s.rafMs) if (g.sample(t, s)) changes++;
  return changes;
}

describe("QualityGovernor", () => {
  it("steps down only after sustained GPU load", () => {
    const g = new QualityGovernor();
    // A single heavy window is ignored.
    feed(g, 0, 0.5, { rafMs: 30, gpuMs: 20, cpuMs: 1 });
    feed(g, 500, 0.5, { rafMs: 16, gpuMs: 8, cpuMs: 1 });
    expect(g.scale).toBe(1);
    // Two consecutive heavy windows lower the scale one step.
    feed(g, 1000, 1.05, { rafMs: 30, gpuMs: 20, cpuMs: 1 });
    expect(g.scale).toBe(0.875);
  });

  it("does not react to a slow frame loop when the GPU is idle", () => {
    const g = new QualityGovernor();
    feed(g, 0, 5, { rafMs: 40, gpuMs: 2, cpuMs: 1 });
    expect(g.scale).toBe(1);
  });

  it("recovers slowly and never oscillates under steady load", () => {
    const g = new QualityGovernor();
    feed(g, 0, 3, { rafMs: 30, gpuMs: 20, cpuMs: 1 });
    const low = g.scale;
    expect(low).toBeLessThan(1);
    // Middle band (after one window that still holds heavy samples): no change.
    feed(g, 3000, 1, { rafMs: 16.7, gpuMs: 9, cpuMs: 1 });
    const settled = g.scale;
    expect(feed(g, 4000, 5, { rafMs: 16.7, gpuMs: 9, cpuMs: 1 })).toBe(0);
    expect(g.scale).toBe(settled);
    // Light load: back up, but no more than one step per 2 s.
    const n = feed(g, 9000, 4.1, { rafMs: 16.7, gpuMs: 3, cpuMs: 1 });
    expect(n).toBeLessThanOrEqual(2);
    expect(g.scale).toBeGreaterThan(settled);
  });

  it("manual modes pin the scale", () => {
    const g = new QualityGovernor();
    g.setMode("low");
    feed(g, 0, 3, { rafMs: 10, gpuMs: 1, cpuMs: 1 });
    expect(g.scale).toBe(0.5);
  });
});

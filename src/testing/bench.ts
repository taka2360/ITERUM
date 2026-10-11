// ITERUM_MODE=bench: fixed scenarios at a fixed internal resolution.
//
// Every scenario: 2 s warm-up, 10 s measurement, repeated 3 times. Frame
// interval (rAF) and GPU time (timestamp queries) are recorded separately.
// Static deep views additionally report time to converge, and the deep view
// is analysed for unresolved pixels per iteration limit and BLA on/off error.

import type { App } from "../main";
import type { Mandelbrot } from "../phenomena/mandelbrot";
import type { Lorenz } from "../phenomena/lorenz";
import { VOYAGES } from "../phenomena/mandelbrot/voyages";
import { DeepCamera } from "../core/camera2d";
import { feFromLog2 } from "../core/floatexp";
import { MODE_DEEP, type EscapeView } from "../phenomena/escape/engine";
import { OrbitManager } from "../phenomena/mandelbrot/orbit";
import { diag, envVar, exitApp, saveFile } from "../platform";
import { outDir, seconds } from "./harness";
import { renderView } from "./gputest";

const RES: [number, number] = [2560, 1440];
const WARMUP_S = 2;
const MEASURE_S = 10;
const RUNS = 3;

interface Scenario {
  name: string;
  phenomenon: number;
  setup: (app: App) => void;
  /** Per-frame motion (seconds since start). */
  motion?: (app: App, t: number, dt: number) => void;
  /** Static scenes: also measure time to converge. */
  converge?: boolean;
}

const mandel = (app: App) => app.phenomena[0] as Mandelbrot;
const at = (re: string, im: string, widthLog2: number) => (app: App) => mandel(app).setView(re, im, widthLog2 - Math.log2(app.canvas.width));
const voyage = (name: string, extraOctaves = 0) => {
  const v = VOYAGES.find((t) => t.name === name)!;
  return { re: v.re, im: v.im, widthLog2: -v.depthLog2 + extraOctaves };
};
/** Zoom in and out around the start depth (±2 octaves, 0.5 octave/s). */
const breathe = (app: App, t: number, dt: number) => mandel(app).cameraState.zoomAt(0, 0, Math.sin(t * 0.25) * 0.5 * dt * 2);
const SEAHORSE = { re: "-0.743643887037158704752191506114774", im: "0.131825904205311970493132056385139" };

function lorenzWith(particles: number) {
  return (app: App) => {
    const l = app.phenomena[3] as Lorenz;
    l.settings = { ...l.settings, particles, stepsPerFrame: 3 };
  };
}

const SCENARIOS: Scenario[] = [
  { name: "mandelbrot-overview-static", phenomenon: 0, setup: (app) => app.phenomena[0].reset(), converge: true },
  { name: "mandelbrot-direct-zooming", phenomenon: 0, setup: at(SEAHORSE.re, SEAHORSE.im, Math.log2(0.01)), motion: breathe },
  { name: "mandelbrot-perturbation-1e15-static", phenomenon: 0, setup: at(SEAHORSE.re, SEAHORSE.im, Math.log2(1e-15)), converge: true },
  { name: "mandelbrot-perturbation-1e15-zooming", phenomenon: 0, setup: at(SEAHORSE.re, SEAHORSE.im, Math.log2(1e-15)), motion: breathe },
  { name: "mandelbrot-floatexp-1e100-static", phenomenon: 0, setup: (a) => { const v = voyage("Elephant Valley", 3); at(v.re, v.im, v.widthLog2)(a); }, converge: true },
  { name: "mandelbrot-floatexp-1e100-zooming", phenomenon: 0, setup: (a) => { const v = voyage("Elephant Valley", 3); at(v.re, v.im, v.widthLog2)(a); }, motion: breathe },
  { name: "mandelbrot-floatexp-1e300-static", phenomenon: 0, setup: (a) => { const v = voyage("Seahorse Abyss", 3); at(v.re, v.im, v.widthLog2)(a); }, converge: true },
  { name: "julia-animated", phenomenon: 1, setup: (app) => app.phenomena[1].reset() },
  { name: "newton-animated", phenomenon: 2, setup: (app) => app.phenomena[2].reset() },
  { name: "lorenz-262k", phenomenon: 3, setup: lorenzWith(1 << 18) },
  { name: "lorenz-524k", phenomenon: 3, setup: lorenzWith(1 << 19) },
  { name: "lorenz-1M", phenomenon: 3, setup: lorenzWith(1 << 20) },
];

const pct = (xs: number[], p: number) => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
};
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
const sd = (xs: number[]) => Math.sqrt(mean(xs.map((x) => (x - mean(xs)) ** 2)));

interface RunStats {
  rafP50: number;
  rafP95: number;
  gpuP50: number;
  gpuP95: number;
  passes: Record<string, number>;
  convergeS?: number;
  frames: number;
}

async function runOnce(app: App, sc: Scenario): Promise<RunStats> {
  app.select(sc.phenomenon);
  await seconds(1.8); // transition
  sc.setup(app);
  const t0 = performance.now();
  let convergeS: number | undefined;
  const raf: number[] = [];
  const gpu: number[] = [];
  const passes = new Map<string, number[]>();
  let lastMeasured = app.timer.frames;
  let measuring = false;
  let last = t0;
  const prev = app.onFrame;
  app.onFrame = (info) => {
    const now = performance.now();
    const t = (now - t0) / 1000;
    sc.motion?.(app, t, (now - last) / 1000);
    last = now;
    if (convergeS === undefined && sc.converge && app.lastStatus === "converged") convergeS = t;
    if (!measuring) return;
    raf.push(info.rafMs);
    // Only count fresh GPU measurements (they arrive a few frames late).
    if (app.timer.frames !== lastMeasured) {
      lastMeasured = app.timer.frames;
      gpu.push(app.timer.frameMs);
      for (const [k, v] of app.timer.passMs) {
        if (!passes.has(k)) passes.set(k, []);
        passes.get(k)!.push(v);
      }
    }
  };
  await seconds(WARMUP_S);
  measuring = true;
  await seconds(MEASURE_S);
  measuring = false;
  // Static scenes: allow extra time to report convergence.
  if (sc.converge && convergeS === undefined) {
    const deadline = performance.now() + 60000;
    while (convergeS === undefined && performance.now() < deadline) await seconds(0.1);
  }
  app.onFrame = prev;
  const pass: Record<string, number> = {};
  for (const [k, v] of passes) pass[k] = pct(v, 0.5);
  return { rafP50: pct(raf, 0.5), rafP95: pct(raf, 0.95), gpuP50: pct(gpu, 0.5), gpuP95: pct(gpu, 0.95), passes: pass, convergeS, frames: raf.length };
}

async function deepAnalysis(app: App): Promise<Record<string, unknown>> {
  // Fixed location and zoom: the Elephant Valley minibrot framed 8× wider.
  const v = voyage("Elephant Valley", 3);
  const w = 640;
  const h = 360;
  const log2s = v.widthLog2 - Math.log2(w);
  const cam = new DeepCamera(v.re, v.im, log2s);
  const orbits = new OrbitManager(app.gpu.device);
  const t0 = performance.now();
  orbits.request({ re: cam.re, im: cam.im, frac: cam.frac }, Math.ceil(-log2s) + 64 + 20, 1 << 18, log2s + Math.log2(w * 2));
  while (!orbits.current) await seconds(0.02);
  const o = orbits.current;
  const orbitMs = performance.now() - t0;
  const view: EscapeView = { mode: MODE_DEEP, spacing: feFromLog2(log2s), center: [0, 0], cmr: [0, 0], juliaC: [0, 0], orbit: o };

  const limits: Record<string, unknown>[] = [];
  for (const maxIter of [1000, 4000, 16000, 64000, 256000]) {
    const t = performance.now();
    const r = await renderView(app, view, maxIter, w, h, true);
    const s = r.engine.stats();
    limits.push({ maxIter, unresolvedPx: s.interior, escapedPx: s.escaped, timeS: +((performance.now() - t) / 1000).toFixed(2) });
    r.engine.dispose();
  }

  // BLA on vs off (same limit, smaller image: without BLA it is slow).
  const bw = 320;
  const bh = 180;
  const blaLimit = 64000;
  const bview = { ...view, spacing: feFromLog2(log2s + 1) };
  const t1 = performance.now();
  const on = await renderView(app, bview, blaLimit, bw, bh, true);
  const onS = (performance.now() - t1) / 1000;
  const t2 = performance.now();
  const off = await renderView(app, { ...bview, orbit: { ...o, levels: 0 } }, blaLimit, bw, bh, true);
  const offS = (performance.now() - t2) / 1000;
  let same = 0;
  let maxDiff = 0;
  let wc = 0;
  let wcSame = 0;
  let statusDiff = 0;
  const diffs: number[] = [];
  for (let p = 0; p < bw * bh; p++) {
    const i = p * 12;
    if (on.u[i + 9] !== off.u[i + 9]) {
      statusDiff++;
      continue;
    }
    // Well-conditioned pixels (≥ 1 px from the set) must not depend on BLA.
    if (off.u[i + 9] === 1 && off.f[i + 1] >= 1) {
      wc++;
      if (Math.abs(on.f[i] - off.f[i]) < 0.02) wcSame++;
    }
    if (on.u[i + 9] !== 1) {
      same++;
      continue;
    }
    const d = Math.abs(on.f[i] - off.f[i]);
    diffs.push(d);
    maxDiff = Math.max(maxDiff, d);
    if (d < 0.02) same++;
  }
  const onStats = on.engine.stats();
  on.engine.dispose();
  off.engine.dispose();
  return {
    location: "Elephant Valley minibrot ×8 (≈1e-98)",
    orbit: { length: o.len, bits: o.bits, periodic: o.periodic, blaLevels: o.levels, roundTripMs: +orbitMs.toFixed(1) },
    unresolvedByIterationLimit: limits,
    bla: {
      image: `${bw}×${bh}, maxIter ${blaLimit}`,
      withBlaS: +onS.toFixed(2),
      withoutBlaS: +offS.toFixed(2),
      pixelsAgreeing: `${((same / (bw * bh)) * 100).toFixed(2)}%`,
      wellConditionedAgreeing: `${wcSame}/${wc}`,
      escapeStatusMismatches: statusDiff,
      medianSmoothDiff: pct(diffs, 0.5),
      maxSmoothDiff: maxDiff,
      blaSkippedIterations: onStats.blaSteps,
      rebases: onStats.rebases,
      anomalies: onStats.anomalies,
    },
  };
}

export async function runBench(app: App): Promise<void> {
  await seconds(2);
  app.forcedResolution = RES;
  const info = app.gpu.adapterInfo;
  const report: Record<string, unknown> = {
    date: new Date().toISOString(),
    adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description, timestamps: app.gpu.hasTimestamps },
    resolution: RES,
    protocol: { warmupS: WARMUP_S, measureS: MEASURE_S, runs: RUNS },
    scenarios: {} as Record<string, unknown>,
  };
  const lines: string[] = [];
  const only = await envVar("ITERUM_BENCH");
  for (const sc of only === "deep" ? [] : SCENARIOS) {
    const runs: RunStats[] = [];
    for (let r = 0; r < RUNS; r++) runs.push(await runOnce(app, sc));
    const agg = (k: keyof RunStats) => {
      const xs = runs.map((x) => x[k] as number).filter((x) => x !== undefined && !Number.isNaN(x));
      return { mean: +mean(xs).toFixed(2), sd: +sd(xs).toFixed(2) };
    };
    const entry = {
      rafP50: agg("rafP50"), rafP95: agg("rafP95"), gpuP50: agg("gpuP50"), gpuP95: agg("gpuP95"),
      convergeS: sc.converge ? agg("convergeS") : undefined,
      passesP50: runs[runs.length - 1].passes,
      frames: runs.map((x) => x.frames),
    };
    (report.scenarios as Record<string, unknown>)[sc.name] = entry;
    const line = `${sc.name}: rAF p50 ${entry.rafP50.mean}±${entry.rafP50.sd} ms, p95 ${entry.rafP95.mean} ms | GPU p50 ${entry.gpuP50.mean}±${entry.gpuP50.sd} ms, p95 ${entry.gpuP95.mean} ms${entry.convergeS ? ` | converge ${entry.convergeS.mean}±${entry.convergeS.sd} s` : ""}`;
    lines.push(line);
    diag(`BENCH ${line}`);
  }
  app.forcedResolution = undefined;
  app.select(0);
  const deep = await deepAnalysis(app);
  report.deepAnalysis = deep;
  diag(`BENCH deep ${JSON.stringify(deep)}`);
  const dir = await outDir();
  await saveFile(`${dir}/${only === "deep" ? "bench-deep" : "bench"}.json`, new TextEncoder().encode(JSON.stringify(report, null, 2)));
  diag("bench done");
  await exitApp(0);
}

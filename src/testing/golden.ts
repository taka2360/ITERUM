// ITERUM_MODE=golden: renders fixed escape-time views offscreen to completion
// and saves the raw samples, so two builds can be compared bit for bit
// (scripts/golden-diff.mjs). Also records how long each view took, which
// measures pure computation throughput independent of the frame loop.
//
// Every view starts at the app's initial iteration limit and lets the engine
// raise it as it would interactively, so the limit-raising path is covered.

import type { App } from "../main";
import { DeepCamera } from "../core/camera2d";
import { feFromLog2 } from "../core/floatexp";
import { GpuTimer } from "../gpu/timing";
import { EscapeEngine, MODE_DEEP, MODE_DIRECT, MODE_JULIA, MODE_PERTURB, MODE_NAMES, type EscapeView } from "../phenomena/escape/engine";
import { OrbitManager } from "../phenomena/mandelbrot/orbit";
import { VOYAGES } from "../phenomena/mandelbrot/voyages";
import { diag, envVar, exitApp, saveFile } from "../platform";
import { outDir, seconds } from "./harness";

// Comparison resolution (samples are saved) and timing resolution (3 runs, median).
const W = 640;
const H = 360;
const TIMING: [number, number] = [2560, 1440];
const TIMING_RUNS = 3;
const ORBIT_ITER = 1 << 20;

interface GoldenView {
  name: string;
  /** Direct views: centre and width; deep views: decimal centre and log2 width. */
  direct?: { mode: typeof MODE_DIRECT | typeof MODE_JULIA; center: [number, number]; width: number; juliaC?: [number, number] };
  deep?: { re: string; im: string; widthLog2: number };
}

const SEAHORSE = { re: "-0.743643887037158704752191506114774", im: "0.131825904205311970493132056385139" };
const voyage = (name: string, extra: number) => {
  const v = VOYAGES.find((t) => t.name === name)!;
  return { re: v.re, im: v.im, widthLog2: -v.depthLog2 + extra };
};

export const GOLDEN_VIEWS: GoldenView[] = [
  { name: "overview", direct: { mode: MODE_DIRECT, center: [-0.6, 0], width: 3.6 } },
  { name: "seahorse-1e-2", direct: { mode: MODE_DIRECT, center: [-0.7436438870371587, 0.1318259042053120], width: 0.01 } },
  { name: "julia", direct: { mode: MODE_JULIA, center: [0, 0], width: 3.4, juliaC: [-0.8, 0.156] } },
  { name: "perturb-1e-15", deep: { ...SEAHORSE, widthLog2: Math.log2(1e-15) } },
  { name: "floatexp-1e-98", deep: voyage("Elephant Valley", 3) },
  { name: "floatexp-1e-300", deep: voyage("Seahorse Abyss", 3) },
];

export interface GoldenResult {
  name: string;
  mode: string;
  wallS: number;
  /** Sum of GPU time of the escape passes (timestamp queries; 0 if unavailable). */
  gpuEscapeMs: number;
  frames: number;
  maxIter: number;
  restarts: number;
  iterations: number;
  escaped: number;
  interior: number;
  cycling: number;
  anomalies: number;
  orbitMs?: number;
}

async function orbitFor(orbits: OrbitManager, cam: DeepCamera, log2s: number, w: number, h: number): Promise<number> {
  orbits.reset();
  const t0 = performance.now();
  orbits.request({ re: cam.re, im: cam.im, frac: cam.frac }, Math.ceil(-log2s) + 64 + 32, ORBIT_ITER, log2s + Math.log2(Math.hypot(w, h) * 2));
  while (!orbits.current && performance.now() - t0 < 300000) await seconds(0.02);
  if (!orbits.current) throw new Error(`orbit failed: ${orbits.lastError ?? "timeout"}`);
  return performance.now() - t0;
}

/** Renders one view to convergence; returns the engine (caller disposes) and timings. */
export async function renderGolden(app: App, gv: GoldenView, orbits: OrbitManager, w = W, h = H, configure?: (e: EscapeEngine) => void): Promise<{ engine: EscapeEngine; result: GoldenResult }> {
  let view: EscapeView;
  let orbitMs: number | undefined;
  if (gv.direct) {
    const d = gv.direct;
    view = { mode: d.mode, spacing: feFromLog2(Math.log2(d.width / w)), center: d.center, cmr: [0, 0], juliaC: d.juliaC ?? [0, 0] };
  } else {
    const d = gv.deep!;
    const log2s = d.widthLog2 - Math.log2(w);
    const cam = new DeepCamera(d.re, d.im, log2s);
    orbitMs = await orbitFor(orbits, cam, log2s, w, h);
    const o = orbits.current!;
    const mode = log2s > -100 && o.minLog2Abs > -100 && o.dcMaxLog2 > -120 ? MODE_PERTURB : MODE_DEEP;
    view = { mode, spacing: feFromLog2(log2s), center: [0, 0], cmr: [0, 0], juliaC: [0, 0], orbit: o };
  }
  const engine = new EscapeEngine(() => app.palette);
  engine.init(app.gpu);
  engine.minIter = engine.maxIter = 800;
  if (view.orbit && !orbits.current!.periodic && !orbits.current!.escaped) engine.maxIterCap = ORBIT_ITER;
  configure?.(engine);
  const device = app.gpu.device;
  const timer = new GpuTimer(device, app.gpu.hasTimestamps);
  let gpuMs = 0;
  let measured = timer.frames;
  const t0 = performance.now();
  let frames = 0;
  for (; frames < 200000 && engine.status !== "converged"; frames++) {
    const encoder = device.createCommandEncoder();
    timer.beginFrame();
    engine.setView(view, null);
    engine.render({ gpu: app.gpu, encoder, timer, time: 0, dt: 1 / 60, paused: false, palette: app.palette.palette, width: w, height: h, pixelRatio: 1 });
    const after = timer.endFrame(encoder);
    device.queue.submit([encoder.finish()]);
    after?.();
    await device.queue.onSubmittedWorkDone();
    await new Promise((r) => setTimeout(r, 0));
    if (timer.frames !== measured) {
      measured = timer.frames;
      gpuMs += timer.passMs.get("escape") ?? 0;
    }
  }
  const wallS = (performance.now() - t0) / 1000;
  const s = engine.lastComplete ?? engine.stats();
  return {
    engine,
    result: {
      name: gv.name,
      mode: MODE_NAMES[view.mode],
      wallS: +wallS.toFixed(3),
      gpuEscapeMs: +gpuMs.toFixed(1),
      frames,
      maxIter: engine.maxIter,
      restarts: engine.restarts,
      iterations: engine.iterationsTotal,
      escaped: s.escaped,
      interior: s.interior,
      cycling: s.cycling,
      anomalies: s.anomalies,
      orbitMs: orbitMs !== undefined ? +orbitMs.toFixed(1) : undefined,
    },
  };
}

export async function runGolden(app: App): Promise<void> {
  await seconds(1);
  const dir = `${await outDir()}`;
  const only = await envVar("ITERUM_ONLY");
  const orbits = new OrbitManager(app.gpu.device);
  const results: (GoldenResult & { timing?: GoldenResult[] })[] = [];
  const timingOff = (await envVar("ITERUM_TIMING")) === "0";
  for (const gv of GOLDEN_VIEWS.filter((v) => !only || only.split(",").includes(v.name))) {
    const { engine, result } = await renderGolden(app, gv, orbits);
    const samples = await engine.readSamples();
    await saveFile(`${dir}/${gv.name}.bin`, new Uint8Array(samples.buffer));
    engine.dispose();
    diag(`GOLDEN ${JSON.stringify(result)}`);
    const timing: GoldenResult[] = [];
    for (let r = 0; r < (timingOff ? 0 : TIMING_RUNS); r++) {
      const t = await renderGolden(app, gv, orbits, TIMING[0], TIMING[1]);
      t.engine.dispose();
      timing.push(t.result);
      diag(`TIMING ${JSON.stringify(t.result)}`);
    }
    results.push({ ...result, timing });
  }
  const info = app.gpu.adapterInfo;
  const report = {
    date: new Date().toISOString(),
    commit: (await envVar("ITERUM_COMMIT")) ?? "unknown",
    adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description, timestamps: app.gpu.hasTimestamps },
    resolution: [W, H],
    timingResolution: TIMING,
    views: GOLDEN_VIEWS,
    results,
  };
  await saveFile(`${dir}/golden.json`, new TextEncoder().encode(JSON.stringify(report, null, 2)));
  diag("golden done");
  await exitApp(0);
}

// ITERUM_MODE=golden: renders fixed escape-time views offscreen to completion
// and saves the raw samples, so two builds can be compared bit for bit
// (scripts/golden-diff.mjs). Also records how long each view took, which
// measures pure computation throughput independent of the frame loop.
//
// Every view starts at the app's initial iteration limit and lets the engine
// raise it as it would interactively, so the limit-raising path is covered.
// "grow" views also grow the reference orbit as the app does (short orbit
// first, extended whenever the limit outgrows it, limit cap 2^24), so they
// measure the orbit pipeline too.
//
// Each view is rendered once at the comparison resolution (samples and the
// per-pixel status and iteration count are saved), then once at the timing
// resolution as warm-up, then ITERUM_RUNS times (default 5) for timing.
// Experiments: ITERUM_CAP caps the iteration limit of every view,
// ITERUM_DUMP_LIMIT=L instead renders each (perturbation) view once at the
// timing resolution, one sample without jitter, with the limit fixed at L and
// no interior proofs, and writes <view>-dump-{packet.bin, view.txt, state.bin,
// webgpu.txt} for the CUDA benchmark (src-tauri/examples/cuda).
// ITERUM_PROVEN=0 disables the proven-interior tests (ITERUM_PATCHES=0 only
// the component proofs), ITERUM_TRAP sets the fewest shading steps of proven
// pixels, ITERUM_SHARE the share of a frame the iteration may take.

import type { App } from "../main";
import { DeepCamera } from "../core/camera2d";
import { feFromLog2 } from "../core/floatexp";
import { GpuTimer } from "../gpu/timing";
import { EscapeEngine, MODE_DEEP, MODE_DIRECT, MODE_JULIA, MODE_PERTURB, MODE_NAMES, type EscapeView } from "../phenomena/escape/engine";
import { OrbitManager, orbitReference, requestPatches, type OrbitInfo } from "../phenomena/mandelbrot/orbit";
import { VOYAGES } from "../phenomena/mandelbrot/voyages";
import { diag, envVar, exitApp, saveFile } from "../platform";
import { outDir, seconds } from "./harness";

// Comparison resolution (samples are saved) and timing resolution (median of the runs).
const W = 640;
const H = 360;
const TIMING: [number, number] = [2560, 1440];
const TIMING_RUNS = 5;
const ORBIT_ITER = 1 << 20;
/** A run is abandoned (and reported as timed out) after this long. */
const RUN_LIMIT_S = 900;

interface GoldenView {
  name: string;
  /** Direct views: centre and width; deep views: decimal centre and log2 width. */
  direct?: { mode: typeof MODE_DIRECT | typeof MODE_JULIA; center: [number, number]; width: number; juliaC?: [number, number] };
  deep?: { re: string; im: string; widthLog2: number };
  /** Grow the reference orbit as the app does (see above). */
  grow?: boolean;
  /**
   * Heavy views for tuning thresholds ("tune"), and views kept aside to judge
   * them ("holdout": fixed before any tuning, never used to choose values).
   */
  role?: "tune" | "holdout";
}

const SEAHORSE = { re: "-0.743643887037158704752191506114774", im: "0.131825904205311970493132056385139" };
const voyage = (name: string, extra: number) => {
  const v = VOYAGES.find((t) => t.name === name)!;
  return { re: v.re, im: v.im, widthLog2: -v.depthLog2 + extra };
};
/** Width of the app's "zoom" (location.ts: zoom = 4 / width). */
const zoomWidthLog2 = (zoom: number) => Math.log2(4 / zoom);

export const GOLDEN_VIEWS: GoldenView[] = [
  { name: "overview", direct: { mode: MODE_DIRECT, center: [-0.6, 0], width: 3.6 } },
  { name: "seahorse-1e-2", direct: { mode: MODE_DIRECT, center: [-0.7436438870371587, 0.1318259042053120], width: 0.01 } },
  { name: "julia", direct: { mode: MODE_JULIA, center: [0, 0], width: 3.4, juliaC: [-0.8, 0.156] } },
  { name: "perturb-1e-15", deep: { ...SEAHORSE, widthLog2: Math.log2(1e-15) } },
  { name: "floatexp-1e-98", deep: voyage("Elephant Valley", 3) },
  { name: "floatexp-1e-300", deep: voyage("Seahorse Abyss", 3) },
  // Reported as slow. Just inside the period-2 disk (|λ| = 1 − 4.3e-9), and
  // just outside the lower period-3 component (the centre escapes after 2.84e6).
  { name: "boundary-p2", deep: { re: "-1.249999772751977185", im: "-0.000336290822967596", widthLog2: zoomWidthLog2(4.772e8) }, grow: true, role: "tune" },
  { name: "boundary-p3", deep: { re: "-0.115435619547424377", im: "-0.837972464768108822", widthLog2: zoomWidthLog2(3.374e8) }, grow: true, role: "tune" },
  { name: "abyss-1e-1000", deep: voyage("Abyssal Plain", 3), grow: true },
  // Hold-out boundary views: the 1/3 root of the main cardioid, the period-3
  // cusp at −1.75 (centre inside), the rim of a deep minibrot.
  { name: "holdout-cardioid-third", deep: { re: "-0.1249999990", im: "0.6495190500", widthLog2: Math.log2(1e-8) }, grow: true, role: "holdout" },
  { name: "holdout-cusp-p3", deep: { re: "-1.750000003", im: "0", widthLog2: Math.log2(1e-8) }, grow: true, role: "holdout" },
  { name: "holdout-minibrot-rim", deep: voyage("Spiral Crossing", 0), grow: true, role: "holdout" },
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
  proven: number;
  anomalies: number;
  orbitMs?: number;
  /** Loop steps executed (a BLA span counts once). */
  steps: number;
  waitFrames: { limit: number; sample: number; orbit: number };
  /** Orbit uploads (initial + extensions), their total time and its Rust part. */
  orbitRequests?: number;
  orbitTotalMs?: number;
  rustMs?: { orbit: number; bla: number; encode: number };
  timedOut?: boolean;
  /** Time to obtain the component proofs (perturbation views). */
  patchMs?: number;
  raises?: { from: number; to: number; escaped: number; late: number; paused: number }[];
}

async function orbitFor(orbits: OrbitManager, cam: DeepCamera, log2s: number, w: number, h: number, grow = false): Promise<number> {
  orbits.reset();
  const t0 = performance.now();
  // Grow: the app's first request (limit 800: 4096 iterations, bits for that limit).
  const bits = grow ? Math.ceil(-log2s) + 32 + Math.ceil(Math.log2(800)) + 32 : Math.ceil(-log2s) + 64 + 32;
  const length = dumpLimit ? dumpLimit + 1 : grow ? 4096 : ORBIT_ITER;
  orbits.request({ re: cam.re, im: cam.im, frac: cam.frac }, bits, length, log2s + Math.log2(Math.hypot(w, h) * 2));
  while (!orbits.current && performance.now() - t0 < 300000) await seconds(0.02);
  if (!orbits.current) throw new Error(`orbit failed: ${orbits.lastError ?? "timeout"}`);
  return performance.now() - t0;
}

const sum = (os: OrbitInfo[], f: (o: OrbitInfo) => number) => +os.reduce((t, o) => t + f(o), 0).toFixed(1);

/** Renders one view to convergence; returns the engine (caller disposes) and timings. */
/** Experiment settings from the environment (see the header). */
let experiment: (e: EscapeEngine) => void = () => {};
let usePatches = true;
let dumpLimit = 0;

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
    orbitMs = await orbitFor(orbits, cam, log2s, w, h, gv.grow && !dumpLimit);
    const o = orbits.current!;
    const mode = log2s > -100 && o.minLog2Abs > -100 && o.dcMaxLog2 > -120 ? MODE_PERTURB : MODE_DEEP;
    view = { mode, spacing: feFromLog2(log2s), center: [0, 0], cmr: [0, 0], juliaC: [0, 0], orbit: o, interiorRef: orbitReference(o) };
  }
  const engine = new EscapeEngine(() => app.palette);
  engine.init(app.gpu);
  engine.minIter = engine.maxIter = 800;
  if (view.orbit && !gv.grow && !orbits.current!.periodic && !orbits.current!.escaped) engine.maxIterCap = ORBIT_ITER;
  if (gv.grow && !dumpLimit) engine.orbitExtends = true;
  experiment(engine);
  if (dumpLimit) {
    engine.minIter = engine.maxIter = engine.maxIterCap = dumpLimit;
    engine.sampleTarget = 1;
    engine.provenInterior = false;
  }
  configure?.(engine);
  const installed: OrbitInfo[] = view.orbit ? [orbits.current!] : [];
  // Component proofs, as the app asks for them (here before the first frame).
  let patchMs: number | undefined;
  if (view.mode === MODE_PERTURB && usePatches && !dumpLimit) {
    const t = performance.now();
    engine.setView(view, null);
    engine.setPatches(await requestPatches(orbits.current!, view, w, h));
    patchMs = performance.now() - t;
  }
  const device = app.gpu.device;
  const timer = new GpuTimer(device, app.gpu.hasTimestamps);
  let gpuMs = 0;
  let measured = timer.frames;
  const t0 = performance.now();
  let frames = 0;
  let timedOut = false;
  for (; engine.status !== "converged"; frames++) {
    if (frames >= 200000 || performance.now() - t0 > RUN_LIMIT_S * 1000) {
      timedOut = true;
      break;
    }
    if (gv.grow) {
      // As Mandelbrot.manageOrbit: extend the orbit once the limit outgrows it.
      const o = orbits.current!;
      if (o !== view.orbit) {
        installed.push(o);
        view = { ...view, orbit: o };
      }
      if (!orbits.busy && !o.escaped && !o.periodic && o.maxIter <= engine.maxIter) {
        orbits.request(o.ref, o.bits, Math.min(Math.max(engine.maxIter * 4, 4096), engine.maxIterCap + 1), o.dcMaxLog2, o);
      }
    }
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
      proven: s.proven,
      anomalies: s.anomalies,
      orbitMs: orbitMs !== undefined ? +orbitMs.toFixed(1) : undefined,
      steps: engine.stepsTotal,
      waitFrames: { ...engine.waitFrames },
      orbitRequests: gv.grow ? installed.length : undefined,
      orbitTotalMs: gv.grow ? sum(installed, (o) => o.computeMs) : undefined,
      rustMs: gv.grow
        ? { orbit: sum(installed, (o) => o.rustMs[0]), bla: sum(installed, (o) => o.rustMs[1]), encode: sum(installed, (o) => o.rustMs[2]) }
        : undefined,
      timedOut: timedOut || undefined,
      patchMs: patchMs !== undefined ? +patchMs.toFixed(1) : undefined,
      raises: engine.raiseLog,
    },
  };
}

/** View names become file names. */
export const fileName = (name: string) => name.replace(/[^\w.-]/g, "_");

/** Per pixel: status and iteration count (u32 pairs); 0xffffffff for pixels of another job. */
async function statusAndCount(engine: EscapeEngine): Promise<Uint8Array<ArrayBuffer>> {
  const st = await engine.readState();
  const n = st.width * st.height;
  const out = new Uint32Array(2 * n);
  for (let i = 0; i < n; i++) {
    const b = i * 12;
    const mine = st.u[b + 8] === engine.jobId;
    out[2 * i] = mine ? st.u[b + 9] : 0xffffffff;
    out[2 * i + 1] = mine ? st.u[b + 7] : 0xffffffff;
  }
  return new Uint8Array(out.buffer);
}

/** ITERUM_DUMP_LIMIT: inputs and WebGPU results for the CUDA benchmark. */
async function dumpView(app: App, gv: GoldenView, orbits: OrbitManager, dir: string): Promise<void> {
  const [w, h] = TIMING;
  const { engine, result } = await renderGolden(app, gv, orbits, w, h);
  const out = `${dir}/${fileName(gv.name)}-dump`; // file prefix
  const o = orbits.current!;
  const log2s = gv.deep!.widthLog2 - Math.log2(w);
  const sp = feFromLog2(log2s);
  if (MODE_NAMES.indexOf(result.mode) === MODE_PERTURB && orbits.lastPacket) {
    await saveFile(`${out}-packet.bin`, new Uint8Array(orbits.lastPacket));
    await saveFile(`${out}-view.txt`, new TextEncoder().encode(`${w} ${h} ${sp.m} ${sp.e} 0 0 ${dumpLimit}\n`));
    await saveFile(`${out}-state.bin`, await statusAndCount(engine));
    await saveFile(`${out}-webgpu.txt`, new TextEncoder().encode(`${JSON.stringify({ gpuEscapeMs: result.gpuEscapeMs, steps: result.steps, wallS: result.wallS, frames: result.frames, orbitLen: o.len })}\n`));
  }
  diag(`DUMP ${JSON.stringify(result)}`);
  engine.dispose();
}

export async function runGolden(app: App): Promise<void> {
  await seconds(1);
  const dir = `${await outDir()}`;
  const only = await envVar("ITERUM_ONLY");
  const orbits = new OrbitManager(app.gpu.device);
  const results: (GoldenResult & { timing?: GoldenResult[] })[] = [];
  const timingOff = (await envVar("ITERUM_TIMING")) === "0";
  const runs = timingOff ? 0 : Number((await envVar("ITERUM_RUNS")) ?? TIMING_RUNS);
  dumpLimit = Number((await envVar("ITERUM_DUMP_LIMIT")) ?? 0);
  const cap = await envVar("ITERUM_CAP");
  const proven = (await envVar("ITERUM_PROVEN")) !== "0";
  const trap = await envVar("ITERUM_TRAP");
  const share = await envVar("ITERUM_SHARE");
  experiment = (e) => {
    if (cap) e.maxIterCap = Math.min(e.maxIterCap, Number(cap));
    e.provenInterior = proven;
    if (trap) e.trapIters = Number(trap);
    if (share) e.stillShare = Number(share);
  };
  usePatches = proven && (await envVar("ITERUM_PATCHES")) !== "0";
  for (const gv of GOLDEN_VIEWS.filter((v) => !only || only.split(",").includes(v.name))) {
    if (dumpLimit) {
      if (gv.deep) await dumpView(app, gv, orbits, dir).catch((e) => diag(`DUMP ERROR ${e?.stack ?? e}`));
      continue;
    }
    const { engine, result } = await renderGolden(app, gv, orbits);
    const samples = await engine.readSamples();
    await saveFile(`${dir}/${fileName(gv.name)}.bin`, new Uint8Array(samples.buffer));
    await saveFile(`${dir}/${fileName(gv.name)}.state.bin`, await statusAndCount(engine));
    engine.dispose();
    diag(`GOLDEN ${JSON.stringify(result)}`);
    const timing: GoldenResult[] = [];
    if (runs > 0) {
      // Warm-up at the timing resolution (pipelines, allocations); not recorded.
      (await renderGolden(app, gv, orbits, TIMING[0], TIMING[1])).engine.dispose();
    }
    for (let r = 0; r < runs; r++) {
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

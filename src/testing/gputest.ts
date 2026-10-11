// ITERUM_MODE=gputest: checks GPU results against independent CPU references.
//
//  floatexp      shader floatexp ops vs the CPU model (incl. exponent-alignment edges)
//  lorenz        RK4 particles vs the CPU integrator
//  escape-*      escape-time pixels vs f64 (direct) or the Rust arbitrary-precision
//                oracle (perturbation / floatexp), incl. near-periodic references
//  modes         the same view in perturbation-f32 and floatexp agrees
//  device-loss   a forced device loss is recovered and every phenomenon draws again
//
// Results are printed as "TEST <name> PASS|FAIL <detail>" and the process
// exits with the number of failures. Readbacks happen only here.

import { invoke } from "@tauri-apps/api/core";
import type { App } from "../main";
import { DeepCamera, scaledToBig } from "../core/camera2d";
import { cfe, cfeAdd, cfeMul, cfeNorm2, feLess, feFromLog2, type CFe } from "../core/floatexp";
import { shaderModule } from "../gpu/device";
import { GpuTimer } from "../gpu/timing";
import { wgsl } from "../shaders";
import { EscapeEngine, MODE_DEEP, MODE_DIRECT, MODE_PERTURB, type EscapeMode, type EscapeView } from "../phenomena/escape/engine";
import { OrbitManager } from "../phenomena/mandelbrot/orbit";
import { VOYAGES } from "../phenomena/mandelbrot/voyages";
import { integrate } from "../phenomena/lorenz/reference";
import lorenzSrc from "../phenomena/lorenz/lorenz.wgsl?raw";
import floatexpTestSrc from "./floatexp_test.wgsl?raw";
import { diag, exitApp } from "../platform";
import { frames, seconds, shot } from "./harness";

interface Result {
  name: string;
  pass: boolean;
  detail: string;
}

const results: Result[] = [];
function report(name: string, pass: boolean, detail: string): void {
  results.push({ name, pass, detail });
  diag(`TEST ${name} ${pass ? "PASS" : "FAIL"} ${detail}`);
}

async function readBuffer(device: GPUDevice, src: GPUBuffer, size: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return out;
}

// ---- floatexp ----------------------------------------------------------------

async function testFloatexp(app: App): Promise<void> {
  const device = app.gpu.device;
  const cases: [CFe, CFe][] = [];
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296) * 2 - 1;
  for (let i = 0; i < 400; i++) cases.push([cfe(rnd(), rnd(), Math.round(rnd() * 3000)), cfe(rnd(), rnd(), Math.round(rnd() * 3000))]);
  // Exponent differences right at the alignment cut-off, both orders.
  for (const d of [28, 29, 30, 31, 32, -28, -29, -30, -31, -32]) cases.push([cfe(0.7, -0.3, 0), cfe(0.6, 0.55, -d)]);
  // Near cancellation (the rebasing comparison), at extreme exponents.
  for (const e of [-9000, -700, 0, 700]) cases.push([cfe(0.75, 0.5, e), cfe(-0.7499, -0.4999, e)]);
  // Zero operands.
  cases.push([cfe(0, 0), cfe(0.5, 0.25, -50)], [cfe(0.5, 0.25, 40), cfe(0, 0)]);

  const data = new ArrayBuffer(cases.length * 32);
  const f = new Float32Array(data);
  const i32 = new Int32Array(data);
  cases.forEach(([a, b], k) => {
    f[k * 8] = a.re; f[k * 8 + 1] = a.im; i32[k * 8 + 2] = a.e;
    f[k * 8 + 4] = b.re; f[k * 8 + 5] = b.im; i32[k * 8 + 6] = b.e;
  });
  const inBuf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, data);
  const outSize = cases.length * 48;
  const outBuf = device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const module = shaderModule(device, "floatexp-test", wgsl(floatexpTestSrc));
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] }));
  pass.dispatchWorkgroups(Math.ceil(cases.length / 64));
  pass.end();
  device.queue.submit([enc.finish()]);
  const raw = await readBuffer(device, outBuf, outSize);
  const of = new Float32Array(raw);
  const oi = new Int32Array(raw);
  const ou = new Uint32Array(raw);

  let worst = 0;
  let bad = 0;
  const close = (gpu: CFe, cpu: CFe, scale: number) => {
    // Compare as values relative to the larger input magnitude (f32 rounding / FMA differences).
    const g = [gpu.re * 2 ** (gpu.e - scale), gpu.im * 2 ** (gpu.e - scale)];
    const c = [cpu.re * 2 ** (cpu.e - scale), cpu.im * 2 ** (cpu.e - scale)];
    const err = Math.hypot(g[0] - c[0], g[1] - c[1]);
    worst = Math.max(worst, err);
    return err < 4e-7;
  };
  cases.forEach(([a, b], k) => {
    const o = k * 12;
    const gs: CFe = { re: of[o], im: of[o + 1], e: oi[o + 2] };
    const gm: CFe = { re: of[o + 4], im: of[o + 5], e: oi[o + 3] };
    const s = cfeAdd(a, b);
    const m = cfeMul(a, b);
    const okAdd = close(gs, s, Math.max(a.e, b.e));
    const okMul = close(gm, m, a.e + b.e);
    const n2 = cfeNorm2(s);
    const okNorm = (n2.m === 0 && of[o + 6] === 0) || (n2.e === oi[o + 7] && Math.abs(n2.m - of[o + 6]) < 1e-6);
    const okLess = (ou[o + 8] === 1) === feLess(cfeNorm2(s), cfeNorm2(b));
    if (!(okAdd && okMul && okNorm && okLess)) bad++;
  });
  report("floatexp", bad === 0, `${cases.length} cases, ${bad} mismatches, worst relative error ${worst.toExponential(2)}`);
  inBuf.destroy();
  outBuf.destroy();
}

// ---- Lorenz -------------------------------------------------------------------

async function testLorenz(app: App): Promise<void> {
  const device = app.gpu.device;
  const n = 256;
  const steps = 300;
  const dt = 0.002;
  const parts = new Float32Array(n * 8);
  const starts: [number, number, number][] = [];
  for (let i = 0; i < n; i++) {
    const p: [number, number, number] = [Math.fround(-10 + (i % 16) * 1.3), Math.fround(-12 + Math.floor(i / 16) * 1.5), Math.fround(15 + (i % 7) * 2)];
    starts.push(p);
    parts.set(p, i * 8);
  }
  const pbuf = device.createBuffer({ size: parts.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(pbuf, 0, parts);
  const u = new ArrayBuffer(160);
  const uu = new Uint32Array(u);
  const uf = new Float32Array(u);
  uu[18] = n; uu[19] = steps; uf[20] = dt; uf[21] = 10; uf[22] = 28; uf[23] = 8 / 3;
  const ubuf = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(ubuf, 0, u);
  const module = shaderModule(device, "lorenz-test", wgsl(lorenzSrc));
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "warmup" } });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubuf } }, { binding: 1, resource: { buffer: pbuf } }] }));
  pass.dispatchWorkgroups(Math.ceil(n / 256));
  pass.end();
  device.queue.submit([enc.finish()]);
  const out = new Float32Array(await readBuffer(device, pbuf, parts.byteLength));
  let worst = 0;
  for (let i = 0; i < n; i++) {
    // f32 emulation of the shader's arithmetic; the flow is chaotic, so the
    // horizon (0.6 time units) is kept short enough for f32 to stay close.
    const ref = integrate(starts[i], Math.fround(dt), steps, Math.fround);
    worst = Math.max(worst, Math.hypot(out[i * 8] - ref[0], out[i * 8 + 1] - ref[1], out[i * 8 + 2] - ref[2]));
  }
  report("lorenz-rk4", worst < 1e-2, `${n} particles × ${steps} RK4 steps, worst deviation ${worst.toExponential(2)}`);
  pbuf.destroy();
  ubuf.destroy();
}

// ---- escape-time ------------------------------------------------------------

const W = 64;
const H = 64;

export async function renderView(app: App, view: EscapeView, maxIter: number, w = W, h = H, pin = false): Promise<{ f: Float32Array; u: Uint32Array; engine: EscapeEngine }> {
  const engine = new EscapeEngine(() => app.palette);
  engine.init(app.gpu);
  engine.sampleTarget = 1;
  engine.maxIter = maxIter;
  engine.minIter = maxIter;
  if (pin) engine.maxIterCap = maxIter;
  const timer = new GpuTimer(app.gpu.device, false);
  for (let i = 0; i < 2000 && engine.status !== "converged"; i++) {
    const encoder = app.gpu.device.createCommandEncoder();
    engine.setView(view, null);
    engine.render({ gpu: app.gpu, encoder, timer, time: 0, dt: 1 / 60, paused: false, palette: app.palette.palette, width: w, height: h, pixelRatio: 1 });
    app.gpu.device.queue.submit([encoder.finish()]);
    await app.gpu.device.queue.onSubmittedWorkDone();
    await new Promise((r) => setTimeout(r, 0));
  }
  const s = await engine.readState();
  return { f: s.f, u: s.u, engine };
}

const smoothOf = (n: number, re: number, im: number) => n + 1 - Math.log2((0.5 * Math.log2(re * re + im * im)) / 8);

interface Ref {
  escaped: boolean;
  smooth: number;
}

/** Sampled pixels (every 4th in x and y). */
function samplePixels(): [number, number][] {
  const out: [number, number][] = [];
  for (let y = 1; y < H; y += 4) for (let x = 2; x < W; x += 4) out.push([x, y]);
  return out;
}

/**
 * Pixels on boundary filaments (distance estimate below a pixel) are
 * ill-conditioned: moving c by an f32 rounding step changes their iteration
 * count, so any f32 renderer differs there from an exact reference. They must
 * agree on escape vs interior; well-conditioned pixels must agree closely.
 */
function compare(name: string, f: Float32Array, u: Uint32Array, pix: [number, number][], refs: Ref[], extra = ""): void {
  let statusBad = 0;
  let good = 0;
  let goodClose = 0;
  const errs: number[] = [];
  pix.forEach(([x, y], k) => {
    const i = (y * W + x) * 12;
    const gpuEscaped = u[i + 9] === 1;
    if (gpuEscaped !== refs[k].escaped) {
      statusBad++;
      return;
    }
    if (!gpuEscaped) return;
    const e = Math.abs(f[i] - refs[k].smooth);
    errs.push(e);
    if (f[i + 1] >= 1) {
      good++;
      if (e < 0.02) goodClose++;
    }
  });
  errs.sort((a, b) => a - b);
  const median = errs.length ? errs[errs.length >> 1] : 0;
  const share = good ? goodClose / good : 1;
  const pass = statusBad <= pix.length * 0.02 && share >= 0.99 && median < 2e-3;
  report(
    name,
    pass,
    `${pix.length} px: escape/interior mismatches ${statusBad}; well-conditioned (DE ≥ 1 px) ${goodClose}/${good} within 0.02; median |Δsmooth| ${median.toExponential(2)}${extra}`,
  );
}

async function testDirect(app: App): Promise<void> {
  const cx = -0.7436;
  const cy = 0.1318;
  const log2s = Math.log2(0.004 / W);
  const maxIter = 2000;
  const { f, u, engine } = await renderView(app, { mode: MODE_DIRECT, spacing: feFromLog2(log2s), center: [cx, cy], cmr: [0, 0], juliaC: [0, 0] }, maxIter);
  const pix = samplePixels();
  const s = 2 ** log2s;
  const fcx = Math.fround(cx);
  const fcy = Math.fround(cy);
  const refs = pix.map(([x, y]) => {
    const cr = fcx + (x + 0.5 - W / 2) * s;
    const ci = fcy - (y + 0.5 - H / 2) * s;
    let zr = 0, zi = 0, n = 0;
    for (; n < engine.maxIter && zr * zr + zi * zi <= 65536; n++) {
      const t = zr * zr - zi * zi + cr;
      zi = 2 * zr * zi + ci;
      zr = t;
    }
    return { escaped: zr * zr + zi * zi > 65536, smooth: smoothOf(n, zr, zi) };
  });
  compare("escape-direct", f, u, pix, refs, " (vs f64)");
  engine.dispose();
}

async function oracle(cam: DeepCamera, log2s: number, pix: [number, number][], maxIter: number): Promise<Ref[]> {
  const points = pix.map(([x, y]) => [
    (cam.re + scaledToBig(x + 0.5 - W / 2, log2s + cam.frac)).toString(16),
    (cam.im - scaledToBig(y + 0.5 - H / 2, log2s + cam.frac)).toString(16),
  ]);
  const bits = Math.ceil(-log2s) + 96;
  const res = await invoke<{ iterations: number; escaped: boolean; finalRe: number; finalIm: number }[]>("direct_batch", {
    req: { points, fracBits: cam.frac, bits, maxIter },
  });
  return res.map((r) => ({ escaped: r.escaped, smooth: smoothOf(r.iterations, r.finalRe, r.finalIm) }));
}

async function deepView(app: App, re: string, im: string, widthLog2: number, mode: EscapeMode | "auto", maxIter: number, label: string, orbits: OrbitManager): Promise<void> {
  const log2s = widthLog2 - Math.log2(W);
  const cam = new DeepCamera(re, im, log2s);
  orbits.reset();
  orbits.request({ re: cam.re, im: cam.im, frac: cam.frac }, Math.ceil(-log2s) + 64 + 20, maxIter, log2s + Math.log2(W * 2));
  const t0 = performance.now();
  while (!orbits.current && performance.now() - t0 < 60000) await seconds(0.05);
  const o = orbits.current;
  if (!o) {
    report(label, false, `orbit not received: ${orbits.lastError ?? "timeout"}`);
    return;
  }
  const m: EscapeMode = mode === "auto" ? (log2s > -100 && o.minLog2Abs > -100 ? MODE_PERTURB : MODE_DEEP) : mode;
  const { f, u, engine } = await renderView(app, { mode: m, spacing: feFromLog2(log2s), center: [0, 0], cmr: [0, 0], juliaC: [0, 0], orbit: o }, maxIter);
  const st = engine.stats();
  const refs = await oracle(cam, log2s, samplePixels(), engine.maxIter);
  compare(label, f, u, samplePixels(), refs, `; mode ${m === MODE_DEEP ? "floatexp" : "f32"}, maxIter ${engine.maxIter}, orbit ${o.len}${o.periodic ? " (periodic)" : ""}, min log2|Z| ${o.minLog2Abs.toFixed(0)}, rebases ${st.rebases}, BLA steps ${st.blaSteps}, anomalies ${st.anomalies}`);
  if (st.anomalies > 0) report(`${label}-anomalies`, false, `${st.anomalies} pixels flagged`);
  engine.dispose();
}

async function testModesAgree(app: App, orbits: OrbitManager): Promise<void> {
  // A view inside the f32 range, rendered with f32 deltas and with floatexp.
  const re = "-0.743643887037158704752191506114774";
  const im = "0.131825904205311970493132056385139";
  const log2s = Math.log2(1e-20) - Math.log2(W);
  const cam = new DeepCamera(re, im, log2s);
  orbits.reset();
  orbits.request({ re: cam.re, im: cam.im, frac: cam.frac }, Math.ceil(-log2s) + 84, 20000, log2s + Math.log2(W * 2));
  while (!orbits.current) await seconds(0.05);
  const o = orbits.current;
  const base: EscapeView = { mode: MODE_PERTURB, spacing: feFromLog2(log2s), center: [0, 0], cmr: [0, 0], juliaC: [0, 0], orbit: o };
  const a = await renderView(app, base, 20000);
  const b = await renderView(app, { ...base, mode: MODE_DEEP }, 20000);
  let agree = 0;
  for (let p = 0; p < W * H; p++) {
    const i = p * 12;
    if (a.u[i + 9] === b.u[i + 9] && (a.u[i + 9] !== 1 || Math.abs(a.f[i] - b.f[i]) < 0.02)) agree++;
  }
  report("modes-agree", agree / (W * H) >= 0.97, `perturbation-f32 vs floatexp at 1e-20: ${((agree / (W * H)) * 100).toFixed(1)}% of ${W * H} px agree`);
  a.engine.dispose();
  b.engine.dispose();
}

// ---- device loss ------------------------------------------------------------

async function meanLuma(png: Uint8Array): Promise<number> {
  const bmp = await createImageBitmap(new Blob([png as Uint8Array<ArrayBuffer>], { type: "image/png" }));
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const g = c.getContext("2d")!;
  g.drawImage(bmp, 0, 0);
  const d = g.getImageData(0, 0, bmp.width, bmp.height).data;
  let s = 0;
  for (let i = 0; i < d.length; i += 4 * 97) s += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
  return s / (d.length / (4 * 97)) / 255;
}

async function testDeviceLoss(app: App): Promise<void> {
  const before = app.deviceGeneration;
  app.loseDevice();
  const t0 = performance.now();
  while (app.deviceGeneration === before && performance.now() - t0 < 10000) await seconds(0.1);
  await frames(app, 20);
  const lumas: string[] = [];
  let ok = app.deviceGeneration > before;
  for (let i = 0; i < app.phenomena.length; i++) {
    app.select(i);
    await seconds(i === 3 ? 5 : 3);
    const png = await app.capture();
    const l = await meanLuma(png);
    lumas.push(`${app.phenomena[i].meta.id} ${l.toFixed(3)}`);
    if (l < 0.01) ok = false;
    await shot(app, `after-device-loss-${app.phenomena[i].meta.id}`);
  }
  report("device-loss", ok, `generation ${before} → ${app.deviceGeneration}; mean luminance after recovery: ${lumas.join(", ")}`);
}

// ---- controls ----------------------------------------------------------------

async function testControls(app: App): Promise<void> {
  const click = (id: string) => (document.getElementById(id) as HTMLButtonElement).click();
  const notes: string[] = [];
  let ok = true;
  // Animated phenomena: play button pauses and resumes their motion.
  for (const i of [1, 2, 3]) {
    app.select(i);
    await seconds(0.3);
    click("play");
    const paused = app.paused;
    click("play");
    const resumed = !app.paused;
    if (!paused || !resumed) ok = false;
    notes.push(`${app.phenomena[i].meta.id} pause ${paused} resume ${resumed}`);
  }
  // Mandelbrot: play button starts and stops Voyage.
  app.select(0);
  await seconds(0.3);
  const m = app.phenomena[0] as import("../phenomena/mandelbrot").Mandelbrot;
  click("play");
  const started = m.playing();
  click("play");
  const stopped = !m.playing();
  if (!started || !stopped) ok = false;
  notes.push(`voyage start ${started} stop ${stopped}`);
  // Reset: a deep view returns to the overview.
  m.setView("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", -120);
  await seconds(0.5);
  click("reset");
  await seconds(1.5);
  const back = m.cameraState.log2Scale > -12;
  if (!back) ok = false;
  notes.push(`reset log2Scale ${m.cameraState.log2Scale.toFixed(1)}`);
  report("controls", ok, notes.join("; "));
}

export async function runGpuTests(app: App): Promise<void> {
  await seconds(1);
  const steps: [string, () => Promise<void>][] = [];
  const orbits = new OrbitManager(app.gpu.device);
  steps.push(["floatexp", () => testFloatexp(app)]);
  steps.push(["lorenz", () => testLorenz(app)]);
  steps.push(["direct", () => testDirect(app)]);
  steps.push(["perturb-1e-12", () => deepView(app, "-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", Math.log2(1e-12), "auto", 20000, "escape-perturbation-1e-12", orbits)]);
  const v = (n: string) => VOYAGES.find((t) => t.name === n)!;
  steps.push(["deep-1e-40", () => deepView(app, v("Seahorse Spiral").re, v("Seahorse Spiral").im, -v("Seahorse Spiral").depthLog2, "auto", 20000, "escape-floatexp-1e-40-nucleus", orbits)]);
  steps.push(["deep-1e-100", () => deepView(app, v("Elephant Valley").re, v("Elephant Valley").im, -v("Elephant Valley").depthLog2, "auto", 50000, "escape-floatexp-1e-100-nucleus", orbits)]);
  steps.push(["deep-1e-98-wide", () => deepView(app, v("Elephant Valley").re, v("Elephant Valley").im, -v("Elephant Valley").depthLog2 + 6, "auto", 50000, "escape-floatexp-1e-98-surroundings", orbits)]);
  steps.push(["deep-1e-300", () => deepView(app, v("Seahorse Abyss").re, v("Seahorse Abyss").im, -v("Seahorse Abyss").depthLog2 + 2, "auto", 50000, "escape-floatexp-1e-300", orbits)]);
  steps.push(["modes", () => testModesAgree(app, orbits)]);
  steps.push(["controls", () => testControls(app)]);
  steps.push(["device-loss", () => testDeviceLoss(app)]);
  for (const [name, fn] of steps) {
    try {
      await fn();
    } catch (e) {
      report(name, false, `threw: ${e}`);
    }
  }
  const failed = results.filter((r) => !r.pass).length;
  diag(`GPU TESTS: ${results.length - failed} passed, ${failed} failed`);
  await exitApp(failed);
}

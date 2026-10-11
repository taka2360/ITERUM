// Resumable, progressive escape-time renderer shared by Mandelbrot and Julia.
//
// A "view" is everything that determines the image (mode, centre, scale,
// iteration limit, orbit). Each view is rendered as a sequence of jittered
// samples; each sample is a job whose pixels advance a bounded number of
// iterations per frame until every pixel has finished. Completion is detected
// from a small counter readback that lags a frame or two; the frame loop never
// waits on it.

import { shaderModule, type GpuContext } from "../../gpu/device";
import type { FrameContext, RenderStatus } from "../../core/phenomenon";
import type { PaletteTexture } from "../../core/palettes";
import { hexToRgb } from "../../core/palettes";
import type { Fe } from "../../core/floatexp";
import { wgsl } from "../../shaders";
import escapeSrc from "./escape.wgsl?raw";
import displaySrc from "./display.wgsl?raw";

export const MODE_DIRECT = 0;
export const MODE_JULIA = 1;
export const MODE_PERTURB = 2;
export const MODE_DEEP = 3;
export type EscapeMode = 0 | 1 | 2 | 3;
export const MODE_NAMES = ["direct f32", "julia f32", "perturbation f32", "perturbation floatexp"];

export interface GpuOrbit {
  id: number;
  orbit: GPUBuffer;
  bla: GPUBuffer;
  blaOff: GPUBuffer;
  len: number;
  levels: number;
}

export interface EscapeView {
  mode: EscapeMode;
  /** Complex units per internal pixel. */
  spacing: Fe;
  /** Centre for direct modes. */
  center: [number, number];
  /** (centre − reference) / spacing, internal pixels; perturbation modes. */
  cmr: [number, number];
  juliaC: [number, number];
  orbit?: GpuOrbit;
}

/** Maps an internal pixel of the new view to the previous frame: old = mid + (p − mid)·scale + offset. */
export interface HistoryMap {
  scale: number;
  offset: [number, number];
}

/** Detailed progress of the current view (see EscapeEngine.progress). */
export interface EscapeProgress {
  /** Completion of each progressive step of the current sample (0..1). */
  coarse: number;
  mid: number;
  fine: number;
  /** Completion of the current sample, and of the whole view (all samples). */
  sample: number;
  overall: number;
  samplesDone: number;
  sampleTarget: number;
  converged: boolean;
  /** The view changes every frame (dragging, zooming, Voyage): figures are per frame. */
  moving: boolean;
  /** Iterations advanced per second (BLA spans counted in full). */
  iterationsPerSecond: number;
  elapsedS: number;
  /** Estimated seconds left, if the trend allows an estimate. */
  etaS?: number;
  maxIter: number;
  maxEscaped: number;
  /** Why the view was started over, if it was (e.g. iteration limit raised). */
  note?: string;
  pixels: number;
}

export interface EscapeStats {
  job: number;
  samplesDone: number;
  sampleTarget: number;
  unfinished: number;
  escaped: number;
  interior: number;
  anomalies: number;
  rebases: number;
  blaSteps: number;
  maxEscapedN: number;
  minEscapedN: number;
  budget: number;
  maxIter: number;
}

const COUNTER_BYTES = 48;
const PARAM_BYTES = 80;
const DISPLAY_BYTES = 96;
const STATE_BYTES = 48;
// display.wgsl averages up to this many samples (unrolled); keep both in step.
const MAX_SAMPLES = 4;

// Jitter offsets for supersampling (first sample at the pixel centre).
const JITTER: [number, number][] = [[0, 0], [0.25, -0.25], [-0.25, 0.25], [0.375, 0.375]];

interface Readback {
  buffer: GPUBuffer;
  busy: boolean;
}

export class EscapeEngine {
  private device!: GPUDevice;
  private computeLayout!: GPUBindGroupLayout;
  private displayLayout!: GPUBindGroupLayout;
  private pipelines: GPUComputePipeline[] = [];
  private displayPipeline!: GPURenderPipeline;
  private params: GPUBuffer[] = [];
  private displayParams!: GPUBuffer;
  private counters!: GPUBuffer;
  private readbacks: Readback[] = [];
  private dummy!: GPUBuffer;
  private state?: GPUBuffer;
  private samples?: GPUBuffer;
  private outputs: GPUTexture[] = [];
  private outIndex = 0;
  private linear!: GPUSampler;
  private computeGroups: GPUBindGroup[] = [];
  private computeGroupKey = "";
  private displayGroups: GPUBindGroup[] = [];
  private displayGroupPal?: GPUTexture;
  private width = 0;
  private height = 0;

  private view?: EscapeView;
  private job = 1;
  private sample = 0;
  private samplesDone = 0;
  private converged = false;
  private histValid = false;
  private hist: HistoryMap = { scale: 1, offset: [0, 0] };
  private budget = 256;
  private framesSinceJob = 0;

  /** Iteration limit; adapted from statistics unless pinned. */
  maxIter = 1000;
  minIter = 500;
  maxIterCap = 1 << 24;
  sampleTarget = MAX_SAMPLES;
  /** Target GPU milliseconds per frame for iteration. */
  targetMs = 9;
  /** density: palette cycles per doubling of (iterations − sMin)/bandScale. */
  look = { offset: 0, density: 0.32, glow: 0.6, sMin: 0, bandScale: 6, interiorGlow: 0.3 };

  private agg = { escaped: 0, interior: 0, anomalies: 0, rebases: 0, blaSteps: 0, maxN: 0, minN: Infinity, late: 0, unfinished: -1 };
  /** Aggregated statistics of the most recently completed sample. */
  lastComplete?: EscapeStats;

  constructor(private palette: () => PaletteTexture) {}

  init(gpu: GpuContext): void {
    const device = (this.device = gpu.device);
    this.outputs = [];
    this.state = this.samples = undefined;
    this.width = this.height = 0;
    this.computeGroupKey = "";
    const ro = { type: "read-only-storage" } as const;
    const rw = { type: "storage" } as const;
    this.computeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: ro },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: ro },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: ro },
      ],
    });
    const module = shaderModule(device, "escape", wgsl(escapeSrc));
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    this.pipelines = [0, 1, 2, 3].map((mode) =>
      device.createComputePipeline({ label: `escape-${mode}`, layout, compute: { module, entryPoint: "iterate", constants: { MODE: mode } } }),
    );
    this.displayLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: ro },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: ro },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      ],
    });
    const dmod = shaderModule(device, "escape-display", wgsl(displaySrc));
    this.displayPipeline = device.createRenderPipeline({
      label: "escape-display",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.displayLayout] }),
      vertex: { module: dmod, entryPoint: "vs" },
      fragment: { module: dmod, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
    });
    this.params = [0, 1, 2].map(() => device.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    this.displayParams = device.createBuffer({ size: DISPLAY_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.counters = device.createBuffer({ size: COUNTER_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.readbacks = [0, 1, 2, 3].map(() => ({
      buffer: device.createBuffer({ size: COUNTER_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      busy: false,
    }));
    this.dummy = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE });
    this.linear = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    this.restart();
  }

  private ensureSize(w: number, h: number): void {
    if (w === this.width && h === this.height && this.state) return;
    this.width = w;
    this.height = h;
    this.state?.destroy();
    this.samples?.destroy();
    for (const t of this.outputs) t.destroy();
    const device = this.device;
    this.state = device.createBuffer({ size: Math.max(64, w * h * STATE_BYTES), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.samples = device.createBuffer({ size: Math.max(64, w * h * 8 * MAX_SAMPLES), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.outputs = [0, 1].map(() =>
      device.createTexture({
        size: [w, h],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
    );
    this.computeGroupKey = "";
    this.displayGroups = [];
    this.histValid = false;
    this.restart();
  }

  private prog = {
    viewStart: performance.now(),
    finishedAt: 0,
    coarse: -1,
    mid: -1,
    rate: [] as { t: number; n: number }[],
    trend: [] as { t: number; f: number }[],
    note: undefined as string | undefined,
    sMinLocked: false,
    anyUnfinished: -1,
    anyCoarse: -1,
    anyMid: -1,
  };

  private newView(): void {
    this.prog.sMinLocked = false;
    this.prog.viewStart = performance.now();
    this.prog.finishedAt = 0;
    this.prog.note = undefined;
    this.prog.trend = [];
  }

  /** Starts the view over (e.g. after the iteration limit changed). */
  restart(): void {
    this.prog.coarse = this.prog.mid = -1;
    this.prog.finishedAt = 0;
    this.prog.trend = [];
    this.job++;
    this.sample = 0;
    this.samplesDone = 0;
    this.converged = false;
    this.framesSinceJob = 0;
    this.resetAgg();
  }

  private resetAgg(): void {
    this.agg = { escaped: 0, interior: 0, anomalies: 0, rebases: 0, blaSteps: 0, maxN: 0, minN: Infinity, late: 0, unfinished: -1 };
  }

  /**
   * Sets the view for this frame. `history` maps into the previous frame when
   * the view changed, or is null when the view is unchanged.
   */
  setView(view: EscapeView, history: HistoryMap | null): void {
    const prev = this.view;
    this.view = view;
    if (history || !prev || prev.mode !== view.mode || prev.orbit !== view.orbit) {
      this.hist = history ?? { scale: 1, offset: [0, 0] };
      this.newView();
      this.restart();
    } else {
      this.hist = { scale: 1, offset: [0, 0] };
    }
  }

  get status(): RenderStatus {
    return this.converged ? "converged" : "refining";
  }

  stats(): EscapeStats {
    return {
      job: this.job,
      samplesDone: this.samplesDone,
      sampleTarget: this.sampleTarget,
      unfinished: this.agg.unfinished,
      escaped: this.agg.escaped,
      interior: this.agg.interior,
      anomalies: this.agg.anomalies,
      rebases: this.agg.rebases,
      blaSteps: this.agg.blaSteps,
      maxEscapedN: this.agg.maxN,
      minEscapedN: this.agg.minN,
      budget: this.budget,
      maxIter: this.maxIter,
    };
  }

  private computeGroupsFor(orbit?: GpuOrbit): GPUBindGroup[] {
    const key = `${this.width}x${this.height}:${orbit?.id ?? "none"}`;
    if (key === this.computeGroupKey) return this.computeGroups;
    this.computeGroupKey = key;
    this.computeGroups = this.params.map((p) =>
      this.device.createBindGroup({
        layout: this.computeLayout,
        entries: [
          { binding: 0, resource: { buffer: p } },
          { binding: 1, resource: { buffer: this.state! } },
          { binding: 2, resource: { buffer: this.samples! } },
          { binding: 3, resource: { buffer: this.counters } },
          { binding: 4, resource: { buffer: orbit?.orbit ?? this.dummy } },
          { binding: 5, resource: { buffer: orbit?.bla ?? this.dummy } },
          { binding: 6, resource: { buffer: orbit?.blaOff ?? this.dummy } },
        ],
      }),
    );
    return this.computeGroups;
  }

  private displayGroup(): GPUBindGroup {
    const pal = this.palette();
    // One group per output texture (each reads the other as history).
    if (this.displayGroups.length === 0 || this.displayGroupPal !== pal.texture) {
      this.displayGroups = [0, 1].map((i) =>
        this.device.createBindGroup({
          layout: this.displayLayout,
          entries: [
            { binding: 0, resource: { buffer: this.displayParams } },
            { binding: 1, resource: { buffer: this.state! } },
            { binding: 2, resource: { buffer: this.samples! } },
            { binding: 3, resource: pal.texture.createView() },
            { binding: 4, resource: pal.sampler },
            { binding: 5, resource: this.outputs[1 - i].createView() },
            { binding: 6, resource: this.linear },
          ],
        }),
      );
      this.displayGroupPal = pal.texture;
    }
    return this.displayGroups[this.outIndex];
  }

  /** Records iteration (unless `compute` is false: reproject only) and display. */
  render(frame: FrameContext, compute = true): { output: GPUTexture; status: RenderStatus } {
    const view = this.view!;
    this.ensureSize(frame.width, frame.height);
    const device = this.device;
    const enc = frame.encoder;
    const w = this.width;
    const h = this.height;

    if (!this.converged && compute) {
      this.framesSinceJob++;
      const groups = this.computeGroupsFor(view.orbit);
      const jit = JITTER[this.sample % JITTER.length];
      for (let step = 0; step < 3; step++) {
        const buf = new ArrayBuffer(PARAM_BYTES);
        const u = new Uint32Array(buf);
        const f = new Float32Array(buf);
        const i = new Int32Array(buf);
        u[0] = w; u[1] = h; u[2] = this.job; u[3] = step;
        f[4] = jit[0]; f[5] = jit[1];
        f[6] = view.cmr[0]; f[7] = view.cmr[1];
        f[8] = view.center[0]; f[9] = view.center[1];
        f[10] = view.juliaC[0]; f[11] = view.juliaC[1];
        f[12] = view.spacing.m; i[13] = view.spacing.e;
        u[14] = this.maxIter;
        // Coarse steps cover fewer pixels, so they get a larger share.
        u[15] = Math.max(16, Math.round(this.budget * [4, 2, 1][step]));
        u[16] = view.orbit?.len ?? 0;
        u[17] = view.orbit?.levels ?? 0;
        u[18] = this.sample;
        device.queue.writeBuffer(this.params[step], 0, buf);
      }
      device.queue.writeBuffer(this.counters, 0, new Uint32Array([0, 0, 0, 0, 0, 0, 0xffffffff, 0, 0, 0, 0, 0]));
      const pass = enc.beginComputePass({ label: "escape", timestampWrites: frame.timer.writes("escape") });
      pass.setPipeline(this.pipelines[view.mode]);
      for (let step = 0; step < 3; step++) {
        const stride = 1 << (2 - step);
        pass.setBindGroup(0, groups[step]);
        pass.dispatchWorkgroups(Math.ceil(Math.ceil(w / stride) / 8), Math.ceil(Math.ceil(h / stride) / 8));
      }
      pass.end();
      this.scheduleReadback(enc);
    }

    // Display.
    const palTex = this.palette();
    const pal = palTex.palette;
    const interior = hexToRgb(pal.interior);
    const dbuf = new ArrayBuffer(DISPLAY_BYTES);
    const du = new Uint32Array(dbuf);
    const df = new Float32Array(dbuf);
    du[0] = w; du[1] = h; du[2] = this.job; du[3] = this.samplesDone;
    df[4] = this.hist.offset[0]; df[5] = this.hist.offset[1]; df[6] = this.hist.scale; du[7] = this.histValid ? 1 : 0;
    df[8] = this.look.offset; df[9] = this.look.density; df[10] = this.look.sMin; df[11] = this.look.glow;
    df[12] = interior[0]; df[13] = interior[1]; df[14] = interior[2]; df[15] = 1;
    df[16] = palTex.mean[0]; df[17] = palTex.mean[1]; df[18] = palTex.mean[2]; df[19] = this.look.bandScale;
    df[20] = this.look.interiorGlow;
    device.queue.writeBuffer(this.displayParams, 0, dbuf);
    this.outIndex = 1 - this.outIndex;
    const target = this.outputs[this.outIndex];
    const rp = enc.beginRenderPass({
      label: "escape-display",
      colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }],
      timestampWrites: frame.timer.writes("display"),
    });
    rp.setPipeline(this.displayPipeline);
    rp.setBindGroup(0, this.displayGroup());
    rp.draw(3);
    rp.end();
    this.histValid = true;
    this.hist = { scale: 1, offset: [0, 0] };
    this.adaptBudget(frame);
    return { output: target, status: this.status };
  }

  private adaptBudget(frame: FrameContext): void {
    const ms = frame.timer.passMs.get("escape");
    if (ms === undefined || this.converged) return;
    // Multiplicative controller towards the target; clamp per-frame change.
    const ratio = Math.max(0.5, Math.min(1.5, this.targetMs / Math.max(ms, 0.05)));
    this.budget = Math.max(32, Math.min(1 << 16, this.budget * ratio));
  }

  private scheduleReadback(enc: GPUCommandEncoder): void {
    const rb = this.readbacks.find((r) => !r.busy);
    if (!rb) return;
    enc.copyBufferToBuffer(this.counters, 0, rb.buffer, 0, COUNTER_BYTES);
    rb.busy = true;
    const job = this.job;
    const sample = this.sample;
    queueMicrotask(() => {
      // mapAsync must be issued after submit; the frame loop submits synchronously
      // after render() returns, and microtasks run after the current task.
      rb.buffer
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const c = new Uint32Array(rb.buffer.getMappedRange().slice(0));
          rb.buffer.unmap();
          rb.busy = false;
          this.onCounters(job, sample, c);
        })
        .catch(() => (rb.busy = false));
    });
  }

  private onCounters(job: number, sample: number, c: Uint32Array): void {
    // Throughput counts all work, including frames of superseded jobs.
    const now = performance.now();
    const pr = this.prog;
    pr.rate.push({ t: now, n: c[11] * 256 });
    while (pr.rate.length > 2 && now - pr.rate[0].t > 1000) pr.rate.shift();
    pr.anyUnfinished = c[0];
    pr.anyCoarse = c[9];
    pr.anyMid = c[10];
    if (job !== this.job || sample !== this.sample) return;
    pr.coarse = c[9];
    pr.mid = c[10];
    const a = this.agg;
    if (c[6] !== 0xffffffff) a.minN = Math.min(a.minN, c[6]);
    // Fix the colour reference (fastest escape) while the coarse preview is
    // still being drawn, so colours never shift when the image completes.
    if (this.sample === 0 && !pr.sMinLocked && a.minN !== Infinity) {
      this.look.sMin = a.minN;
      if (c[9] === 0) pr.sMinLocked = true;
    }
    a.unfinished = c[0];
    a.escaped += c[1];
    a.interior += c[2];
    a.anomalies += c[3];
    a.rebases += c[4];
    a.maxN = Math.max(a.maxN, c[5]);
    a.blaSteps += c[7];
    a.late += c[8];
    if (c[0] !== 0) return;
    // Sample complete.
    this.lastComplete = this.stats();
    if (this.sample === 0) {
      if (this.wantsMoreIterations()) {
        this.maxIter = Math.min(this.maxIterCap, this.maxIter * 2);
        this.prog.note = `反復上限を ${this.maxIter.toLocaleString()} に引き上げて再計算中`;
        this.restart();
        return;
      }
    }
    this.samplesDone++;
    if (this.samplesDone >= this.sampleTarget) {
      this.converged = true;
      this.prog.finishedAt = performance.now();
      this.prog.note = undefined;
      return;
    }
    this.sample = this.samplesDone;
    this.job++;
    this.framesSinceJob = 0;
    this.resetAgg();
  }

  /**
   * Pixels still iterating at the limit are drawn as interior. If a noticeable
   * share of escapes happened in the upper half of the range, more of those
   * "interior" pixels are probably exterior too, so the limit doubles.
   */
  private wantsMoreIterations(): boolean {
    const a = this.agg;
    if (this.maxIter >= this.maxIterCap || a.interior === 0) return false;
    // Nothing escaped: we cannot tell interior from slow exterior. Keep looking,
    // but not forever: a view entirely inside the set stops at 2^17.
    if (a.escaped === 0) return this.maxIter < 1 << 17;
    return a.late > Math.max(16, a.escaped * 0.002);
  }

  progress(): EscapeProgress {
    const w = this.width;
    const h = this.height;
    const pixels = w * h;
    const half = Math.ceil(w / 2) * Math.ceil(h / 2);
    const nCoarse = Math.ceil(w / 4) * Math.ceil(h / 4);
    const nMid = half - nCoarse;
    const nFine = pixels - half;
    const pr = this.prog;
    const a = this.agg;
    const now = performance.now();
    let known = a.unfinished >= 0 && pr.coarse >= 0;
    let [left, leftCoarse, leftMid] = [a.unfinished, pr.coarse, pr.mid];
    // While the view keeps changing, readbacks always belong to an earlier
    // frame's view; report those per-frame figures instead of a stuck 0%.
    const moving = !known && !this.converged && now - pr.viewStart < 400 && pr.anyUnfinished >= 0;
    if (moving) {
      known = true;
      [left, leftCoarse, leftMid] = [pr.anyUnfinished, pr.anyCoarse, pr.anyMid];
    }
    const frac = (l: number, n: number) => (n > 0 ? 1 - Math.min(Math.max(l, 0), n) / n : 1);
    const coarse = this.converged ? 1 : known ? frac(leftCoarse, nCoarse) : 0;
    const mid = this.converged ? 1 : known ? frac(leftMid, nMid) : 0;
    const fine = this.converged ? 1 : known ? frac(left - leftCoarse - leftMid, nFine) : 0;
    const sample = this.converged ? 1 : known ? frac(left, pixels) : 0;
    const overall = this.converged ? 1 : Math.min(1, (this.samplesDone + sample) / this.sampleTarget);
    const rate =
      pr.rate.length > 1 && !this.converged
        ? pr.rate.slice(1).reduce((s, r) => s + r.n, 0) / Math.max(1e-3, (pr.rate[pr.rate.length - 1].t - pr.rate[0].t) / 1000)
        : 0;
    // ETA from the trend of overall completion over the last few seconds.
    pr.trend.push({ t: now, f: overall });
    while (pr.trend.length > 2 && now - pr.trend[0].t > 3000) pr.trend.shift();
    let etaS: number | undefined;
    const first = pr.trend[0];
    if (!this.converged && first && now - first.t > 500 && overall > first.f) {
      etaS = ((1 - overall) * (now - first.t)) / (overall - first.f) / 1000;
    }
    return {
      coarse, mid, fine, sample, overall,
      samplesDone: this.samplesDone,
      sampleTarget: this.sampleTarget,
      converged: this.converged,
      moving,
      iterationsPerSecond: rate,
      elapsedS: ((this.converged ? pr.finishedAt : now) - pr.viewStart) / 1000,
      etaS,
      maxIter: this.maxIter,
      maxEscaped: a.maxN,
      note: pr.note,
      pixels,
    };
  }

  /** Lowers the iteration limit after zooming out (called on view changes). */
  relaxIterations(): void {
    const s = this.lastComplete;
    if (s && s.maxEscapedN > 0 && s.maxEscapedN * 4 < this.maxIter) {
      this.maxIter = Math.max(this.minIter, Math.ceil(s.maxEscapedN * 2));
    }
  }

  /** Copies the per-pixel state to the CPU (tests only; never in the frame loop). */
  async readState(): Promise<{ f: Float32Array; u: Uint32Array; width: number; height: number }> {
    const size = this.width * this.height * STATE_BYTES;
    const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.state!, 0, staging, 0, size);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return { f: new Float32Array(copy), u: new Uint32Array(copy), width: this.width, height: this.height };
  }

  /** Copies the finished samples to the CPU (tests only). */
  async readSamples(): Promise<Float32Array> {
    const size = this.width * this.height * 8 * MAX_SAMPLES;
    const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.samples!, 0, staging, 0, size);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** Current job id (tests use it to know which pixels belong to the view). */
  get jobId(): number {
    return this.job;
  }

  dispose(): void {
    this.state?.destroy();
    this.samples?.destroy();
    for (const t of this.outputs) t.destroy();
    this.counters?.destroy();
    for (const r of this.readbacks) r.buffer.destroy();
  }
}

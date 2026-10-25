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
import { interiorTerms } from "./interior";
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
  /**
   * Id of an earlier orbit this one extends: same reference, precision and
   * |δc| bound, so its points and BLA blocks over the earlier length are
   * bit-identical. Switching to it keeps every pixel's state.
   */
  continues?: number;
  /** The reference escaped or returned to 0: the orbit is complete. */
  escaped?: boolean;
  periodic?: boolean;
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
  /**
   * Perturbation modes: the reference point as an exact binary fraction
   * (re, im) / 2^frac. Enables the closed-form interior tests there.
   */
  interiorRef?: { re: bigint; im: bigint; frac: number };
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
  /** Loop steps executed per second (a BLA span counts once). */
  stepsPerSecond: number;
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
  /** Interior pixels found cycling (subset of interior). */
  cycling: number;
  /** Interior pixels proven by the closed-form tests (subset of interior). */
  proven: number;
  anomalies: number;
  rebases: number;
  blaSteps: number;
  maxEscapedN: number;
  minEscapedN: number;
  budget: number;
  maxIter: number;
  /** Escapes in the upper half of the iteration limit (they decide limit raises). */
  late: number;
  /** Only the remaining pixels are dispatched (few left). */
  listMode: boolean;
  /** Component proofs apply to the current view. */
  patches: boolean;
}

// Counters (escape.wgsl): words 0..16 count this frame's events and are
// cleared every frame; words 17.. belong to the job (see "Iteration limit").
const FRAME_WORDS = 17;
const C_MAX_ITER = 17;
const C_RAISES = 18;
const C_RAISED_NOW = 19;
const C_FRAME_MAX_ITER = 20;
const C_AGG = 21; // escaped, late, interior, cycling, proven
const C_RAISE = 26; // escaped, late, paused at the latest raise
const COUNTER_BYTES = 29 * 4;
// Coefficients of the closed-form interior tests (interior.ts).
const TERM_BYTES = 21 * 16;
// Component proofs (component.rs): one record per tile of an 8 × 8 grid.
export const PATCH_GRID = 8;
const PATCH_COUNT = PATCH_GRID * PATCH_GRID;
export const PATCH_BYTES = 32 + 15 * 16;
// Largest per-pixel budget of a full-grid pass (list mode allows more).
const FULL_BUDGET_MAX = 1 << 16;
const PARAM_BYTES = 112;
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
  // Compaction (see escape.wgsl): per-mode list pipelines, the indirect-args
  // pipeline, two index lists, their counts and the dispatch arguments.
  private buildPipelines: GPUComputePipeline[] = [];
  private listPipelines: GPUComputePipeline[] = [];
  private argsPipeline!: GPUComputePipeline;
  private argsLayout!: GPUBindGroupLayout;
  private lists: GPUBuffer[] = [];
  private listCtl!: GPUBuffer;
  private listArgs!: GPUBuffer;
  private listParams: GPUBuffer[] = [];
  private listGroups: GPUBindGroup[] = [];
  private argsGroups: GPUBindGroup[] = [];
  // decide_limit: one group per parameter buffer it may run with.
  private decideLayout!: GPUBindGroupLayout;
  private decidePipeline!: GPUComputePipeline;
  private decideGroups = new Map<GPUBuffer, GPUBindGroup>();
  /** Job whose limit and aggregates the GPU counters currently hold. */
  private countersJob = -1;
  /** Raises of the limit by the GPU seen in this job's readbacks. */
  private raisesSeen = 0;
  private listMode = false;
  private listBuild = false;
  private listCur = 0;
  private displayPipeline!: GPURenderPipeline;
  private params: GPUBuffer[] = [];
  private displayParams!: GPUBuffer;
  private counters!: GPUBuffer;
  private readbacks: Readback[] = [];
  private dummy!: GPUBuffer;
  private terms!: GPUBuffer;
  // Component proofs for the current view (escape.wgsl, "Proven interior").
  private patches!: GPUBuffer;
  private patchesOn = false;
  private retestPending = false;
  private retestPipelines: GPUComputePipeline[] = [];
  private termsKey = "";
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

  /**
   * A longer continuation of a truncated orbit will be supplied when needed
   * (see GpuOrbit.continues): pixels reaching the orbit's end wait for it.
   */
  orbitExtends = false;
  /** Iteration limit; adapted from statistics unless pinned. */
  maxIter = 1000;
  minIter = 500;
  maxIterCap = 1 << 24;
  sampleTarget = MAX_SAMPLES;
  /**
   * Switch to dispatching only unfinished pixels (compaction) once fewer than
   * this fraction of the pixels is still running; 0 disables it.
   */
  compactBelow = 0.25;
  /**
   * Closed-form interior tests (escape.wgsl, "Proven interior"); a proven
   * pixel iterates trapIters steps for its shading.
   */
  provenInterior = true;
  trapIters = 1024;
  /**
   * GPU time per frame for iteration, as a share of the measured frame
   * interval: more while the view is still (only the image refines), less
   * while it moves, and never above movingMaxMs then.
   */
  stillShare = 0.72;
  movingShare = 0.55;
  movingMaxMs = 9;
  private frameMsAvg = 16.7;
  /** density: palette cycles per doubling of (iterations − sMin)/bandScale. */
  look = { offset: 0, density: 0.32, glow: 0.6, sMin: 0, bandScale: 6, interiorGlow: 0.3, relief: 0.2 };

  private agg = { escaped: 0, interior: 0, anomalies: 0, rebases: 0, blaSteps: 0, maxN: 0, minN: Infinity, late: 0, unfinished: -1, cycling: 0, proven: 0 };
  /** Aggregated statistics of the most recently completed sample. */
  lastComplete?: EscapeStats;
  /** Iterations advanced since init (BLA spans counted in full); benchmarks. */
  iterationsTotal = 0;
  /** Times the iteration limit was raised (benchmarks). */
  restarts = 0;
  /** The limit raises of sample 0 since init and what they were decided on (benchmarks). */
  raiseLog: { from: number; to: number; escaped: number; late: number; paused: number }[] = [];
  /** Loop steps actually executed since init (a BLA span counts once); benchmarks. */
  stepsTotal = 0;
  /** Frames that recorded iteration work since init. */
  computeFrames = 0;
  /**
   * Frames recorded after every pixel of a sample had stopped, before the CPU
   * acted on it (benchmarks): `limit` when it then raised the iteration limit,
   * `sample` when it moved on to the next sample. `orbit`: frames whose only
   * unfinished pixels were waiting for a longer reference orbit.
   */
  waitFrames = { limit: 0, sample: 0, orbit: 0 };

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
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.argsLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: rw },
      ],
    });
    const module = shaderModule(device, "escape", wgsl(escapeSrc));
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    const pipe = (entryPoint: string, mode: number) =>
      device.createComputePipeline({ label: `escape-${entryPoint}-${mode}`, layout, compute: { module, entryPoint, constants: { MODE: mode } } });
    this.pipelines = [0, 1, 2, 3].map((mode) => pipe("iterate", mode));
    this.listPipelines = [0, 1, 2, 3].map((mode) => pipe("iterate_list", mode));
    this.buildPipelines = [0, 1, 2, 3].map((mode) => pipe("build_list", mode));
    this.argsPipeline = device.createComputePipeline({
      label: "escape-list-args",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.argsLayout] }),
      compute: { module, entryPoint: "list_args" },
    });
    this.listCtl = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.listArgs = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.decideLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: rw },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: rw },
      ],
    });
    this.decidePipeline = device.createComputePipeline({
      label: "escape-decide-limit",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.decideLayout] }),
      compute: { module, entryPoint: "decide_limit" },
    });
    this.decideGroups.clear();
    this.listParams = [0, 1].map(() => device.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    this.argsGroups = this.listParams.map((p) =>
      device.createBindGroup({
        layout: this.argsLayout,
        entries: [
          { binding: 0, resource: { buffer: p } },
          { binding: 9, resource: { buffer: this.listCtl } },
          { binding: 10, resource: { buffer: this.listArgs } },
        ],
      }),
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
    this.terms = device.createBuffer({ size: TERM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.patches = device.createBuffer({ size: PATCH_COUNT * PATCH_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.retestPipelines = [0, 1, 2, 3].map((mode) => pipe("retest", mode));
    this.patchesOn = false;
    this.termsKey = "";
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
    for (const l of this.lists) l.destroy();
    const device = this.device;
    this.lists = [0, 1].map(() => device.createBuffer({ size: Math.max(64, w * h * 4), usage: GPUBufferUsage.STORAGE }));
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
    rate: [] as { t: number; n: number; s: number }[],
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
    // Component proofs belong to a view.
    this.patchesOn = false;
    this.retestPending = false;
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
    this.leaveListMode();
    this.resetAgg();
  }

  private resetAgg(): void {
    this.agg = { escaped: 0, interior: 0, anomalies: 0, rebases: 0, blaSteps: 0, maxN: 0, minN: Infinity, late: 0, unfinished: -1, cycling: 0, proven: 0 };
  }

  /**
   * Sets the view for this frame. `history` maps into the previous frame when
   * the view changed, or is null when the view is unchanged.
   */
  setView(view: EscapeView, history: HistoryMap | null): void {
    const prev = this.view;
    this.view = view;
    const extended = view.orbit?.continues !== undefined && view.orbit.continues === prev?.orbit?.id;
    if (history || !prev || prev.mode !== view.mode || (prev.orbit !== view.orbit && !extended)) {
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
      cycling: this.agg.cycling,
      proven: this.agg.proven,
      anomalies: this.agg.anomalies,
      rebases: this.agg.rebases,
      blaSteps: this.agg.blaSteps,
      maxEscapedN: this.agg.maxN,
      minEscapedN: this.agg.minN,
      budget: this.budget,
      maxIter: this.maxIter,
      late: this.agg.late,
      listMode: this.listMode,
      patches: this.patchesOn,
    };
  }

  private computeGroupsFor(orbit?: GpuOrbit): GPUBindGroup[] {
    const key = `${this.width}x${this.height}:${orbit?.id ?? "none"}`;
    if (key === this.computeGroupKey) return this.computeGroups;
    this.computeGroupKey = key;
    const group = (p: GPUBuffer) =>
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
          { binding: 7, resource: { buffer: this.lists[0] } },
          { binding: 8, resource: { buffer: this.lists[1] } },
          { binding: 9, resource: { buffer: this.listCtl } },
          { binding: 11, resource: { buffer: this.terms } },
          { binding: 13, resource: { buffer: this.patches } },
        ],
      });
    this.computeGroups = this.params.map(group);
    this.listGroups = this.listParams.map(group);
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
      this.computeFrames++;
      const groups = this.computeGroupsFor(view.orbit);
      const jit = JITTER[this.sample % JITTER.length];
      const writeParams = (target: GPUBuffer, step: number, listIn: number) => {
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
        u[14] = this.maxIterCap;
        // Budget of the finest step; the shader gives coarser steps (fewer
        // pixels) 2× and 4× as much.
        u[15] = Math.max(16, Math.round(this.budget));
        u[16] = view.orbit?.len ?? 0;
        u[17] = view.orbit?.levels ?? 0;
        u[18] = this.sample;
        const o = view.orbit;
        u[20] = this.orbitExtends && o && !o.escaped && !o.periodic ? 1 : 0;
        u[19] = listIn;
        const ref = view.interiorRef;
        if (ref) {
          f[21] = Number(ref.re) / 2 ** ref.frac;
          f[22] = Number(ref.im) / 2 ** ref.frac;
        }
        u[23] = this.provenInterior && (view.mode === 0 || ((view.mode === 2 || view.mode === 3) && ref)) ? this.trapIters : 0;
        u[24] = this.raisesSeen;
        u[25] = Math.max(16, Math.round(Math.min(this.budget, FULL_BUDGET_MAX)));
        u[26] = this.listMode ? 1 : 0;
        u[27] = this.patchesOn ? 1 : 0;
        device.queue.writeBuffer(target, 0, buf);
      };
      // Clear this frame's counters; when a job starts, also set its limit
      // and clear its aggregates.
      const words = this.countersJob === this.job ? FRAME_WORDS : COUNTER_BYTES / 4;
      const cw = new Uint32Array(words);
      cw[6] = 0xffffffff;
      if (words > FRAME_WORDS) {
        cw[C_MAX_ITER] = cw[C_FRAME_MAX_ITER] = this.maxIter;
        this.countersJob = this.job;
        this.raisesSeen = 0;
      }
      device.queue.writeBuffer(this.counters, 0, cw);
      this.writeTerms(view);
      for (let step = 0; step < 3; step++) writeParams(this.params[step], step, 0);
      if (this.retestPending) {
        this.retestPending = false;
        const pass = enc.beginComputePass({ label: "escape-retest" });
        pass.setPipeline(this.retestPipelines[view.mode]);
        pass.setBindGroup(0, groups[2]);
        pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
        pass.end();
      }
      if (!this.listMode) {
        const pass = enc.beginComputePass({ label: "escape", timestampWrites: frame.timer.writes("escape") });
        pass.setPipeline(this.pipelines[view.mode]);
        for (let step = 0; step < 3; step++) {
          const stride = 1 << (2 - step);
          pass.setBindGroup(0, groups[step]);
          pass.dispatchWorkgroups(Math.ceil(Math.ceil(w / stride) / 8), Math.ceil(Math.ceil(h / stride) / 8));
        }
        pass.setPipeline(this.decidePipeline);
        pass.setBindGroup(0, this.decideGroup(this.params[2]));
        pass.dispatchWorkgroups(1);
        pass.end();
      } else {
        // listGroups[k] reads list k and appends survivors to list 1 − k.
        const cur = this.listCur;
        const other = 1 - cur;
        writeParams(this.listParams[0], 2, 0);
        writeParams(this.listParams[1], 2, 1);
        if (this.listBuild) {
          this.listBuild = false;
          enc.clearBuffer(this.listCtl, cur * 4, 4);
          const groupsTotal = Math.ceil((w * h) / 64);
          const x = Math.min(groupsTotal, 32768);
          const pass = enc.beginComputePass({ label: "escape-build", timestampWrites: frame.timer.writes("escape") });
          pass.setPipeline(this.buildPipelines[view.mode]);
          pass.setBindGroup(0, this.listGroups[cur]);
          pass.dispatchWorkgroups(x, Math.ceil(groupsTotal / x));
          pass.end();
        }
        enc.clearBuffer(this.listCtl, other * 4, 4);
        const pass = enc.beginComputePass({ label: "escape", timestampWrites: frame.timer.writes("escape") });
        pass.setPipeline(this.argsPipeline);
        pass.setBindGroup(0, this.argsGroups[cur]);
        pass.dispatchWorkgroups(1);
        pass.setPipeline(this.listPipelines[view.mode]);
        pass.setBindGroup(0, this.listGroups[cur]);
        pass.dispatchWorkgroupsIndirect(this.listArgs, cur * 16);
        // A raise of the limit refills the next list from the full grid.
        pass.setPipeline(this.decidePipeline);
        pass.setBindGroup(0, this.decideGroup(this.listParams[cur]));
        pass.dispatchWorkgroups(1);
        pass.setPipeline(this.buildPipelines[view.mode]);
        pass.setBindGroup(0, this.listGroups[other]);
        pass.dispatchWorkgroupsIndirect(this.listArgs, 32);
        pass.end();
        this.listCur = other;
      }
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
    df[20] = this.look.interiorGlow; df[21] = this.look.relief;
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

  private decideGroup(params: GPUBuffer): GPUBindGroup {
    let g = this.decideGroups.get(params);
    if (!g) {
      g = this.device.createBindGroup({
        layout: this.decideLayout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 3, resource: { buffer: this.counters } },
          { binding: 9, resource: { buffer: this.listCtl } },
          { binding: 10, resource: { buffer: this.listArgs } },
        ],
      });
      this.decideGroups.set(params, g);
    }
    return g;
  }

  private adaptBudget(frame: FrameContext): void {
    if (frame.dt > 0) this.frameMsAvg += (Math.min(frame.dt * 1000, 50) - this.frameMsAvg) * 0.1;
    const ms = frame.timer.passMs.get("escape");
    if (ms === undefined || this.converged) return;
    const still = performance.now() - this.prog.viewStart > 300;
    const target = still ? this.frameMsAvg * this.stillShare : Math.min(this.movingMaxMs, this.frameMsAvg * this.movingShare);
    // Multiplicative controller towards the target; clamp per-frame change.
    // Few pixels remain in list mode, so each may take a much larger share.
    const ratio = Math.max(0.5, Math.min(1.5, target / Math.max(ms, 0.05)));
    this.budget = Math.max(32, Math.min(this.listMode ? 1 << 22 : FULL_BUDGET_MAX, this.budget * ratio));
  }

  /** Back to full-grid passes: never start those with a list-mode budget. */
  private leaveListMode(): void {
    this.listMode = false;
    this.budget = Math.min(this.budget, FULL_BUDGET_MAX);
  }

  private scheduleReadback(enc: GPUCommandEncoder): void {
    const rb = this.readbacks.find((r) => !r.busy);
    if (!rb) return;
    enc.copyBufferToBuffer(this.counters, 0, rb.buffer, 0, COUNTER_BYTES);
    rb.busy = true;
    const job = this.job;
    const sample = this.sample;
    const frameNo = this.computeFrames;
    queueMicrotask(() => {
      // mapAsync must be issued after submit; the frame loop submits synchronously
      // after render() returns, and microtasks run after the current task.
      rb.buffer
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const c = new Uint32Array(rb.buffer.getMappedRange().slice(0));
          rb.buffer.unmap();
          rb.busy = false;
          this.onCounters(job, sample, c, frameNo);
        })
        .catch(() => (rb.busy = false));
    });
  }

  private onCounters(job: number, sample: number, c: Uint32Array, frameNo: number): void {
    // Throughput counts all work, including frames of superseded jobs.
    const now = performance.now();
    const pr = this.prog;
    pr.rate.push({ t: now, n: c[11] * 256, s: c[14] * 4294967296 + c[13] });
    this.iterationsTotal += c[11] * 256;
    this.stepsTotal += c[14] * 4294967296 + c[13];
    while (pr.rate.length > 2 && now - pr.rate[0].t > 1000) pr.rate.shift();
    pr.anyUnfinished = c[0];
    pr.anyCoarse = c[9];
    pr.anyMid = c[10];
    if (job !== this.job || sample !== this.sample) return;
    // The GPU raised the limit (perhaps several frames ago): follow it.
    if (c[C_RAISES] > this.raisesSeen) this.noteRaises(c);
    // Frames recorded before the latest raise belong to an older limit.
    if (c[C_FRAME_MAX_ITER] !== this.maxIter && !c[C_RAISED_NOW]) return;
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
    a.anomalies += c[3];
    a.rebases += c[4];
    a.maxN = Math.max(a.maxN, c[5]);
    a.blaSteps += c[7];
    a.late = c[C_AGG + 1];
    if (c[0] !== 0 && c[0] === c[15]) this.waitFrames.orbit++;
    // Few pixels left: from now on dispatch only those.
    if (c[0] !== 0 && !this.listMode && c[0] < this.width * this.height * this.compactBelow) {
      this.listMode = true;
      this.listBuild = true;
    }
    // Job totals from the GPU (exact: every frame counted).
    a.escaped = c[C_AGG];
    a.interior = c[C_AGG + 2];
    a.cycling = c[C_AGG + 3];
    a.proven = c[C_AGG + 4];
    // No pixel running, and the GPU kept the limit: the sample is complete.
    if (c[0] !== 0 || c[C_RAISED_NOW]) return;
    const idle = this.computeFrames - frameNo;
    this.lastComplete = this.stats();
    this.waitFrames.sample += idle;
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
    this.leaveListMode();
    this.resetAgg();
  }

  /**
   * The GPU doubled the iteration limit (escape.wgsl, "Iteration limit"):
   * pixels stopped at the old limit continue from where they are, finished
   * pixels keep their results. Many pixels may rejoin at once, so the budget
   * returns to a full-grid one (the GPU caps it until this is seen).
   */
  private noteRaises(c: Uint32Array): void {
    this.raiseLog.push({ from: this.maxIter, to: c[C_MAX_ITER], escaped: c[C_RAISE], late: c[C_RAISE + 1], paused: c[C_RAISE + 2] });
    this.restarts += c[C_RAISES] - this.raisesSeen;
    this.raisesSeen = c[C_RAISES];
    this.maxIter = c[C_MAX_ITER];
    this.prog.note = `反復上限を ${this.maxIter.toLocaleString()} に引き上げて計算を継続中`;
    this.budget = Math.min(this.budget, FULL_BUDGET_MAX);
    this.agg.unfinished = -1;
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
    const perSecond = (f: (r: { n: number; s: number }) => number) =>
      pr.rate.length > 1 && !this.converged
        ? pr.rate.slice(1).reduce((s, r) => s + f(r), 0) / Math.max(1e-3, (pr.rate[pr.rate.length - 1].t - pr.rate[0].t) / 1000)
        : 0;
    const rate = perSecond((r) => r.n);
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
      stepsPerSecond: perSecond((r) => r.s),
      elapsedS: ((this.converged ? pr.finishedAt : now) - pr.viewStart) / 1000,
      etaS,
      maxIter: this.maxIter,
      maxEscaped: a.maxN,
      note: pr.note,
      pixels,
    };
  }

  /**
   * Component proofs for the current view (component.rs records, one per
   * tile), or null to drop them. Pixels already running are checked again.
   * The caller ensures they belong to the view being rendered.
   */
  setPatches(data: ArrayBuffer | null): void {
    if (!data) {
      this.patchesOn = false;
      return;
    }
    if (data.byteLength !== PATCH_COUNT * PATCH_BYTES) throw new Error("bad patch data");
    this.device.queue.writeBuffer(this.patches, 0, data);
    this.patchesOn = true;
    this.retestPending = true;
  }

  /** Coefficients of the closed-form interior tests around the view's reference point. */
  private writeTerms(view: EscapeView): void {
    const ref = view.mode === 0 ? { re: 0n, im: 0n, frac: 0 } : view.interiorRef;
    if (!ref) return;
    const key = `${ref.re}/${ref.im}/${ref.frac}`;
    if (key === this.termsKey) return;
    this.termsKey = key;
    this.device.queue.writeBuffer(this.terms, 0, interiorTerms(ref.re, ref.im, ref.frac));
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
    for (const l of this.lists) l.destroy();
    this.listCtl?.destroy();
    this.listArgs?.destroy();
    this.terms?.destroy();
    this.patches?.destroy();
    for (const r of this.readbacks) r.buffer.destroy();
  }
}

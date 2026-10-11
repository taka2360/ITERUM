// The Lorenz attractor, drawn as hundreds of thousands of particles that each
// follow the flow and leave fading light behind them.

import { OrbitCamera } from "../../core/camera3d";
import type { InputFrame } from "../../core/input";
import type { FrameContext, Phenomenon, PhenomenonMeta, ProgressInfo } from "../../core/phenomenon";
import { bigCount, int, percent } from "../../ui/format";
import { hexToRgb, type PaletteTexture } from "../../core/palettes";
import { shaderModule, type GpuContext } from "../../gpu/device";
import { wgsl } from "../../shaders";
import simSrc from "./lorenz.wgsl?raw";
import displaySrc from "./display.wgsl?raw";

const UNIFORM_BYTES = 160;
const PARTICLE_BYTES = 32;
const LEADERS = 56;
const MAX_PARTICLES = 1 << 20;

export interface LorenzSettings {
  particles: number;
  stepsPerFrame: number;
  dt: number;
  decay: number;
}

export const LORENZ_DEFAULTS: LorenzSettings = { particles: 1 << 19, stepsPerFrame: 3, dt: 0.0012, decay: 0.93 };

export class Lorenz implements Phenomenon {
  readonly meta: PhenomenonMeta = {
    id: "lorenz",
    title: "Lorenz Attractor",
    titleJa: "ローレンツ・アトラクタ",
    formula:
      '<math><mtable columnalign="left"><mtr><mtd><mover><mi>x</mi><mo>˙</mo></mover><mo>=</mo><mi>σ</mi><mo>(</mo><mi>y</mi><mo>−</mo><mi>x</mi><mo>)</mo></mtd></mtr><mtr><mtd><mover><mi>y</mi><mo>˙</mo></mover><mo>=</mo><mi>x</mi><mo>(</mo><mi>ρ</mi><mo>−</mo><mi>z</mi><mo>)</mo><mo>−</mo><mi>y</mi></mtd></mtr><mtr><mtd><mover><mi>z</mi><mo>˙</mo></mover><mo>=</mo><mi>x</mi><mi>y</mi><mo>−</mo><mi>β</mi><mi>z</mi></mtd></mtr></mtable><mspace width="1em"/><mi>σ</mi><mo>=</mo><mn>10</mn><mo>,</mo><mi>ρ</mi><mo>=</mo><mn>28</mn><mo>,</mo><mi>β</mi><mo>=</mo><mn>8</mn><mo>/</mo><mn>3</mn></math>',
    description:
      "大気の対流を極端に単純化した三つの方程式。わずかに異なる位置から出発した点はたちまち別々の道を進むのに、全体はいつも同じ蝶の翅のような形の上に留まり続ける。ここでは数十万の粒子がその流れに乗って光の跡を残している。",
    hints: ["ドラッグ — 回転", "ホイール — 近づく・離れる", "Space — 流れを止める"],
    motion: "animation",
  };

  settings: LorenzSettings = { ...LORENZ_DEFAULTS };
  exposure = 0.14;
  private device!: GPUDevice;
  private sim!: GPUComputePipeline;
  private warm!: GPUComputePipeline;
  private decayPipe!: GPUComputePipeline;
  private display!: GPURenderPipeline;
  private uniform!: GPUBuffer;
  private dispUniform!: GPUBuffer;
  private particles!: GPUBuffer;
  private accum?: GPUBuffer;
  private output?: GPUTexture;
  private simGroup?: GPUBindGroup;
  private dispGroup?: GPUBindGroup;
  private groupPal?: GPUTexture;
  private camera = Lorenz.homeCamera();
  private seeded = false;
  private preroll = 0;
  private palOffset = 0;

  constructor(private palette: () => PaletteTexture) {}

  /** Starts close to the classic view along the y axis, where both wings open. */
  private static homeCamera(): OrbitCamera {
    const c = new OrbitCamera([0, 0, 24], 78);
    c.yaw = -1.25;
    c.pitch = 0.12;
    return c;
  }

  init(gpu: GpuContext): void {
    const device = (this.device = gpu.device);
    const sim = shaderModule(device, "lorenz", wgsl(simSrc));
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      ],
    });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.sim = device.createComputePipeline({ label: "lorenz-integrate", layout: pl, compute: { module: sim, entryPoint: "integrate" } });
    this.warm = device.createComputePipeline({ label: "lorenz-warmup", layout: pl, compute: { module: sim, entryPoint: "warmup" } });
    this.decayPipe = device.createComputePipeline({ label: "lorenz-decay", layout: pl, compute: { module: sim, entryPoint: "decay" } });
    const disp = shaderModule(device, "lorenz-display", wgsl(displaySrc));
    this.display = device.createRenderPipeline({
      label: "lorenz-display",
      layout: "auto",
      vertex: { module: disp, entryPoint: "vs" },
      fragment: { module: disp, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
    });
    this.uniform = device.createBuffer({ size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dispUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.particles = device.createBuffer({ size: MAX_PARTICLES * PARTICLE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.accum = undefined;
    this.output = undefined;
    this.simGroup = this.dispGroup = undefined;
    this.seeded = false;
  }

  reset(): void {
    this.camera = Lorenz.homeCamera();
  }

  /**
   * Seeds particles near the attractor with staggered lifetimes; warm-up and
   * pre-roll (see render) then settle them into comet trails.
   */
  private seed(): void {
    const n = MAX_PARTICLES;
    const data = new Float32Array(n * 8);
    let s = 0x2545f491;
    const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    for (let i = 0; i < n; i++) {
      const o = i * 8;
      data[o] = 1 + (rnd() - 0.5) * 8;
      data[o + 1] = 1 + (rnd() - 0.5) * 8;
      data[o + 2] = 20 + (rnd() - 0.5) * 8;
      data[o + 3] = rnd() - 0.5;
      const life = 4 + 12 * rnd() * rnd();
      data[o + 4] = life * (1 + rnd()); // past its life: reborn at once
      data[o + 5] = i < LEADERS ? Infinity : life;
    }
    this.device.queue.writeBuffer(this.particles, 0, data);
  }

  update(input: InputFrame, ctx: { time: number; dt: number; paused: boolean; viewW: number; viewH: number }): void {
    const dt = Math.min(ctx.dt, 0.1);
    if (input.dragging || input.altDragging) this.camera.drag(input.dragX + input.altDragX, input.dragY + input.altDragY, dt);
    if (input.zoom !== 0) this.camera.zoom(input.zoom);
    if (input.dblclick) this.camera.zoom(1);
    this.camera.update(dt, input.dragging || input.altDragging, ctx.paused);
    if (!ctx.paused) this.palOffset += dt * 0.003;
  }

  private frameNo = 0;

  private writeUniforms(frame: FrameContext, steps: number, dt: number, count: number, frameDt = 0, draw = true): void {
    const pal = this.palette().palette;
    const bg = hexToRgb(pal.background);
    const buf = new ArrayBuffer(UNIFORM_BYTES);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    f.set(this.camera.matrix(frame.width / frame.height), 0);
    u[16] = frame.width; u[17] = frame.height; u[18] = count; u[19] = steps;
    f[20] = dt; f[21] = 10; f[22] = 28; f[23] = 8 / 3;
    f[24] = this.settings.decay; f[25] = 1; f[26] = this.palOffset; f[27] = 1 / 180;
    f[28] = bg[0]; f[29] = bg[1]; f[30] = bg[2]; f[31] = 1;
    const eye = this.camera.eye();
    f[32] = eye[0]; f[33] = eye[1]; f[34] = eye[2]; f[35] = this.camera.distance;
    u[36] = LEADERS; f[37] = frameDt; u[38] = this.frameNo++; u[39] = draw ? 1 : 0;
    this.device.queue.writeBuffer(this.uniform, 0, buf);
  }

  render(frame: FrameContext): { output: GPUTexture; status: "refining" | "converged" } {
    const device = this.device;
    const { width, height } = frame;
    if (!this.output || this.output.width !== width || this.output.height !== height) {
      this.output?.destroy();
      this.accum?.destroy();
      this.output = device.createTexture({ size: [width, height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      this.accum = device.createBuffer({ size: width * height * 12, usage: GPUBufferUsage.STORAGE });
      this.simGroup = this.dispGroup = undefined;
    }
    const pal = this.palette();
    if (!this.simGroup || this.groupPal !== pal.texture) {
      this.simGroup = device.createBindGroup({
        layout: this.sim.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: this.particles } },
          { binding: 2, resource: { buffer: this.accum! } },
          { binding: 3, resource: pal.texture.createView() },
          { binding: 4, resource: pal.sampler },
        ],
      });
      this.dispGroup = device.createBindGroup({
        layout: this.display.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.dispUniform } },
          { binding: 1, resource: { buffer: this.accum! } },
        ],
      });
      this.groupPal = pal.texture;
    }
    // Keep one frame's additions per pixel below 2^30 (see lorenz.wgsl).
    const count = Math.min(this.settings.particles, MAX_PARTICLES);
    const steps = Math.max(1, Math.min(this.settings.stepsPerFrame, Math.floor((1 << 22) / count)));
    const enc = frame.encoder;

    if (!this.seeded) {
      this.seed();
      this.writeUniforms(frame, 4000, 0.004, MAX_PARTICLES);
      const p = enc.beginComputePass({ label: "lorenz-warmup" });
      p.setPipeline(this.warm);
      p.setBindGroup(0, this.simGroup);
      p.dispatchWorkgroups(Math.ceil(MAX_PARTICLES / 256));
      p.end();
      this.seeded = true;
      this.preroll = 24;
      return { output: this.output, status: "refining" };
    }
    if (this.preroll > 0) {
      // Simulate ~1 s per frame without drawing until the comet trails exist.
      this.preroll--;
      this.writeUniforms(frame, steps * 60, this.settings.dt, count, 1.0, false);
      const p = enc.beginComputePass({ label: "lorenz-preroll" });
      p.setPipeline(this.sim);
      p.setBindGroup(0, this.simGroup);
      p.dispatchWorkgroups(Math.ceil(count / 256));
      p.end();
      return { output: this.output, status: "refining" };
    }

    const speed = Math.min(frame.dt * 60, 3);
    this.paused = frame.paused;
    if (!frame.paused) {
      this.simTime += this.settings.dt * speed * steps;
      const inst = (count * steps) / Math.max(frame.dt, 1e-3);
      this.stepRate += (inst - this.stepRate) * 0.1;
    } else {
      this.stepRate = 0;
    }
    this.writeUniforms(frame, steps, frame.paused ? 0 : this.settings.dt * speed, count, frame.paused ? 0 : frame.dt);
    const pal2 = pal.palette;
    const bg = hexToRgb(pal2.background);
    const exposure = this.exposure * (width * height) / (1920 * 1080) * (LORENZ_DEFAULTS.particles / count) * (LORENZ_DEFAULTS.stepsPerFrame / steps);
    device.queue.writeBuffer(this.dispUniform, 0, new Float32Array([0, 0, exposure, 0, bg[0], bg[1], bg[2], 1]));
    device.queue.writeBuffer(this.dispUniform, 0, new Uint32Array([width, height]));

    const sim = enc.beginComputePass({ label: "lorenz-integrate", timestampWrites: frame.timer.writes("lorenz-integrate") });
    sim.setPipeline(this.sim);
    sim.setBindGroup(0, this.simGroup);
    sim.dispatchWorkgroups(Math.ceil(count / 256));
    sim.end();

    const rp = enc.beginRenderPass({
      label: "lorenz-display",
      colorAttachments: [{ view: this.output.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }],
      timestampWrites: frame.timer.writes("lorenz-display"),
    });
    rp.setPipeline(this.display);
    rp.setBindGroup(0, this.dispGroup!);
    rp.draw(3);
    rp.end();

    const dec = enc.beginComputePass({ label: "lorenz-decay", timestampWrites: frame.timer.writes("lorenz-decay") });
    dec.setPipeline(this.decayPipe);
    dec.setBindGroup(0, this.simGroup);
    dec.dispatchWorkgroups(Math.ceil((width * height * 3) / 256));
    dec.end();
    return { output: this.output, status: "refining" };
  }

  private simTime = 0;
  private stepRate = 0;
  private paused = false;

  progress(): ProgressInfo {
    const s = this.settings;
    if (this.preroll > 0 || !this.seeded) {
      const f = this.seeded ? 1 - this.preroll / 24 : 0;
      return {
        fraction: f,
        headline: "準備中 — 粒子をアトラクタになじませています",
        rows: [{ label: "準備", value: percent(f), bar: f }],
      };
    }
    return {
      headline: this.paused ? "停止中" : "流れに沿って毎フレーム積分しています",
      rows: [
        { label: "粒子", value: `${int(s.particles)} 個（先導粒子 56 と、その後を追う粒子）` },
        { label: "積分", value: `RK4 · ${s.stepsPerFrame} ステップ/フレーム · Δt ${s.dt}` },
        { label: "積分速度", value: `${bigCount(this.stepRate)} 粒子ステップ/秒` },
        { label: "シミュレーション時間", value: `t = ${this.simTime.toFixed(1)}` },
        { label: "残光", value: `1 フレームごとに ${(s.decay * 100).toFixed(0)}% に減衰` },
      ],
    };
  }

  debug(): Record<string, string> {
    const s = this.settings;
    return {
      particles: `${s.particles.toLocaleString()} × RK4 ${s.stepsPerFrame} steps/frame (dt ${s.dt})`,
      decay: `${s.decay}`,
      camera: `yaw ${this.camera.yaw.toFixed(2)} pitch ${this.camera.pitch.toFixed(2)} dist ${this.camera.distance.toFixed(1)}`,
    };
  }

  dispose(): void {
    this.output?.destroy();
    this.accum?.destroy();
    this.particles?.destroy();
  }
}

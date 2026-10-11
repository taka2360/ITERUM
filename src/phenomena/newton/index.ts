// Newton fractals: basins of attraction of Newton's method for polynomials
// whose roots drift slowly, so the boundaries between basins keep weaving.

import { Camera2D } from "../../core/camera2d";
import type { InputFrame } from "../../core/input";
import type { FrameContext, Phenomenon, PhenomenonMeta } from "../../core/phenomenon";
import { hexToRgb, type PaletteTexture } from "../../core/palettes";
import { shaderModule, type GpuContext } from "../../gpu/device";
import { wgsl } from "../../shaders";
import src from "./newton.wgsl?raw";

const UNIFORM_BYTES = 128;

interface Preset {
  name: string;
  /** Root count; roots sit on a circle, with individual slow wobble. */
  n: number;
  radius: number;
  wobble: number;
}

const PRESETS: Preset[] = [
  { name: "z³ − 1", n: 3, radius: 1, wobble: 0.16 },
  { name: "z⁴ − 1", n: 4, radius: 1, wobble: 0.12 },
  { name: "z⁵ − 1", n: 5, radius: 1, wobble: 0.1 },
  { name: "z⁶ − 1", n: 6, radius: 1, wobble: 0.08 },
];

export class Newton implements Phenomenon {
  readonly meta: PhenomenonMeta = {
    id: "newton",
    title: "Newton Fractal",
    titleJa: "ニュートン・フラクタル",
    formula:
      '<math><msub><mi>z</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>z</mi><mi>n</mi></msub><mo>−</mo><mfrac><mrow><mi>p</mi><mo>(</mo><msub><mi>z</mi><mi>n</mi></msub><mo>)</mo></mrow><mrow><msup><mi>p</mi><mo>′</mo></msup><mo>(</mo><msub><mi>z</mi><mi>n</mi></msub><mo>)</mo></mrow></mfrac><mo>,</mo><mspace width="0.6em"/><mi>p</mi><mo>(</mo><mi>z</mi><mo>)</mo><mo>=</mo><munder><mo>∏</mo><mi>k</mi></munder><mo>(</mo><mi>z</mi><mo>−</mo><msub><mi>r</mi><mi>k</mi></msub><mo>)</mo></math>',
    description:
      "方程式の根を探すニュートン法を、平面のあらゆる点から出発させる。どの根にたどり着くかで塗り分けると、領域の境目には三つ以上の色が無限に入れ子になって接する。根はゆっくりと巡り、境界は絶えず編み直される。",
    hints: ["ドラッグ — 移動", "ホイール — 拡大・縮小", "N — 根の数を変える", "Space — 動きを止める"],
    motion: "animation",
  };

  private device!: GPUDevice;
  private pipeline!: GPURenderPipeline;
  private uniform!: GPUBuffer;
  private group?: GPUBindGroup;
  private groupPal?: GPUTexture;
  private output?: GPUTexture;
  private camera = new Camera2D(0, 0, 0);
  private needsHome = true;
  private preset = 0;
  private phase = 0;
  private zoomImpulse = 0;
  private zoomAnchor: [number, number] = [0, 0];
  private velocity: [number, number] = [0, 0];
  private lastKey = "";

  constructor(private palette: () => PaletteTexture) {}

  init(gpu: GpuContext): void {
    this.device = gpu.device;
    const module = shaderModule(gpu.device, "newton", wgsl(src));
    this.pipeline = gpu.device.createRenderPipeline({
      label: "newton",
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
    });
    this.uniform = gpu.device.createBuffer({ size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.group = undefined;
    this.output = undefined;
    this.lastKey = "";
  }

  reset(): void {
    this.needsHome = true;
    this.velocity = [0, 0];
    this.zoomImpulse = 0;
  }

  onKey(key: string): boolean {
    if (key === "n" || key === "N") {
      this.preset = (this.preset + 1) % PRESETS.length;
      this.updateFormulaName();
      return true;
    }
    return false;
  }

  private updateFormulaName(): void {
    const n = PRESETS[this.preset].n;
    const el = document.querySelector("#caption .ja");
    if (el && document.querySelector("#caption .en")?.textContent === this.meta.title) {
      el.textContent = `${this.meta.titleJa} · 根 ${n} 個`;
    }
  }

  setActive(active: boolean): void {
    if (active) setTimeout(() => this.updateFormulaName(), 600);
  }

  update(input: InputFrame, ctx: { time: number; dt: number; paused: boolean; viewW: number; viewH: number }): void {
    const dt = Math.min(ctx.dt, 0.1);
    if (this.needsHome) {
      this.camera = new Camera2D(0, 0, Math.log2(4.4 / Math.max(1, Math.min(ctx.viewW, ctx.viewH * 1.5))));
      this.needsHome = false;
    }
    const cam = this.camera;
    if (input.dragging) {
      cam.pan(input.dragX, input.dragY);
      const a = 1 - Math.exp(-dt * 30);
      this.velocity[0] += (input.dragX / Math.max(dt, 1e-3) - this.velocity[0]) * a;
      this.velocity[1] += (input.dragY / Math.max(dt, 1e-3) - this.velocity[1]) * a;
    } else if (Math.hypot(...this.velocity) > 2) {
      cam.pan(this.velocity[0] * dt, this.velocity[1] * dt);
      const k = Math.exp(-dt * 4.5);
      this.velocity = [this.velocity[0] * k, this.velocity[1] * k];
    }
    if (input.zoom !== 0) {
      this.zoomImpulse += input.zoom * 0.5;
      this.zoomAnchor = [input.pointerX - ctx.viewW / 2, input.pointerY - ctx.viewH / 2];
    }
    if (input.dblclick) {
      this.zoomImpulse += 1.5;
      this.zoomAnchor = [input.dblclick.x - ctx.viewW / 2, input.dblclick.y - ctx.viewH / 2];
    }
    if (Math.abs(this.zoomImpulse) > 1e-4) {
      const step = this.zoomImpulse * (1 - Math.exp(-dt * 7));
      this.zoomImpulse -= step;
      cam.zoomAt(this.zoomAnchor[0], this.zoomAnchor[1], step);
    }
    const minScale = Math.log2(Math.max(Math.abs(cam.cx), Math.abs(cam.cy), 0.5)) - 18;
    if (cam.log2Scale < minScale) {
      cam.log2Scale += (minScale - cam.log2Scale) * (1 - Math.exp(-dt * 6));
      this.zoomImpulse = Math.min(this.zoomImpulse, 0);
    }
    const maxScale = Math.log2(10 / Math.min(ctx.viewW, ctx.viewH));
    if (cam.log2Scale > maxScale) cam.log2Scale = maxScale;
    if (!ctx.paused) this.phase += dt;
  }

  private roots(): [number, number][] {
    const p = PRESETS[this.preset];
    const t = this.phase;
    const out: [number, number][] = [];
    for (let k = 0; k < p.n; k++) {
      // Each root has its own slow wobble; the whole set turns very slowly.
      const a = (2 * Math.PI * k) / p.n + t * 0.02 + p.wobble * Math.sin(t * (0.07 + 0.013 * k) + k * 1.7);
      const r = p.radius * (1 + p.wobble * 0.8 * Math.sin(t * (0.05 + 0.011 * k) + k * 2.3));
      out.push([r * Math.cos(a), r * Math.sin(a)]);
    }
    return out;
  }

  render(frame: FrameContext): { output: GPUTexture; status: "refining" | "converged" } {
    const device = this.device;
    if (!this.output || this.output.width !== frame.width || this.output.height !== frame.height) {
      this.output?.destroy();
      this.output = device.createTexture({
        size: [frame.width, frame.height],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.lastKey = "";
    }
    const pal = this.palette();
    if (!this.group || this.groupPal !== pal.texture) {
      this.group = device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: pal.texture.createView() },
          { binding: 2, resource: pal.sampler },
        ],
      });
      this.groupPal = pal.texture;
    }
    const cam = this.camera;
    const roots = this.roots();
    const samples = frame.pixelRatio > 1.01 ? 1 : 4;
    const relax = 1 + 0.06 * Math.sin(this.phase * 0.031);
    const palOffset = this.phase * 0.002;
    const interior = hexToRgb(pal.palette.interior);
    const key = JSON.stringify([cam.cx, cam.cy, cam.log2Scale, roots, samples, relax, palOffset, pal.index, frame.width, frame.height]);
    if (key !== this.lastKey) {
      this.lastKey = key;
      const buf = new ArrayBuffer(UNIFORM_BYTES);
      const u = new Uint32Array(buf);
      const f = new Float32Array(buf);
      u[0] = frame.width; u[1] = frame.height; u[2] = roots.length; u[3] = samples;
      f[4] = cam.cx; f[5] = cam.cy; f[6] = 2 ** (cam.log2Scale + Math.log2(frame.pixelRatio)); f[7] = relax;
      f[8] = palOffset; u[9] = 64; f[10] = 0.6;
      roots.forEach(([x, y], k) => { f[12 + 2 * k] = x; f[13 + 2 * k] = y; });
      f[28] = interior[0]; f[29] = interior[1]; f[30] = interior[2]; f[31] = 1;
      device.queue.writeBuffer(this.uniform, 0, buf);
      const pass = frame.encoder.beginRenderPass({
        label: "newton",
        colorAttachments: [{ view: this.output.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }],
        timestampWrites: frame.timer.writes("newton"),
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.group);
      pass.draw(3);
      pass.end();
    }
    return { output: this.output, status: "converged" };
  }

  debug(): Record<string, string> {
    return { polynomial: PRESETS[this.preset].name, roots: this.roots().map(([x, y]) => `${x.toFixed(3)}${y < 0 ? "" : "+"}${y.toFixed(3)}i`).join(" ") };
  }

  dispose(): void {
    this.output?.destroy();
  }
}

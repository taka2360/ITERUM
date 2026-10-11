// Julia sets: the same iteration as the Mandelbrot set, with c held fixed and
// z0 taken from the plane. The parameter c drifts slowly along the edge of the
// Mandelbrot set, where Julia sets are most intricate, so the form breathes.

import { Camera2D } from "../../core/camera2d";
import { feFromLog2 } from "../../core/floatexp";
import type { InputFrame } from "../../core/input";
import type { FrameContext, Phenomenon, PhenomenonMeta } from "../../core/phenomenon";
import type { PaletteTexture } from "../../core/palettes";
import type { GpuContext } from "../../gpu/device";
import { EscapeEngine, MODE_JULIA, type HistoryMap } from "../escape/engine";

/** Point on the main cardioid's boundary at angle θ, pushed outward by `r` (1 = boundary). */
export function cardioid(theta: number, r: number): [number, number] {
  // c = e^{iθ}/2 − e^{2iθ}/4
  const x = Math.cos(theta) / 2 - Math.cos(2 * theta) / 4;
  const y = Math.sin(theta) / 2 - Math.sin(2 * theta) / 4;
  return [x * r, y * r];
}

export class Julia implements Phenomenon {
  readonly meta: PhenomenonMeta = {
    id: "julia",
    title: "Julia Set",
    titleJa: "ジュリア集合",
    formula:
      '<math><msub><mi>z</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><msubsup><mi>z</mi><mi>n</mi><mn>2</mn></msubsup><mo>+</mo><mi>c</mi><mo>,</mo><mspace width="0.6em"/><mi>c</mi><mo>=</mo><mtext class="c-value"></mtext></math>',
    description:
      "マンデルブロ集合と同じ式で、c を固定し出発点 z を動かしたときに逃げない点の集まり。c がマンデルブロ集合の縁をゆっくり巡るにつれ、形はつながった島から塵のような星屑へと姿を変える。",
    hints: ["ドラッグ — 移動", "ホイール — 拡大・縮小", "右ドラッグ — c を動かす", "Space — 変化を止める"],
  };

  private engine: EscapeEngine;
  private camera = new Camera2D(0, 0, 0);
  private theta = 2.2;
  private c: [number, number] = [0, 0];
  private manualOffset: [number, number] = [0, 0];
  private zoomImpulse = 0;
  private zoomAnchor: [number, number] = [0, 0];
  private velocity: [number, number] = [0, 0];
  private prev?: { cx: number; cy: number; s: number; pr: number; c: [number, number] };
  private needsHome = true;
  private drift = 0;

  constructor(palette: () => PaletteTexture) {
    this.engine = new EscapeEngine(palette);
    this.engine.minIter = 400;
    this.engine.maxIter = 600;
    this.engine.maxIterCap = 1 << 16;
    this.engine.look.density = 0.16;
    this.engine.look.glow = 0.5;
  }

  init(gpu: GpuContext): void {
    this.engine.init(gpu);
    this.prev = undefined;
  }

  reset(): void {
    this.needsHome = true;
    this.manualOffset = [0, 0];
    this.velocity = [0, 0];
    this.zoomImpulse = 0;
  }

  update(input: InputFrame, ctx: { time: number; dt: number; paused: boolean; viewW: number; viewH: number }): void {
    const dt = Math.min(ctx.dt, 0.1);
    if (this.needsHome) {
      this.camera = new Camera2D(0, 0, Math.log2(3.3 / Math.max(1, Math.min(ctx.viewW, ctx.viewH * 1.5))));
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
    // f32 iteration: stop gently before pixels become indistinguishable.
    const minScale = Math.log2(Math.max(Math.abs(cam.cx), Math.abs(cam.cy), 0.5)) - 19;
    if (cam.log2Scale < minScale) {
      cam.log2Scale += (minScale - cam.log2Scale) * (1 - Math.exp(-dt * 6));
      this.zoomImpulse = Math.min(this.zoomImpulse, 0);
    }
    const maxScale = Math.log2(8 / Math.min(ctx.viewW, ctx.viewH));
    if (cam.log2Scale > maxScale) cam.log2Scale = maxScale;

    if (input.altDragging) {
      this.manualOffset[0] += input.altDragX * 0.0008;
      this.manualOffset[1] -= input.altDragY * 0.0008;
    }
    if (!ctx.paused) {
      // One revolution every ~3 minutes, with a slow radial breath.
      this.theta += dt * 0.035;
      this.drift += dt * 0.006;
    }
    const breath = 1.0 + 0.045 * Math.sin(this.theta * 2.3) - 0.02;
    const [x, y] = cardioid(this.theta, breath);
    this.c = [x + this.manualOffset[0], y + this.manualOffset[1]];
    const el = document.querySelector(".c-value");
    if (el) el.textContent = `${this.c[0].toFixed(4)} ${this.c[1] < 0 ? "−" : "+"} ${Math.abs(this.c[1]).toFixed(4)}i`;
  }

  render(frame: FrameContext): { output: GPUTexture; status: "refining" | "converged" } {
    const cam = this.camera;
    const s = cam.log2Scale + Math.log2(frame.pixelRatio);
    let history: HistoryMap | null = null;
    const p = this.prev;
    if (!p || p.pr !== frame.pixelRatio) {
      history = { scale: 1, offset: [0, 0] };
    } else if (p.cx !== cam.cx || p.cy !== cam.cy || p.s !== s || p.c[0] !== this.c[0] || p.c[1] !== this.c[1]) {
      const sp = 2 ** p.s;
      history = { scale: 2 ** (s - p.s), offset: [(cam.cx - p.cx) / sp, -(cam.cy - p.cy) / sp] };
    }
    this.prev = { cx: cam.cx, cy: cam.cy, s, pr: frame.pixelRatio, c: [...this.c] };
    this.engine.look.offset = this.drift;
    this.engine.setView(
      { mode: MODE_JULIA, spacing: feFromLog2(s), center: [cam.cx, cam.cy], cmr: [0, 0], juliaC: this.c },
      history,
    );
    return this.engine.render(frame);
  }

  debug(): Record<string, string> {
    const s = this.engine.stats();
    return {
      c: `${this.c[0].toFixed(6)}, ${this.c[1].toFixed(6)}`,
      iterations: `${s.maxIter} (max escaped ${s.maxEscapedN})`,
      samples: `${s.samplesDone}/${s.sampleTarget}`,
      budget: `${Math.round(s.budget)} it/px/frame`,
    };
  }

  dispose(): void {
    this.engine.dispose();
  }
}

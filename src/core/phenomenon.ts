// The contract every phenomenon implements. Adding a phenomenon means one
// folder under src/phenomena plus one line in src/core/registry.ts.

import type { GpuContext } from "../gpu/device";
import type { GpuTimer } from "../gpu/timing";
import type { InputFrame } from "./input";
import type { Palette } from "./palettes";

export interface PhenomenonMeta {
  id: string;
  /** Display title, e.g. "Mandelbrot Set". */
  title: string;
  /** Japanese caption name. */
  titleJa: string;
  /** Formula as HTML (MathML allowed). */
  formula: string;
  /** One or two sentences shown on demand. */
  description: string;
  /** Short interaction hints shown in the help overlay. */
  hints: string[];
  /**
   * What the play control does: "animation" pauses/resumes the phenomenon's
   * own motion; "voyage" starts/stops an automatic journey (setPlaying).
   */
  motion: "animation" | "voyage";
}

export interface FrameContext {
  gpu: GpuContext;
  encoder: GPUCommandEncoder;
  timer: GpuTimer;
  /** Seconds since start, paused time excluded. */
  time: number;
  dt: number;
  paused: boolean;
  palette: Palette;
  /** Internal resolution chosen by the quality governor. */
  width: number;
  height: number;
  /** Device pixels per internal pixel. */
  pixelRatio: number;
}

export type RenderStatus = "refining" | "converged";

export interface Phenomenon {
  readonly meta: PhenomenonMeta;
  /** Creates GPU resources; called again after device loss with a new context. */
  init(gpu: GpuContext): Promise<void> | void;
  /** Applies input and advances animation. Input coordinates are device pixels. */
  update(input: InputFrame, ctx: { time: number; dt: number; paused: boolean; viewW: number; viewH: number }): void;
  /** Records GPU work and returns the texture holding the finished image. */
  render(frame: FrameContext): { output: GPUTexture; status: RenderStatus };
  /** Handles a key; returns true if consumed. */
  onKey?(key: string): boolean;
  /** Key/value lines for the debug overlay. */
  debug?(): Record<string, string>;
  /** Short status line shown subtly (e.g. "computing reference"). */
  notice?(): string | undefined;
  /** Returns the view (camera) to its starting point; the simulation keeps running. */
  reset(): void;
  /** For motion "voyage": whether the journey is running, and control of it. */
  playing?(): boolean;
  setPlaying?(on: boolean): void;
  /** Called when the phenomenon becomes visible or hidden. */
  setActive?(active: boolean): void;
  dispose(): void;
}

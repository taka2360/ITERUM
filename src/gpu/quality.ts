// Quality governor: picks the internal resolution scale from measured load.
//
// Signals are kept separate: rAF interval (what the viewer feels), GPU time of
// the fixed-cost passes (timestamp-query, when available), and CPU time spent
// building the frame. Elastic work, such as escape-time iteration that simply
// fills whatever budget it gets, is excluded by the caller. Decisions use the
// median of 0.5 s windows plus hysteresis so the resolution never flickers.

export type QualityMode = "auto" | "low" | "medium" | "high";

const WINDOW_MS = 500;
const STEP = 0.125;
const MIN_SCALE = 0.5;
const MAX_SCALE = 1.0;
const FIXED: Record<Exclude<QualityMode, "auto">, number> = { low: 0.5, medium: 0.75, high: 1.0 };

export interface LoadSample {
  rafMs: number;
  /** GPU time of fixed-cost passes, if timestamps are available. */
  gpuMs?: number;
  cpuMs: number;
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[s.length >> 1];
}

export class QualityGovernor {
  mode: QualityMode = "auto";
  scale = MAX_SCALE;
  private windowStart = -1;
  private raf: number[] = [];
  private gpu: number[] = [];
  private cpu: number[] = [];
  private overloaded = 0;
  private underloaded = 0;
  lastWindow = { rafMs: 0, gpuMs: 0, cpuMs: 0 };

  setMode(mode: QualityMode): void {
    this.mode = mode;
    if (mode !== "auto") this.scale = FIXED[mode];
    this.overloaded = this.underloaded = 0;
  }

  /** Feeds one frame; returns true when the scale changed. */
  sample(now: number, s: LoadSample): boolean {
    if (this.windowStart < 0) this.windowStart = now;
    this.raf.push(s.rafMs);
    if (s.gpuMs !== undefined) this.gpu.push(s.gpuMs);
    this.cpu.push(s.cpuMs);
    if (now - this.windowStart < WINDOW_MS) return false;
    const raf = median(this.raf);
    const gpu = this.gpu.length ? median(this.gpu) : undefined;
    const cpu = median(this.cpu);
    this.lastWindow = { rafMs: raf, gpuMs: gpu ?? 0, cpuMs: cpu };
    this.raf = [];
    this.gpu = [];
    this.cpu = [];
    this.windowStart = now;
    if (this.mode !== "auto") return false;

    // GPU-bound only if the GPU itself is busy; a slow rAF with an idle GPU
    // (e.g. CPU stalls, background tab) must not lower the resolution.
    const heavy = gpu !== undefined ? gpu > 13 : raf > 22;
    const light = gpu !== undefined ? gpu < 6 && raf < 19 : raf < 18;
    if (heavy) {
      this.overloaded++;
      this.underloaded = 0;
    } else if (light) {
      this.underloaded++;
      this.overloaded = 0;
    } else {
      this.overloaded = this.underloaded = 0;
    }
    const before = this.scale;
    if (this.overloaded >= 2 && this.scale > MIN_SCALE) {
      this.scale = Math.max(MIN_SCALE, this.scale - STEP);
      this.overloaded = 0;
    } else if (this.underloaded >= 4 && this.scale < MAX_SCALE) {
      this.scale = Math.min(MAX_SCALE, this.scale + STEP);
      this.underloaded = 0;
    }
    return this.scale !== before;
  }
}

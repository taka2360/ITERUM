// ITERUM — a quiet museum of mathematical phenomena.

import { createGpu, type GpuContext } from "./gpu/device";
import { GpuTimer, InFlight } from "./gpu/timing";
import { QualityGovernor, type QualityMode } from "./gpu/quality";
import { Compositor } from "./gpu/compositor";
import { InputCollector } from "./core/input";
import { PaletteTexture, PALETTES } from "./core/palettes";
import { createPhenomena } from "./core/registry";
import type { Phenomenon } from "./core/phenomenon";
import { Overlay } from "./ui/overlay";
import { diag, launchMode, toggleFullscreen } from "./platform";

const TRANSITION_S = 1.6;
const QUALITY_ORDER: QualityMode[] = ["auto", "low", "medium", "high"];
const QUALITY_NAMES: Record<QualityMode, string> = { auto: "品質：自動", low: "品質：低", medium: "品質：中", high: "品質：高" };

export class App {
  gpu!: GpuContext;
  palette!: PaletteTexture;
  timer!: GpuTimer;
  inFlight!: InFlight;
  compositor!: Compositor;
  readonly governor = new QualityGovernor();
  readonly phenomena: Phenomenon[];
  readonly overlay = new Overlay();
  readonly input: InputCollector;
  index = 0;
  private outgoing?: { texture: GPUTexture };
  private transition = 1;
  private lastOutput?: GPUTexture;
  private time = 0;
  private last = performance.now();
  paused = false;
  private generation = 0;
  private recovering = false;
  private fadeIn = 0;
  private captureRequest?: (png: Uint8Array) => void;
  lastStatus: "refining" | "converged" = "refining";
  /** Fixed internal resolution (benchmarks); undefined = governor decides. */
  forcedResolution?: [number, number];
  /** Called once per frame after submission (benchmarks hook in here). */
  onFrame?: (info: { now: number; rafMs: number; cpuMs: number }) => void;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.phenomena = createPhenomena(() => this.palette);
    this.input = new InputCollector(canvas);
    this.input.onActivity = () => this.overlay.wake();
    this.overlay.setPhenomena(this.phenomena.map((p) => p.meta));
    this.overlay.onSelect = (i) => this.select(i);
    window.addEventListener("keydown", (e) => this.onKey(e));
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  async start(): Promise<boolean> {
    if (!(await this.initGpu())) return false;
    this.overlay.show(this.phenomena[0].meta, 0);
    this.phenomena[0].setActive?.(true);
    requestAnimationFrame((t) => this.frame(t));
    return true;
  }

  private async initGpu(): Promise<boolean> {
    const gpu = await createGpu(this.canvas, ++this.generation, (info) => this.onDeviceLost(info));
    if (typeof gpu === "string") {
      const reasons = {
        "no-webgpu": "この環境では WebGPU が利用できないため、描画を開始できません。WebGPU 対応の GPU とドライバをご用意ください。",
        "no-adapter": "利用できる GPU アダプタが見つかりませんでした。GPU ドライバを更新すると表示できる場合があります。",
        "device-failed": "GPU デバイスの初期化に失敗しました。",
      };
      this.overlay.unsupported(reasons[gpu]);
      diag(`gpu init failed: ${gpu}`);
      return false;
    }
    this.gpu = gpu;
    const info = gpu.adapterInfo;
    diag(`webgpu adapter vendor="${info.vendor}" architecture="${info.architecture}" description="${info.description}" timestamps=${gpu.hasTimestamps}`);
    this.palette = new PaletteTexture(gpu.device);
    this.timer = new GpuTimer(gpu.device, gpu.hasTimestamps);
    this.inFlight = new InFlight(gpu.device);
    this.compositor = new Compositor(gpu);
    for (const p of this.phenomena) await p.init(gpu);
    this.outgoing = undefined;
    this.lastOutput = undefined;
    return true;
  }

  private onDeviceLost(info: GPUDeviceLostInfo): void {
    diag(`device lost: reason=${info.reason} message=${info.message}`);
    if (this.recovering) return;
    this.recovering = true;
    this.overlay.setNotice("GPU を再初期化しています");
    // Recreate everything on a fresh device; phenomena rebuild their resources in init().
    this.initGpu().then((ok) => {
      this.recovering = false;
      this.overlay.setNotice(undefined);
      diag(`device recovered: ${ok}`);
    });
  }

  /** Forces a device loss (used by the GPU test to verify recovery). */
  loseDevice(): void {
    this.gpu.device.destroy();
  }

  get deviceGeneration(): number {
    return this.generation;
  }

  get active(): Phenomenon {
    return this.phenomena[this.index];
  }

  private resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  select(i: number): void {
    const n = this.phenomena.length;
    i = ((i % n) + n) % n;
    if (i === this.index) return;
    if (this.lastOutput) this.outgoing = { texture: this.lastOutput };
    this.active.setActive?.(false);
    this.index = i;
    this.active.setActive?.(true);
    this.transition = 0;
    this.overlay.show(this.active.meta, i);
  }

  private onKey(e: KeyboardEvent): void {
    if (e.repeat && !["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    this.overlay.wake();
    if (this.active.onKey?.(e.key)) return;
    switch (e.key) {
      case "ArrowRight": this.select(this.index + 1); break;
      case "ArrowLeft": this.select(this.index - 1); break;
      case "f": case "F": case "F11": e.preventDefault(); toggleFullscreen(); break;
      case "Escape": if (document.fullscreenElement) document.exitFullscreen(); break;
      case " ": e.preventDefault(); this.paused = !this.paused; this.overlay.toast(this.paused ? "一時停止" : "再開"); break;
      case "r": case "R": this.active.reset(); break;
      case "c": case "C": this.palette.set(this.palette.index + 1); this.overlay.toast(`配色：${PALETTES[this.palette.index].name}`); break;
      case "h": case "H": this.overlay.toggleHidden(); break;
      case "i": case "I": this.overlay.toggle("info"); break;
      case "?": case "/": this.overlay.toggle("help"); break;
      case "d": case "D": this.overlay.toggle("debug"); break;
      case "q": case "Q": {
        const next = QUALITY_ORDER[(QUALITY_ORDER.indexOf(this.governor.mode) + 1) % QUALITY_ORDER.length];
        this.governor.setMode(next);
        this.overlay.toast(QUALITY_NAMES[next]);
        break;
      }
      default:
        if (/^[1-9]$/.test(e.key)) this.select(Number(e.key) - 1);
    }
  }

  private frame(now: number): void {
    requestAnimationFrame((t) => this.frame(t));
    const rafMs = now - this.last;
    this.last = now;
    if (this.recovering) return;
    const cpu0 = performance.now();
    const dt = Math.min(rafMs / 1000, 0.1);
    if (!this.paused) this.time += dt;
    this.fadeIn = Math.min(1, this.fadeIn + dt / 2.5);

    const viewW = this.canvas.width;
    const viewH = this.canvas.height;
    const input = this.input.take();
    const active = this.active;
    active.update(input, { time: this.time, dt, paused: this.paused, viewW, viewH });

    // Backpressure: never queue more than two frames of GPU work.
    if (this.inFlight.count > 2) return;

    const scale = this.governor.scale;
    const width = this.forcedResolution?.[0] ?? Math.max(1, Math.round(viewW * scale));
    const height = this.forcedResolution?.[1] ?? Math.max(1, Math.round(viewH * scale));
    const device = this.gpu.device;
    const encoder = device.createCommandEncoder();
    this.timer.beginFrame();
    const r = active.render({
      gpu: this.gpu, encoder, timer: this.timer, time: this.time, dt, paused: this.paused,
      palette: this.palette.palette, width, height, pixelRatio: viewW / width,
    });
    this.lastOutput = r.output;
    if (this.transition < 1) {
      this.transition = Math.min(1, this.transition + dt / TRANSITION_S);
      if (this.transition >= 1) this.outgoing = undefined;
    }
    const target = this.gpu.context.getCurrentTexture();
    this.compositor.draw(encoder, this.timer, target, r.output, this.outgoing?.texture, this.transition, this.fadeIn);
    const capture = this.captureRequest ? this.recordCapture(encoder, target, this.captureRequest) : undefined;
    this.captureRequest = undefined;
    const after = this.timer.endFrame(encoder);
    device.queue.submit([encoder.finish()]);
    this.inFlight.track();
    after?.();
    capture?.();
    this.lastStatus = r.status;

    const cpuMs = performance.now() - cpu0;
    // Elastic iteration ("escape") is excluded: it fills whatever budget it is given.
    let fixedGpu: number | undefined;
    if (this.timer.enabled && this.timer.frames > 0) {
      fixedGpu = 0;
      for (const [k, v] of this.timer.passMs) if (k !== "escape") fixedGpu += v;
    }
    this.governor.sample(now, { rafMs, gpuMs: fixedGpu, cpuMs });
    this.overlay.setNotice(active.notice?.());
    if (this.overlay.isOpen("debug")) this.updateDebug(rafMs, cpuMs, width, height);
    this.onFrame?.({ now, rafMs, cpuMs });
  }

  /** PNG of the next presented frame (test harness). */
  capture(): Promise<Uint8Array> {
    return new Promise((resolve) => (this.captureRequest = resolve));
  }

  private recordCapture(enc: GPUCommandEncoder, tex: GPUTexture, done: (png: Uint8Array) => void): () => void {
    const w = tex.width;
    const h = tex.height;
    const bpr = Math.ceil((w * 4) / 256) * 256;
    const buf = this.gpu.device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
    const bgra = tex.format === "bgra8unorm";
    return () => {
      buf.mapAsync(GPUMapMode.READ).then(async () => {
        const src = new Uint8Array(buf.getMappedRange());
        const img = new ImageData(w, h);
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const i = y * bpr + x * 4;
            const o = (y * w + x) * 4;
            img.data[o] = src[i + (bgra ? 2 : 0)];
            img.data[o + 1] = src[i + 1];
            img.data[o + 2] = src[i + (bgra ? 0 : 2)];
            img.data[o + 3] = 255;
          }
        }
        buf.unmap();
        buf.destroy();
        const c = new OffscreenCanvas(w, h);
        c.getContext("2d")!.putImageData(img, 0, 0);
        const blob = await c.convertToBlob({ type: "image/png" });
        done(new Uint8Array(await blob.arrayBuffer()));
      });
    };
  }

  private updateDebug(rafMs: number, cpuMs: number, w: number, h: number): void {
    const info = this.gpu.adapterInfo;
    const passes = [...this.timer.passMs].map(([k, v]) => `${k} ${v.toFixed(2)}`).join("  ");
    const g = this.governor.lastWindow;
    this.overlay.setDebug({
      adapter: `${info.vendor || "?"} ${info.architecture || ""} ${info.description || ""}`.trim(),
      frame: `${(1000 / Math.max(rafMs, 1e-3)).toFixed(0)} fps  raf ${g.rafMs.toFixed(1)} ms  cpu ${cpuMs.toFixed(2)} ms`,
      gpu: this.timer.enabled ? `${this.timer.frameMs.toFixed(2)} ms  [${passes}]` : "timestamps unavailable",
      resolution: `${w}×${h} (${this.governor.mode}, scale ${this.governor.scale})`,
      ...(this.active.debug?.() ?? {}),
    });
  }
}

async function main(): Promise<void> {
  const canvas = document.getElementById("stage") as HTMLCanvasElement;
  const app = new App(canvas);
  (window as unknown as { iterum: App }).iterum = app;
  const ok = await app.start();
  if (!ok) return;
  const mode = await launchMode();
  if (mode === "gputest") (await import("./testing/gputest")).runGpuTests(app);
  if (mode === "shots") (await import("./testing/shots")).runShots(app);
  if (mode === "bench") (await import("./testing/bench")).runBench(app);
}

main().catch((e) => {
  console.error(e);
  diag(`fatal: ${e}`);
});

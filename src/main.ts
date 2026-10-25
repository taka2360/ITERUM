// ITERUM — a quiet museum of mathematical phenomena.

import { createGpu, type GpuContext } from "./gpu/device";
import { GpuTimer, InFlight } from "./gpu/timing";
import { QualityGovernor, type QualityMode } from "./gpu/quality";
import { Compositor } from "./gpu/compositor";
import { InputCollector } from "./core/input";
import { PaletteTexture, PALETTES } from "./core/palettes";
import { createPhenomena } from "./core/registry";
import type { Phenomenon, ProgressRow } from "./core/phenomenon";
import { Overlay } from "./ui/overlay";
import { diag, exitFullscreen, launchMode, toggleFullscreen } from "./platform";
import { keyBinding } from "./ui/keys";
import { formatZoom, locationText, log2ScaleFromZoom, log2ZoomFromScale, normalizeDecimal, parseZoom, type Location } from "./ui/location";

const TRANSITION_S = 1.6;
const QUALITY_ORDER: QualityMode[] = ["auto", "low", "medium", "high"];
const QUALITY_NAMES: Record<QualityMode, string> = { auto: "品質：自動", low: "品質：低", medium: "品質：中", high: "品質：高" };
/** GPU timer pass names as the progress panel shows them. */
const PASS_NAMES: Record<string, string> = {
  escape: "反復計算", display: "着色", composite: "合成と表示", newton: "ニュートン法",
  "lorenz-integrate": "粒子の積分", "lorenz-display": "粒子の描画", "lorenz-decay": "残光の減衰",
};

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
  private transitionS = TRANSITION_S;
  private lastOutput?: GPUTexture;
  private time = 0;
  private last = performance.now();
  /** Pause state per phenomenon (only those whose motion is "animation"). */
  private pausedBy: boolean[] = [];
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
    this.overlay.onPlay = () => this.togglePlay();
    this.overlay.onReset = () => this.resetView();
    this.overlay.onLocationGo = (re, im, zoom) => this.goTo(re, im, zoom);
    this.overlay.onLocationCopy = () => this.copyLocation();
    this.overlay.progressRefresh = () => (this.lastProgressUpdate = 0);
    window.addEventListener("keydown", (e) => this.onKey(e));
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  async start(): Promise<boolean> {
    if (!(await this.initGpu())) return false;
    this.overlay.show(this.phenomena[0].meta, 0);
    this.phenomena[0].setActive?.(true);
    this.overlay.setLocationAvailable(!!this.active.view);
    this.syncControls();
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

  get paused(): boolean {
    return this.pausedBy[this.index] ?? false;
  }

  /** Play/stop for the active phenomenon: its animation, or its voyage. */
  togglePlay(): void {
    const p = this.active;
    if (p.meta.motion === "voyage") {
      p.setPlaying?.(!p.playing?.());
      this.overlay.toast(p.playing?.() ? "Voyage" : "Voyage を止めました");
    } else {
      this.pausedBy[this.index] = !this.paused;
      this.overlay.toast(this.paused ? "停止" : "再生");
    }
    this.syncControls();
  }

  /** Returns the view to its start, dissolving from the current image. */
  resetView(): void {
    if (this.lastOutput) this.outgoing = { texture: this.lastOutput };
    this.transition = 0;
    this.transitionS = 0.9;
    this.active.reset();
    this.syncControls();
  }

  /** Centre and zoom of the active view, if it is a complex-plane view. */
  location(): Location | undefined {
    const v = this.active.view?.();
    if (!v) return undefined;
    return { re: v.re, im: v.im, log2Zoom: log2ZoomFromScale(v.log2Scale, this.canvas.width, this.canvas.height) };
  }

  /** Jumps to a typed location; returns the fields that could not be read. */
  goTo(reText: string, imText: string, zoomText: string): ("re" | "im" | "zoom")[] {
    const p = this.active;
    if (!p.setView) return [];
    const bad: ("re" | "im" | "zoom")[] = [];
    const read = <T>(k: "re" | "im" | "zoom", f: () => T): T | undefined => {
      try {
        return f();
      } catch {
        bad.push(k);
        return undefined;
      }
    };
    const re = read("re", () => normalizeDecimal(reText));
    const im = read("im", () => normalizeDecimal(imText));
    const log2Zoom = read("zoom", () => parseZoom(zoomText));
    if (bad.length) {
      this.overlay.toast("読み取れない値があります");
      return bad;
    }
    if (this.lastOutput) this.outgoing = { texture: this.lastOutput };
    this.transition = 0;
    this.transitionS = 0.9;
    p.setView(re!, im!, log2ScaleFromZoom(log2Zoom!, this.canvas.width, this.canvas.height));
    this.syncControls();
    this.overlay.toast(`移動しました — 拡大率 ${formatZoom(log2Zoom!)}`);
    return [];
  }

  private async copyLocation(): Promise<void> {
    const loc = this.location();
    if (!loc) return;
    const ok = await this.overlay.copyText(locationText(loc));
    this.overlay.toast(ok ? "座標と拡大率をコピーしました" : "コピーできませんでした");
  }

  private syncControls(): void {
    const p = this.active;
    const playing = p.meta.motion === "voyage" ? !!p.playing?.() : !this.paused;
    this.overlay.setControls(p.meta.motion, playing);
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
    this.transitionS = TRANSITION_S;
    this.overlay.show(this.active.meta, i);
    this.overlay.setLocationAvailable(!!this.active.view);
    this.syncControls();
  }

  /** Keys come from the table in ui/keys.ts; the active phenomenon sees each key first. */
  private onKey(e: KeyboardEvent): void {
    // Typing in a text field (location panel) is not a shortcut.
    if (e.target instanceof HTMLInputElement) return;
    if (e.ctrlKey || e.altKey || e.metaKey) return;
    const binding = keyBinding(e.key);
    if (e.repeat && !binding?.repeat) return;
    this.overlay.wake();
    if (this.active.onKey?.(e.key)) return;
    if (!binding?.action) return;
    switch (binding.action) {
      case "step": this.select(this.index + (e.key === "ArrowLeft" ? -1 : 1)); break;
      case "number": this.select(Number(e.key) - 1); break;
      case "play": e.preventDefault(); this.togglePlay(); break;
      case "reset": this.resetView(); break;
      case "palette": this.palette.set(this.palette.index + 1); this.overlay.toast(`配色：${PALETTES[this.palette.index].name}`); break;
      case "quality": {
        const next = QUALITY_ORDER[(QUALITY_ORDER.indexOf(this.governor.mode) + 1) % QUALITY_ORDER.length];
        this.governor.setMode(next);
        this.overlay.toast(QUALITY_NAMES[next]);
        break;
      }
      case "fullscreen": e.preventDefault(); toggleFullscreen(); break;
      case "hideUi": this.overlay.toggleHidden(); break;
      case "info": this.overlay.toggle("info"); break;
      case "help": this.overlay.toggle("help"); break;
      case "debug": this.overlay.toggle("debug"); break;
      case "progress": this.overlay.cycleProgress(); break;
      case "location":
        if (this.active.view) {
          this.overlay.toggle("location");
          this.lastLocationUpdate = 0;
        } else {
          this.overlay.toast("この現象には座標がありません");
        }
        break;
      case "close": if (!this.overlay.closePanels()) exitFullscreen(); break;
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
      this.transition = Math.min(1, this.transition + dt / this.transitionS);
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
    this.syncControls();
    if (this.overlay.isOpen("debug")) this.updateDebug(rafMs, cpuMs, width, height);
    this.updateProgress(now, rafMs, cpuMs, width, height);
    this.updateLocation(now);
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

  private lastLocationUpdate = 0;

  private updateLocation(now: number): void {
    if (!this.overlay.isOpen("location") || now - this.lastLocationUpdate < 120) return;
    this.lastLocationUpdate = now;
    const loc = this.location();
    if (loc) this.overlay.setLocation(loc.re, loc.im, formatZoom(loc.log2Zoom));
  }

  private lastProgressUpdate = 0;
  private fps = 60;
  private cpuMs = 0;

  /** Progress line every frame; the panel ~8 times per second. */
  private updateProgress(now: number, rafMs: number, cpuMs: number, w: number, h: number): void {
    this.fps += (1000 / Math.max(rafMs, 1) - this.fps) * 0.05;
    this.cpuMs += (cpuMs - this.cpuMs) * 0.05;
    const info = this.active.progress?.();
    this.overlay.setProgressLine(info?.fraction);
    if (!info || !this.overlay.isOpen("progress") || now - this.lastProgressUpdate < 120) return;
    this.lastProgressUpdate = now;
    this.overlay.setProgress(info.headline, [...info.rows, ...this.renderRows(w, h)]);
  }

  /** The "描画" group every phenomenon shares: GPU, frame rate, resolution. */
  private renderRows(w: number, h: number): ProgressRow[] {
    const group = "描画";
    const timed = this.timer.enabled && this.timer.frames > 0;
    const gpu = timed ? `${this.timer.frameMs.toFixed(1)} ms` : "計測不可";
    const rows: ProgressRow[] = [
      { group, label: "GPU 時間 / フレーム", value: `${gpu} · ${this.fps.toFixed(0)} fps · 内部解像度 ${w}×${h}` },
    ];
    if (timed) {
      const total = Math.max(this.timer.frameMs, 1e-6);
      for (const [k, v] of [...this.timer.passMs].sort((a, b) => b[1] - a[1])) {
        rows.push({ group, detail: true, label: `　${PASS_NAMES[k] ?? k}`, value: `${v.toFixed(2)} ms（${((v / total) * 100).toFixed(0)}%）`, bar: v / total });
      }
    }
    const frameMs = 1000 / Math.max(this.fps, 1e-3);
    const pixels = w * h;
    const shown = this.canvas.width * this.canvas.height;
    rows.push(
      { group, detail: true, label: "フレーム間隔", value: `${frameMs.toFixed(1)} ms · CPU ${this.cpuMs.toFixed(2)} ms` },
      { group, detail: true, label: "品質", value: `${QUALITY_NAMES[this.governor.mode].replace("品質：", "")} · 解像度の倍率 ${this.governor.scale}` },
      {
        group, detail: true, label: "解像度",
        value: `内部 ${w}×${h}（${(pixels / 1e6).toFixed(2)} M 画素）→ 表示 ${this.canvas.width}×${this.canvas.height}（${(shown / 1e6).toFixed(2)} M 画素）· DPR ${window.devicePixelRatio || 1}`,
      },
    );
    const a = this.gpu.adapterInfo;
    const adapter = [a.vendor, a.architecture, a.description].filter(Boolean).join(" ");
    rows.push({ group, detail: true, label: "GPU", value: `${adapter || "不明"}${this.gpu.hasTimestamps ? "" : "（時間計測なし）"}` });
    return rows;
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
  if (mode === "golden") (await import("./testing/golden")).runGolden(app);
}

main().catch((e) => {
  console.error(e);
  diag(`fatal: ${e}`);
});

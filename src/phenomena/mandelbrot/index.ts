// The Mandelbrot set, with deep zoom.
//
// Mode selection uses the numeric state of the view, not the zoom factor:
//   direct f32    while a pixel is wider than ~16 f32 ulps of the centre;
//   perturbation  while pixel size, |δc| and the orbit's smallest |Z| stay
//                 inside the f32 exponent range (with margin);
//   floatexp      otherwise, at any depth.
// When the orbit for the current depth is not ready yet, the previous image is
// reprojected and held rather than showing a wrong one.

import { DeepCamera, bigScaled, rescale, type DeepPoint } from "../../core/camera2d";
import { feFromLog2 } from "../../core/floatexp";
import type { InputFrame } from "../../core/input";
import type { FrameContext, Phenomenon, PhenomenonMeta, ProgressInfo, ProgressRow } from "../../core/phenomenon";
import { int, percent, pow10FromLog2, seconds } from "../../ui/format";
import { formatZoom, log2ZoomFromScale } from "../../ui/location";
import { iterationRows, pixelRows, stepRows } from "../escape/progress";
import type { PaletteTexture } from "../../core/palettes";
import type { GpuContext } from "../../gpu/device";
import {
  EscapeEngine, MODE_DEEP, MODE_DIRECT, MODE_NAMES, MODE_PERTURB,
  type EscapeMode, type EscapeView, type HistoryMap,
} from "../escape/engine";
import { OrbitManager, hasTauri, orbitReference, requestPatches } from "./orbit";
import { VOYAGES, type VoyageTarget } from "./voyages";

const SUPERSCRIPT = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const toSuperscript = (n: number) => String(n).replace(/\d/g, (d) => SUPERSCRIPT[Number(d)]);

const HOME = { re: "-0.6", im: "0", width: 3.6 };
/** Exponent margin kept from the f32 range limits (2^±126). */
const F32_SAFE_LOG2 = -100;

interface Snapshot extends DeepPoint {
  log2Scale: number;
  pixelRatio: number;
}

export class Mandelbrot implements Phenomenon {
  readonly meta: PhenomenonMeta = {
    id: "mandelbrot",
    title: "Mandelbrot Set",
    titleJa: "マンデルブロ集合",
    formula:
      '<math><msub><mi>z</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><msubsup><mi>z</mi><mi>n</mi><mn>2</mn></msubsup><mo>+</mo><mi>c</mi><mo>,</mo><mspace width="0.6em"/><msub><mi>z</mi><mn>0</mn></msub><mo>=</mo><mn>0</mn></math>',
    description:
      "複素数 c ごとに z を二乗して c を足す操作を繰り返し、値が無限大へ逃げない c の集まり。境界はどこまで拡大しても尽きることのない細部を持ち、その奥には自分自身の小さな写しが無数に潜んでいる。",
    controls: [["ダブルクリック", "その点へ近づく"], ["V ／ Space", "Voyage — 深い場所へ自動で潜る（もう一度で止める）"]],
    motion: "voyage",
  };

  private engine: EscapeEngine;
  private orbits?: OrbitManager;
  private camera = new DeepCamera(HOME.re, HOME.im, 0);
  private viewW = 1;
  private viewH = 1;
  private prev?: Snapshot;
  private zoomImpulse = 0;
  private zoomAnchor: [number, number] = [0, 0];
  private velocity: [number, number] = [0, 0];
  private glide?: DeepPoint;
  private voyage?: { target: VoyageTarget; index: number; phase: "approach" | "dive" | "rest" | "return"; t: number };
  private mode: EscapeMode = MODE_DIRECT;
  private holding = false;
  private needsHome = true;
  private lastSpacingLog2 = 0;
  // Component proofs (orbit.ts, requestPatches): the view they are for, when
  // it last changed, and whether they were requested.
  private patchKey = "";
  private patchSince = 0;
  private patchAsked = false;

  constructor(palette: () => PaletteTexture) {
    this.engine = new EscapeEngine(palette);
    this.engine.minIter = 800;
    // manageOrbit() extends a truncated orbit whenever the limit outgrows it.
    this.engine.orbitExtends = true;
    this.engine.maxIter = 800;
  }

  init(gpu: GpuContext): void {
    this.engine.init(gpu);
    this.orbits = new OrbitManager(gpu.device);
    this.prev = undefined;
  }

  reset(): void {
    this.needsHome = true;
    this.voyage = undefined;
    this.glide = undefined;
    this.zoomImpulse = 0;
    this.velocity = [0, 0];
    this.engine.maxIter = this.engine.minIter;
  }

  private home(): void {
    this.camera.set(HOME.re, HOME.im, Math.log2(HOME.width / Math.max(1, Math.min(this.viewW, this.viewH * 1.6))));
    this.needsHome = false;
  }

  get deepAvailable(): boolean {
    return hasTauri();
  }

  /** Smallest log2 scale (device px) reachable: unlimited with deep zoom. */
  private minLog2Scale(pixelRatio: number): number {
    if (this.deepAvailable) return -1e7;
    const [cx, cy] = this.camera.centerNumber();
    return Math.log2(Math.max(Math.abs(cx), Math.abs(cy), 1e-3)) - 19 - Math.log2(pixelRatio);
  }

  onKey(key: string): boolean {
    if (key === "v" || key === "V") {
      this.toggleVoyage();
      return true;
    }
    return false;
  }

  toggleVoyage(index?: number): void {
    if (this.voyage && index === undefined) {
      this.voyage = undefined;
      return;
    }
    this.startVoyage(index ?? Math.floor(Math.random() * VOYAGES.length));
  }

  playing(): boolean {
    return this.voyage !== undefined;
  }

  setPlaying(on: boolean): void {
    if (on !== this.playing()) this.toggleVoyage();
  }

  get voyaging(): boolean {
    return this.voyage !== undefined;
  }

  private startVoyage(index: number): void {
    const target = VOYAGES[index % VOYAGES.length];
    this.voyage = { target, index, phase: "approach", t: 0 };
    const cam = new DeepCamera(target.re, target.im, -target.depthLog2);
    this.glide = { re: cam.re, im: cam.im, frac: cam.frac };
  }

  /**
   * Orbit length to request: at least 4096, and never beyond what the
   * iteration cap can use (one more than the cap, so that no pixel ever waits
   * at the end of an orbit the limit cannot outgrow).
   */
  private orbitLength(want: number): number {
    return Math.min(Math.max(want, 4096), this.engine.maxIterCap + 1);
  }

  /**
   * Once a perturbation view has stayed unchanged for a moment, asks for
   * component proofs of its tiles; they apply only if it is still the view.
   */
  private managePatches(view: EscapeView, width: number, height: number): void {
    const o = this.orbits?.current;
    if (view.mode !== MODE_PERTURB || !o || view.orbit !== o) return;
    const key = `${o.rustId}:${view.spacing.m}:${view.spacing.e}:${view.cmr[0]}:${view.cmr[1]}:${width}x${height}`;
    const now = performance.now();
    if (key !== this.patchKey) {
      this.patchKey = key;
      this.patchSince = now;
      this.patchAsked = false;
    }
    if (this.patchAsked || now - this.patchSince < 250) return;
    this.patchAsked = true;
    requestPatches(o, view, width, height)
      .then((buf) => {
        if (this.patchKey === key) this.engine.setPatches(buf);
      })
      .catch(() => {});
  }

  /** Places the view at an exact location (used by benchmarks and tests). */
  setView(re: string, im: string, log2Scale: number): void {
    this.needsHome = false;
    this.voyage = undefined;
    this.glide = undefined;
    this.zoomImpulse = 0;
    this.velocity = [0, 0];
    this.engine.maxIter = this.engine.minIter;
    this.camera.set(re, im, log2Scale);
  }

  view(): { re: string; im: string; log2Scale: number } {
    const [re, im] = this.camera.centerDecimal();
    return { re, im, log2Scale: this.camera.log2Scale };
  }

  get cameraState(): DeepCamera {
    return this.camera;
  }

  update(input: InputFrame, ctx: { time: number; dt: number; paused: boolean; viewW: number; viewH: number }): void {
    this.viewW = ctx.viewW;
    this.viewH = ctx.viewH;
    if (this.needsHome) this.home();
    const dt = Math.min(ctx.dt, 0.1);
    const cam = this.camera;
    const userActive = input.dragging || input.zoom !== 0 || input.dblclick !== undefined;
    if (userActive && this.voyage) this.voyage = undefined;

    // Drag with inertia.
    if (input.dragging) {
      cam.pan(input.dragX, input.dragY);
      const a = 1 - Math.exp(-dt * 30);
      this.velocity[0] += (input.dragX / Math.max(dt, 1e-3) - this.velocity[0]) * a;
      this.velocity[1] += (input.dragY / Math.max(dt, 1e-3) - this.velocity[1]) * a;
      this.glide = undefined;
    } else if (Math.hypot(this.velocity[0], this.velocity[1]) > 2) {
      cam.pan(this.velocity[0] * dt, this.velocity[1] * dt);
      const k = Math.exp(-dt * 4.5);
      this.velocity[0] *= k;
      this.velocity[1] *= k;
    } else {
      this.velocity = [0, 0];
    }

    // Wheel zoom about the pointer; 1 notch = half an octave, eased.
    if (input.zoom !== 0) {
      this.zoomImpulse += input.zoom * 0.5;
      this.zoomAnchor = [input.pointerX - ctx.viewW / 2, input.pointerY - ctx.viewH / 2];
      this.glide = undefined;
    }
    if (input.dblclick) {
      const ox = input.dblclick.x - ctx.viewW / 2;
      const oy = input.dblclick.y - ctx.viewH / 2;
      this.glide = cam.pointAt(ox, oy);
      this.zoomImpulse += 1.5;
      this.zoomAnchor = [0, 0];
    }

    this.updateVoyage(dt);

    if (this.glide) {
      const [gx, gy] = cam.offsetPx(this.glide);
      const a = 1 - Math.exp(-dt * (this.voyage ? 1.2 : 4));
      cam.pan(-gx * a, gy * a);
      if (Math.hypot(gx, gy) < 0.5 && !this.voyage) this.glide = undefined;
    }

    if (Math.abs(this.zoomImpulse) > 1e-4) {
      const step = this.zoomImpulse * (1 - Math.exp(-dt * 7));
      this.zoomImpulse -= step;
      const anchor: [number, number] = this.glide ? [0, 0] : this.zoomAnchor;
      cam.zoomAt(anchor[0], anchor[1], step);
    }
    // Zooming out past the overview is pointless; ease back.
    const maxScale = Math.log2(8 / Math.max(1, Math.min(ctx.viewW, ctx.viewH)));
    if (cam.log2Scale > maxScale) cam.zoomAt(0, 0, cam.log2Scale - maxScale);
  }

  private updateVoyage(dt: number): void {
    const v = this.voyage;
    if (!v) return;
    v.t += dt;
    const cam = this.camera;
    const [gx, gy] = this.glide ? cam.offsetPx(this.glide) : [0, 0];
    const dist = Math.hypot(gx, gy);
    const depth = -cam.log2Scale - Math.log2(Math.max(this.viewW, this.viewH));
    switch (v.phase) {
      case "approach":
        // Centre on the target first, zooming gently as it comes into view.
        if (dist < Math.min(this.viewW, this.viewH) * 0.25) v.phase = "dive";
        else this.zoomImpulse = Math.max(this.zoomImpulse, Math.min(0.02, (0.25 * Math.min(this.viewW, this.viewH)) / (dist + 1)));
        break;
      case "dive": {
        // Steady dive, slowing near the destination; never wait for computation
        // longer than needed (holding the image counts as waiting).
        const remaining = v.target.depthLog2 - depth;
        if (remaining <= 0.05) {
          v.phase = "rest";
          v.t = 0;
          break;
        }
        const rate = this.holding ? 0.05 : Math.min(v.target.rate ?? 0.75, 0.25 + remaining * 0.3);
        this.zoomImpulse = rate * dt * 8;
        break;
      }
      case "rest":
        if (v.t > 6) {
          v.phase = "return";
          v.t = 0;
        }
        break;
      case "return":
        // Rise back towards the overview, then pick the next destination.
        this.glide = undefined;
        this.zoomImpulse = -Math.max(1.5, depth * 0.02) * dt * 8;
        if (depth < 3) this.startVoyage(v.index + 1);
        break;
    }
  }

  private historyMap(snap: Snapshot): HistoryMap | null {
    const prev = this.prev;
    if (!prev || prev.pixelRatio !== snap.pixelRatio) return { scale: 1, offset: [0, 0] };
    if (prev.re === snap.re && prev.im === snap.im && prev.frac === snap.frac && prev.log2Scale === snap.log2Scale) return null;
    const a = rescale(prev, Math.max(prev.frac, snap.frac));
    const b = rescale(snap, Math.max(prev.frac, snap.frac));
    // Offset of the new centre in previous internal pixels (screen y down).
    const s = -a.frac - prev.log2Scale - Math.log2(prev.pixelRatio);
    return {
      scale: 2 ** (snap.log2Scale - prev.log2Scale),
      offset: [bigScaled(b.re - a.re, s), -bigScaled(b.im - a.im, s)],
    };
  }

  render(frame: FrameContext): { output: GPUTexture; status: "refining" | "converged" } {
    const cam = this.camera;
    const minScale = this.minLog2Scale(frame.pixelRatio);
    if (cam.log2Scale < minScale) cam.zoomAt(0, 0, cam.log2Scale - minScale);
    const snap: Snapshot = { ...cam.snapshot(), pixelRatio: frame.pixelRatio };
    const history = this.historyMap(snap);
    if (history && this.prev && snap.log2Scale > this.prev.log2Scale + 0.5) this.engine.relaxIterations();
    this.prev = snap;

    const spacingLog2 = cam.log2Scale + Math.log2(frame.pixelRatio);
    this.lastSpacingLog2 = spacingLog2;
    const [cx, cy] = cam.centerNumber();
    const ulpLog2 = Math.log2(Math.max(Math.abs(cx), Math.abs(cy), 2 ** -20)) - 23;
    const directOk = spacingLog2 > ulpLog2 + 4 && spacingLog2 > F32_SAFE_LOG2;
    const view: EscapeView = {
      mode: MODE_DIRECT,
      spacing: feFromLog2(spacingLog2),
      center: [cx, cy],
      cmr: [0, 0],
      juliaC: [0, 0],
    };
    this.holding = false;
    const orbits = this.orbits;
    // Prefetch the orbit shortly before direct precision runs out.
    const needOrbit = !directOk || spacingLog2 < ulpLog2 + 10;
    if (needOrbit && orbits && this.deepAvailable) this.manageOrbit(frame, spacingLog2);
    if (!directOk) {
      const o = orbits?.current;
      const usable = o && o.bits >= Math.ceil(-spacingLog2) + 16 && this.refOffset(o.ref, frame.pixelRatio);
      if (o && usable) {
        const cmr = usable;
        const range = spacingLog2 > F32_SAFE_LOG2 && o.minLog2Abs > F32_SAFE_LOG2 && o.dcMaxLog2 > F32_SAFE_LOG2 - 20;
        view.mode = range ? MODE_PERTURB : MODE_DEEP;
        view.cmr = cmr;
        view.orbit = o;
        view.interiorRef = orbitReference(o);
      } else {
        this.holding = true;
      }
    }
    this.mode = view.mode;
    if (this.holding) {
      this.engine.setView(view, history ?? { scale: 1, offset: [0, 0] });
      return this.engine.render(frame, false);
    }
    this.engine.setView(view, history);
    this.managePatches(view, frame.width, frame.height);
    return this.engine.render(frame);
  }

  /** (centre − reference) in internal pixels, or undefined if too far to be accurate in f32. */
  private refOffset(ref: DeepPoint, pixelRatio: number): [number, number] | undefined {
    const [ox, oy] = this.camera.offsetPx(ref);
    const x = -ox / pixelRatio;
    const y = -oy / pixelRatio;
    if (Math.hypot(x, y) > 2 ** 22) return undefined;
    return [x, y];
  }

  private manageOrbit(frame: FrameContext, spacingLog2: number): void {
    const orbits = this.orbits!;
    if (orbits.busy) return;
    const cam = this.camera;
    const maxIter = this.engine.maxIter;
    const needBits = Math.ceil(-spacingLog2) + 32 + Math.ceil(Math.log2(maxIter));
    const halfDiag = 0.5 * Math.hypot(frame.width, frame.height);
    const o = orbits.current;
    const off = o ? this.refOffset(o.ref, frame.pixelRatio) : undefined;
    const tooFar = !off || Math.hypot(off[0], off[1]) > halfDiag * 3;
    const tooShallow = !o || o.bits < needBits - 24;
    // Pixels wait at the end of a truncated orbit (index o.maxIter, reached at
    // iteration o.maxIter at the earliest), and the limit is not raised while
    // any pixel waits: an orbit only as long as the limit must be extended too.
    const tooShort = o !== undefined && !o.escaped && !o.periodic && o.maxIter <= maxIter;
    if (o && tooShort && !tooFar && !tooShallow) {
      // Same reference, longer: the GPU keeps computing with the current orbit
      // (pixels rebase at its end) and switches without starting over.
      orbits.request(o.ref, o.bits, this.orbitLength(maxIter * 4), o.dcMaxLog2, o);
      return;
    }
    if (!o || tooFar || tooShallow) {
      // New reference at the view centre, with headroom for further zoom.
      const dcMaxLog2 = spacingLog2 + Math.log2(halfDiag * 4);
      orbits.request({ re: cam.re, im: cam.im, frac: cam.frac }, needBits + 32, this.orbitLength(maxIter * 2), dcMaxLog2);
      return;
    }
    // The BLA table is valid while |δc| stays below the bound it was built for;
    // rebuild it when the view outgrows it or has shrunk far below it.
    const dcNow = spacingLog2 + Math.log2(halfDiag + Math.hypot(off![0], off![1]));
    if (dcNow > o.dcMaxLog2 || dcNow < o.dcMaxLog2 - 16) {
      orbits.rebuildBla(dcNow + 2);
    }
  }

  progress(): ProgressInfo {
    const e = this.engine.progress();
    const s = this.engine.stats();
    const orbits = this.orbits;
    const rows: ProgressRow[] = [...stepRows(e)];
    const MODE_JA = ["f32 で直接計算", "", "摂動法（f32 の差分）", "摂動法（floatexp の差分）"];

    // Where the view is and how precisely it is computed.
    let group = "位置と精度";
    rows.push({
      group, label: "精度モード",
      value: this.holding ? "基準軌道を待っています（直前の画像を表示中）" : `${MODE_JA[this.mode]} · 画素の幅 2^${this.lastSpacingLog2.toFixed(0)} (${pow10FromLog2(this.lastSpacingLog2)})`,
    });
    const [re, im] = this.camera.centerDecimal();
    rows.push(
      { group, detail: true, label: "拡大率", value: formatZoom(log2ZoomFromScale(this.camera.log2Scale, this.viewW, this.viewH)) },
      { group, detail: true, label: "中心（実部）", value: digits(re) },
      { group, detail: true, label: "中心（虚部）", value: digits(im) },
      { group, detail: true, label: "座標の保持精度", value: `${int(this.camera.frac)} bit（10 進で約 ${int(this.camera.frac * Math.LOG10E * Math.LN2)} 桁）` },
      { group, detail: true, label: "深いズーム", value: this.deepAvailable ? "利用可（倍率の上限なし）" : "利用不可（ブラウザ実行のため f32 の範囲まで）" },
    );

    // Reference orbit.
    group = "基準軌道（任意精度）";
    const p = orbits?.pending;
    const pr = orbits?.progress;
    const o = orbits?.current;
    if (p && p.kind === "orbit" && pr) {
      const label = pr.phase === "orbit" ? "計算中" : pr.phase === "bla" ? "BLA 表を構築中" : "転送中";
      const frac = pr.phase === "orbit" ? pr.iterations / Math.max(1, pr.maxIter) : 1;
      rows.push({
        group, label: "基準軌道（任意精度）",
        value: `${label}　${int(pr.iterations)} / ${int(pr.maxIter)} 反復 · ${int(pr.bits)} bit · ${seconds(pr.elapsedMs / 1000)}`,
        bar: frac,
      });
    } else if (p && p.kind === "bla") {
      rows.push({ group, label: "基準軌道（任意精度）", value: "BLA 表を現在の範囲に合わせて再構築中" });
    } else if (this.mode === MODE_DIRECT && !this.holding) {
      rows.push({ group, label: "基準軌道（任意精度）", value: "不要（この倍率では f32 で直接計算）" });
    } else if (o) {
      rows.push({
        group, label: "基準軌道（任意精度）",
        value: `${int(o.len)} 点 · ${int(o.bits)} bit${o.periodic ? " · 周期点" : ""} · BLA ${o.levels} 段 · ${seconds(o.computeMs / 1000)}`,
      });
    }
    if (o && this.mode !== MODE_DIRECT) {
      rows.push(
        {
          group, detail: true, label: "軌道の終わり",
          value: o.escaped ? "基準点が発散した（軌道はここで完結）" : o.periodic ? "周期軌道（0 に戻るので先頭から繰り返す）" : "途中まで（反復上限が伸びれば延長）",
        },
        { group, detail: true, label: "軌道の長さ / 反復上限", value: `${int(o.len)} 点 / ${int(e.maxIter)} 反復` },
        { group, detail: true, label: "BLA 表", value: `${o.levels} 段 · |δc| ≤ 2^${o.dcMaxLog2.toFixed(0)} で有効` },
        { group, detail: true, label: "基準点の乗り換え（リベース）", value: `${int(s.rebases)} 回（この 1 枚）` },
        { group, detail: true, label: "BLA で飛ばした回数", value: `${int(s.blaSteps)} 回（この 1 枚）` },
        { group, detail: true, label: "誤差の兆候（グリッチ）", value: s.anomalies ? `${int(s.anomalies)} 画素で検出・補正` : "なし" },
        { group, detail: true, label: "成分の証明（内部の高速判定）", value: s.patches ? "この視点に適用中" : "未適用（視点が止まると計算）" },
      );
    }

    rows.push(...iterationRows(this.engine, e), ...pixelRows(this.engine, e));

    if (this.voyage) {
      group = "Voyage";
      const v = this.voyage;
      const PHASE_JA = { approach: "目的地へ移動", dive: "潜っている", rest: "最深部で休止", return: "浮上して戻る" };
      rows.push(
        { group, detail: true, label: "目的地", value: `${v.target.name}（${v.index + 1} / ${VOYAGES.length}）` },
        { group, detail: true, label: "段階", value: PHASE_JA[v.phase] },
        {
          group, detail: true, label: "目標の深さ",
          value: `${pow10FromLog2(-v.target.depthLog2)} · 毎秒 ×${(2 ** (v.target.rate ?? 1)).toFixed(2)} で拡大${v.target.period ? ` · 周期 ${v.target.period} の minibrot` : ""}`,
        },
      );
    }

    let headline = e.converged ? "完成" : e.moving ? `視点の動きに合わせて毎フレーム描き直し中 — ${percent(e.sample)}` : `計算中 — ${percent(e.overall)}`;
    if (this.holding) headline = "基準軌道を計算中";
    else if (e.note && !e.converged) headline = e.note;
    if (this.voyage) headline = `Voyage 中 — ${headline}`;
    // No bottom line while moving: the per-frame figure would only flicker.
    return { fraction: e.moving ? undefined : this.holding ? 0 : e.overall, headline, rows };
  }

  notice(): string | undefined {
    const depth = -this.lastSpacingLog2 - Math.log2(Math.max(this.viewW, this.viewH));
    const exp10 = Math.max(0, Math.floor(depth * Math.LOG10E * Math.LN2));
    const parts: string[] = [];
    if (this.voyage) parts.push(`Voyage — ${this.voyage.target.name}`);
    if (this.voyage || exp10 >= 6) parts.push(`10⁻${toSuperscript(exp10)}`);
    if (this.holding) parts.push("基準軌道を計算しています");
    return parts.length ? parts.join("　·　") : undefined;
  }

  debug(): Record<string, string> {
    const s = this.engine.stats();
    const o = this.orbits?.current;
    const depth = -this.lastSpacingLog2 + Math.log2(Math.max(this.viewW, this.viewH)) - 2;
    return {
      mode: this.holding ? "hold (orbit pending)" : MODE_NAMES[this.mode],
      zoom: `2^${depth.toFixed(1)} (1e${(depth * Math.LOG10E * Math.LN2).toFixed(1)})`,
      "pixel size": `2^${this.lastSpacingLog2.toFixed(1)}`,
      iterations: `${s.maxIter} (max escaped ${s.maxEscapedN})`,
      samples: `${s.samplesDone}/${s.sampleTarget}, unfinished ${s.unfinished}`,
      budget: `${Math.round(s.budget)} it/px/frame`,
      orbit: o ? `${o.len} pts, ${o.bits} bits, ${o.levels} BLA levels, ${o.computeMs.toFixed(0)} ms` : "—",
      "rebase / bla": `${s.rebases} / ${s.blaSteps}`,
      anomalies: `${s.anomalies}`,
      center: this.camera.centerDecimal().map((x) => x.slice(0, 22)).join(", "),
      ...(this.voyage && this.glide ? { voyage: `${this.voyage.phase}, target offset ${this.camera.offsetPx(this.glide).map((v) => v.toExponential(2)).join(", ")} px` } : {}),
    };
  }

  dispose(): void {
    this.engine.dispose();
  }
}

/** A long decimal, shortened for the panel with its full digit count. */
function digits(x: string): string {
  const n = x.replace(/^-|\./g, "").length;
  return x.length > 64 ? `${x.slice(0, 62)}…（全 ${n} 桁、L でコピー）` : x;
}

// The quiet layer over the image: caption, formula, description, selector,
// help and debug. Everything fades away after a few seconds without input.

import type { PhenomenonMeta } from "../core/phenomenon";

const IDLE_MS = 3500;

const KEYS: [string, string][] = [
  ["← →", "現象を切り替える"],
  ["ドラッグ", "移動 ／ 回転"],
  ["ホイール", "拡大・縮小"],
  ["ダブルクリック", "その点へ近づく"],
  ["右ドラッグ", "パラメータを動かす（ジュリア集合）"],
  ["V", "Voyage — 自動で深く潜る（マンデルブロ集合）"],
  ["Space", "再生 ／ 停止（マンデルブロ集合では Voyage）"],
  ["C", "配色を変える"],
  ["F", "全画面"],
  ["I", "解説"],
  ["H", "表示をすべて隠す"],
  ["R", "視点を戻す"],
  ["Q", "品質（自動・低・中・高）"],
  ["P", "計算の進み具合"],
  ["D", "計測情報"],
];

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;

export class Overlay {
  private idleTimer = 0;
  private toastTimer = 0;
  private selector = $("#selector");
  onSelect: (i: number) => void = () => {};
  onPlay: () => void = () => {};
  onReset: () => void = () => {};
  private controlsKey = "";

  constructor() {
    for (const id of ["#mark", "#caption", "#info", "#notice", "#controls"]) $(id).classList.add("fades");
    $("#play").addEventListener("click", () => this.onPlay());
    $("#reset").addEventListener("click", () => this.onReset());
    $("#progress-toggle").addEventListener("click", () => this.toggle("progress"));
    // Buttons must not keep keyboard focus (Space would click them again).
    for (const id of ["#play", "#reset", "#progress-toggle"]) $(id).addEventListener("pointerup", (e) => (e.currentTarget as HTMLElement).blur());
    const dl = $("#help").querySelector("dl")!;
    for (const [k, v] of KEYS) {
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = v;
      dl.append(dt, dd);
    }
    window.addEventListener("pointermove", (e) => {
      this.wake();
      const near = e.clientY > window.innerHeight - 120;
      this.selector.classList.toggle("show", near);
    });
    this.wake();
  }

  setPhenomena(list: PhenomenonMeta[]): void {
    this.selector.replaceChildren(
      ...list.map((m, i) => {
        const b = document.createElement("button");
        b.textContent = m.title;
        b.addEventListener("click", () => this.onSelect(i));
        return b;
      }),
    );
  }

  show(meta: PhenomenonMeta, index: number): void {
    const cap = $("#caption");
    cap.style.opacity = "0";
    setTimeout(() => {
      cap.querySelector(".en")!.textContent = meta.title;
      cap.querySelector(".ja")!.textContent = meta.titleJa;
      cap.querySelector(".formula")!.innerHTML = meta.formula;
      $("#info").querySelector(".description")!.textContent = meta.description;
      cap.style.opacity = "";
    }, 500);
    this.selector.querySelectorAll("button").forEach((b, i) => b.classList.toggle("active", i === index));
    this.wake();
  }

  /** Play button: pause/play icon for animations, a labelled toggle for Voyage. */
  setControls(motion: "animation" | "voyage", playing: boolean): void {
    const key = `${motion}:${playing}`;
    if (key === this.controlsKey) return;
    this.controlsKey = key;
    const b = $("#play");
    const pause = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect class="solid" x="7" y="6" width="3" height="12" rx="0.6"/><rect class="solid" x="14" y="6" width="3" height="12" rx="0.6"/></svg>';
    const play = '<svg viewBox="0 0 24 24" aria-hidden="true"><path class="solid" d="M8 5.5v13l10.5-6.5z"/></svg>';
    if (motion === "voyage") {
      b.classList.add("wide");
      b.innerHTML = `${playing ? pause : play}<span class="label">Voyage</span>`;
      b.title = playing ? "Voyage を止める（Space / V）" : "Voyage — 自動で深く潜る（Space / V）";
    } else {
      b.classList.remove("wide");
      b.innerHTML = playing ? pause : play;
      b.title = playing ? "停止（Space）" : "再生（Space）";
    }
    b.setAttribute("aria-label", b.title);
  }

  /** Marks activity; captions return and fade again after a pause. */
  wake(): void {
    document.body.classList.remove("idle");
    clearTimeout(this.idleTimer);
    this.idleTimer = window.setTimeout(() => {
      document.body.classList.add("idle");
      this.selector.classList.remove("show");
    }, IDLE_MS);
  }

  toggle(id: "info" | "help" | "debug" | "progress"): boolean {
    const el = $("#" + id);
    el.hidden = !el.hidden;
    if (id === "progress") $("#progress-toggle").classList.toggle("on", !el.hidden);
    this.wake();
    return !el.hidden;
  }

  toggleHidden(): void {
    document.body.classList.toggle("hidden-ui");
  }

  isOpen(id: "debug" | "progress"): boolean {
    return !$("#" + id).hidden;
  }

  setDebug(lines: Record<string, string>): void {
    const keys = Object.keys(lines);
    const w = Math.max(...keys.map((k) => k.length));
    $("#debug").textContent = keys.map((k) => `${k.padEnd(w)}  ${lines[k]}`).join("\n");
  }

  private lineTimer = 0;
  private refiningSince = 0;

  /** Thin line along the bottom edge while an image is being computed. */
  setProgressLine(fraction: number | undefined): void {
    const line = $("#progressline");
    const fill = line.querySelector(".fill") as HTMLElement;
    if (fraction === undefined) {
      line.classList.remove("show");
      this.refiningSince = 0;
      return;
    }
    fill.style.width = `${(fraction * 100).toFixed(2)}%`;
    if (fraction < 1) {
      clearTimeout(this.lineTimer);
      this.lineTimer = 0;
      // Only computations that take a moment get a line; quick ones would flicker.
      if (!this.refiningSince) this.refiningSince = performance.now();
      if (performance.now() - this.refiningSince > 450) line.classList.add("show");
    } else {
      this.refiningSince = 0;
    }
    if (fraction >= 1 && line.classList.contains("show") && !this.lineTimer) {
      // Let the full line be seen briefly, then fade.
      this.lineTimer = window.setTimeout(() => {
        line.classList.remove("show");
        this.lineTimer = 0;
      }, 700);
    }
  }

  setProgress(headline: string, rows: { label: string; value: string; bar?: number }[]): void {
    const panel = $("#progress");
    panel.querySelector(".headline")!.textContent = headline;
    const box = panel.querySelector(".rows") as HTMLElement;
    while (box.children.length < rows.length) {
      const r = document.createElement("div");
      r.className = "row";
      r.innerHTML = '<div class="label"></div><div class="value"></div><div class="bar"><div></div></div>';
      box.append(r);
    }
    while (box.children.length > rows.length) box.lastElementChild!.remove();
    rows.forEach((row, i) => {
      const el = box.children[i] as HTMLElement;
      (el.querySelector(".label") as HTMLElement).textContent = row.label;
      (el.querySelector(".value") as HTMLElement).textContent = row.value;
      const bar = el.querySelector(".bar") as HTMLElement;
      bar.hidden = row.bar === undefined;
      if (row.bar !== undefined) (bar.firstElementChild as HTMLElement).style.width = `${(Math.max(0, Math.min(1, row.bar)) * 100).toFixed(1)}%`;
    });
  }

  setNotice(text: string | undefined): void {
    const n = $("#notice");
    if (text) n.textContent = text;
    n.classList.toggle("show", !!text);
  }

  toast(text: string): void {
    const t = $("#toast");
    t.textContent = text;
    t.classList.add("show");
    clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => t.classList.remove("show"), 1800);
  }

  unsupported(reason: string): void {
    const u = $("#unsupported");
    u.hidden = false;
    u.querySelector(".reason")!.textContent = reason;
    for (const id of ["#mark", "#caption", "#selector"]) $(id).hidden = true;
  }
}

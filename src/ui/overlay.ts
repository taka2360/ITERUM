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
  ["Space", "動きを止める ／ 再開"],
  ["C", "配色を変える"],
  ["F", "全画面"],
  ["I", "解説"],
  ["H", "表示をすべて隠す"],
  ["R", "はじめの視点へ"],
  ["Q", "品質（自動・低・中・高）"],
  ["D", "計測情報"],
];

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;

export class Overlay {
  private idleTimer = 0;
  private toastTimer = 0;
  private selector = $("#selector");
  onSelect: (i: number) => void = () => {};

  constructor() {
    for (const id of ["#mark", "#caption", "#info", "#notice"]) $(id).classList.add("fades");
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

  /** Marks activity; captions return and fade again after a pause. */
  wake(): void {
    document.body.classList.remove("idle");
    clearTimeout(this.idleTimer);
    this.idleTimer = window.setTimeout(() => {
      document.body.classList.add("idle");
      this.selector.classList.remove("show");
    }, IDLE_MS);
  }

  toggle(id: "info" | "help" | "debug"): boolean {
    const el = $("#" + id);
    el.hidden = !el.hidden;
    this.wake();
    return !el.hidden;
  }

  toggleHidden(): void {
    document.body.classList.toggle("hidden-ui");
  }

  isOpen(id: "debug"): boolean {
    return !$("#" + id).hidden;
  }

  setDebug(lines: Record<string, string>): void {
    const keys = Object.keys(lines);
    const w = Math.max(...keys.map((k) => k.length));
    $("#debug").textContent = keys.map((k) => `${k.padEnd(w)}  ${lines[k]}`).join("\n");
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

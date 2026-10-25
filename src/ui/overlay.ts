// The quiet layer over the image: caption, formula, description, selector,
// help and debug. Everything fades away after a few seconds without input.

import type { PhenomenonMeta, ProgressRow } from "../core/phenomenon";
import { KEY_GROUPS } from "./keys";
import { parseLocationText } from "./location";

const IDLE_MS = 3500;

export type PanelId = "info" | "help" | "debug" | "progress" | "location";

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;

export class Overlay {
  private idleTimer = 0;
  private toastTimer = 0;
  private selector = $("#selector");
  onSelect: (i: number) => void = () => {};
  onPlay: () => void = () => {};
  onReset: () => void = () => {};
  /** Jump request from the location panel; returns the fields that could not be read. */
  onLocationGo: (re: string, im: string, zoom: string) => ("re" | "im" | "zoom")[] = () => [];
  onLocationCopy: () => void = () => {};
  /** The location fields hold the user's edits; live updates pause until Enter or Esc. */
  private locationDirty = false;
  private controlsKey = "";
  /** The progress panel shows a summary or every figure (P cycles summary → detail → closed). */
  progressDetail = false;
  private phenomenonCount = 0;

  constructor() {
    for (const id of ["#mark", "#caption", "#info", "#notice", "#controls"]) $(id).classList.add("fades");
    $("#play").addEventListener("click", () => this.onPlay());
    $("#reset").addEventListener("click", () => this.onReset());
    $("#progress-toggle").addEventListener("click", () => this.cycleProgress());
    $("#location-toggle").addEventListener("click", () => this.toggle("location"));
    this.initLocation();
    // Buttons must not keep keyboard focus (Space would click them again).
    for (const id of ["#play", "#reset", "#progress-toggle", "#location-toggle", "#loc-copy", "#loc-go", "#progress .mode"]) $(id).addEventListener("pointerup", (e) => (e.currentTarget as HTMLElement).blur());
    $("#progress .mode").addEventListener("click", () => this.setProgressDetail(!this.progressDetail));
    window.addEventListener("pointermove", (e) => {
      this.wake();
      const near = e.clientY > window.innerHeight - 120;
      this.selector.classList.toggle("show", near);
    });
    this.wake();
  }

  setPhenomena(list: PhenomenonMeta[]): void {
    this.phenomenonCount = list.length;
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
    this.renderHelp(meta);
    this.wake();
  }

  /** The shortcut list: the common groups, then what only the current phenomenon has. */
  private renderHelp(meta: PhenomenonMeta): void {
    const groups = KEY_GROUPS.map((g) => ({
      title: g.title,
      items: g.items.map((it): [string, string] => [it.action === "number" ? `1–${this.phenomenonCount}` : it.label, it.text]),
    }));
    groups.push({ title: `この現象だけの操作 — ${meta.titleJa}`, items: meta.controls });
    const box = $("#help .groups");
    box.replaceChildren(
      ...groups.map((g) => {
        const sec = document.createElement("section");
        const h = document.createElement("h3");
        h.textContent = g.title;
        const dl = document.createElement("dl");
        for (const [k, v] of g.items) {
          const dt = document.createElement("dt");
          for (const part of k.split(" ／ ")) {
            if (dt.childElementCount) dt.append(" ");
            const kbd = document.createElement("kbd");
            kbd.textContent = part;
            dt.append(kbd);
          }
          const dd = document.createElement("dd");
          dd.textContent = v;
          dl.append(dt, dd);
        }
        sec.append(h, dl);
        if (g === groups[groups.length - 1]) sec.classList.add("current");
        return sec;
      }),
    );
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

  toggle(id: PanelId): boolean {
    return this.setOpen(id, $("#" + id).hidden);
  }

  setOpen(id: PanelId, open: boolean): boolean {
    const el = $("#" + id);
    el.hidden = !open;
    if (id === "progress" || id === "location") $(`#${id}-toggle`).classList.toggle("on", open);
    if (id === "location" && !open) this.leaveLocationEdit();
    if (id === "progress" && !open) this.setProgressDetail(false);
    this.wake();
    return open;
  }

  /** P and the panel button: closed → summary → detail → closed. */
  cycleProgress(): void {
    if (!this.isOpen("progress")) this.setOpen("progress", true);
    else if (!this.progressDetail) this.setProgressDetail(true);
    else this.setOpen("progress", false);
  }

  setProgressDetail(on: boolean): void {
    this.progressDetail = on;
    const panel = $("#progress");
    panel.classList.toggle("detail", on);
    panel.querySelector(".mode")!.textContent = on ? "概要に戻す" : "詳細を見る";
    this.progressRefresh();
  }

  /** Asks the app for fresh figures right away (set by main.ts). */
  progressRefresh: () => void = () => {};

  /** Closes every open panel; returns false if none was open. */
  closePanels(): boolean {
    let any = false;
    for (const id of ["help", "progress", "location", "info", "debug"] as const) {
      if (!$("#" + id).hidden) {
        this.setOpen(id, false);
        any = true;
      }
    }
    return any;
  }

  toggleHidden(): void {
    document.body.classList.toggle("hidden-ui");
  }

  isOpen(id: PanelId): boolean {
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

  /**
   * Fills the progress panel. The summary shows rows without `detail`; the
   * detailed view shows every row under its group heading.
   */
  setProgress(headline: string, rows: ProgressRow[]): void {
    const panel = $("#progress");
    panel.querySelector(".headline")!.textContent = headline;
    const box = panel.querySelector(".rows") as HTMLElement;
    const shown = this.progressDetail ? rows : rows.filter((r) => !r.detail);
    // Group headings become rows of their own so the list can be reused in place.
    const items: ({ heading: string } | ProgressRow)[] = [];
    let group: string | undefined;
    for (const r of shown) {
      if (this.progressDetail && r.group && r.group !== group) items.push({ heading: r.group });
      group = r.group;
      items.push(r);
    }
    while (box.children.length < items.length) box.append(document.createElement("div"));
    while (box.children.length > items.length) box.lastElementChild!.remove();
    items.forEach((item, i) => {
      let el = box.children[i] as HTMLElement;
      const kind = "heading" in item ? "heading" : "row";
      if (el.className !== kind) {
        const fresh = document.createElement("div");
        fresh.className = kind;
        if (kind === "row") fresh.innerHTML = '<div class="label"></div><div class="value"></div><div class="bar"><div></div></div>';
        el.replaceWith(fresh);
        el = fresh;
      }
      if ("heading" in item) {
        el.textContent = item.heading;
        return;
      }
      (el.querySelector(".label") as HTMLElement).textContent = item.label;
      (el.querySelector(".value") as HTMLElement).textContent = item.value;
      const bar = el.querySelector(".bar") as HTMLElement;
      bar.hidden = item.bar === undefined;
      if (item.bar !== undefined) (bar.firstElementChild as HTMLElement).style.width = `${(Math.max(0, Math.min(1, item.bar)) * 100).toFixed(1)}%`;
    });
  }

  private locInputs = { re: $<HTMLInputElement>("#loc-re"), im: $<HTMLInputElement>("#loc-im"), zoom: $<HTMLInputElement>("#loc-zoom") };

  private initLocation(): void {
    const inputs = Object.values(this.locInputs);
    const go = () => {
      const { re, im, zoom } = this.locInputs;
      const bad = this.onLocationGo(re.value, im.value, zoom.value);
      for (const [k, el] of Object.entries(this.locInputs)) el.classList.toggle("invalid", bad.includes(k as "re"));
      if (bad.length === 0) this.leaveLocationEdit();
    };
    for (const el of inputs) {
      el.addEventListener("input", () => {
        this.locationDirty = true;
        el.classList.remove("invalid");
      });
      el.addEventListener("keydown", (e) => {
        if (e.key === "Enter") go();
        else if (e.key === "Escape") this.leaveLocationEdit();
      });
      el.addEventListener("paste", (e) => {
        const p = parseLocationText(e.clipboardData?.getData("text") ?? "");
        // A single value goes into the focused field as usual.
        if ([p.re, p.im, p.zoom].filter((v) => v !== undefined).length < 2) return;
        e.preventDefault();
        for (const k of ["re", "im", "zoom"] as const) {
          const v = p[k];
          if (v !== undefined) this.locInputs[k].value = v;
          this.locInputs[k].classList.remove("invalid");
        }
        this.locationDirty = true;
      });
      // Show the beginning of long coordinates when not editing.
      el.addEventListener("blur", () => (el.scrollLeft = 0));
    }
    $("#loc-go").addEventListener("click", go);
    $("#loc-copy").addEventListener("click", () => this.onLocationCopy());
  }

  /** Drops edits and lets the fields follow the view again. */
  private leaveLocationEdit(): void {
    this.locationDirty = false;
    for (const el of Object.values(this.locInputs)) {
      el.classList.remove("invalid");
      if (document.activeElement === el) el.blur();
    }
  }

  /** Live view; ignored while the user is editing a field. */
  setLocation(re: string, im: string, zoom: string): void {
    if (this.locationDirty || Object.values(this.locInputs).includes(document.activeElement as HTMLInputElement)) return;
    this.locInputs.re.value = re;
    this.locInputs.im.value = im;
    this.locInputs.zoom.value = zoom;
  }

  /** Hides the location control for phenomena without a complex-plane view. */
  setLocationAvailable(on: boolean): void {
    $("#location-toggle").hidden = !on;
    if (!on && !$("#location").hidden) this.toggle("location");
  }

  async copyText(text: string): Promise<boolean> {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.cssText = "position:fixed;opacity:0";
      document.body.append(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    }
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

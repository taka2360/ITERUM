// Collects pointer and wheel events and hands them to the active phenomenon
// once per frame. All coordinates are device pixels relative to the canvas.

export interface InputFrame {
  /** Left-button drag movement since the last frame. */
  dragX: number;
  dragY: number;
  dragging: boolean;
  /** Right-button (or shift + left) drag movement since the last frame. */
  altDragX: number;
  altDragY: number;
  altDragging: boolean;
  /** Wheel steps since last frame; positive = zoom in. */
  zoom: number;
  pointerX: number;
  pointerY: number;
  pointerInside: boolean;
  /** Double-click position, if one happened this frame. */
  dblclick?: { x: number; y: number };
  /** True when any input arrived this frame. */
  active: boolean;
}

export class InputCollector {
  private f: InputFrame = InputCollector.empty();
  private buttons = 0;
  private shift = false;
  private lastX = 0;
  private lastY = 0;
  onActivity: () => void = () => {};

  static empty(): InputFrame {
    return {
      dragX: 0, dragY: 0, dragging: false,
      altDragX: 0, altDragY: 0, altDragging: false,
      zoom: 0, pointerX: 0, pointerY: 0, pointerInside: false, active: false,
    };
  }

  constructor(private el: HTMLElement) {
    el.addEventListener("pointerdown", (e) => {
      el.setPointerCapture(e.pointerId);
      this.buttons = e.buttons;
      this.shift = e.shiftKey;
      [this.lastX, this.lastY] = this.pos(e);
      this.touch();
    });
    el.addEventListener("pointermove", (e) => {
      const [x, y] = this.pos(e);
      const dx = x - this.lastX;
      const dy = y - this.lastY;
      this.lastX = x;
      this.lastY = y;
      this.f.pointerX = x;
      this.f.pointerY = y;
      this.f.pointerInside = true;
      if (e.buttons & 2 || (e.buttons & 1 && this.shift)) {
        this.f.altDragX += dx;
        this.f.altDragY += dy;
      } else if (e.buttons & 1) {
        this.f.dragX += dx;
        this.f.dragY += dy;
      }
      this.buttons = e.buttons;
      this.touch();
    });
    const up = (e: PointerEvent) => {
      this.buttons = e.buttons;
      this.touch();
    };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("pointerleave", () => (this.f.pointerInside = false));
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        // Normalize: one mouse notch (~100 px) = 1 step; trackpads give fractions.
        const unit = e.deltaMode === 1 ? 1 / 3 : e.deltaMode === 2 ? 3 : 1 / 100;
        const steps = -e.deltaY * unit * (e.ctrlKey ? 2 : 1);
        this.f.zoom += Math.max(-4, Math.min(4, steps));
        [this.f.pointerX, this.f.pointerY] = this.pos(e);
        this.f.pointerInside = true;
        this.touch();
      },
      { passive: false },
    );
    el.addEventListener("dblclick", (e) => {
      const [x, y] = this.pos(e);
      this.f.dblclick = { x, y };
      this.touch();
    });
  }

  private pos(e: MouseEvent): [number, number] {
    const r = this.el.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    return [(e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr];
  }

  private touch(): void {
    this.f.active = true;
    this.onActivity();
  }

  /** Returns this frame's input and resets accumulators. */
  take(): InputFrame {
    const out = this.f;
    out.dragging = (this.buttons & 1) !== 0 && !this.shift;
    out.altDragging = (this.buttons & 2) !== 0 || ((this.buttons & 1) !== 0 && this.shift);
    this.f = InputCollector.empty();
    this.f.pointerX = out.pointerX;
    this.f.pointerY = out.pointerY;
    this.f.pointerInside = out.pointerInside;
    return out;
  }
}

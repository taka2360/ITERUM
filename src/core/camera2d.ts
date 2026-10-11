// Cameras for the complex plane. Screen coordinates are device pixels with y
// pointing down; the complex plane has imaginary pointing up.
//
// DeepCamera keeps the centre as arbitrary-precision fixed point (BigInt) so
// that zoom depth is limited only by memory, while `log2Scale` (complex units
// per device pixel) is an ordinary float. Viewing transforms therefore never
// lose precision; the numeric depth of each renderer is handled separately.

const FRAC_STEP = 64;

export function bitLength(b: bigint): number {
  if (b < 0n) b = -b;
  if (b === 0n) return 0;
  const hex = b.toString(16);
  return (hex.length - 1) * 4 + (32 - Math.clz32(parseInt(hex[0], 16)));
}

/** round(x * 2^log2) as BigInt, for any finite log2. */
export function scaledToBig(x: number, log2: number): bigint {
  if (x === 0 || !Number.isFinite(x)) return 0n;
  const whole = Math.floor(log2);
  const xm = x * 2 ** (log2 - whole); // |xm| in [|x|, 2|x|)
  // xm = mi * 2^-52 * 2^k exactly enough: take a 53-bit integer mantissa.
  const k = Math.floor(Math.log2(Math.abs(xm)));
  const mi = BigInt(Math.round(xm * 2 ** (52 - k)));
  const sh = whole + k - 52;
  return sh >= 0 ? mi << BigInt(sh) : roundShift(mi, -sh);
}

function roundShift(b: bigint, s: number): bigint {
  const half = 1n << BigInt(s - 1);
  return b >= 0n ? (b + half) >> BigInt(s) : -((-b + half) >> BigInt(s));
}

/** b * 2^log2 as a number (accurate to f64 precision). */
export function bigScaled(b: bigint, log2: number): number {
  if (b === 0n) return 0;
  const bl = bitLength(b);
  let s = 0;
  if (bl > 60) {
    s = bl - 60;
    b = b < 0n ? -((-b) >> BigInt(s)) : b >> BigInt(s);
  }
  // Number(b) < 2^60; scale in two steps to stay inside the f64 range.
  return (Number(b) * 2 ** -60) * 2 ** (log2 + s + 60);
}

export function decimalToBig(s: string, frac: number): bigint {
  const t = s.trim();
  const neg = t.startsWith("-");
  const body = neg || t.startsWith("+") ? t.slice(1) : t;
  const [ip, fp = ""] = body.split(".");
  if (!/^\d*$/.test(ip) || !/^\d*$/.test(fp) || (ip + fp).length === 0) throw new Error(`bad decimal: ${s}`);
  const n = BigInt((ip || "0") + fp);
  const v = (n << BigInt(frac)) / 10n ** BigInt(fp.length);
  return neg ? -v : v;
}

export function bigToDecimal(b: bigint, frac: number, digits: number): string {
  const neg = b < 0n;
  if (neg) b = -b;
  const ip = b >> BigInt(frac);
  let rem = b - (ip << BigInt(frac));
  let out = "";
  for (let i = 0; i < digits; i++) {
    rem *= 10n;
    out += (rem >> BigInt(frac)).toString();
    rem &= (1n << BigInt(frac)) - 1n;
  }
  return `${neg ? "-" : ""}${ip}.${out}`;
}

export interface DeepPoint {
  re: bigint;
  im: bigint;
  frac: number;
}

export function rescale(p: DeepPoint, frac: number): DeepPoint {
  if (frac === p.frac) return p;
  const d = frac - p.frac;
  return d > 0
    ? { re: p.re << BigInt(d), im: p.im << BigInt(d), frac }
    : { re: roundShift(p.re, -d), im: roundShift(p.im, -d), frac };
}

export class DeepCamera implements DeepPoint {
  re = 0n;
  im = 0n;
  frac = FRAC_STEP;
  /** log2 of complex-plane units per device pixel. */
  log2Scale = 0;

  constructor(re: string, im: string, log2Scale: number) {
    this.set(re, im, log2Scale);
  }

  set(re: string, im: string, log2Scale: number): void {
    this.log2Scale = log2Scale;
    this.frac = this.requiredFrac();
    this.re = decimalToBig(re, this.frac);
    this.im = decimalToBig(im, this.frac);
  }

  private requiredFrac(): number {
    return Math.max(FRAC_STEP, Math.ceil((-this.log2Scale + 64) / FRAC_STEP) * FRAC_STEP);
  }

  private ensureFrac(): void {
    const need = this.requiredFrac();
    if (need > this.frac) {
      const p = rescale(this, need);
      this.re = p.re;
      this.im = p.im;
      this.frac = need;
    }
  }

  /** Moves the view so content follows a screen drag of (dx, dy) device pixels. */
  pan(dx: number, dy: number): void {
    this.re -= scaledToBig(dx, this.log2Scale + this.frac);
    this.im += scaledToBig(dy, this.log2Scale + this.frac);
  }

  /** Zooms by 2^log2Factor keeping the point at screen offset (ox, oy) from the centre fixed. */
  zoomAt(ox: number, oy: number, log2Factor: number): void {
    const before = this.log2Scale;
    this.log2Scale -= log2Factor;
    this.ensureFrac();
    // centre' = p - (p - centre) * s'/s, i.e. move by offset * (s - s').
    const k = 1 - 2 ** -log2Factor;
    this.re += scaledToBig(ox * k, before + this.frac);
    this.im -= scaledToBig(oy * k, before + this.frac);
  }

  /** Offset of point `p` from the centre in device pixels (x right, y up). */
  offsetPx(p: DeepPoint): [number, number] {
    const q = rescale(p, this.frac);
    const s = -this.frac - this.log2Scale;
    return [bigScaled(q.re - this.re, s), bigScaled(q.im - this.im, s)];
  }

  /** Point at screen offset (ox, oy) device pixels from the centre (y down). */
  pointAt(ox: number, oy: number): DeepPoint {
    return {
      re: this.re + scaledToBig(ox, this.log2Scale + this.frac),
      im: this.im - scaledToBig(oy, this.log2Scale + this.frac),
      frac: this.frac,
    };
  }

  snapshot(): DeepPoint & { log2Scale: number } {
    return { re: this.re, im: this.im, frac: this.frac, log2Scale: this.log2Scale };
  }

  centerNumber(): [number, number] {
    return [bigScaled(this.re, -this.frac), bigScaled(this.im, -this.frac)];
  }

  centerDecimal(): [string, string] {
    const digits = Math.ceil((-this.log2Scale + 16) * Math.LOG10E * Math.LN2) + 2;
    return [bigToDecimal(this.re, this.frac, Math.max(6, digits)), bigToDecimal(this.im, this.frac, Math.max(6, digits))];
  }
}

/** Plain double-precision camera for phenomena with a bounded zoom range. */
export class Camera2D {
  constructor(public cx: number, public cy: number, public log2Scale: number) {}

  pan(dx: number, dy: number): void {
    const s = 2 ** this.log2Scale;
    this.cx -= dx * s;
    this.cy += dy * s;
  }

  zoomAt(ox: number, oy: number, log2Factor: number): void {
    const s = 2 ** this.log2Scale;
    const k = 1 - 2 ** -log2Factor;
    this.cx += ox * s * k;
    this.cy -= oy * s * k;
    this.log2Scale -= log2Factor;
  }
}

// Text form of a view: centre as decimal strings and a zoom factor.
//
// Zoom is defined independently of the window size: zoom = 4 / (width of the
// view's shorter side in complex units), so zoom 1 shows a 4-unit-tall strip.

export interface Location {
  re: string;
  im: string;
  /** log2 of the zoom factor. */
  log2Zoom: number;
}

const LOG2_10 = Math.log2(10);
const SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹";

export function log2ZoomFromScale(log2Scale: number, viewW: number, viewH: number): number {
  return 2 - log2Scale - Math.log2(Math.max(1, Math.min(viewW, viewH)));
}

export function log2ScaleFromZoom(log2Zoom: number, viewW: number, viewH: number): number {
  return 2 - log2Zoom - Math.log2(Math.max(1, Math.min(viewW, viewH)));
}

/** "1.778", "12346", or "1.234e30" for large factors. */
export function formatZoom(log2Zoom: number): string {
  const log10 = log2Zoom / LOG2_10;
  if (log10 < 6) return String(Number((2 ** log2Zoom).toPrecision(5)));
  let e = Math.floor(log10);
  let m = 10 ** (log10 - e);
  if (Number(m.toFixed(3)) >= 10) {
    m /= 10;
    e += 1;
  }
  return `${m.toFixed(3)}e${e}`;
}

/** Accepts "1.5e100", "1.5×10^100", "1.5 × 10¹⁰⁰", "10^100" or "2^300"; returns log2. */
export function parseZoom(s: string): number {
  const t = s
    .replace(/\s+/g, "")
    .replace(/[×xX]/g, "*")
    .replace(/−/g, "-")
    .replace(/[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+/g, (m) => "^" + m.replace("⁻", "-").replace("⁺", "").replace(/./gu, (c) => (SUPERSCRIPTS.includes(c) ? String(SUPERSCRIPTS.indexOf(c)) : c)));
  let log2: number | undefined;
  const pow = /^(?:(\d*\.?\d+)\*)?(10|2)\^([+-]?\d*\.?\d+)$/.exec(t);
  const sci = /^(\d*\.?\d+)(?:e([+-]?\d+))?$/i.exec(t);
  if (pow) log2 = Math.log2(Number(pow[1] ?? 1)) + Number(pow[3]) * (pow[2] === "10" ? LOG2_10 : 1);
  else if (sci) log2 = Math.log2(Number(sci[1])) + Number(sci[2] ?? 0) * LOG2_10;
  if (log2 === undefined || !Number.isFinite(log2)) throw new Error(`bad zoom: ${s}`);
  return log2;
}

/** "-1.25e-3" → "-0.00125"; plain decimals pass through. Throws on anything else. */
export function normalizeDecimal(s: string): string {
  const t = s.replace(/[\s_]+/g, "").replace(/−/g, "-");
  const m = /^([+-]?)(\d*)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(t);
  if (!m || (m[2] + (m[3] ?? "")).length === 0) throw new Error(`bad decimal: ${s}`);
  const sign = m[1] === "-" ? "-" : "";
  const digits = m[2] + (m[3] ?? "");
  const exp = Number(m[4] ?? 0);
  if (Math.abs(exp) > 1e6) throw new Error(`exponent out of range: ${s}`);
  let point = m[2].length + exp;
  let body = digits;
  if (point <= 0) {
    body = "0".repeat(1 - point) + body;
    point = 1;
  } else if (point > body.length) {
    body = body + "0".repeat(point - body.length);
  }
  const ip = body.slice(0, point).replace(/^0+(?=\d)/, "");
  const fp = body.slice(point);
  return `${sign}${ip}${fp ? "." + fp : ""}`;
}

/** A double as a plain decimal with enough digits for pixel size 2^log2Scale. */
export function numberToDecimal(x: number, log2Scale: number): string {
  const digits = Math.min(100, Math.max(0, Math.ceil((-log2Scale + 4) / LOG2_10)));
  return x.toFixed(digits);
}

export function locationText(loc: Location): string {
  return `re: ${loc.re}\nim: ${loc.im}\nzoom: ${formatZoom(loc.log2Zoom)}`;
}

/**
 * Reads pasted text: either "re: … im: … zoom: …" (any order, ":" or "=") or
 * two or three bare values "re im [zoom]". Returns only what was found.
 */
export function parseLocationText(text: string): { re?: string; im?: string; zoom?: string } {
  const field = (keys: string) => new RegExp(`(?<![A-Za-z])(?:${keys})\\s*[:=]\\s*([^\\s,;]+)`, "i").exec(text)?.[1];
  const re = field("re|real|x|実部");
  const im = field("im|imag|y|虚部");
  const zoom = field("zoom|z|拡大率");
  if (re !== undefined || im !== undefined || zoom !== undefined) return { re, im, zoom };
  const tokens = text.trim().split(/[\s,;]+/).filter(Boolean);
  if (tokens.length === 2 || tokens.length === 3) return { re: tokens[0], im: tokens[1], zoom: tokens[2] };
  return {};
}

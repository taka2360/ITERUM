// Curated cyclic palettes, interpolated in OKLab for even, calm gradients.

export interface Palette {
  name: string;
  /** Cyclic key colours: [position 0..1, sRGB hex]. */
  stops: [number, string][];
  /** Colour of points inside the set / unresolved regions (sRGB hex). */
  interior: string;
  /** Background for particle systems (sRGB hex). */
  background: string;
}

export const PALETTES: Palette[] = [
  {
    name: "Nocturne",
    stops: [
      [0.0, "#040613"], [0.14, "#0d1f47"], [0.3, "#094e8c"], [0.44, "#4097bd"],
      [0.54, "#acdbe3"], [0.61, "#faf1df"], [0.69, "#f5c076"], [0.77, "#d77343"],
      [0.86, "#89334c"], [0.94, "#32153e"], [1.0, "#040613"],
    ],
    interior: "#020306",
    background: "#03040a",
  },
  {
    name: "Ember",
    stops: [
      [0.0, "#0f0302"], [0.16, "#430709"], [0.32, "#892312"], [0.46, "#cb5906"],
      [0.58, "#f4a437"], [0.67, "#fbe2a7"], [0.73, "#fbf5df"], [0.82, "#de8f57"],
      [0.91, "#7b2432"], [1.0, "#0f0302"],
    ],
    interior: "#040202",
    background: "#060302",
  },
  {
    name: "Verdigris",
    stops: [
      [0.0, "#000a0b"], [0.16, "#032d2f"], [0.32, "#06625f"], [0.46, "#3ba888"],
      [0.56, "#a6daaf"], [0.63, "#eff1da"], [0.71, "#e4c87c"], [0.8, "#b38236"],
      [0.9, "#24462c"], [1.0, "#000a0b"],
    ],
    interior: "#010404",
    background: "#020505",
  },
  {
    name: "Graphite",
    stops: [
      [0.0, "#060607"], [0.22, "#1c1c22"], [0.42, "#484954"], [0.58, "#a7a097"],
      [0.68, "#efe4c7"], [0.78, "#d2a249"], [0.9, "#543a19"], [1.0, "#060607"],
    ],
    interior: "#030303",
    background: "#040405",
  },
  {
    name: "Aurora",
    stops: [
      [0.0, "#060313"], [0.15, "#22125a"], [0.3, "#373ba7"], [0.43, "#0886b0"],
      [0.53, "#2ac3bb"], [0.61, "#acefd1"], [0.68, "#fbecff"], [0.77, "#e18fc6"],
      [0.87, "#8c399e"], [0.94, "#34125a"], [1.0, "#060313"],
    ],
    interior: "#020106",
    background: "#03020a",
  },
];

type Vec3 = [number, number, number];

export function hexToRgb(hex: string): Vec3 {
  const v = parseInt(hex.slice(1), 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

const toLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toSrgb = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

export function srgbToOklab([r, g, b]: Vec3): Vec3 {
  const lr = toLinear(r), lg = toLinear(g), lb = toLinear(b);
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

export function oklabToSrgb([L, a, b]: Vec3): Vec3 {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const lin: Vec3 = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  return lin.map((c) => Math.min(1, Math.max(0, toSrgb(Math.min(1, Math.max(0, c)))))) as Vec3;
}

/** Samples a palette at t (wrapped to [0, 1)) with smooth OKLab interpolation. */
export function samplePalette(p: Palette, t: number): Vec3 {
  t -= Math.floor(t);
  const stops = p.stops;
  let i = 0;
  while (i < stops.length - 2 && t > stops[i + 1][0]) i++;
  const [t0, c0] = stops[i];
  const [t1, c1] = stops[i + 1];
  const u = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
  const s = u * u * (3 - 2 * u); // ease between stops: no visible kinks
  const a = srgbToOklab(hexToRgb(c0));
  const b = srgbToOklab(hexToRgb(c1));
  return oklabToSrgb([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s]);
}

export const LUT_SIZE = 1024;

/** RGBA8 lookup table (sRGB-encoded) for upload as a 1D-like 2D texture. */
export function paletteLut(p: Palette): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i++) {
    const [r, g, b] = samplePalette(p, i / LUT_SIZE);
    out[i * 4] = Math.round(r * 255);
    out[i * 4 + 1] = Math.round(g * 255);
    out[i * 4 + 2] = Math.round(b * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** Average colour of a palette (sRGB), used where bands are finer than a pixel. */
export function paletteMean(p: Palette): [number, number, number] {
  const lut = paletteLut(p);
  const m = [0, 0, 0];
  for (let i = 0; i < LUT_SIZE; i++) for (let c = 0; c < 3; c++) m[c] += lut[i * 4 + c] / 255;
  return [m[0] / LUT_SIZE, m[1] / LUT_SIZE, m[2] / LUT_SIZE];
}

/**
 * Positions of `n` vivid, mutually distinct colours of a palette (for basins
 * that each need their own colour). Greedy: the most chromatic mid-light
 * colour first, then each next one as far as possible from those chosen,
 * weighted by its own chroma, so pale and near-black stops are avoided.
 */
export function distinctPositions(p: Palette, n: number): number[] {
  const N = 256;
  const cand: { t: number; lab: Vec3; c: number }[] = [];
  for (let i = 0; i < N; i++) {
    const lab = srgbToOklab(samplePalette(p, i / N));
    const c = Math.hypot(lab[1], lab[2]);
    if (lab[0] > 0.4 && lab[0] < 0.85) cand.push({ t: i / N, lab, c });
  }
  if (cand.length === 0) return Array.from({ length: n }, (_, k) => 0.32 + (0.46 * k) / Math.max(n - 1, 1));
  const chosen = [cand.reduce((a, b) => (b.c > a.c ? b : a))];
  while (chosen.length < n) {
    let best = cand[0];
    let bestScore = -1;
    for (const x of cand) {
      const d = Math.min(...chosen.map((y) => Math.hypot(x.lab[0] - y.lab[0], x.lab[1] - y.lab[1], x.lab[2] - y.lab[2])));
      const score = d * d * (0.02 + x.c);
      if (score > bestScore) { bestScore = score; best = x; }
    }
    chosen.push(best);
  }
  return chosen.map((x) => x.t).sort((a, b) => a - b);
}

/** GPU texture + sampler for the active palette, shared by all phenomena. */
export class PaletteTexture {
  readonly texture: GPUTexture;
  readonly sampler: GPUSampler;
  index = 0;
  mean: [number, number, number] = [0, 0, 0];

  constructor(private device: GPUDevice) {
    this.texture = device.createTexture({
      size: [LUT_SIZE, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "clamp-to-edge",
    });
    this.set(0);
  }

  get palette(): Palette {
    return PALETTES[this.index];
  }

  set(i: number): void {
    this.index = ((i % PALETTES.length) + PALETTES.length) % PALETTES.length;
    this.mean = paletteMean(this.palette);
    this.device.queue.writeTexture({ texture: this.texture }, paletteLut(this.palette), { bytesPerRow: LUT_SIZE * 4 }, [LUT_SIZE, 1]);
  }
}

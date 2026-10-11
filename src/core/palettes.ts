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
      [0.0, "#03050b"], [0.17, "#0c1c36"], [0.36, "#28537c"], [0.52, "#9cc0d8"],
      [0.61, "#f2ead8"], [0.71, "#d9a35f"], [0.84, "#5e2f45"], [1.0, "#03050b"],
    ],
    interior: "#020306",
    background: "#03040a",
  },
  {
    name: "Ember",
    stops: [
      [0.0, "#080403"], [0.2, "#2b100b"], [0.4, "#6f2b17"], [0.56, "#c8742f"],
      [0.68, "#f4d39b"], [0.75, "#fbf3e0"], [0.88, "#7c3f25"], [1.0, "#080403"],
    ],
    interior: "#040202",
    background: "#060302",
  },
  {
    name: "Verdigris",
    stops: [
      [0.0, "#020707"], [0.2, "#0b2a2a"], [0.4, "#1f5f5a"], [0.56, "#7fb8a4"],
      [0.67, "#e8efe0"], [0.77, "#c9b37a"], [0.89, "#3d4c33"], [1.0, "#020707"],
    ],
    interior: "#010404",
    background: "#020505",
  },
  {
    name: "Graphite",
    stops: [
      [0.0, "#050506"], [0.22, "#1c1c21"], [0.42, "#4a4a54"], [0.58, "#a8a29a"],
      [0.68, "#efe4c8"], [0.78, "#c8a050"], [0.9, "#4f3c1c"], [1.0, "#050506"],
    ],
    interior: "#030303",
    background: "#040405",
  },
  {
    name: "Aurora",
    stops: [
      [0.0, "#04030a"], [0.2, "#1b1240"], [0.38, "#3b3f9a"], [0.53, "#4fa3b8"],
      [0.64, "#c9f0e4"], [0.72, "#f7e9ff"], [0.86, "#7c55a8"], [1.0, "#04030a"],
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

/** GPU texture + sampler for the active palette, shared by all phenomena. */
export class PaletteTexture {
  readonly texture: GPUTexture;
  readonly sampler: GPUSampler;
  index = 0;

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
    this.device.queue.writeTexture({ texture: this.texture }, paletteLut(this.palette), { bytesPerRow: LUT_SIZE * 4 }, [LUT_SIZE, 1]);
  }
}

// Generates the ITERUM app icon (a small Mandelbrot set on deep ink) as a PNG.
import { deflateSync } from "node:zlib";
import { writeFileSync } from "node:fs";

const N = 1024;
const raw = Buffer.alloc((N * 4 + 1) * N);
const crcTable = new Int32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c;
});
const crc = (buf) => {
  let c = -1;
  for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
};
const chunk = (type, data) => {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type), data]);
  const c = Buffer.alloc(4); c.writeUInt32BE(crc(td));
  return Buffer.concat([len, td, c]);
};
for (let y = 0; y < N; y++) {
  raw[y * (N * 4 + 1)] = 0;
  for (let x = 0; x < N; x++) {
    const o = y * (N * 4 + 1) + 1 + x * 4;
    const u = (x + 0.5) / N * 2 - 1, v = (y + 0.5) / N * 2 - 1;
    const r = Math.hypot(u, v);
    // rounded square mask
    const q = Math.max(Math.abs(u), Math.abs(v));
    const corner = Math.hypot(Math.max(Math.abs(u) - 0.78, 0), Math.max(Math.abs(v) - 0.78, 0));
    const inside = q < 0.96 && corner < 0.18;
    if (!inside) { raw.writeUInt32BE(0, o); continue; }
    const cr = u * 1.35 - 0.55, ci = v * 1.35;
    let zr = 0, zi = 0, i = 0;
    for (; i < 200 && zr * zr + zi * zi < 256; i++) { const t = zr * zr - zi * zi + cr; zi = 2 * zr * zi + ci; zr = t; }
    let R = 8, G = 10, B = 16;
    if (i < 200) {
      const s = i + 1 - Math.log2(Math.log2(zr * zr + zi * zi) / 2);
      const t = Math.min(1, Math.pow(s / 40, 0.8));
      R = 8 + 190 * t * t; G = 10 + 170 * t; B = 16 + 200 * Math.sqrt(t);
    } else { R = 3; G = 4; B = 8; }
    raw[o] = R; raw[o + 1] = G; raw[o + 2] = B; raw[o + 3] = 255;
    void r;
  }
}
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(N, 0); ihdr.writeUInt32BE(N, 4); ihdr[8] = 8; ihdr[9] = 6;
const png = Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
writeFileSync(process.argv[2] ?? "assets/icon.png", png);

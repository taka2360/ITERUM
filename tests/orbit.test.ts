import { describe, expect, it } from "vitest";
import { blaLevels, blaOffsets, decodePacket, HEADER_WORDS } from "../src/phenomena/mandelbrot/orbit";

describe("orbit packet decoding", () => {
  it("reads the layout written by format.rs", () => {
    const len = 5;
    const levels = 2;
    const blaEntries = 3;
    const buf = new ArrayBuffer(HEADER_WORDS * 4 + len * 16 + blaEntries * 32);
    const h = new Uint32Array(buf, 0, HEADER_WORDS);
    const hf = new Float32Array(buf, 0, HEADER_WORDS);
    h[0] = 0x4d525449; h[1] = 1; h[2] = 42; h[3] = len; h[4] = 1; h[5] = levels;
    hf[6] = -17.5; h[7] = 320; h[8] = 0; h[9] = 2; h[10] = 3; hf[42] = -250;
    const orbit = new Float32Array(buf, HEADER_WORDS * 4, len * 4);
    orbit[4 * 3] = 0.625;
    const p = decodePacket(buf);
    expect(p).toMatchObject({ id: 42, len, escaped: true, periodic: false, levels, minLog2Abs: -17.5, bits: 320, dcMaxLog2: -250 });
    expect([...p.offsets]).toEqual([0, 2, 3]);
    expect(p.bla.byteLength).toBe(blaEntries * 32);
    expect(new Float32Array(p.orbit.buffer, p.orbit.byteOffset, len * 4)[12]).toBe(0.625);
  });

  it("rejects foreign data", () => {
    expect(() => decodePacket(new ArrayBuffer(HEADER_WORDS * 4))).toThrow();
  });
});

describe("BLA table layout", () => {
  // As bla.rs builds it: level 0 has len − 2 entries, each level pairs the
  // previous one (chunks_exact) while it has at least two; level 0 is not stored.
  function reference(len: number): number[] {
    if (len < 3) return [0];
    const counts = [len - 2];
    while (counts[counts.length - 1] >= 2) counts.push(Math.floor(counts[counts.length - 1] / 2));
    const stored = counts.slice(1, 33);
    const off = [0];
    for (const c of stored) off.push(off[off.length - 1] + c);
    return off;
  }

  it("matches the table built by bla.rs for every length", () => {
    for (const len of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1025, 1026, 4097, 25601, 1 << 20, (1 << 24) + 1, (1 << 24) + 2]) {
      expect([...blaOffsets(len)]).toEqual(reference(len));
      expect(blaLevels(len)).toBe(reference(len).length - 1);
    }
  });
});

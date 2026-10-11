import { describe, expect, it } from "vitest";
import { decodePacket, HEADER_WORDS } from "../src/phenomena/mandelbrot/orbit";

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

// Requests reference orbits from the Rust side and uploads them to the GPU.
// Layout: see src-tauri/src/deepzoom/format.rs.

import { invoke } from "@tauri-apps/api/core";
import type { DeepPoint } from "../../core/camera2d";
import { PATCH_GRID, type GpuOrbit } from "../escape/engine";
import type { Fe } from "../../core/floatexp";

export const HEADER_WORDS = 48;
const MAGIC = 0x4d525449;

export interface OrbitInfo extends GpuOrbit {
  /** Request id of the orbit on the Rust side (kept across BLA rebuilds). */
  rustId: number;
  ref: DeepPoint;
  bits: number;
  escaped: boolean;
  /** The reference is a nucleus: one period suffices for any iteration count. */
  periodic: boolean;
  minLog2Abs: number;
  dcMaxLog2: number;
  maxIter: number;
  computeMs: number;
  /** Time spent on the Rust side: orbit, BLA table, encoding (ms). */
  rustMs: [number, number, number];
}

export interface DecodedPacket {
  id: number;
  /** Total orbit length (for a delta: including the base). */
  len: number;
  /**
   * Delta packet: extends the orbit of request `baseId` (length `baseLen`);
   * `orbit` holds points baseLen..len and `bla`/`offsets` only the entries of
   * each stored level beyond those of the base.
   */
  delta?: { baseLen: number; baseId: number };
  escaped: boolean;
  periodic: boolean;
  levels: number;
  minLog2Abs: number;
  bits: number;
  dcMaxLog2: number;
  rustMs: [number, number, number];
  offsets: Uint32Array<ArrayBuffer>;
  orbit: Uint8Array<ArrayBuffer>;
  bla: Uint8Array<ArrayBuffer>;
}

export function decodePacket(buf: ArrayBuffer): DecodedPacket {
  const h = new Uint32Array(buf, 0, HEADER_WORDS);
  const hf = new Float32Array(buf, 0, HEADER_WORDS);
  if (h[0] !== MAGIC) throw new Error("bad orbit packet");
  const len = h[3];
  const levels = h[5];
  const offsets = new Uint32Array(levels + 1);
  for (let i = 0; i <= levels; i++) offsets[i] = h[8 + i];
  const points = h[4] & 4 ? len - h[46] : len;
  const orbitStart = HEADER_WORDS * 4;
  const blaStart = orbitStart + points * 16;
  return {
    id: h[2],
    len,
    delta: h[4] & 4 ? { baseLen: h[46], baseId: h[47] } : undefined,
    escaped: (h[4] & 1) !== 0,
    periodic: (h[4] & 2) !== 0,
    levels,
    minLog2Abs: hf[6],
    bits: h[7],
    dcMaxLog2: hf[42],
    rustMs: [hf[43], hf[44], hf[45]],
    offsets,
    orbit: new Uint8Array(buf, orbitStart, points * 16),
    bla: new Uint8Array(buf, blaStart, offsets[levels] * 32),
  };
}

/**
 * The point the orbit was actually computed for: Rust holds the reference in
 * fixed point with 64·⌈bits/64⌉ fractional bits and truncates (towards zero)
 * any finer bits of the request (fixed.rs, Fixed::from_hex).
 */
export function orbitReference(o: { ref: DeepPoint; bits: number }): DeepPoint {
  const frac = 64 * Math.ceil(o.bits / 64);
  if (frac >= o.ref.frac) return o.ref;
  const shift = BigInt(o.ref.frac - frac);
  const trunc = (v: bigint) => (v < 0n ? -(-v >> shift) : v >> shift);
  return { re: trunc(o.ref.re), im: trunc(o.ref.im), frac };
}

/**
 * Asks Rust to prove the tiles of a perturbation view interior to hyperbolic
 * components (component.rs). Tiles follow escape.wgsl: a PATCH_GRID² grid
 * over the image; each patch is a disc around the tile centre covering it.
 */
export function requestPatches(o: OrbitInfo, view: { spacing: Fe; cmr: [number, number] }, width: number, height: number): Promise<ArrayBuffer> {
  const sp = view.spacing.m * 2 ** view.spacing.e;
  const tiles: [number, number][] = [];
  for (let ty = 0; ty < PATCH_GRID; ty++) {
    for (let tx = 0; tx < PATCH_GRID; tx++) {
      // Tile centre in rel_pos coordinates (pixels from the image centre, y up).
      const x = ((tx + 0.5) * width) / PATCH_GRID - width / 2;
      const y = -(((ty + 0.5) * height) / PATCH_GRID - height / 2);
      tiles.push([(x + view.cmr[0]) * sp, (y + view.cmr[1]) * sp]);
    }
  }
  // Half the tile diagonal, plus a pixel for the sample jitter and the tile edges.
  const reach = (Math.hypot(width / PATCH_GRID, height / PATCH_GRID) / 2 + 2) * sp;
  const req = {
    re: o.ref.re.toString(16),
    im: o.ref.im.toString(16),
    fracBits: o.ref.frac,
    bits: o.bits,
    tiles,
    rhoLog2: Math.ceil(Math.log2(reach)),
    maxPeriod: 4096,
  };
  return invoke<ArrayBuffer>("component_patches", { req });
}

export const hasTauri = (): boolean => "__TAURI_INTERNALS__" in window;

// A function, not a constant: GPUBufferUsage does not exist where WebGPU is absent (unit tests).
const usage = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

function upload(device: GPUDevice, data: Uint8Array<ArrayBuffer>, label: string): GPUBuffer {
  const size = Math.max(64, Math.ceil(data.byteLength / 4) * 4);
  const buffer = device.createBuffer({ label, size, usage: usage() });
  if (data.byteLength > 0) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

/**
 * Entries of stored BLA level l (1-based) for an orbit of `len` points, as
 * bla.rs builds them: level 0 has len − 2, each level half the one below.
 */
export function blaLevelCount(len: number, l: number): number {
  return Math.floor(Math.max(0, len - 2) / 2 ** l);
}

/** Stored BLA levels for an orbit of `len` points (format.rs: at most 32). */
export function blaLevels(len: number): number {
  const c0 = Math.max(0, len - 2);
  return c0 < 2 ? 0 : Math.min(32, Math.floor(Math.log2(c0)));
}

/** Level offsets (in entries) of a full table for `len` points; last = total. */
export function blaOffsets(len: number): Uint32Array<ArrayBuffer> {
  const k = blaLevels(len);
  const off = new Uint32Array(k + 1);
  for (let l = 1; l <= k; l++) off[l] = off[l - 1] + blaLevelCount(len, l);
  return off;
}

export interface OrbitProgress {
  id: number;
  phase: "orbit" | "bla" | "encode" | "idle";
  iterations: number;
  maxIter: number;
  bits: number;
  elapsedMs: number;
}

export class OrbitManager {
  current?: OrbitInfo;
  /** Live progress of the pending computation (polled from Rust). */
  progress?: OrbitProgress;
  private poller = 0;
  pending?: { id: number; bits: number; ref: DeepPoint; started: number; kind: "orbit" | "bla"; extend?: OrbitInfo };
  lastError?: string;
  /** The latest full packet received (benchmarks dump it). */
  lastPacket?: ArrayBuffer;
  private nextId = 1;

  constructor(private device: GPUDevice) {}

  get busy(): boolean {
    return this.pending !== undefined;
  }

  /**
   * Requests a reference orbit. With `extend`, the request repeats that
   * orbit's reference, precision and |δc| bound with a longer length, and the
   * result is marked as its continuation (see GpuOrbit.continues).
   */
  request(ref: DeepPoint, bits: number, maxIter: number, dcMaxLog2: number, extend?: OrbitInfo): void {
    // Rust continues its cached orbit and sends only the new part when the
    // base is the orbit currently installed here (see install()).
    const extendOf = extend && extend === this.current ? extend.rustId : undefined;
    const id = this.nextId++;
    if (extend) {
      ref = extend.ref;
      bits = extend.bits;
      dcMaxLog2 = extend.dcMaxLog2;
    }
    this.pending = { id, bits, ref, started: performance.now(), kind: "orbit", extend };
    const req = {
      id,
      re: ref.re.toString(16),
      im: ref.im.toString(16),
      fracBits: ref.frac,
      bits,
      maxIter,
      dcMaxLog2,
      extendOf,
    };
    this.progress = { id, phase: "orbit", iterations: 0, maxIter, bits, elapsedMs: 0 };
    this.startPolling();
    invoke<ArrayBuffer>("reference_orbit", { req })
      .then((buf) => {
        if (this.pending?.id !== id) return;
        const p = decodePacket(buf);
        if (!p.delta) this.lastPacket = buf;
        const ext = this.pending.extend;
        const continues = ext && this.current === ext && p.len >= ext.len && p.bits === ext.bits && p.dcMaxLog2 === ext.dcMaxLog2 ? ext.id : undefined;
        if (p.delta && (continues === undefined || p.delta.baseId !== ext!.rustId || p.delta.baseLen !== ext!.len)) {
          // The base is no longer installed: the delta cannot be applied.
          this.pending = undefined;
          this.lastError = "orbit delta without its base";
          return;
        }
        this.install(p, ref, maxIter, performance.now() - this.pending.started, continues);
        this.pending = undefined;
      })
      .catch((e) => {
        if (this.pending?.id === id) {
          this.pending = undefined;
          this.lastError = String(e);
        }
      });
  }

  private startPolling(): void {
    clearInterval(this.poller);
    this.poller = window.setInterval(() => {
      if (!this.pending) {
        clearInterval(this.poller);
        return;
      }
      invoke<OrbitProgress>("orbit_progress")
        .then((p) => {
          if (this.pending && p.id === this.pending.id) this.progress = p;
        })
        .catch(() => {});
    }, 200);
  }

  /** Rebuilds the BLA table of the current orbit for a new |δc| bound. */
  rebuildBla(dcMaxLog2: number): void {
    const cur = this.current;
    if (!cur || this.pending) return;
    const id = this.nextId++;
    this.pending = { id, bits: cur.bits, ref: cur.ref, started: performance.now(), kind: "bla" };
    invoke<ArrayBuffer>("bla_table", { req: { orbitId: cur.rustId, dcMaxLog2 } })
      .then((buf) => {
        if (this.pending?.id !== id || this.current !== cur) return;
        const p = decodePacket(buf);
        this.install(p, cur.ref, cur.maxIter, cur.computeMs);
        this.pending = undefined;
      })
      .catch((e) => {
        if (this.pending?.id === id) {
          this.pending = undefined;
          this.lastError = String(e);
        }
      });
  }

  private install(p: DecodedPacket, ref: DeepPoint, maxIter: number, ms: number, continues?: number): void {
    const old = this.current;
    const buffers = p.delta && old ? this.merge(old, p) : {
      orbit: upload(this.device, p.orbit, "orbit"),
      bla: upload(this.device, p.bla, "bla"),
      blaOff: upload(this.device, new Uint8Array(p.offsets.buffer), "bla-offsets"),
    };
    this.current = {
      // Unique per upload: GPU bind groups are cached by this id.
      id: this.nextId++,
      rustId: p.id,
      ...buffers,
      len: p.len,
      levels: p.levels,
      continues,
      ref,
      bits: p.bits,
      escaped: p.escaped,
      periodic: p.periodic,
      minLog2Abs: p.minLog2Abs,
      dcMaxLog2: p.dcMaxLog2,
      maxIter,
      computeMs: ms,
      rustMs: p.rustMs,
    };
    if (old) {
      // Let in-flight frames finish before freeing.
      setTimeout(() => {
        old.orbit.destroy();
        old.bla.destroy();
        old.blaOff.destroy();
      }, 500);
    }
  }

  /**
   * Buffers for the base orbit `old` extended by the delta packet `p`: the
   * base's points and BLA entries are copied on the GPU into the layout of the
   * full table, and only the new ones are uploaded.
   */
  private merge(old: OrbitInfo, p: DecodedPacket): { orbit: GPUBuffer; bla: GPUBuffer; blaOff: GPUBuffer } {
    const device = this.device;
    const baseLen = p.delta!.baseLen;
    const offNew = blaOffsets(p.len);
    const offOld = blaOffsets(baseLen);
    const k = offNew.length - 1;
    const orbit = device.createBuffer({ label: "orbit", size: Math.max(64, p.len * 16), usage: usage() });
    const bla = device.createBuffer({ label: "bla", size: Math.max(64, offNew[k] * 32), usage: usage() });
    const enc = device.createCommandEncoder({ label: "orbit-extend" });
    if (baseLen > 0) enc.copyBufferToBuffer(old.orbit, 0, orbit, 0, baseLen * 16);
    for (let l = 1; l <= k; l++) {
      const n = l < offOld.length ? blaLevelCount(baseLen, l) : 0;
      if (n > 0) enc.copyBufferToBuffer(old.bla, offOld[l - 1] * 32, bla, offNew[l - 1] * 32, n * 32);
    }
    device.queue.submit([enc.finish()]);
    device.queue.writeBuffer(orbit, baseLen * 16, p.orbit);
    for (let l = 1; l <= k; l++) {
      const n = l < offOld.length ? blaLevelCount(baseLen, l) : 0;
      const from = p.offsets[l - 1] * 32;
      const to = p.offsets[l] * 32;
      if (to > from) device.queue.writeBuffer(bla, (offNew[l - 1] + n) * 32, p.bla, from, to - from);
    }
    return { orbit, bla, blaOff: upload(device, new Uint8Array(offNew.buffer), "bla-offsets") };
  }

  reset(): void {
    this.pending = undefined;
    this.current = undefined;
  }
}

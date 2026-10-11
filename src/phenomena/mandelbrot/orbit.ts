// Requests reference orbits from the Rust side and uploads them to the GPU.
// Layout: see src-tauri/src/deepzoom/format.rs.

import { invoke } from "@tauri-apps/api/core";
import type { DeepPoint } from "../../core/camera2d";
import type { GpuOrbit } from "../escape/engine";

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
}

export interface DecodedPacket {
  id: number;
  len: number;
  escaped: boolean;
  periodic: boolean;
  levels: number;
  minLog2Abs: number;
  bits: number;
  dcMaxLog2: number;
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
  const orbitStart = HEADER_WORDS * 4;
  const blaStart = orbitStart + len * 16;
  return {
    id: h[2],
    len,
    escaped: (h[4] & 1) !== 0,
    periodic: (h[4] & 2) !== 0,
    levels,
    minLog2Abs: hf[6],
    bits: h[7],
    dcMaxLog2: hf[42],
    offsets,
    orbit: new Uint8Array(buf, orbitStart, len * 16),
    bla: new Uint8Array(buf, blaStart, offsets[levels] * 32),
  };
}

export const hasTauri = (): boolean => "__TAURI_INTERNALS__" in window;

function upload(device: GPUDevice, data: Uint8Array<ArrayBuffer>, label: string): GPUBuffer {
  const size = Math.max(64, Math.ceil(data.byteLength / 4) * 4);
  const buffer = device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  if (data.byteLength > 0) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
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
  pending?: { id: number; bits: number; ref: DeepPoint; started: number; kind: "orbit" | "bla" };
  lastError?: string;
  private nextId = 1;

  constructor(private device: GPUDevice) {}

  get busy(): boolean {
    return this.pending !== undefined;
  }

  request(ref: DeepPoint, bits: number, maxIter: number, dcMaxLog2: number): void {
    const id = this.nextId++;
    this.pending = { id, bits, ref, started: performance.now(), kind: "orbit" };
    const req = {
      id,
      re: ref.re.toString(16),
      im: ref.im.toString(16),
      fracBits: ref.frac,
      bits,
      maxIter,
      dcMaxLog2,
    };
    this.progress = { id, phase: "orbit", iterations: 0, maxIter, bits, elapsedMs: 0 };
    this.startPolling();
    invoke<ArrayBuffer>("reference_orbit", { req })
      .then((buf) => {
        if (this.pending?.id !== id) return;
        const p = decodePacket(buf);
        this.install(p, ref, maxIter, performance.now() - this.pending.started);
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

  private install(p: DecodedPacket, ref: DeepPoint, maxIter: number, ms: number): void {
    const old = this.current;
    this.current = {
      // Unique per upload: GPU bind groups are cached by this id.
      id: this.nextId++,
      rustId: p.id,
      orbit: upload(this.device, p.orbit, "orbit"),
      bla: upload(this.device, p.bla, "bla"),
      blaOff: upload(this.device, new Uint8Array(p.offsets.buffer), "bla-offsets"),
      len: p.len,
      levels: p.levels,
      ref,
      bits: p.bits,
      escaped: p.escaped,
      periodic: p.periodic,
      minLog2Abs: p.minLog2Abs,
      dcMaxLog2: p.dcMaxLog2,
      maxIter,
      computeMs: ms,
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

  reset(): void {
    this.pending = undefined;
    this.current = undefined;
  }
}

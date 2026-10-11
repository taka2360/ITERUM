// GPU pass timing via the optional `timestamp-query` feature, plus a tracker
// of submitted-but-unfinished work. Readbacks are asynchronous and lag a few
// frames; nothing here ever blocks the frame loop.

const MAX_PAIRS = 16;
const RING = 4;

interface Slot {
  buffer: GPUBuffer;
  busy: boolean;
  labels: string[];
}

export class GpuTimer {
  readonly enabled: boolean;
  private querySet?: GPUQuerySet;
  private resolve?: GPUBuffer;
  private slots: Slot[] = [];
  private labels: string[] = [];
  /** Latest measured milliseconds per pass label. */
  readonly passMs = new Map<string, number>();
  /** Sum of all passes of the latest measured frame. */
  frameMs = 0;
  private measuredFrames = 0;

  constructor(device: GPUDevice, enabled: boolean) {
    this.enabled = enabled;
    if (!enabled) return;
    this.querySet = device.createQuerySet({ type: "timestamp", count: MAX_PAIRS * 2 });
    this.resolve = device.createBuffer({
      size: MAX_PAIRS * 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    for (let i = 0; i < RING; i++) {
      this.slots.push({
        buffer: device.createBuffer({ size: MAX_PAIRS * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        busy: false,
        labels: [],
      });
    }
  }

  get frames(): number {
    return this.measuredFrames;
  }

  beginFrame(): void {
    this.labels = [];
  }

  /** Timestamp writes for one pass, or undefined when unavailable or full. */
  writes(label: string): GPUComputePassTimestampWrites | undefined {
    if (!this.querySet || this.labels.length >= MAX_PAIRS) return undefined;
    const i = this.labels.length;
    this.labels.push(label);
    return { querySet: this.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
  }

  endFrame(encoder: GPUCommandEncoder): (() => void) | undefined {
    if (!this.querySet || !this.resolve || this.labels.length === 0) return undefined;
    const slot = this.slots.find((s) => !s.busy);
    if (!slot) return undefined;
    const n = this.labels.length;
    encoder.resolveQuerySet(this.querySet, 0, n * 2, this.resolve, 0);
    encoder.copyBufferToBuffer(this.resolve, 0, slot.buffer, 0, n * 16);
    slot.busy = true;
    slot.labels = this.labels;
    // Called after submit.
    return () => {
      slot.buffer
        .mapAsync(GPUMapMode.READ, 0, n * 16)
        .then(() => {
          const t = new BigUint64Array(slot.buffer.getMappedRange(0, n * 16));
          let total = 0;
          const sums = new Map<string, number>();
          for (let i = 0; i < n; i++) {
            const ms = Number(t[2 * i + 1] - t[2 * i]) / 1e6;
            if (ms >= 0 && ms < 1e4) {
              total += ms;
              sums.set(slot.labels[i], (sums.get(slot.labels[i]) ?? 0) + ms);
            }
          }
          slot.buffer.unmap();
          slot.busy = false;
          this.passMs.clear();
          for (const [k, v] of sums) this.passMs.set(k, v);
          this.frameMs = total;
          this.measuredFrames++;
        })
        .catch(() => {
          slot.busy = false;
        });
    };
  }
}

/** Counts command submissions whose GPU work has not completed yet. */
export class InFlight {
  count = 0;
  constructor(private device: GPUDevice) {}

  track(): void {
    this.count++;
    this.device.queue.onSubmittedWorkDone().then(
      () => this.count--,
      () => this.count--,
    );
  }
}

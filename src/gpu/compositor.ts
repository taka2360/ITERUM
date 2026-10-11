// Final pass to the canvas: scales the phenomenon's internal-resolution image,
// crossfades during transitions, adds a faint vignette, and dithers so that
// the dark, slow gradients never band.

import { shaderModule, type GpuContext } from "./device";
import { wgsl } from "../shaders";
import type { GpuTimer } from "./timing";

const SRC = /* wgsl */ `
struct C { mix: f32, vignette: f32, seed: f32, fade: f32 }
@group(0) @binding(0) var<uniform> U: C;
@group(0) @binding(1) var a: texture_2d<f32>;
@group(0) @binding(2) var b: texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;
#include fullscreen

fn hash(p: vec2<f32>) -> f32 {
  let q = fract(p * vec2<f32>(0.1031, 0.1030));
  let r = q + dot(q, q.yx + 33.33);
  return fract((r.x + r.y) * r.x);
}

@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
  var c = textureSampleLevel(a, smp, in.uv, 0.0).rgb;
  if (U.mix < 1.0) {
    let o = textureSampleLevel(b, smp, in.uv, 0.0).rgb;
    // Ease through a slightly darker midpoint: a quiet, gallery-like dissolve.
    let t = smoothstep(0.0, 1.0, U.mix);
    c = mix(o, c, t) * (1.0 - 0.18 * sin(3.14159 * t));
  }
  let d = in.uv - 0.5;
  c *= 1.0 - U.vignette * dot(d, d) * 1.6;
  c *= U.fade;
  // Triangular dither, one 8-bit step wide.
  let n = hash(in.pos.xy + U.seed) + hash(in.pos.yx * 1.37 + U.seed) - 1.0;
  return vec4<f32>(c + n / 255.0, 1.0);
}
`;

export class Compositor {
  private pipeline: GPURenderPipeline;
  private uniform: GPUBuffer;
  private sampler: GPUSampler;
  private layout: GPUBindGroupLayout;
  private cache = new Map<string, GPUBindGroup>();
  private ids = new WeakMap<GPUTexture, number>();
  private nextId = 1;

  constructor(private gpu: GpuContext) {
    const device = gpu.device;
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      ],
    });
    const module = shaderModule(device, "compositor", wgsl(SRC));
    this.pipeline = device.createRenderPipeline({
      label: "compositor",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: gpu.format }] },
    });
    this.uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  }

  private id(t: GPUTexture): number {
    let i = this.ids.get(t);
    if (i === undefined) {
      i = this.nextId++;
      this.ids.set(t, i);
    }
    return i;
  }

  private group(a: GPUTexture, b: GPUTexture): GPUBindGroup {
    const key = `${this.id(a)}:${this.id(b)}`;
    let g = this.cache.get(key);
    if (!g) {
      if (this.cache.size > 64) this.cache.clear();
      g = this.gpu.device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: a.createView() },
          { binding: 2, resource: b.createView() },
          { binding: 3, resource: this.sampler },
        ],
      });
      this.cache.set(key, g);
    }
    return g;
  }

  /** Forget cached bind groups (call when textures are destroyed). */
  invalidate(): void {
    this.cache.clear();
  }

  draw(enc: GPUCommandEncoder, timer: GpuTimer, target: GPUTexture, current: GPUTexture, previous: GPUTexture | undefined, mix: number, fade: number): void {
    this.gpu.device.queue.writeBuffer(this.uniform, 0, new Float32Array([previous ? mix : 1, 0.22, Math.random() * 100, fade]));
    const pass = enc.beginRenderPass({
      label: "composite",
      colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }],
      timestampWrites: timer.writes("composite"),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group(current, previous ?? current));
    pass.draw(3);
    pass.end();
  }
}

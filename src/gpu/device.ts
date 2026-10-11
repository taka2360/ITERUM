// WebGPU initialization and device-loss recovery.

export interface GpuContext {
  device: GPUDevice;
  adapterInfo: GPUAdapterInfo;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  hasTimestamps: boolean;
  /** Increments every time the device is recreated. */
  generation: number;
}

export type GpuInitFailure = "no-webgpu" | "no-adapter" | "device-failed";

export async function createGpu(
  canvas: HTMLCanvasElement,
  generation: number,
  onLost: (info: GPUDeviceLostInfo) => void,
): Promise<GpuContext | GpuInitFailure> {
  if (!("gpu" in navigator) || !navigator.gpu) return "no-webgpu";
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return "no-adapter";
  const hasTimestamps = adapter.features.has("timestamp-query");
  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({
      requiredFeatures: hasTimestamps ? ["timestamp-query"] : [],
      requiredLimits: {
        // Escape-time state buffers are large at high resolutions.
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
      },
    });
  } catch {
    return "device-failed";
  }
  device.lost.then(onLost);
  device.addEventListener("uncapturederror", (e) => {
    console.error("[iterum] GPU error:", (e as GPUUncapturedErrorEvent).error.message);
  });
  const context = canvas.getContext("webgpu");
  if (!context) return "no-webgpu";
  const format = navigator.gpu.getPreferredCanvasFormat();
  // COPY_SRC lets the test harness capture exactly what is presented.
  context.configure({ device, format, alphaMode: "opaque", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  return { device, adapterInfo: adapter.info, canvas, context, format, hasTimestamps, generation };
}

/** Compiles a WGSL module and reports compile errors with line numbers. */
export function shaderModule(device: GPUDevice, label: string, code: string): GPUShaderModule {
  const module = device.createShaderModule({ label, code });
  module.getCompilationInfo().then((info) => {
    for (const m of info.messages) {
      if (m.type === "error") console.error(`[iterum] WGSL ${label}:${m.lineNum}:${m.linePos} ${m.message}`);
    }
  });
  return module;
}

export const FULLSCREEN_VS = /* wgsl */ `
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VsOut;
  o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2<f32>(p.x, 1.0 - p.y);
  return o;
}
`;

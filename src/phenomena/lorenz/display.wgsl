// Tone-maps the Lorenz accumulation buffer (fixed point, 8 fractional bits).

struct D { size: vec2<u32>, exposure: f32, _pad: f32, background: vec4<f32> }

@group(0) @binding(0) var<uniform> P: D;
@group(0) @binding(1) var<storage, read> accum: array<u32>;

#include fullscreen

fn aces(x: vec3<f32>) -> vec3<f32> {
  // Narkowicz's fit of the ACES filmic curve.
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let p = vec2<u32>(in.pos.xy);
  let i = (p.y * P.size.x + p.x) * 3u;
  let v = vec3<f32>(f32(accum[i]), f32(accum[i + 1u]), f32(accum[i + 2u])) / 256.0;
  // Logarithmic exposure keeps both faint wisps and dense cores legible.
  let e = log2(1.0 + v * P.exposure) * 0.32;
  let c = aces(e);
  // The light is accumulated in sRGB-ish values; lift it over the background.
  return vec4<f32>(P.background.rgb + c * (1.0 - P.background.rgb), 1.0);
}

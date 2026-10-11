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
  // Tone-map luminance only and keep the palette's hue, so dense cores glow in
  // colour instead of bleaching; logarithmic exposure keeps faint wisps visible.
  let l = dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
  let lt = aces(vec3<f32>(log2(1.0 + l * P.exposure) * 0.3)).x;
  let hue = v / max(l, 1e-6);
  var c = hue * lt;
  // Only the very brightest cores drift towards white.
  c = mix(c, vec3<f32>(lt), smoothstep(0.75, 1.0, lt) * 0.45);
  c = min(c, vec3<f32>(1.0));
  return vec4<f32>(P.background.rgb + c * (1.0 - P.background.rgb), 1.0);
}

// Colours escape-time results into the phenomenon's output texture.
//
// Finished samples of the current view are averaged (jittered supersampling).
// Pixels the current job has not reached yet borrow from coarser progressive
// pixels, and failing that from the previous frame reprojected into the new
// view, so motion stays smooth while computation catches up.

struct DisplayParams {
  size: vec2<u32>,
  job: u32,
  samples_done: u32,
  hist_offset: vec2<f32>,
  hist_scale: f32,
  hist_valid: u32,
  pal_offset: f32,
  pal_density: f32,
  s_min: f32,
  glow: f32,
  interior: vec4<f32>,
}

struct PixelState {
  dr: f32, di: f32, de: i32, m: u32,
  gr: f32, gi: f32, ge: i32, n: u32,
  job: u32, status: u32, rebases: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> D: DisplayParams;
@group(0) @binding(1) var<storage, read> state: array<PixelState>;
@group(0) @binding(2) var<storage, read> samples: array<vec2<f32>>;
@group(0) @binding(3) var palette: texture_2d<f32>;
@group(0) @binding(4) var pal_sampler: sampler;
@group(0) @binding(5) var history: texture_2d<f32>;
@group(0) @binding(6) var hist_sampler: sampler;

#include fullscreen

fn shade(v: vec2<f32>) -> vec3<f32> {
  if (v.x < 0.0) { return D.interior.rgb; }
  let t = D.pal_density * sqrt(max(v.x - D.s_min, 0.0)) + D.pal_offset;
  let base = textureSampleLevel(palette, pal_sampler, vec2<f32>(t, 0.5), 0.0).rgb;
  // Filaments closer than ~1 px to the set catch a soft light.
  let near = 1.0 - smoothstep(0.0, 3.0, v.y);
  let lifted = mix(base, sqrt(base) * 1.05, near * D.glow);
  // Very close to the set the colour settles towards the interior tone.
  let inside = 1.0 - smoothstep(0.0, 0.35, v.y);
  return mix(lifted, D.interior.rgb, inside * 0.55);
}

fn current(idx: u32) -> vec2<f32> {
  let s = state[idx];
  if (s.job == D.job && s.status != 0u) { return vec2<f32>(s.dr, s.di); }
  return vec2<f32>(-3.0, 0.0);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let p = vec2<u32>(in.pos.xy);
  let w = D.size.x;
  let idx = p.y * w + p.x;
  let plane = w * D.size.y;

  var acc = vec3<f32>(0.0);
  var count = 0.0;
  for (var k = 0u; k < D.samples_done; k++) {
    acc += shade(samples[k * plane + idx]);
    count += 1.0;
  }
  let cur = current(idx);
  if (cur.x > -3.0) {
    acc += shade(cur);
    count += 1.0;
  }
  if (count > 0.0) { return vec4<f32>(acc / count, 1.0); }

  // Not reached yet: borrow from the progressive parents of this job.
  let p1 = p & vec2<u32>(~1u);
  let c1 = current(p1.y * w + p1.x);
  if (c1.x > -3.0) { return vec4<f32>(shade(c1), 1.0); }
  let p2 = p & vec2<u32>(~3u);
  let c2 = current(p2.y * w + p2.x);
  if (c2.x > -3.0) { return vec4<f32>(shade(c2), 1.0); }

  // Otherwise reproject the previous frame.
  if (D.hist_valid != 0u) {
    let mid = vec2<f32>(D.size) * 0.5;
    let q = mid + (in.pos.xy - mid) * D.hist_scale + D.hist_offset;
    if (all(q >= vec2<f32>(0.0)) && all(q < vec2<f32>(D.size))) {
      return vec4<f32>(textureSampleLevel(history, hist_sampler, q / vec2<f32>(D.size), 0.0).rgb, 1.0);
    }
  }
  return vec4<f32>(D.interior.rgb, 1.0);
}

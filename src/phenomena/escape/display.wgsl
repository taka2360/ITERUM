// Colours escape-time results into the phenomenon's output texture.
//
// Finished samples of the current view are averaged (jittered supersampling).
// Pixels the current job has not reached yet borrow from coarser progressive
// pixels, and failing that from the previous frame reprojected into the new
// view, so motion stays smooth while computation catches up.
//
// Colour: the continuous iteration count is mapped logarithmically (bands
// widen gently towards the set instead of crowding into moiré) and filaments
// near the set catch a soft light. Colouring never depends on how far the
// computation has progressed, so the image keeps its colours when it finishes.

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
  mean: vec4<f32>,        // w: band scale (iterations); rgb unused
  interior_glow: f32,
  _p0: f32, _p1: f32, _p2: f32,
}

struct PixelState {
  dr: f32, di: f32, de: i32, m: u32,
  gr: f32, gi: f32, ge: i32, n: u32,
  job: u32, status: u32, rebases: u32, trap: f32,
}

@group(0) @binding(0) var<uniform> D: DisplayParams;
@group(0) @binding(1) var<storage, read> state: array<PixelState>;
@group(0) @binding(2) var<storage, read> samples: array<vec2<f32>>;
@group(0) @binding(3) var palette: texture_2d<f32>;
@group(0) @binding(4) var pal_sampler: sampler;
@group(0) @binding(5) var history: texture_2d<f32>;
@group(0) @binding(6) var hist_sampler: sampler;

#include fullscreen

const NONE: f32 = -3.0;

// The fastest escapes start a little into the palette, not at its darkest stop.
const BAND_LIFT: f32 = 0.16;

fn band(s: f32) -> f32 {
  return D.pal_density * log2(1.0 + max(s - D.s_min, 0.0) / D.mean.w) + D.pal_offset + BAND_LIFT;
}

fn pal(t: f32) -> vec3<f32> {
  return textureSampleLevel(palette, pal_sampler, vec2<f32>(t, 0.5), 0.0).rgb;
}

fn shade(v: vec2<f32>) -> vec3<f32> {
  if (v.x < -1.5) { return D.interior.rgb; }  // numeric anomaly: never shown as detail
  if (v.x < 0.0) {
    // Inside: a faint glow where orbits pass close to 0 (centres of components).
    // v.y is the smallest |z| the orbit reached: soft, smooth light that
    // gathers towards the centres of components and attracting cycles.
    let r = clamp(v.y, 0.0, 2.0);
    let w = D.interior_glow * exp(-3.0 * r);
    let tint = pal(0.27 + 0.18 * r + D.pal_offset * 0.5);
    return mix(D.interior.rgb, tint * 0.55, w);
  }
  let base = pal(band(v.x));
  // Filaments closer than a few pixels to the set catch a soft light.
  let near = 1.0 - smoothstep(0.0, 4.0, v.y);
  let lifted = mix(base, sqrt(base) * 1.04, near * near * D.glow);
  // Right at the set the colour settles towards the interior tone.
  let edge = 1.0 - smoothstep(0.0, 0.3, v.y);
  return mix(lifted, D.interior.rgb, edge * 0.5);
}

fn current(idx: u32) -> vec2<f32> {
  let s = state[idx];
  if (s.job != D.job || s.status == 0u) { return vec2<f32>(NONE, 0.0); }
  // Stopped at the iteration limit (state kept for resuming) or proven interior.
  if (s.status == 2u || s.status == 4u) { return vec2<f32>(-1.0, sqrt(s.trap)); }
  return vec2<f32>(s.dr, s.di);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let p = vec2<u32>(in.pos.xy);
  let w = D.size.x;
  let idx = p.y * w + p.x;
  let plane = w * D.size.y;

  var acc = vec3<f32>(0.0);
  var count = 0.0;
  // Written out rather than looped. A loop over samples[k * plane + idx]
  // produced wrong (darker) values for k > 0 on the D3D12 backend, although the
  // buffer contents were correct; the unrolled form reads them reliably.
  // Keep in step with MAX_SAMPLES in engine.ts.
  if (D.samples_done > 0u) { acc += shade(samples[idx]); count += 1.0; }
  if (D.samples_done > 1u) { acc += shade(samples[plane + idx]); count += 1.0; }
  if (D.samples_done > 2u) { acc += shade(samples[2u * plane + idx]); count += 1.0; }
  if (D.samples_done > 3u) { acc += shade(samples[3u * plane + idx]); count += 1.0; }
  let cur = current(idx);
  if (cur.x > NONE) {
    acc += shade(cur);
    count += 1.0;
  }
  if (count > 0.0) { return vec4<f32>(acc / count, 1.0); }

  // Not reached yet: borrow from the progressive parents of this job.
  let p1 = p & vec2<u32>(~1u);
  let c1 = current(p1.y * w + p1.x);
  if (c1.x > NONE) { return vec4<f32>(shade(c1), 1.0); }
  let p2 = p & vec2<u32>(~3u);
  let c2 = current(p2.y * w + p2.x);
  if (c2.x > NONE) { return vec4<f32>(shade(c2), 1.0); }

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

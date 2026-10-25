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
//
// Relief: the continuous iteration count is treated as height, its slope read
// from the neighbouring pixels of the same sample (all pixels of a sample
// share its jitter), and lit from the upper left, so the set rises like
// embossed metal. The count follows the smooth potential, so the light has no
// seams; only its direction is used, so it is equally soft at every depth.

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
  relief: f32,
  _p1: f32, _p2: f32,
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
const BAND_LIFT: f32 = 0.2;

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

// Towards the light (upper left of the screen).
const LIGHT: vec2<f32> = vec2<f32>(-0.7071068, -0.7071068);

// Height of a neighbour; one inside the set takes the centre's (one-sided slope).
fn height(n: vec2<f32>, v: vec2<f32>) -> f32 { return select(n.x, v.x, n.x < 0.0); }

// Brightness factor for an escaped value given its four neighbours.
fn relief(v: vec2<f32>, l: vec2<f32>, r: vec2<f32>, u: vec2<f32>, d: vec2<f32>) -> f32 {
  if (v.x < 0.0) { return 1.0; }
  // Uphill, towards the set; the surface faces the other way.
  let g = vec2<f32>(height(r, v) - height(l, v), height(d, v) - height(u, v));
  let len = length(g);
  if (!(len > 1e-20)) { return 1.0; }
  return 1.0 - D.relief * dot(g / len, LIGHT);
}

// Shaded and lit value of finished sample plane `b`.
fn lit(b: u32, n: vec4<u32>, idx: u32) -> vec3<f32> {
  let v = samples[b + idx];
  return shade(v) * relief(v, samples[b + n.x], samples[b + n.y], samples[b + n.z], samples[b + n.w]);
}

fn current(idx: u32) -> vec2<f32> {
  let s = state[idx];
  // Running, or waiting for a longer reference orbit: not finished yet.
  if (s.job != D.job || s.status == 0u || s.status == 5u) { return vec2<f32>(NONE, 0.0); }
  // Stopped at the iteration limit (state kept for resuming), cycling or proven interior.
  if (s.status == 2u || s.status == 4u || s.status == 6u) { return vec2<f32>(-1.0, sqrt(s.trap)); }
  return vec2<f32>(s.dr, s.di);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let p = vec2<u32>(in.pos.xy);
  let w = D.size.x;
  let idx = p.y * w + p.x;
  let plane = w * D.size.y;

  // Neighbours (left, right, up, down), clamped at the edges.
  let n = vec4<u32>(
    select(idx - 1u, idx, p.x == 0u), select(idx + 1u, idx, p.x + 1u >= w),
    select(idx - w, idx, p.y == 0u), select(idx + w, idx, p.y + 1u >= D.size.y));

  var acc = vec3<f32>(0.0);
  var count = 0.0;
  // Written out rather than looped. A loop over samples[k * plane + idx]
  // produced wrong (darker) values for k > 0 on the D3D12 backend, although the
  // buffer contents were correct; the unrolled form reads them reliably.
  // Keep in step with MAX_SAMPLES in engine.ts.
  if (D.samples_done > 0u) { acc += lit(0u, n, idx); count += 1.0; }
  if (D.samples_done > 1u) { acc += lit(plane, n, idx); count += 1.0; }
  if (D.samples_done > 2u) { acc += lit(2u * plane, n, idx); count += 1.0; }
  if (D.samples_done > 3u) { acc += lit(3u * plane, n, idx); count += 1.0; }
  let cur = current(idx);
  if (cur.x > NONE) {
    // Neighbours of the running job may not have finished yet: light only
    // when all four have.
    let cl = current(n.x); let cr = current(n.y); let cu = current(n.z); let cd = current(n.w);
    var f = 1.0;
    if (min(min(cl.x, cr.x), min(cu.x, cd.x)) > NONE) { f = relief(cur, cl, cr, cu, cd); }
    acc += shade(cur) * f;
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

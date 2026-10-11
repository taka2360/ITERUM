// Escape-time iteration for the Mandelbrot and Julia sets.
//
// Work is resumable: every pixel keeps its iteration state in `state`, and each
// dispatch advances it by at most `budget` steps. A frame therefore never
// stalls on deep regions, and the image refines over several frames.
//
// MODE 0  Mandelbrot, direct f32 iteration
// MODE 1  Julia, direct f32 iteration
// MODE 2  Mandelbrot, perturbation around a reference orbit, f32 deltas
// MODE 3  Mandelbrot, perturbation with floatexp deltas, orbit and BLA (any depth)
//
// Perturbation: z = Z_m + δ, δ' = (2 Z_m + δ) δ + δc. When |z| < |δ| or the
// orbit ends, the pixel rebases (δ ← z, m ← 0), so one reference serves the
// whole image without glitches. BLA tables skip runs of steps while |δ| < R.
//
// Status: 0 running, 1 escaped, 2 stopped at the iteration limit (not proven
// interior; it resumes when the limit is raised, so its full state is kept),
// 3 numeric anomaly, 4 cycling (see below), 5 waiting at the end of a
// truncated reference orbit for its extension (see orbit_wait). BLA spans may
// run past the current limit, so the path a pixel takes depends only on its
// state (z, or δ and m), never on n or the limit: raising the limit and
// resuming equals starting with the higher limit.
//
// Cycles: the iteration is deterministic, so once a pixel's complete state
// repeats bit for bit it will repeat forever and never escape. Its result
// (interior, smallest |z|² over the cycle, all of which has been visited) is
// then final, exactly what running to any limit would give. Brent's method
// finds the repeat with one saved state and one comparison per step.
//
// Progressive order: STEP 0 handles every 4th pixel in x and y, STEP 1 the
// remaining even pixels, STEP 2 the rest; the display pass fills gaps from
// coarser pixels.

override MODE: u32 = 0u;

const BAIL2: f32 = 65536.0; // escape radius 256, shared with reference.rs

struct Params {
  size: vec2<u32>,
  job: u32,
  step: u32,
  jitter: vec2<f32>,
  cmr: vec2<f32>,        // (centre - reference) / spacing, in pixels
  center: vec2<f32>,     // centre (direct modes)
  julia_c: vec2<f32>,
  spacing_m: f32,        // complex units per pixel, floatexp
  spacing_e: i32,
  max_iter: u32,
  budget: u32,
  orbit_len: u32,
  bla_levels: u32,
  sample: u32,
  _p3: u32,
  // 1: the orbit is truncated and a longer one with the same prefix is coming.
  // Rebasing at its end would replace δ by the full z (|z| ~ 1), losing the
  // pixel's offset in f32, so pixels wait there instead and continue exactly
  // when the extension arrives. 0: rebase at the end (complete orbits).
  orbit_wait: u32,
  _p0: u32, _p1: u32, _p2: u32,
}

struct PixelState {
  dr: f32, di: f32, de: i32,   // z (direct) or δ (perturbation); on finish: smooth, distance
  m: u32,                      // reference index
  gr: f32, gi: f32, ge: i32,   // derivative dz/dc · spacing (pixel units)
  n: u32,                      // iterations so far
  job: u32,
  status: u32,                 // see "Status" above
  rebases: u32,
  trap: f32,                   // smallest |z|² seen (interior shading)
}

struct OrbitPt { re: f32, im: f32, e: i32, _pad: u32 }
struct Bla { ar: f32, ai: f32, ae: i32, br: f32, bi: f32, be: i32, r2m: f32, r2e: i32 }

struct Counters {
  unfinished: atomic<u32>,
  escaped: atomic<u32>,
  interior: atomic<u32>,
  anomalies: atomic<u32>,
  rebases: atomic<u32>,
  max_escaped_n: atomic<u32>,
  min_escaped_n: atomic<u32>,
  bla_steps: atomic<u32>,
  late_escapes: atomic<u32>,   // escaped after more than max_iter / 2 iterations
  unfinished_coarse: atomic<u32>, // still running after this frame, step 0 pixels
  unfinished_mid: atomic<u32>,    // ... step 1 pixels
  iterations256: atomic<u32>,     // iterations advanced this frame / 256 (BLA spans count fully)
  cycling: atomic<u32>,           // pixels found cycling (also counted in interior)
}

var<workgroup> wg_iterations: atomic<u32>;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> state: array<PixelState>;
@group(0) @binding(2) var<storage, read_write> samples: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> counters: Counters;
@group(0) @binding(4) var<storage, read> orbit: array<OrbitPt>;
@group(0) @binding(5) var<storage, read> bla: array<Bla>;
@group(0) @binding(6) var<storage, read> bla_off: array<u32>;

#include floatexp

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Inf or NaN, tested on the bit pattern so the compiler cannot fold it away.
fn bad(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u; }
fn bad2(v: vec2<f32>) -> bool { return bad(v.x) || bad(v.y); }

fn rel_pos(pix: vec2<u32>) -> vec2<f32> {
  let p = vec2<f32>(pix) + 0.5 + P.jitter - vec2<f32>(P.size) * 0.5;
  return vec2<f32>(p.x, -p.y);
}

fn spacing_fe() -> FE { return FE(P.spacing_m, P.spacing_e); }
fn spacing_f() -> f32 { return ldexp(P.spacing_m, P.spacing_e); }

fn orbit_cf(m: u32) -> CF { let o = orbit[m]; return CF(vec2<f32>(o.re, o.im), o.e); }
fn orbit_f(m: u32) -> vec2<f32> { let o = orbit[m]; return ldexp(vec2<f32>(o.re, o.im), vec2<i32>(o.e)); }

// Records a finished pixel. `z` is the escaped value (|z| > 256),
// `log2_g` is log2 |dz/dc · spacing|.
fn finish_escaped(s: ptr<function, PixelState>, idx: u32, z: vec2<f32>, log2_g: f32) {
  let r2 = dot(z, z);
  let log2_abs = 0.5 * log2(r2);
  // Continuous iteration count: n + 1 - log2(log2|z| / log2(256)).
  let sm = f32((*s).n) + 1.0 - log2(log2_abs / 8.0);
  // Distance estimate in pixels: |z| ln|z| / |dz/dc| / spacing.
  let log2_de = log2_abs + log2(log2_abs * 0.69314718) - log2_g;
  let de = exp2(clamp(log2_de, -60.0, 60.0));
  (*s).dr = sm;
  (*s).di = de;
  (*s).status = 1u;
  samples[P.sample * P.size.x * P.size.y + idx] = vec2<f32>(sm, de);
  atomicAdd(&counters.escaped, 1u);
  atomicMax(&counters.max_escaped_n, (*s).n);
  atomicMin(&counters.min_escaped_n, (*s).n);
  if ((*s).n > P.max_iter / 2u) { atomicAdd(&counters.late_escapes, 1u); }
}

// Stops at the iteration limit. The caller has stored the iteration state, so
// the pixel can resume if the limit is raised; meanwhile it shows as interior.
fn pause_at_limit(s: ptr<function, PixelState>, idx: u32) {
  (*s).status = 2u;
  samples[P.sample * P.size.x * P.size.y + idx] = vec2<f32>(-1.0, sqrt((*s).trap));
  atomicAdd(&counters.interior, 1u);
}

// The pixel's state has repeated exactly: it cycles forever (see "Cycles").
fn finish_cycling(s: ptr<function, PixelState>, idx: u32) {
  (*s).status = 4u;
  samples[P.sample * P.size.x * P.size.y + idx] = vec2<f32>(-1.0, sqrt((*s).trap));
  atomicAdd(&counters.interior, 1u);
  atomicAdd(&counters.cycling, 1u);
}

fn same_bits(a: vec2<f32>, b: vec2<f32>) -> bool {
  return all(bitcast<vec2<u32>>(a) == bitcast<vec2<u32>>(b));
}

fn finish_anomaly(s: ptr<function, PixelState>, idx: u32) {
  (*s).dr = -2.0;
  (*s).di = 0.0;
  (*s).status = 3u;
  samples[P.sample * P.size.x * P.size.y + idx] = vec2<f32>(-2.0, 0.0);
  atomicAdd(&counters.anomalies, 1u);
}

fn init_state(pix: vec2<u32>) -> PixelState {
  var s: PixelState;
  s.job = P.job;
  s.status = 0u;
  s.n = 0u;
  s.m = 0u;
  s.rebases = 0u;
  s.trap = 1e30;
  s.de = 0;
  s.ge = 0;
  s.dr = 0.0; s.di = 0.0; s.gr = 0.0; s.gi = 0.0;
  if (MODE == 1u) {
    // Julia: z0 is the pixel, dz/dz0 = 1 (times spacing).
    let z0 = P.center + rel_pos(pix) * spacing_f();
    s.dr = z0.x; s.di = z0.y;
    s.gr = spacing_f();
  }
  return s;
}

// ---- direct iteration (MODE 0, 1) -------------------------------------------

fn iterate_direct(s: ptr<function, PixelState>, pix: vec2<u32>, idx: u32) {
  var z = vec2<f32>((*s).dr, (*s).di);
  var g = vec2<f32>((*s).gr, (*s).gi);
  var n = (*s).n;
  let sp = spacing_f();
  var c = P.julia_c;
  var gc = 0.0;
  if (MODE == 0u) { c = P.center + rel_pos(pix) * sp; gc = sp; }
  var cyc = z;          // Brent: saved state, steps since saving, power of two
  var cyc_len = 0u;
  var cyc_pow = 1u;
  for (var k = 0u; k < P.budget; k++) {
    if (n >= P.max_iter) {
      (*s).dr = z.x; (*s).di = z.y; (*s).gr = g.x; (*s).gi = g.y; (*s).n = n;
      pause_at_limit(s, idx);
      return;
    }
    g = 2.0 * cmul(z, g) + vec2<f32>(gc, 0.0);
    z = cmul(z, z) + c;
    n++;
    (*s).trap = min((*s).trap, dot(z, z));
    if (dot(z, z) > BAIL2) {
      (*s).n = n;
      var lg = 0.5 * log2(dot(g, g));
      if (bad2(g)) { lg = 120.0; }
      finish_escaped(s, idx, z, lg);
      return;
    }
    if (same_bits(z, cyc) && !bad2(z)) { (*s).n = n; finish_cycling(s, idx); return; }
    cyc_len++;
    if (cyc_len == cyc_pow) { cyc = z; cyc_pow <<= 1u; cyc_len = 0u; }
  }
  if (bad2(g)) { g = vec2<f32>(1e30, 0.0); }
  (*s).dr = z.x; (*s).di = z.y; (*s).gr = g.x; (*s).gi = g.y; (*s).n = n;
  atomicAdd(&counters.unfinished, 1u);
}

// ---- perturbation with f32 deltas (MODE 2) ---------------------------------

fn iterate_perturb(s: ptr<function, PixelState>, pix: vec2<u32>, idx: u32) {
  var d = vec2<f32>((*s).dr, (*s).di);
  var g = vec2<f32>((*s).gr, (*s).gi);
  var m = (*s).m;
  var n = (*s).n;
  var rebases = 0u;
  var skipped = 0u;
  let sp = spacing_f();
  let dc = (rel_pos(pix) + P.cmr) * sp;
  let last = P.orbit_len - 1u;
  var cyc_d = d;        // Brent: saved state (δ, m)
  var cyc_m = m;
  var cyc_len = 0u;
  var cyc_pow = 1u;
  for (var k = 0u; k < P.budget; k++) {
    if (n >= P.max_iter) {
      (*s).dr = d.x; (*s).di = d.y; (*s).gr = g.x; (*s).gi = g.y; (*s).m = m; (*s).n = n;
      (*s).rebases += rebases;
      pause_at_limit(s, idx);
      atomicAdd(&counters.rebases, rebases);
      if (skipped > 0u) { atomicAdd(&counters.bla_steps, skipped); }
      return;
    }
    var stepped = false;
    if (m > 0u && P.bla_levels > 0u) {
      let mm = m - 1u;
      var l = min(countTrailingZeros(mm), P.bla_levels);
      let d2 = dot(d, d);
      loop {
        if (l == 0u) { break; }
        let span = 1u << l;
        let i = bla_off[l - 1u] + (mm >> l);
        if (i < bla_off[l]) {
          let e = bla[i];
          if (e.ae < 100 && e.be < 100 && e.r2e > -120 && d2 < ldexp(e.r2m, e.r2e)) {
            let a = ldexp(vec2<f32>(e.ar, e.ai), vec2<i32>(e.ae));
            let b = ldexp(vec2<f32>(e.br, e.bi), vec2<i32>(e.be));
            d = cmul(a, d) + cmul(b, dc);
            g = cmul(a, g) + b * sp;
            m += span;
            n += span;
            skipped += span;
            stepped = true;
            break;
          }
        }
        l--;
      }
    }
    if (!stepped) {
      let zm = orbit_f(m);
      g = 2.0 * cmul(zm + d, g) + vec2<f32>(sp, 0.0);
      d = cmul(2.0 * zm + d, d) + dc;
      m++;
      n++;
    }
    let z = orbit_f(m) + d;
    let r2 = dot(z, z);
    (*s).trap = min((*s).trap, r2);
    if (r2 > BAIL2) {
      (*s).n = n;
      var lg = 0.5 * log2(dot(g, g));
      if (bad2(g)) { lg = 120.0; }
      finish_escaped(s, idx, z, lg);
      atomicAdd(&counters.rebases, rebases);
      if (skipped > 0u) { atomicAdd(&counters.bla_steps, skipped); }
      return;
    }
    if (m >= last && r2 >= dot(d, d) && P.orbit_wait != 0u) {
      (*s).dr = d.x; (*s).di = d.y; (*s).gr = g.x; (*s).gi = g.y; (*s).m = m; (*s).n = n;
      (*s).rebases += rebases;
      (*s).status = 5u;
      atomicAdd(&counters.rebases, rebases);
      if (skipped > 0u) { atomicAdd(&counters.bla_steps, skipped); }
      atomicAdd(&counters.unfinished, 1u);
      return;
    }
    if (r2 < dot(d, d) || m >= last) {
      d = z;
      m = 0u;
      rebases++;
    }
    if (bad2(d)) { (*s).n = n; finish_anomaly(s, idx); return; }
    if (m == cyc_m && same_bits(d, cyc_d)) {
      (*s).n = n;
      (*s).rebases += rebases;
      finish_cycling(s, idx);
      atomicAdd(&counters.rebases, rebases);
      if (skipped > 0u) { atomicAdd(&counters.bla_steps, skipped); }
      return;
    }
    cyc_len++;
    if (cyc_len == cyc_pow) { cyc_d = d; cyc_m = m; cyc_pow <<= 1u; cyc_len = 0u; }
  }
  if (bad2(g)) { g = vec2<f32>(1e30, 0.0); }
  (*s).dr = d.x; (*s).di = d.y; (*s).gr = g.x; (*s).gi = g.y; (*s).m = m; (*s).n = n;
  (*s).rebases += rebases;
  atomicAdd(&counters.rebases, rebases);
  if (skipped > 0u) { atomicAdd(&counters.bla_steps, skipped); }
  atomicAdd(&counters.unfinished, 1u);
}

// ---- perturbation with floatexp deltas (MODE 3) ----------------------------

fn iterate_deep(s: ptr<function, PixelState>, pix: vec2<u32>, idx: u32) {
  var d = CF(vec2<f32>((*s).dr, (*s).di), (*s).de);
  var g = CF(vec2<f32>((*s).gr, (*s).gi), (*s).ge);
  var m = (*s).m;
  var n = (*s).n;
  var rebases = 0u;
  var skipped = 0u;
  let sp = spacing_fe();
  let sp_c = CF(vec2<f32>(sp.m, 0.0), sp.e);
  let dc = cf_scale(cf_make(rel_pos(pix) + P.cmr, 0), sp);
  let last = P.orbit_len - 1u;
  var cyc_d = d;        // Brent: saved state (δ, m)
  var cyc_m = m;
  var cyc_len = 0u;
  var cyc_pow = 1u;
  for (var k = 0u; k < P.budget; k++) {
    if (n >= P.max_iter) {
      (*s).dr = d.v.x; (*s).di = d.v.y; (*s).de = d.e;
      (*s).gr = g.v.x; (*s).gi = g.v.y; (*s).ge = g.e;
      (*s).m = m; (*s).n = n;
      (*s).rebases += rebases;
      pause_at_limit(s, idx);
      atomicAdd(&counters.rebases, rebases);
      if (skipped > 0u) { atomicAdd(&counters.bla_steps, skipped); }
      return;
    }
    var stepped = false;
    if (m > 0u && P.bla_levels > 0u) {
      let mm = m - 1u;
      var l = min(countTrailingZeros(mm), P.bla_levels);
      let d2 = cf_norm2(d);
      loop {
        if (l == 0u) { break; }
        let span = 1u << l;
        let i = bla_off[l - 1u] + (mm >> l);
        if (i < bla_off[l]) {
          let e = bla[i];
          if (fe_less_pos(d2, FE(e.r2m, e.r2e))) {
            let a = CF(vec2<f32>(e.ar, e.ai), e.ae);
            let b = CF(vec2<f32>(e.br, e.bi), e.be);
            d = cf_add(cf_mul(a, d), cf_mul(b, dc));
            g = cf_add(cf_mul(a, g), cf_mul(b, sp_c));
            m += span;
            n += span;
            skipped += span;
            stepped = true;
            break;
          }
        }
        l--;
      }
    }
    if (!stepped) {
      let zm = orbit_cf(m);
      g = cf_add(cf_mul(cf_add(zm, d), cf_scale(g, FE(0.5, 2))), sp_c);
      d = cf_add(cf_mul(cf_add(cf_scale(zm, FE(0.5, 2)), d), d), dc);
      m++;
      n++;
    }
    let z = cf_add(orbit_cf(m), d);
    if (z.e < -40) { (*s).trap = 0.0; }
    if (z.e >= -40) {
      let zf = cf_to_vec2(z);
      (*s).trap = min((*s).trap, dot(zf, zf));
      if (dot(zf, zf) > BAIL2) {
        (*s).n = n;
        let lg = log2(length(g.v)) + f32(g.e);
        finish_escaped(s, idx, zf, lg);
        atomicAdd(&counters.rebases, rebases);
        if (skipped > 0u) { atomicAdd(&counters.bla_steps, skipped); }
        return;
      }
    }
    let closer = fe_less_pos(cf_norm2(z), cf_norm2(d));
    if (m >= last && !closer && P.orbit_wait != 0u) {
      (*s).dr = d.v.x; (*s).di = d.v.y; (*s).de = d.e;
      (*s).gr = g.v.x; (*s).gi = g.v.y; (*s).ge = g.e;
      (*s).m = m; (*s).n = n;
      (*s).rebases += rebases;
      (*s).status = 5u;
      atomicAdd(&counters.rebases, rebases);
      if (skipped > 0u) { atomicAdd(&counters.bla_steps, skipped); }
      atomicAdd(&counters.unfinished, 1u);
      return;
    }
    if (closer || m >= last) {
      d = z;
      m = 0u;
      rebases++;
    }
    if (bad2(d.v)) { (*s).n = n; finish_anomaly(s, idx); return; }
    if (m == cyc_m && d.e == cyc_d.e && same_bits(d.v, cyc_d.v)) {
      (*s).n = n;
      (*s).rebases += rebases;
      finish_cycling(s, idx);
      atomicAdd(&counters.rebases, rebases);
      if (skipped > 0u) { atomicAdd(&counters.bla_steps, skipped); }
      return;
    }
    cyc_len++;
    if (cyc_len == cyc_pow) { cyc_d = d; cyc_m = m; cyc_pow <<= 1u; cyc_len = 0u; }
  }
  (*s).dr = d.v.x; (*s).di = d.v.y; (*s).de = d.e;
  (*s).gr = g.v.x; (*s).gi = g.v.y; (*s).ge = g.e;
  (*s).m = m; (*s).n = n;
  (*s).rebases += rebases;
  atomicAdd(&counters.rebases, rebases);
  if (skipped > 0u) { atomicAdd(&counters.bla_steps, skipped); }
  atomicAdd(&counters.unfinished, 1u);
}

// Advances one pixel; returns the number of iterations it moved forward.
fn process(gid: vec3<u32>) -> u32 {
  let stride = 1u << (2u - P.step);
  let pix = gid.xy * stride;
  if (pix.x >= P.size.x || pix.y >= P.size.y) { return 0u; }
  if (P.step > 0u) {
    let coarse = stride * 2u;
    if (pix.x % coarse == 0u && pix.y % coarse == 0u) { return 0u; }
  }
  let idx = pix.y * P.size.x + pix.x;
  var s = state[idx];
  if (s.job != P.job) { s = init_state(pix); }
  // Resume pixels stopped at an earlier, lower limit.
  if (s.status == 2u && s.n < P.max_iter) { s.status = 0u; }
  // Waiting at the end of a truncated orbit: continue once it is longer (or
  // no extension is coming).
  if (s.status == 5u) {
    if (s.m + 1u < P.orbit_len || P.orbit_wait == 0u) {
      s.status = 0u;
    } else {
      atomicAdd(&counters.unfinished, 1u);
      if (P.step == 0u) { atomicAdd(&counters.unfinished_coarse, 1u); }
      if (P.step == 1u) { atomicAdd(&counters.unfinished_mid, 1u); }
      return 0u;
    }
  }
  if (s.status != 0u) { return 0u; }
  let n0 = s.n;
  if (MODE <= 1u) {
    iterate_direct(&s, pix, idx);
  } else if (MODE == 2u) {
    iterate_perturb(&s, pix, idx);
  } else {
    iterate_deep(&s, pix, idx);
  }
  state[idx] = s;
  if (s.status == 0u || s.status == 5u) {
    if (P.step == 0u) { atomicAdd(&counters.unfinished_coarse, 1u); }
    if (P.step == 1u) { atomicAdd(&counters.unfinished_mid, 1u); }
  }
  return min(s.n - n0, 1u << 25u);
}

@compute @workgroup_size(8, 8)
fn iterate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) { atomicStore(&wg_iterations, 0u); }
  workgroupBarrier();
  // One global atomic per workgroup keeps the throughput counter cheap.
  atomicAdd(&wg_iterations, process(gid));
  workgroupBarrier();
  if (li == 0u) { atomicAdd(&counters.iterations256, atomicLoad(&wg_iterations) >> 8u); }
}

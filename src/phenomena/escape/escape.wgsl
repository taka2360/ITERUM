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
// Iteration limit: kept on the GPU (Counters.max_iter) and raised there by
// decide_limit, so the GPU never waits for the CPU to read the counters back
// before it continues. Each invocation reads it once at its start. The rule is
// the CPU's former one: on sample 0, once no pixel is running, double the
// limit if a noticeable share of escapes came late (in the upper half of the
// range), or if nothing escaped yet (up to 2^17); never beyond the cap, and
// never when no pixel stopped at the limit. The job aggregates it uses count
// each event once: escaped/late/interior/cycling when a pixel reaches that
// status; raising the limit resets late and returns interior to the pixels
// that cannot resume (cycling).
//
// Status: 0 running, 1 escaped, 2 stopped at the iteration limit (not proven
// interior; it resumes when the limit is raised, so its full state is kept),
// 3 numeric anomaly, 4 cycling (see below), 5 waiting at the end of a
// truncated reference orbit for its extension (see orbit_wait), 6 proven
// interior (see "Proven interior"). BLA spans may run past the current limit,
// so the path a pixel takes depends only on its state (z, or δ and m), never
// on n or the limit: raising the limit and resuming equals starting with the
// higher limit.
//
// Proven interior: when a pixel starts, the closed-form tests for the main
// cardioid and the period-2 disk are evaluated with a rigorous error bound
// (PROOF.md, section 1); in MODE 2 also the patch of its tile, if the CPU
// proved one: |λ|² − 1 of a component's cycle as a polynomial with a bound
// (section 2). Patches arrive while the image is computed; the retest pass
// then checks the pixels that have not finished. A pixel that passes is
// interior for certain. It
// iterates only as long as its interior shading needs (shading_steps), then
// finishes with status 6 whatever the iteration limit. The proof decides the
// status; the shading of such a pixel is an approximation (PROOF.md, section 3).
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
//
// Compaction: once few pixels are still running, a list of their indices is
// built (build_list) and only those are dispatched (iterate_list, indirect),
// each appending itself to the next list if it is still running, so SIMD
// lanes are not spent on finished pixels. Results do not depend on the order
// or grouping in which pixels advance.

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
  max_iter_cap: u32,     // the limit is never raised beyond this
  budget: u32,
  orbit_len: u32,
  bla_levels: u32,
  sample: u32,
  list_in: u32,          // list mode: which of the two lists is read (0 or 1)
  // 1: the orbit is truncated and a longer one with the same prefix is coming.
  // Rebasing at its end would replace δ by the full z (|z| ~ 1), losing the
  // pixel's offset in f32, so pixels wait there instead and continue exactly
  // when the extension arrives. 0: rebase at the end (complete orbits).
  orbit_wait: u32,
  ref_x: f32, ref_y: f32,  // reference point, rounded (perturbation modes; shading only)
  // Fewest steps a proven-interior pixel iterates for its shading; 0 disables
  // the closed-form interior tests (Julia, or no exact reference point).
  trap_iters: u32,
  // Raises of the limit the CPU has seen; until it sees a new one (and lowers
  // its budget for the pixels that rejoin), budgets are capped by budget_safe.
  acked_raises: u32,
  budget_safe: u32,
  list_mode: u32,        // decide_limit: rebuild list 1 − list_in after a raise
  patches_on: u32,       // the patch uniform holds proofs for this view
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
  // Loop steps actually executed (a BLA span counts as one), 64-bit: low, high word.
  steps_lo: atomic<u32>,
  steps_hi: atomic<u32>,
  waiting: atomic<u32>,           // unfinished pixels that wait for a longer orbit
  proven: atomic<u32>,            // pixels finished as proven interior (also counted in interior)
  // Per job (the CPU clears these when a job starts; only decide_limit writes them).
  max_iter: u32,                  // current iteration limit
  raises: u32,                    // times decide_limit raised it in this job
  raised_now: u32,                // decide_limit raised it in this frame
  frame_max_iter: u32,            // the limit this frame's iteration used
  agg_escaped: u32,
  agg_late: u32,
  agg_interior: u32,
  agg_cycling: u32,
  agg_proven: u32,
  // The totals the latest raise was decided on (diagnostics).
  raise_escaped: u32,
  raise_late: u32,
  raise_paused: u32,
}

var<workgroup> wg_iterations: atomic<u32>;
var<workgroup> wg_steps: atomic<u32>;
var<workgroup> wg_keep: atomic<u32>;
var<workgroup> wg_base: u32;
// Iteration budget of the pixel being processed (per progressive step).
var<private> budget: u32;
// Iteration limit, read once per invocation (see "Iteration limit").
var<private> max_iter: u32;
// Iterations skipped by BLA beyond one step per span (pixel being processed).
var<private> bla_saved: u32;
// Loop steps executed for the pixel being processed (set by process_pixel).
var<private> loop_steps: u32;

// List entries: pixel index | step << 30. ctl.count: entries in list 0, 1.
// args (bound only to list_args, used as the indirect buffer): dispatch
// arguments (x, y, z, _) for list 0, then for list 1.
struct ListCtl { count: array<atomic<u32>, 4> }
// args[8..11]: indirect arguments of the list rebuild decide_limit requests.
const LIST_GROUP: u32 = 64u;
const MAX_GROUPS_X: u32 = 32768u;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> state: array<PixelState>;
@group(0) @binding(2) var<storage, read_write> samples: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> counters: Counters;
@group(0) @binding(4) var<storage, read> orbit: array<OrbitPt>;
@group(0) @binding(5) var<storage, read> bla: array<Bla>;
@group(0) @binding(6) var<storage, read> bla_off: array<u32>;
@group(0) @binding(7) var<storage, read_write> list0: array<u32>;
@group(0) @binding(8) var<storage, read_write> list1: array<u32>;
@group(0) @binding(9) var<storage, read_write> ctl: ListCtl;
// Closed-form interior tests: coefficients of g (G_TERMS) then h (H_TERMS)
// around the reference point (interior.ts).
struct Term { m: f32, e: i32, i: u32, j: u32 }
const G_TERMS: u32 = 6u;
const H_TERMS: u32 = 15u;
@group(0) @binding(11) var<uniform> terms: array<Term, 21>;
// Component proofs, one per tile of a PATCH_GRID² grid over the image
// (component.rs, PATCH_BYTES): |λ(u)|² − 1 for u = (δc − d0) / 2^rho_e,
// |u| ≤ 1, with `bound` covering everything the 15 terms leave out.
struct PatchHead { d0: vec2<f32>, rho_e: i32, period: u32, bound_m: f32, bound_e: i32, _a: u32, _b: u32 }
struct PatchRec { head: PatchHead, t: array<Term, 15> }
const PATCH_GRID: u32 = 8u;
@group(0) @binding(13) var<uniform> patches: array<PatchRec, 64>;
@group(0) @binding(10) var<storage, read_write> args: array<u32, 12>;

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

// The pixel's parameter, exactly as the iteration uses it: c (MODE 0), or
// δc = c − reference (MODE 2 f32, MODE 3 floatexp).
fn direct_c(pix: vec2<u32>) -> vec2<f32> { return P.center + rel_pos(pix) * spacing_f(); }
fn perturb_dc(pix: vec2<u32>) -> vec2<f32> { return (rel_pos(pix) + P.cmr) * spacing_f(); }
fn deep_dc(pix: vec2<u32>) -> CF { return cf_scale(cf_make(rel_pos(pix) + P.cmr, 0), spacing_fe()); }

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
  if ((*s).n > max_iter / 2u) { atomicAdd(&counters.late_escapes, 1u); }
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

// ---- proven interior (PROOF.md, section 1) ----------------------------------

const PROVEN_CARDIOID: u32 = 0x80000000u; // flags in PixelState.rebases
const PROVEN_P2: u32 = 0x40000000u;
const PROVEN_COMPONENT: u32 = 0x20000000u;
const PROVEN_ANY: u32 = 0xe0000000u;

fn fe_mul(a: FE, b: FE) -> FE { return fe_make(a.m * b.m, a.e + b.e); }
fn fe_add(a: FE, b: FE) -> FE {
  let r = cf_add(CF(vec2<f32>(a.m, 0.0), a.e), CF(vec2<f32>(b.m, 0.0), b.e));
  return FE(r.v.x, r.e);
}

// Value v of the polynomial terms[start..start+count] at (dx, dy) and the sum
// t of the absolute values of its terms.
struct PolyVal { v: FE, t: FE }

fn poly_eval(start: u32, count: u32, dx: FE, dy: FE) -> PolyVal {
  var px: array<FE, 5>;
  var py: array<FE, 5>;
  px[0] = FE(0.5, 1);
  py[0] = FE(0.5, 1);
  for (var k = 1u; k < 5u; k++) {
    px[k] = fe_mul(px[k - 1u], dx);
    py[k] = fe_mul(py[k - 1u], dy);
  }
  var v = FE(0.0, 0);
  var t = FE(0.0, 0);
  for (var k = 0u; k < count; k++) {
    let a = terms[start + k];
    if (a.m == 0.0) { continue; }
    let term = fe_mul(fe_mul(FE(a.m, a.e), px[a.i]), py[a.j]);
    v = fe_add(v, term);
    t = fe_add(t, FE(abs(term.m), term.e));
  }
  return PolyVal(v, t);
}

// True only if the polynomial is certainly negative at (dx, dy): computed
// value plus the error bound 2^-16 · Σ|terms| < 0.
fn poly_negative(start: u32, count: u32, dx: FE, dy: FE) -> bool {
  let p = poly_eval(start, count, dx, dy);
  if (p.t.m == 0.0) { return false; }
  let w = fe_add(p.v, FE(p.t.m, p.t.e - 16));
  return w.m < 0.0;
}

// The pixel's c (MODE 0) or δc (MODE 2, 3) as floatexp components; ok is
// false if a component is subnormal (fe_make could flush it).
struct PixelD { dx: FE, dy: FE, ok: bool }

fn pixel_d(pix: vec2<u32>) -> PixelD {
  var d = vec2<f32>(0.0);
  var e = 0;
  if (MODE == 0u) {
    d = direct_c(pix);
  } else if (MODE == 2u) {
    d = perturb_dc(pix);
  } else {
    let dc = deep_dc(pix);
    d = dc.v;
    e = dc.e;
  }
  return PixelD(fe_make(d.x, e), fe_make(d.y, e), !subnormal(d.x) && !subnormal(d.y));
}

fn subnormal(x: f32) -> bool { return x != 0.0 && abs(x) < 1.17549435e-38; }

// Closed-form interior flags of the pixel (0 if not proven).
fn closed_form_interior(pix: vec2<u32>) -> u32 {
  let d = pixel_d(pix);
  // A subnormal input could be flushed by fe_make (frexp): no proof then.
  if (!d.ok) { return 0u; }
  if (poly_negative(G_TERMS, H_TERMS, d.dx, d.dy)) { return PROVEN_CARDIOID; }
  if (poly_negative(0u, G_TERMS, d.dx, d.dy)) { return PROVEN_P2; }
  return 0u;
}

// Steps a proven pixel iterates for its shading (smallest |z|² of the orbit).
// Near the boundary of a component (multiplier |λ| = 1 − η, η small) the
// orbit keeps passing close to 0 for a while before it settles: measured at
// the period-2 boundary, the last such pass came within 5/√η steps. So the
// pixel runs 16/√η steps (at least P.trap_iters).
// η comes from the polynomial value, which is accurate where f32 c is not:
// period 2: |λ| = 4|c+1| = √(1 + 16 g) ≈ 1 + 8g; cardioid: η ≈ −h |λ'(c)| / |∇h|
// with λ'(c) = 2 / √(1 − 4c).
const NO_SHORTCUT: u32 = 0xffffffffu;

fn shading_steps(pix: vec2<u32>, flags: u32) -> u32 {
  let d = pixel_d(pix);
  var v: FE;
  var factor = 8.0;
  if ((flags & PROVEN_COMPONENT) != 0u) {
    // v = |λ|² − 1 ≈ −2η.
    v = patch_eval(pix).v;
    factor = 0.5;
  } else if ((flags & PROVEN_CARDIOID) != 0u) {
    v = poly_eval(G_TERMS, H_TERMS, d.dx, d.dy).v;
    let c = approx_c(pix);
    let u = c.x - 0.25;
    let q = u * u + c.y * c.y;
    let grad = length(vec2<f32>((2.0 * q + u) * 2.0 * u + q, (2.0 * q + u) * 2.0 * c.y - 0.5 * c.y));
    let s = length(csqrt(vec2<f32>(1.0, 0.0) - 4.0 * c));
    // Near the cusp at 1/4 this estimate breaks down: no shortcut there.
    if (grad < 1e-3 || s < 1e-3) { return NO_SHORTCUT; }
    factor = 2.0 / (s * grad);
  } else {
    v = poly_eval(0u, G_TERMS, d.dx, d.dy).v;
  }
  if (v.m >= 0.0) { return NO_SHORTCUT; }
  let eta_log2 = log2(-v.m) + f32(v.e) + log2(factor);
  let steps_log2 = 4.0 - 0.5 * eta_log2;
  if (steps_log2 >= 31.0) { return NO_SHORTCUT; }
  return max(P.trap_iters, u32(exp2(max(steps_log2, 0.0))));
}

// The pixel's patch: |λ|² − 1 (v) and its error bound (e), if it has one.
struct PatchVal { ok: bool, v: FE, e: FE }

fn patch_eval(pix: vec2<u32>) -> PatchVal {
  let none = PatchVal(false, FE(0.0, 0), FE(0.0, 0));
  if (MODE != 2u || P.patches_on == 0u) { return none; }
  let tx = min(pix.x * PATCH_GRID / P.size.x, PATCH_GRID - 1u);
  let ty = min(pix.y * PATCH_GRID / P.size.y, PATCH_GRID - 1u);
  let rec = ty * PATCH_GRID + tx;
  let h = patches[rec].head;
  if (h.period == 0u) { return none; }
  // u differs from the exact (δc − d0)/ρ by at most 2^-21 (f32 rounding of
  // d0 and of the difference; the CPU added that to the bound), so
  // |u|² ≤ 0.9999 keeps the exact u inside the proven disc.
  let u = ldexp(perturb_dc(pix) - h.d0, vec2<i32>(-h.rho_e));
  if (dot(u, u) > 0.9999 || subnormal(u.x) || subnormal(u.y)) { return none; }
  var px: array<FE, 5>;
  var py: array<FE, 5>;
  px[0] = FE(0.5, 1);
  py[0] = FE(0.5, 1);
  let ux = fe_make(u.x, 0);
  let uy = fe_make(u.y, 0);
  for (var k = 1u; k < 5u; k++) {
    px[k] = fe_mul(px[k - 1u], ux);
    py[k] = fe_mul(py[k - 1u], uy);
  }
  var v = FE(0.0, 0);
  var t = FE(0.0, 0);
  for (var k = 0u; k < 15u; k++) {
    let a = patches[rec].t[k];
    if (a.m == 0.0) { continue; }
    let term = fe_mul(fe_mul(FE(a.m, a.e), px[a.i]), py[a.j]);
    v = fe_add(v, term);
    t = fe_add(t, FE(abs(term.m), term.e));
  }
  // Evaluation error (as for the closed forms) plus the patch's bound; the
  // CPU sends the bound with 2^-18 to spare, more than this addition can lose.
  return PatchVal(true, v, fe_add(FE(t.m, t.e - 16), FE(h.bound_m, h.bound_e)));
}

fn patch_interior(pix: vec2<u32>) -> u32 {
  let p = patch_eval(pix);
  if (!p.ok) { return 0u; }
  let w = fe_add(p.v, p.e);
  if (w.m < 0.0) { return PROVEN_COMPONENT; }
  return 0u;
}

fn csqrt(a: vec2<f32>) -> vec2<f32> {
  let r = length(a);
  let x = sqrt(max(0.0, 0.5 * (r + a.x)));
  var y = sqrt(max(0.0, 0.5 * (r - a.x)));
  if (a.y < 0.0) { y = -y; }
  return vec2<f32>(x, y);
}

// The pixel's parameter, rounded to f32 (shading only).
fn approx_c(pix: vec2<u32>) -> vec2<f32> {
  if (MODE == 0u) { return direct_c(pix); }
  if (MODE == 2u) { return vec2<f32>(P.ref_x, P.ref_y) + perturb_dc(pix); }
  return vec2<f32>(P.ref_x, P.ref_y) + cf_to_vec2(deep_dc(pix));
}

// Smallest |z|² on the attracting cycle (fixed point, or the 2-cycle; other
// components: not known here).
fn cycle_trap(c: vec2<f32>, flags: u32) -> f32 {
  if ((flags & PROVEN_COMPONENT) != 0u) { return 1e30; }
  if ((flags & PROVEN_CARDIOID) != 0u) {
    let z = 0.5 * (vec2<f32>(1.0, 0.0) - csqrt(vec2<f32>(1.0, 0.0) - 4.0 * c));
    return dot(z, z);
  }
  let w = csqrt(vec2<f32>(-3.0, 0.0) - 4.0 * c);
  let a = 0.5 * (vec2<f32>(-1.0, 0.0) + w);
  let b = 0.5 * (vec2<f32>(-1.0, 0.0) - w);
  return min(dot(a, a), dot(b, b));
}

// Iteration count at which the pixel stops: the limit, or for a proven pixel
// its shading steps if they come first.
fn stop_at(flags: u32, pix: vec2<u32>) -> u32 {
  if ((flags & PROVEN_ANY) != 0u) { return min(max_iter, shading_steps(pix, flags)); }
  return max_iter;
}

// Stops at stop_at(): a proven pixel that has iterated its shading steps
// finishes; any other pauses at the limit (a proven one too: its shading is
// not complete, so it resumes like any pixel when the limit is raised).
fn stop_pixel(s: ptr<function, PixelState>, pix: vec2<u32>, idx: u32) {
  if (((*s).rebases & PROVEN_ANY) != 0u && (*s).n >= shading_steps(pix, (*s).rebases)) {
    (*s).status = 6u;
    (*s).trap = min((*s).trap, cycle_trap(approx_c(pix), (*s).rebases));
    samples[P.sample * P.size.x * P.size.y + idx] = vec2<f32>(-1.0, sqrt((*s).trap));
    atomicAdd(&counters.interior, 1u);
    atomicAdd(&counters.proven, 1u);
    return;
  }
  pause_at_limit(s, idx);
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
  if (MODE == 0u) { c = direct_c(pix); gc = sp; }
  let stop = stop_at((*s).rebases, pix);
  var cyc = z;          // Brent: saved state, steps since saving, power of two
  var cyc_len = 0u;
  var cyc_pow = 1u;
  for (var k = 0u; k < budget; k++) {
    if (n >= stop) {
      (*s).dr = z.x; (*s).di = z.y; (*s).gr = g.x; (*s).gi = g.y; (*s).n = n;
      stop_pixel(s, pix, idx);
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
  let dc = perturb_dc(pix);
  let last = P.orbit_len - 1u;
  let stop = stop_at((*s).rebases, pix);
  // Z_m, carried from step to step: each step reads only the next point.
  var zm = orbit_f(m);
  var cyc_d = d;        // Brent: saved state (δ, m)
  var cyc_m = m;
  var cyc_len = 0u;
  var cyc_pow = 1u;
  for (var k = 0u; k < budget; k++) {
    if (n >= stop) {
      (*s).dr = d.x; (*s).di = d.y; (*s).gr = g.x; (*s).gi = g.y; (*s).m = m; (*s).n = n;
      (*s).rebases += rebases;
      stop_pixel(s, pix, idx);
      atomicAdd(&counters.rebases, rebases);
      if (skipped > 0u) { atomicAdd(&counters.bla_steps, skipped); }
      return;
    }
    var stepped = false;
    if (m > 0u && P.bla_levels > 0u) {
      let mm = m - 1u;
      var l = min(countTrailingZeros(mm), P.bla_levels);
      let d2 = dot(d, d);
      // Radii shrink with the level at a given start (merge: R ≤ R of the
      // first half), so if the 2-step entry rejects δ, every longer one does.
      if (l > 1u) {
        let i1 = bla_off[0] + (mm >> 1u);
        if (i1 < bla_off[1]) {
          let e1 = bla[i1];
          if (!(d2 < ldexp(e1.r2m, e1.r2e))) { l = 0u; }
        }
      }
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
            bla_saved += span - 1u;
            stepped = true;
            break;
          }
        }
        l--;
      }
    }
    if (!stepped) {
      g = 2.0 * cmul(zm + d, g) + vec2<f32>(sp, 0.0);
      d = cmul(2.0 * zm + d, d) + dc;
      m++;
      n++;
    }
    zm = orbit_f(m);
    let z = zm + d;
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
      zm = orbit_f(0u);
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
  let dc = deep_dc(pix);
  let last = P.orbit_len - 1u;
  let stop = stop_at((*s).rebases, pix);
  // Z_m, carried from step to step: each step reads only the next point.
  var zm = orbit_cf(m);
  var cyc_d = d;        // Brent: saved state (δ, m)
  var cyc_m = m;
  var cyc_len = 0u;
  var cyc_pow = 1u;
  for (var k = 0u; k < budget; k++) {
    if (n >= stop) {
      (*s).dr = d.v.x; (*s).di = d.v.y; (*s).de = d.e;
      (*s).gr = g.v.x; (*s).gi = g.v.y; (*s).ge = g.e;
      (*s).m = m; (*s).n = n;
      (*s).rebases += rebases;
      stop_pixel(s, pix, idx);
      atomicAdd(&counters.rebases, rebases);
      if (skipped > 0u) { atomicAdd(&counters.bla_steps, skipped); }
      return;
    }
    var stepped = false;
    if (m > 0u && P.bla_levels > 0u) {
      let mm = m - 1u;
      var l = min(countTrailingZeros(mm), P.bla_levels);
      let d2 = cf_norm2(d);
      // As in MODE 2: a rejection by the 2-step entry rejects all longer ones.
      if (l > 1u) {
        let i1 = bla_off[0] + (mm >> 1u);
        if (i1 < bla_off[1]) {
          let e1 = bla[i1];
          if (!fe_less_pos(d2, FE(e1.r2m, e1.r2e))) { l = 0u; }
        }
      }
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
            bla_saved += span - 1u;
            stepped = true;
            break;
          }
        }
        l--;
      }
    }
    if (!stepped) {
      g = cf_add(cf_mul(cf_add(zm, d), cf_scale(g, FE(0.5, 2))), sp_c);
      d = cf_add(cf_mul(cf_add(cf_scale(zm, FE(0.5, 2)), d), d), dc);
      m++;
      n++;
    }
    zm = orbit_cf(m);
    let z = cf_add(zm, d);
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
      zm = orbit_cf(0u);
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

// Running (or waiting for a longer orbit) after this pass: stays in the list.
var<private> still_active: bool;

fn step_of(pix: vec2<u32>) -> u32 {
  if ((pix.x & 3u) == 0u && (pix.y & 3u) == 0u) { return 0u; }
  if ((pix.x & 1u) == 0u && (pix.y & 1u) == 0u) { return 1u; }
  return 2u;
}

// Advances one pixel of progressive step `step` (the budget P.budget is the
// step-2 budget; coarser steps get more); returns the number of iterations it moved forward.
fn process_pixel(pix: vec2<u32>, step: u32) -> u32 {
  still_active = false;
  bla_saved = 0u;
  loop_steps = 0u;
  budget = max(16u, P.budget << (2u - step));
  if (counters.raises != P.acked_raises) { budget = min(budget, max(16u, P.budget_safe << (2u - step))); }
  let idx = pix.y * P.size.x + pix.x;
  var s = state[idx];
  if (s.job != P.job) {
    s = init_state(pix);
    if (P.trap_iters != 0u && MODE != 1u) {
      s.rebases = closed_form_interior(pix);
      if (s.rebases == 0u) { s.rebases = patch_interior(pix); }
    }
  }
  // Resume pixels stopped at an earlier, lower limit.
  if (s.status == 2u && s.n < max_iter) { s.status = 0u; }
  // Waiting at the end of a truncated orbit: continue once it is longer (or
  // no extension is coming).
  if (s.status == 5u) {
    if (s.m + 1u < P.orbit_len || P.orbit_wait == 0u) {
      s.status = 0u;
    } else {
      atomicAdd(&counters.unfinished, 1u);
      atomicAdd(&counters.waiting, 1u);
      if (step == 0u) { atomicAdd(&counters.unfinished_coarse, 1u); }
      if (step == 1u) { atomicAdd(&counters.unfinished_mid, 1u); }
      still_active = true;
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
  loop_steps = s.n - n0 - bla_saved;
  if (s.status == 5u) { atomicAdd(&counters.waiting, 1u); }
  if (s.status == 0u || s.status == 5u) {
    if (step == 0u) { atomicAdd(&counters.unfinished_coarse, 1u); }
    if (step == 1u) { atomicAdd(&counters.unfinished_mid, 1u); }
    still_active = true;
  }
  return min(s.n - n0, 1u << 25u);
}

// Full-grid pass over the pixels of progressive step P.step.
fn process(gid: vec3<u32>) -> u32 {
  let stride = 1u << (2u - P.step);
  let pix = gid.xy * stride;
  if (pix.x >= P.size.x || pix.y >= P.size.y) { return 0u; }
  if (P.step > 0u) {
    let coarse = stride * 2u;
    if (pix.x % coarse == 0u && pix.y % coarse == 0u) { return 0u; }
  }
  return process_pixel(pix, P.step);
}

// Adds to the 64-bit step counter (carry detected from the returned old value).
fn add_steps(v: u32) {
  if (v == 0u) { return; }
  let old = atomicAdd(&counters.steps_lo, v);
  if (old > 0xffffffffu - v) { atomicAdd(&counters.steps_hi, 1u); }
}

@compute @workgroup_size(8, 8)
fn iterate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  max_iter = counters.max_iter;
  if (li == 0u) { atomicStore(&wg_iterations, 0u); atomicStore(&wg_steps, 0u); }
  workgroupBarrier();
  // One global atomic per workgroup keeps the throughput counter cheap.
  loop_steps = 0u;
  atomicAdd(&wg_iterations, process(gid));
  atomicAdd(&wg_steps, loop_steps);
  workgroupBarrier();
  if (li == 0u) {
    atomicAdd(&counters.iterations256, atomicLoad(&wg_iterations) >> 8u);
    add_steps(atomicLoad(&wg_steps));
  }
}

fn list_read(which: u32, i: u32) -> u32 {
  if (which == 0u) { return list0[i]; }
  return list1[i];
}

fn list_write(which: u32, i: u32, v: u32) {
  if (which == 0u) { list0[i] = v; } else { list1[i] = v; }
}

// Appends `entry` (if `keep`) to list `out` with one global atomic per workgroup.
// Must be called in uniform control flow by every invocation of the group.
fn list_append(li: u32, keep: bool, entry: u32, out: u32) {
  if (li == 0u) { atomicStore(&wg_keep, 0u); }
  workgroupBarrier();
  var slot = 0u;
  if (keep) { slot = atomicAdd(&wg_keep, 1u); }
  workgroupBarrier();
  if (li == 0u) {
    let n = atomicLoad(&wg_keep);
    if (n > 0u) { wg_base = atomicAdd(&ctl.count[out], n); }
  }
  workgroupBarrier();
  if (keep) { list_write(out, wg_base + slot, entry); }
}

// Builds list P.list_in from the full grid: every pixel the current job still
// has to advance. Dispatched 2D like iterate_list.
@compute @workgroup_size(64)
fn build_list(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  max_iter = counters.max_iter;
  let total = P.size.x * P.size.y;
  let idx = (wid.y * nwg.x + wid.x) * LIST_GROUP + li;
  var keep = false;
  var entry = 0u;
  if (idx < total) {
    let pix = vec2<u32>(idx % P.size.x, idx / P.size.x);
    let s = state[idx];
    keep = s.job != P.job || s.status == 0u || s.status == 5u || (s.status == 2u && s.n < max_iter);
    entry = idx | (step_of(pix) << 30u);
  }
  list_append(li, keep, entry, P.list_in);
}

// Advances the pixels of list P.list_in; those still running go to the other list.
@compute @workgroup_size(64)
fn iterate_list(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  max_iter = counters.max_iter;
  let i = (wid.y * nwg.x + wid.x) * LIST_GROUP + li;
  let count = atomicLoad(&ctl.count[P.list_in]);
  if (li == 0u) { atomicStore(&wg_iterations, 0u); atomicStore(&wg_steps, 0u); }
  workgroupBarrier();
  var entry = 0u;
  var keep = false;
  if (i < count) {
    entry = list_read(P.list_in, i);
    let idx = entry & 0x3fffffffu;
    let pix = vec2<u32>(idx % P.size.x, idx / P.size.x);
    atomicAdd(&wg_iterations, process_pixel(pix, entry >> 30u));
    atomicAdd(&wg_steps, loop_steps);
    keep = still_active;
  }
  workgroupBarrier();
  if (li == 0u) {
    atomicAdd(&counters.iterations256, atomicLoad(&wg_iterations) >> 8u);
    add_steps(atomicLoad(&wg_steps));
  }
  list_append(li, keep, entry, 1u - P.list_in);
}

// Component proofs arrived: checks the current job's pixels that have not
// finished. A pixel stopped at the limit that is proven and has iterated its
// shading steps finishes now (it was counted as interior when it stopped).
@compute @workgroup_size(8, 8)
fn retest(@builtin(global_invocation_id) gid: vec3<u32>) {
  max_iter = counters.max_iter;
  let pix = gid.xy;
  if (pix.x >= P.size.x || pix.y >= P.size.y || P.trap_iters == 0u) { return; }
  let idx = pix.y * P.size.x + pix.x;
  var s = state[idx];
  if (s.job != P.job || (s.rebases & PROVEN_ANY) != 0u) { return; }
  if (s.status != 0u && s.status != 2u && s.status != 5u) { return; }
  let f = patch_interior(pix);
  if (f == 0u) { return; }
  s.rebases |= f;
  if (s.status == 2u && s.n >= shading_steps(pix, s.rebases)) {
    s.status = 6u;
    samples[P.sample * P.size.x * P.size.y + idx] = vec2<f32>(-1.0, sqrt(s.trap));
    atomicAdd(&counters.proven, 1u);
  }
  state[idx] = s;
}

// Dispatch size (x, y) for `groups` workgroups (x ≤ MAX_GROUPS_X).
fn groups_2d(groups: u32) -> vec2<u32> {
  let x = min(groups, MAX_GROUPS_X);
  var y = 0u;
  if (x > 0u) { y = (groups + x - 1u) / x; }
  return vec2<u32>(x, y);
}

// After this frame's iteration: adds its events to the job aggregates and
// raises the limit when the rule says so (see "Iteration limit"). In list
// mode a raise also empties the list for the next frame and requests its
// rebuild from the full grid (pixels stopped at the old limit rejoin).
@compute @workgroup_size(1)
fn decide_limit() {
  let cur = counters.max_iter;
  counters.frame_max_iter = cur;
  counters.agg_escaped += atomicLoad(&counters.escaped);
  counters.agg_late += atomicLoad(&counters.late_escapes);
  counters.agg_interior += atomicLoad(&counters.interior);
  counters.agg_cycling += atomicLoad(&counters.cycling);
  counters.agg_proven += atomicLoad(&counters.proven);
  counters.raised_now = 0u;
  args[8] = 0u; args[9] = 1u; args[10] = 1u;
  if (P.sample != 0u || atomicLoad(&counters.unfinished) != 0u) { return; }
  if (cur >= P.max_iter_cap || counters.agg_interior == counters.agg_cycling + counters.agg_proven) { return; }
  var more: bool;
  if (counters.agg_escaped == 0u) {
    // Nothing escaped: interior cannot be told from slow exterior. Keep
    // looking, but not forever.
    more = cur < (1u << 17u);
  } else {
    more = counters.agg_late > 16u && counters.agg_late * 500u > counters.agg_escaped;
  }
  if (!more) { return; }
  counters.max_iter = min(P.max_iter_cap, cur * 2u);
  counters.raises += 1u;
  counters.raise_escaped = counters.agg_escaped;
  counters.raise_late = counters.agg_late;
  counters.raise_paused = counters.agg_interior - counters.agg_cycling - counters.agg_proven;
  counters.raised_now = 1u;
  counters.agg_interior = counters.agg_cycling + counters.agg_proven;
  counters.agg_late = 0u;
  if (P.list_mode != 0u) {
    atomicStore(&ctl.count[1u - P.list_in], 0u);
    let g = groups_2d((P.size.x * P.size.y + LIST_GROUP - 1u) / LIST_GROUP);
    args[8] = g.x; args[9] = g.y;
  }
}

// Indirect dispatch arguments for list P.list_in (2D: x ≤ MAX_GROUPS_X).
@compute @workgroup_size(1)
fn list_args() {
  let which = P.list_in;
  let g = groups_2d((atomicLoad(&ctl.count[which]) + LIST_GROUP - 1u) / LIST_GROUP);
  args[which * 4u] = g.x;
  args[which * 4u + 1u] = g.y;
  args[which * 4u + 2u] = 1u;
}

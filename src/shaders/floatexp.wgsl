// Extended-exponent floats ("floatexp"): value = m * 2^e.
// Shared layout and convention with src-tauri/src/deepzoom/format.rs and the
// CPU model in src/core/floatexp.ts: mantissas follow frexp, |m| in [0.5, 1),
// zero is (0, 0). Complex values share one exponent.

struct FE { m: f32, e: i32 }
struct CF { v: vec2<f32>, e: i32 }

fn fe_make(m: f32, e: i32) -> FE {
  if (m == 0.0) { return FE(0.0, 0); }
  let f = frexp(m);
  return FE(f.fract, e + f.exp);
}

fn cf_make(v: vec2<f32>, e: i32) -> CF {
  let m = max(abs(v.x), abs(v.y));
  if (m == 0.0) { return CF(vec2<f32>(0.0), 0); }
  let k = frexp(m).exp;
  return CF(ldexp(v, vec2<i32>(-k)), e + k);
}

fn cf_is_zero(a: CF) -> bool { return a.v.x == 0.0 && a.v.y == 0.0; }

fn cf_mul(a: CF, b: CF) -> CF {
  return cf_make(vec2<f32>(a.v.x * b.v.x - a.v.y * b.v.y, a.v.x * b.v.y + a.v.y * b.v.x), a.e + b.e);
}

fn cf_sqr(a: CF) -> CF {
  return cf_make(vec2<f32>(a.v.x * a.v.x - a.v.y * a.v.y, 2.0 * a.v.x * a.v.y), 2 * a.e);
}

fn cf_scale(a: CF, s: FE) -> CF {
  return cf_make(a.v * s.m, a.e + s.e);
}

fn cf_add(a: CF, b: CF) -> CF {
  if (cf_is_zero(a)) { return b; }
  if (cf_is_zero(b)) { return a; }
  let d = a.e - b.e;
  if (d > 30) { return a; }
  if (d < -30) { return b; }
  if (d >= 0) { return cf_make(a.v + ldexp(b.v, vec2<i32>(-d)), a.e); }
  return cf_make(ldexp(a.v, vec2<i32>(d)) + b.v, b.e);
}

fn cf_norm2(a: CF) -> FE { return fe_make(dot(a.v, a.v), 2 * a.e); }

// a < b for non-negative normalized values (e.g. squared magnitudes).
fn fe_less_pos(a: FE, b: FE) -> bool {
  if (a.m == 0.0) { return b.m > 0.0; }
  if (b.m == 0.0) { return false; }
  if (a.e != b.e) { return a.e < b.e; }
  return a.m < b.m;
}

fn cf_to_vec2(a: CF) -> vec2<f32> { return ldexp(a.v, vec2<i32>(a.e)); }
fn fe_to_f32(a: FE) -> f32 { return ldexp(a.m, a.e); }
fn fe_log2(a: FE) -> f32 { return log2(abs(a.m)) + f32(a.e); }

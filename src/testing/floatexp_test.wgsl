// GPU side of the floatexp unit test: evaluates the shared floatexp routines
// on test vectors so the CPU model (src/core/floatexp.ts) can check them.

struct Case { ar: f32, ai: f32, ae: i32, _p0: u32, br: f32, bi: f32, be: i32, _p1: u32 }

@group(0) @binding(0) var<storage, read> cases: array<Case>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<u32>>;

#include floatexp

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&cases)) { return; }
  let c = cases[i];
  let a = cf_make(vec2<f32>(c.ar, c.ai), c.ae);
  let b = cf_make(vec2<f32>(c.br, c.bi), c.be);
  let s = cf_add(a, b);
  let m = cf_mul(a, b);
  let n = cf_norm2(s);
  let less = fe_less_pos(cf_norm2(s), cf_norm2(b));
  out[i * 3u] = vec4<u32>(bitcast<u32>(s.v.x), bitcast<u32>(s.v.y), bitcast<u32>(s.e), bitcast<u32>(m.e));
  out[i * 3u + 1u] = vec4<u32>(bitcast<u32>(m.v.x), bitcast<u32>(m.v.y), bitcast<u32>(n.m), bitcast<u32>(n.e));
  out[i * 3u + 2u] = vec4<u32>(select(0u, 1u, less), 0u, 0u, 0u);
}

// Lorenz attractor as comets of light.
//
// The first `leaders` particles simply follow the flow forever. Every other
// particle belongs to one leader; when its life runs out it is reborn at its
// leader's position with a tiny offset. Chaos then stretches each newborn
// group into a ribbon that trails behind the leader, so the attractor is drawn
// by many slowly diverging threads instead of a uniform haze.
//
// Pass order inside one frame (separate dispatches, so they are ordered):
//   1. integrate: each particle takes `steps` RK4 steps and, after each, adds its
//      colour to the accumulation buffer with atomicAdd (fixed point, 8
//      fractional bits);
//   2. display (fragment): tone-maps the accumulated light;
//   3. decay: every element is scaled by `decay` by exactly one invocation and
//      clamped to 2^30, so the next frame's additions cannot overflow u32
//      (one frame adds at most particles·steps·256 ≤ 2^30 to a pixel).

struct U {
  view_proj: mat4x4<f32>,
  size: vec2<u32>,
  count: u32,
  steps: u32,
  dt: f32,
  sigma: f32,
  rho: f32,
  beta: f32,
  decay: f32,
  exposure: f32,
  pal_offset: f32,
  speed_scale: f32,
  background: vec4<f32>,
  cam_pos: vec4<f32>,
  leaders: u32,
  frame_dt: f32,
  frame: u32,
  draw: u32,   // 0 during pre-roll: simulate without drawing
}

struct Particle { p: vec3<f32>, seed: f32, age: f32, life: f32, _a: f32, _b: f32 }

@group(0) @binding(0) var<uniform> P: U;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(3) var palette: texture_2d<f32>;
@group(0) @binding(4) var pal_sampler: sampler;

fn lorenz(p: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(P.sigma * (p.y - p.x), p.x * (P.rho - p.z) - p.y, p.x * p.y - P.beta * p.z);
}

fn rk4(p: vec3<f32>, h: f32) -> vec3<f32> {
  let k1 = lorenz(p);
  let k2 = lorenz(p + 0.5 * h * k1);
  let k3 = lorenz(p + 0.5 * h * k2);
  let k4 = lorenz(p + h * k3);
  return p + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

fn splat(p: vec3<f32>, speed: f32, seed: f32, alpha: f32) {
  let clip = P.view_proj * vec4<f32>(p, 1.0);
  if (clip.w <= 0.1) { return; }
  let ndc = clip.xy / clip.w;
  let px = vec2<i32>(floor((vec2<f32>(ndc.x, -ndc.y) * 0.5 + 0.5) * vec2<f32>(P.size)));
  if (px.x < 0 || px.y < 0 || px.x >= i32(P.size.x) || px.y >= i32(P.size.y)) { return; }
  // Colour from speed (slow arcs cool, fast swings warm), nudged per particle.
  let t = P.pal_offset + 0.3 + 0.42 * clamp(speed * P.speed_scale, 0.0, 1.0) + 0.04 * seed;
  let col = textureSampleLevel(palette, pal_sampler, vec2<f32>(t, 0.5), 0.0).rgb;
  // Nearer particles are slightly brighter: a gentle depth cue.
  let depth = clamp(1.6 - clip.w / P.cam_pos.w, 0.35, 1.0);
  let w = col * depth * alpha * 256.0;
  let i = (u32(px.y) * P.size.x + u32(px.x)) * 3u;
  atomicAdd(&accum[i], u32(w.r));
  atomicAdd(&accum[i + 1u], u32(w.g));
  atomicAdd(&accum[i + 2u], u32(w.b));
}

fn hash3(n: u32) -> vec3<f32> {
  var x = n * 747796405u + 2891336453u;
  var v = vec3<f32>(0.0);
  for (var i = 0u; i < 3u; i++) {
    x = x * 747796405u + 2891336453u;
    var w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    w = (w >> 22u) ^ w;
    v[i] = f32(w) / 4294967295.0 - 0.5;
  }
  return v;
}

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.count) { return; }
  var q = particles[i];
  if (i >= P.leaders) {
    q.age += P.frame_dt;
    if (q.age > q.life) {
      // Rebirth beside the leader (racing with its own update is harmless).
      let leader = particles[i % P.leaders];
      q.p = leader.p + hash3(i ^ (P.frame * 0x9e3779b9u)) * 0.02;
      q.age = 0.0;
    }
  }
  var p = q.p;
  for (var s = 0u; s < P.steps; s++) {
    p = rk4(p, P.dt);
    // Newborn particles fade in so rebirths never flash.
    let fade = clamp(q.age * 0.8, 0.0, 1.0);
    if (fade > 0.0 && P.draw != 0u) { splat(p, length(lorenz(p)), q.seed, fade); }
  }
  q.p = p;
  particles[i] = q;
}

// Moves freshly seeded particles onto the attractor before they are shown.
@compute @workgroup_size(256)
fn warmup(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.count) { return; }
  var q = particles[i];
  var p = q.p;
  for (var s = 0u; s < P.steps; s++) { p = rk4(p, P.dt); }
  q.p = p;
  particles[i] = q;
}

@compute @workgroup_size(256)
fn decay(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.size.x * P.size.y * 3u) { return; }
  let v = atomicLoad(&accum[i]);
  atomicStore(&accum[i], min(u32(f32(v) * P.decay), 1073741824u));
}

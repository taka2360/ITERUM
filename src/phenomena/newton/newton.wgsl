// Newton's method on p(z) = Π (z − r_k), coloured by the root each starting
// point converges to and how quickly it gets there.
// Using p'/p = Σ 1/(z − r_k), one step is z ← z − a / Σ 1/(z − r_k).

struct U {
  size: vec2<u32>,
  roots_n: u32,
  samples: u32,
  center: vec2<f32>,
  spacing: f32,
  relax: f32,
  pal_offset: f32,
  max_iter: u32,
  glow: f32,
  _pad: f32,
  roots: array<vec4<f32>, 4>,   // up to 8 roots, two per vec4
  interior: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: U;
@group(0) @binding(1) var palette: texture_2d<f32>;
@group(0) @binding(2) var pal_sampler: sampler;

#include fullscreen

fn root(k: u32) -> vec2<f32> {
  let v = P.roots[k / 2u];
  return select(v.xy, v.zw, (k & 1u) == 1u);
}

fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b);
}

const TOL: f32 = 1e-3;

fn sample_at(z0: vec2<f32>) -> vec3<f32> {
  var z = z0;
  var prev = z0;
  for (var n = 0u; n < P.max_iter; n++) {
    var s = vec2<f32>(0.0);
    var hit = 99u;
    for (var k = 0u; k < P.roots_n; k++) {
      let d = z - root(k);
      let d2 = dot(d, d);
      if (d2 < TOL * TOL) { hit = k; }
      s += vec2<f32>(d.x, -d.y) / max(d2, 1e-30);
    }
    if (hit != 99u) {
      // Continuous step count: interpolate the TOL crossing in log-log space
      // between the previous distance (> TOL) and this one (< TOL).
      let r = root(hit);
      let ld = log(max(length(z - r), 1e-30));
      let lp = log(max(length(prev - r), TOL));
      let lt = log(TOL);
      var frac = 0.0;
      if (n > 0u && lp > ld) { frac = clamp((lt - lp) / (ld - lp), 0.0, 1.0); }
      let t = max(f32(n) - 1.0 + frac, 0.0);
      // Basin colours come from the luminous middle of the palette, spread evenly.
      let hue = 0.32 + 0.46 * f32(hit) / max(f32(P.roots_n) - 1.0, 1.0);
      let base = textureSampleLevel(palette, pal_sampler, vec2<f32>(hue + 0.012 * t + P.pal_offset, 0.5), 0.0).rgb;
      // Points that settle quickly sink into shadow; slow ones, near basin
      // boundaries, carry the colour and a soft light.
      let g = 1.0 - exp(-0.18 * t);
      let shade = 0.07 + 0.93 * smoothstep(0.0, 1.0, g);
      let col = mix(P.interior.rgb, base, shade);
      return col + sqrt(base) * P.glow * pow(g, 6.0) * 0.5;
    }
    prev = z;
    z -= P.relax * cdiv(vec2<f32>(1.0, 0.0), s);
  }
  return P.interior.rgb;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let mid = vec2<f32>(P.size) * 0.5;
  var acc = vec3<f32>(0.0);
  // Rotated-grid supersampling.
  let offs = array<vec2<f32>, 4>(vec2<f32>(0.125, 0.375), vec2<f32>(-0.375, 0.125), vec2<f32>(0.375, -0.125), vec2<f32>(-0.125, -0.375));
  for (var i = 0u; i < P.samples; i++) {
    var o = vec2<f32>(0.0);
    if (P.samples > 1u) { o = offs[i]; }
    let p = in.pos.xy + o - mid;
    acc += sample_at(P.center + vec2<f32>(p.x, -p.y) * P.spacing);
  }
  return vec4<f32>(acc / f32(P.samples), 1.0);
}

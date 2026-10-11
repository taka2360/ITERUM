// CUDA port of escape.wgsl's MODE 2 (perturbation, f32 deltas) for the
// CUDA-versus-WebGPU benchmark (see cuda_bench.md). Same arithmetic, same
// resumable state, same BLA lookup; no progressive steps or samples: one
// sample (jitter 0) of every pixel, run until all pixels finish.
//
//   nvcc -O3 -arch=sm_89 --fmad=false -o perturb perturb.cu
//   perturb <dump prefix> [budget]
//
// The files <prefix>-* are written by the golden harness (ITERUM_DUMP_LIMIT):
// packet.bin (orbit packet, format.rs), view.txt (w h spacing_m spacing_e
// cmr_x cmr_y limit) and, for comparison, state.bin (status, n per pixel).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

struct Bla { float ar, ai; int ae; float br, bi; int be; float r2m; int r2e; };
struct OrbitPt { float re, im; int e; unsigned pad; };

struct Params {
  unsigned w, h;
  float sp_m; int sp_e;
  float cmr_x, cmr_y;
  unsigned max_iter, budget, orbit_len, bla_levels;
};

struct State {
  float dr, di, gr, gi, trap;
  unsigned m, n, status;  // status: 0 running, 1 escaped, 2 limit, 3 anomaly, 4 cycling
};

__constant__ unsigned c_bla_off[34];

__device__ __forceinline__ bool bad(float x) { return (__float_as_uint(x) & 0x7f800000u) == 0x7f800000u; }

__global__ void iterate(Params P, const OrbitPt* __restrict__ orbit, const Bla* __restrict__ bla,
                        State* __restrict__ st, float2* __restrict__ samples,
                        unsigned* __restrict__ active, unsigned long long* __restrict__ steps) {
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= P.w * P.h) return;
  State s = st[idx];
  if (s.status != 0u) return;
  unsigned px = idx % P.w, py = idx / P.w;
  float sp = ldexpf(P.sp_m, P.sp_e);
  float rx = (float)px + 0.5f - (float)P.w * 0.5f;
  float ry = -((float)py + 0.5f - (float)P.h * 0.5f);
  float dcx = (rx + P.cmr_x) * sp, dcy = (ry + P.cmr_y) * sp;
  float dx = s.dr, dy = s.di, gx = s.gr, gy = s.gi, trap = s.trap;
  unsigned m = s.m, n = s.n, n0 = s.n, saved = 0;
  unsigned last = P.orbit_len - 1u;
  OrbitPt o = orbit[m];
  float zmx = ldexpf(o.re, o.e), zmy = ldexpf(o.im, o.e);
  float cdx = dx, cdy = dy; unsigned cm = m, clen = 0, cpow = 1;
  for (unsigned k = 0; k < P.budget; k++) {
    if (n >= P.max_iter) { s.status = 2u; break; }
    bool stepped = false;
    if (m > 0u && P.bla_levels > 0u) {
      unsigned mm = m - 1u;
      unsigned l = min(mm == 0u ? 32u : (unsigned)(__ffs(mm) - 1), P.bla_levels);
      float d2 = dx * dx + dy * dy;
      if (l > 1u) {
        unsigned i1 = c_bla_off[0] + (mm >> 1u);
        if (i1 < c_bla_off[1]) { Bla e1 = bla[i1]; if (!(d2 < ldexpf(e1.r2m, e1.r2e))) l = 0u; }
      }
      while (l != 0u) {
        unsigned span = 1u << l;
        unsigned i = c_bla_off[l - 1u] + (mm >> l);
        if (i < c_bla_off[l]) {
          Bla e = bla[i];
          if (e.ae < 100 && e.be < 100 && e.r2e > -120 && d2 < ldexpf(e.r2m, e.r2e)) {
            float ax = ldexpf(e.ar, e.ae), ay = ldexpf(e.ai, e.ae), bx = ldexpf(e.br, e.be), by = ldexpf(e.bi, e.be);
            float ndx = (ax * dx - ay * dy) + (bx * dcx - by * dcy);
            float ndy = (ax * dy + ay * dx) + (bx * dcy + by * dcx);
            float ngx = (ax * gx - ay * gy) + bx * sp;
            float ngy = (ax * gy + ay * gx) + by * sp;
            dx = ndx; dy = ndy; gx = ngx; gy = ngy;
            m += span; n += span; saved += span - 1u; stepped = true;
            break;
          }
        }
        l--;
      }
    }
    if (!stepped) {
      float ux = zmx + dx, uy = zmy + dy;
      float ngx = 2.0f * (ux * gx - uy * gy) + sp;
      float ngy = 2.0f * (ux * gy + uy * gx);
      float vx = 2.0f * zmx + dx, vy = 2.0f * zmy + dy;
      float ndx = (vx * dx - vy * dy) + dcx;
      float ndy = (vx * dy + vy * dx) + dcy;
      gx = ngx; gy = ngy; dx = ndx; dy = ndy;
      m++; n++;
    }
    o = orbit[m];
    zmx = ldexpf(o.re, o.e); zmy = ldexpf(o.im, o.e);
    float zx = zmx + dx, zy = zmy + dy;
    float r2 = zx * zx + zy * zy;
    trap = fminf(trap, r2);
    if (r2 > 65536.0f) {
      float log2_abs = 0.5f * log2f(r2);
      float sm = (float)n + 1.0f - log2f(log2_abs / 8.0f);
      float lg = 0.5f * log2f(gx * gx + gy * gy);
      if (bad(gx) || bad(gy)) lg = 120.0f;
      float log2_de = log2_abs + log2f(log2_abs * 0.69314718f) - lg;
      samples[idx] = make_float2(sm, exp2f(fminf(fmaxf(log2_de, -60.0f), 60.0f)));
      s.status = 1u;
      break;
    }
    if (r2 < dx * dx + dy * dy || m >= last) {
      dx = zx; dy = zy; m = 0u;
      o = orbit[0]; zmx = ldexpf(o.re, o.e); zmy = ldexpf(o.im, o.e);
    }
    if (bad(dx) || bad(dy)) { s.status = 3u; samples[idx] = make_float2(-2.0f, 0.0f); break; }
    if (m == cm && __float_as_uint(dx) == __float_as_uint(cdx) && __float_as_uint(dy) == __float_as_uint(cdy)) {
      s.status = 4u; break;
    }
    clen++;
    if (clen == cpow) { cdx = dx; cdy = dy; cm = m; cpow <<= 1u; clen = 0u; }
  }
  if (s.status == 2u || s.status == 4u) samples[idx] = make_float2(-1.0f, sqrtf(trap));
  s.dr = dx; s.di = dy; s.gr = gx; s.gi = gy; s.trap = trap; s.m = m; s.n = n;
  st[idx] = s;
  if (s.status == 0u) atomicAdd(active, 1u);
  atomicAdd(steps, (unsigned long long)(n - n0 - saved));
}

static std::vector<uint8_t> read_file(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> b(n); fread(b.data(), 1, n, f); fclose(f);
  return b;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: perturb <dump prefix> [budget]\n"); return 2; }
  std::string dir = argv[1];
  unsigned budget = argc > 2 ? (unsigned)atoi(argv[2]) : 1u << 16;
  auto pk = read_file(dir + "-packet.bin");
  const uint32_t* hw = (const uint32_t*)pk.data();
  unsigned len = hw[3], levels = hw[5];
  unsigned off[34] = {0};
  for (unsigned i = 0; i <= levels && i < 34; i++) off[i] = hw[8 + i];
  const uint8_t* orbit_bytes = pk.data() + 48 * 4;
  const uint8_t* bla_bytes = orbit_bytes + (size_t)len * 16;
  Params P{};
  FILE* vf = fopen((dir + "-view.txt").c_str(), "r");
  if (!vf || fscanf(vf, "%u %u %f %d %f %f %u", &P.w, &P.h, &P.sp_m, &P.sp_e, &P.cmr_x, &P.cmr_y, &P.max_iter) != 7) { fprintf(stderr, "bad view.txt\n"); return 1; }
  fclose(vf);
  P.budget = budget; P.orbit_len = len; P.bla_levels = levels;
  unsigned npx = P.w * P.h;

  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
  printf("device %s, driver API %d, runtime %d\n", prop.name, drv, rt);

  OrbitPt* d_orbit; Bla* d_bla; State* d_st; float2* d_samples; unsigned* d_active; unsigned long long* d_steps;
  CK(cudaMalloc(&d_orbit, (size_t)len * 16));
  CK(cudaMalloc(&d_bla, (size_t)std::max(1u, off[levels]) * 32));
  CK(cudaMalloc(&d_st, (size_t)npx * sizeof(State)));
  CK(cudaMalloc(&d_samples, (size_t)npx * 8));
  CK(cudaMalloc(&d_active, 4));
  CK(cudaMalloc(&d_steps, 8));
  CK(cudaMemcpy(d_orbit, orbit_bytes, (size_t)len * 16, cudaMemcpyHostToDevice));
  if (off[levels]) CK(cudaMemcpy(d_bla, bla_bytes, (size_t)off[levels] * 32, cudaMemcpyHostToDevice));
  CK(cudaMemcpyToSymbol(c_bla_off, off, sizeof(off)));
  std::vector<State> init(npx);
  for (auto& s : init) { s = State{0, 0, 0, 0, 1e30f, 0, 0, 0}; }
  CK(cudaMemcpy(d_st, init.data(), (size_t)npx * sizeof(State), cudaMemcpyHostToDevice));
  CK(cudaMemset(d_steps, 0, 8));

  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  float total_ms = 0; unsigned launches = 0;
  auto wall0 = std::chrono::steady_clock::now();
  for (;;) {
    CK(cudaMemset(d_active, 0, 4));
    cudaEventRecord(e0);
    iterate<<<(npx + 63) / 64, 64>>>(P, d_orbit, d_bla, d_st, d_samples, d_active, d_steps);
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms; cudaEventElapsedTime(&ms, e0, e1); total_ms += ms; launches++;
    unsigned active; CK(cudaMemcpy(&active, d_active, 4, cudaMemcpyDeviceToHost));
    if (active == 0) break;
  }
  double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
  unsigned long long steps; CK(cudaMemcpy(&steps, d_steps, 8, cudaMemcpyDeviceToHost));
  printf("pixels %u, limit %u, launches %u, kernel %.1f ms, wall %.1f ms, steps %.3e, %.2f Gsteps/s\n",
         npx, P.max_iter, launches, total_ms, wall * 1e3, (double)steps, steps / (total_ms * 1e6));

  // Comparison with the WebGPU state (status, n), if dumped.
  std::vector<State> st(npx);
  CK(cudaMemcpy(st.data(), d_st, (size_t)npx * sizeof(State), cudaMemcpyDeviceToHost));
  FILE* sf = fopen((dir + "-state.bin").c_str(), "rb");
  if (sf) {
    std::vector<uint32_t> ref(2 * (size_t)npx);
    fread(ref.data(), 4, ref.size(), sf); fclose(sf);
    unsigned same = 0, status_diff = 0, n_diff = 0, cyc = 0;
    for (unsigned i = 0; i < npx; i++) {
      unsigned rs = ref[2 * i], rn = ref[2 * i + 1];
      if (rs == 4 && st[i].status == 4) { cyc++; same++; continue; }  // detection time depends on budgets
      if (rs != st[i].status) status_diff++;
      else if (rn != st[i].n) n_diff++;
      else same++;
    }
    printf("vs WebGPU: identical %u of %u (cycling counted by status: %u), status differs %u, n differs %u\n", same, npx, cyc, status_diff, n_diff);
  }
  return 0;
}

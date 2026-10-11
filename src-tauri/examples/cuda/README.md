# CUDA versus WebGPU: perturbation kernel benchmark

Decides whether a CUDA path for heavy views is worth adding. `perturb.cu`
ports escape.wgsl's MODE 2 (perturbation with f32 deltas: BLA lookup with the
2-step early rejection, rebasing, Brent cycle detection, smooth colouring,
distance estimate) and runs one sample of every pixel to a fixed limit,
resuming in launches of `budget` steps per pixel as the shader does across
frames.

## Running

```sh
# 1. WebGPU side: render the views at 2560×1440, one sample, limit fixed, no
#    interior proofs; writes artifacts/cudadump/<view>-dump-*
ITERUM_ONLY=boundary-p2,boundary-p3,perturb-1e-15 ITERUM_DUMP_LIMIT=65536 npm run golden -- artifacts/cudadump

# 2. CUDA side (MSVC environment: vcvars64.bat; nvcc 13.x)
nvcc -O3 -arch=sm_89 --fmad=false -o perturb_nofma.exe perturb.cu
nvcc -O3 -arch=sm_89 --fmad=true  -o perturb_fma.exe   perturb.cu
perturb_nofma.exe artifacts/cudadump/boundary-p2-dump 65536
```

It prints the kernel time, steps per second and how many pixels match the
WebGPU result (status and iteration count; cycling pixels by status only,
since when Brent's method finds a cycle depends on how the work is split).

## Result (2026-10-11)

RTX 4070, driver 617.42 (CUDA driver API 13.4), CUDA 13.3 (V13.3.33),
sm_89 SASS (no PTX JIT). 2560×1440, limit 2^16, one sample.

| view | WebGPU escape passes | CUDA, `--fmad=false` | CUDA, `--fmad=true` |
|---|---|---|---|
| boundary-p2 | 4362 ms | 2982 ms (×1.46), all pixels identical | 2803 ms (×1.56), 2410 px differ |
| boundary-p3 | 5260 ms | 3860 ms (×1.36), 53 px differ in status | 3605 ms (×1.46), 9379 px differ |
| perturb-1e-15 | 199 ms | 179 ms (×1.11) | 145 ms (×1.37) |

Kernel time only. A CUDA path would also pay for the transfer to the
WebView: Tauri IPC moved orbit packets at about 120 MB/s here, so the four
2560×1440 samples (118 MB) cost about a second, plus every progressive
preview. MODE 3 (floatexp) is not ported.

**Decision: not adopted.** The adoption rule fixed in advance was a geometric
mean of at least ×1.5 over the heavy views, transfer included, with no view
slower. The bit-identical build reaches about ×1.41 on the kernel alone. The
FMA build (not identical, so it would also need the approximation tolerance)
reaches about ×1.51 on the kernel alone; with a second of transfer on these
3–5 s renders it falls well below, and only very long renders would amortise
it — for those, the scheduling gain below is available without a second
implementation.

What the comparison shows instead: the same arithmetic runs about 30–45%
faster as one long kernel than as the app's frame-sized dispatches, so most of
the gap is how WebGPU work is scheduled (each frame's dispatch waits for its
slowest pixels), not the compiler.

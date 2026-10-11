// CPU reference of the RK4 integrator in lorenz.wgsl (tests and GPU checks).

export type V3 = [number, number, number];
export const LORENZ = { sigma: 10, rho: 28, beta: 8 / 3 };

export function lorenz([x, y, z]: V3, f: (v: number) => number = (v) => v): V3 {
  const { sigma, rho, beta } = LORENZ;
  return [f(sigma * (y - x)), f(x * (rho - z) - y), f(x * y - f(beta) * z)];
}

export function rk4(p: V3, h: number, f: (v: number) => number = (v) => v): V3 {
  const add = (a: V3, b: V3, s: number): V3 => [f(a[0] + s * b[0]), f(a[1] + s * b[1]), f(a[2] + s * b[2])];
  const k1 = lorenz(p, f);
  const k2 = lorenz(add(p, k1, 0.5 * h), f);
  const k3 = lorenz(add(p, k2, 0.5 * h), f);
  const k4 = lorenz(add(p, k3, h), f);
  return [0, 1, 2].map((i) => f(p[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]))) as V3;
}

export function integrate(p: V3, h: number, steps: number, f?: (v: number) => number): V3 {
  for (let i = 0; i < steps; i++) p = rk4(p, h, f);
  return p;
}

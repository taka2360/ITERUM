// ITERUM_MODE=shots: captures a fixed set of scenes for visual review.

import type { App } from "../main";
import type { Mandelbrot } from "../phenomena/mandelbrot";
import { diag, envVar, exitApp } from "../platform";
import { converged, seconds, shot } from "./harness";
import { VOYAGES } from "../phenomena/mandelbrot/voyages";

export interface Scene {
  name: string;
  phenomenon: number;
  setup?: (app: App) => void;
  /** Seconds to let animation run (animated phenomena never converge). */
  wait?: number;
  timeout?: number;
}

export function mandelbrotAt(re: string, im: string, widthLog2: number) {
  return (app: App) => {
    const m = app.phenomena[0] as Mandelbrot;
    m.setView(re, im, widthLog2 - Math.log2(app.canvas.width));
  };
}

/** Final view of a Voyage destination. */
export function voyageView(name: string) {
  const v = VOYAGES.find((t) => t.name === name)!;
  return mandelbrotAt(v.re, v.im, -v.depthLog2);
}

export const SCENES: Scene[] = [
  { name: "01-mandelbrot-home", phenomenon: 0, setup: (app) => app.phenomena[0].reset() },
  { name: "02-mandelbrot-seahorse", phenomenon: 0, setup: mandelbrotAt("-0.7436438870371587", "0.1318259042053120", Math.log2(0.004)) },
  { name: "03-mandelbrot-1e-12", phenomenon: 0, setup: mandelbrotAt("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", Math.log2(1e-12)) },
  { name: "04-mandelbrot-1e-25", phenomenon: 0, setup: mandelbrotAt("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", Math.log2(1e-25)) },
  { name: "08-deep-1e-40", phenomenon: 0, setup: voyageView("Seahorse Spiral"), timeout: 60 },
  { name: "09-deep-1e-100", phenomenon: 0, setup: voyageView("Elephant Valley"), timeout: 60 },
  { name: "10-deep-1e-300", phenomenon: 0, setup: voyageView("Seahorse Abyss"), timeout: 90 },
  { name: "11-deep-1e-1000", phenomenon: 0, setup: voyageView("Abyssal Plain"), timeout: 120 },
  { name: "12-voyage-20s", phenomenon: 0, setup: (app) => { app.phenomena[0].reset(); setTimeout(() => (app.phenomena[0] as Mandelbrot).toggleVoyage(2), 1500); }, wait: 20 },
  { name: "13-voyage-60s", phenomenon: 0, wait: 40 },
  { name: "14-voyage-120s", phenomenon: 0, wait: 60 },
  { name: "05-julia", phenomenon: 1, wait: 4 },
  { name: "06-newton", phenomenon: 2, wait: 3 },
  { name: "07-lorenz", phenomenon: 3, wait: 6 },
];

export async function runShots(app: App, scenes = SCENES): Promise<void> {
  const filter = await envVar("ITERUM_SCENES");
  if (filter) scenes = scenes.filter((s) => filter.split(",").some((f) => s.name.includes(f)));
  await seconds(1.5);
  for (const s of scenes) {
    app.select(s.phenomenon);
    s.setup?.(app);
    await seconds(2.0); // let the transition finish
    if (s.wait) await seconds(s.wait);
    else {
      const ok = await converged(app, s.timeout ?? 30);
      if (!ok) diag(`scene ${s.name}: did not converge`);
    }
    await shot(app, s.name);
    diag(`debug ${s.name}: ${JSON.stringify(app.active.debug?.() ?? {})}`);
  }
  diag("shots done");
  await exitApp(0);
}

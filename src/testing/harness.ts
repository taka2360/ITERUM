// Helpers shared by the scripted test modes (run with ITERUM_MODE=...).

import type { App } from "../main";
import { diag, envVar, saveFile } from "../platform";

export function frames(app: App, n: number): Promise<void> {
  return new Promise((resolve) => {
    let k = 0;
    const prev = app.onFrame;
    app.onFrame = (info) => {
      prev?.(info);
      if (++k >= n) {
        app.onFrame = prev;
        resolve();
      }
    };
  });
}

export function seconds(s: number): Promise<void> {
  return new Promise((r) => setTimeout(r, s * 1000));
}

/** Waits until the active phenomenon reports a converged image (or timeout). */
export async function converged(app: App, timeoutS: number): Promise<boolean> {
  const t0 = performance.now();
  while (performance.now() - t0 < timeoutS * 1000) {
    await frames(app, 1);
    if (app.lastStatus === "converged") return true;
  }
  return false;
}

export async function outDir(): Promise<string> {
  return (await envVar("ITERUM_OUT")) ?? ".";
}

export async function shot(app: App, name: string): Promise<void> {
  const png = await app.capture();
  const dir = await outDir();
  await saveFile(`${dir}/${name}.png`, png);
  diag(`capture ${name}.png (${png.byteLength} bytes)`);
}

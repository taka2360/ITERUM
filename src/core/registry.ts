// The collection, in viewing order. Add new phenomena here.

import type { Phenomenon } from "./phenomenon";
import type { PaletteTexture } from "./palettes";
import { Mandelbrot } from "../phenomena/mandelbrot";
import { Julia } from "../phenomena/julia";
import { Newton } from "../phenomena/newton";
import { Lorenz } from "../phenomena/lorenz";

export function createPhenomena(palette: () => PaletteTexture): Phenomenon[] {
  return [new Mandelbrot(palette), new Julia(palette), new Newton(palette), new Lorenz(palette)];
}

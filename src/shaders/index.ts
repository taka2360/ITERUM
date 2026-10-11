// Tiny WGSL preprocessor: replaces `#include name` lines with shared snippets.

import floatexp from "./floatexp.wgsl?raw";
import { FULLSCREEN_VS } from "../gpu/device";

const SNIPPETS: Record<string, string> = {
  floatexp,
  fullscreen: FULLSCREEN_VS,
};

export function wgsl(code: string): string {
  return code.replace(/^#include (\w+)\s*$/gm, (_, name: string) => {
    const s = SNIPPETS[name];
    if (s === undefined) throw new Error(`unknown WGSL include: ${name}`);
    return s;
  });
}

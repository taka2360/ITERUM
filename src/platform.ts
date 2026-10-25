// Thin bridge to the Tauri host; every call degrades gracefully in a browser.

import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";

export const isTauri = (): boolean => "__TAURI_INTERNALS__" in window;

/** Logs a line to the console and, under Tauri, to the host's stdout. */
export function diag(line: string): void {
  console.log(`[iterum] ${line}`);
  if (isTauri()) invoke("diag", { line }).catch(() => {});
}

export async function launchMode(): Promise<string> {
  const q = new URLSearchParams(location.search).get("mode");
  if (q) return q;
  if (!isTauri()) return "";
  return invoke<string>("launch_mode").catch(() => "");
}

export async function exitApp(code: number): Promise<void> {
  if (isTauri()) await invoke("exit_app", { code });
}

export async function toggleFullscreen(): Promise<void> {
  if (isTauri()) {
    const w = getCurrentWindow();
    await w.setFullscreen(!(await w.isFullscreen()));
  } else if (document.fullscreenElement) {
    await document.exitFullscreen();
  } else {
    await document.documentElement.requestFullscreen();
  }
}

/** Leaves fullscreen if the window is in it. */
export async function exitFullscreen(): Promise<void> {
  if (isTauri()) {
    const w = getCurrentWindow();
    if (await w.isFullscreen()) await w.setFullscreen(false);
  } else if (document.fullscreenElement) {
    await document.exitFullscreen();
  }
}

export async function envVar(name: string): Promise<string | undefined> {
  if (!isTauri()) return new URLSearchParams(location.search).get(name.toLowerCase()) ?? undefined;
  return (await invoke<string | null>("env_var", { name })) ?? undefined;
}

export async function saveFile(path: string, data: Uint8Array): Promise<void> {
  if (!isTauri()) return;
  await invoke("save_file", { path, data: Array.from(data) });
}

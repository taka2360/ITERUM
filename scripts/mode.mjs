// Runs the desktop app in a scripted mode and exits with its result.
//   node scripts/mode.mjs gputest|bench|shots|golden [outDir]
// Output files (captures, bench.json) go to outDir (default: ./artifacts).
import { execSync, spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const [mode, out = "artifacts"] = process.argv.slice(2);
if (!["gputest", "bench", "shots", "golden"].includes(mode)) {
  console.error("usage: node scripts/mode.mjs gputest|bench|shots|golden [outDir]");
  process.exit(2);
}
const dir = resolve(out);
mkdirSync(dir, { recursive: true });
let commit = "unknown";
try {
  commit = execSync("git rev-parse --short HEAD").toString().trim();
  if (execSync("git status --porcelain --untracked-files=no").toString().trim()) commit += "-dirty";
} catch {}
const child = spawn("npx", ["tauri", "dev"], {
  stdio: ["ignore", "pipe", "pipe"],
  shell: true,
  env: { ...process.env, ITERUM_MODE: mode, ITERUM_COMMIT: commit, ITERUM_OUT: dir.replaceAll("\\\\", "/") },
});
const relay = (buf) => {
  for (const line of buf.toString().split(/\r?\n/)) if (line.includes("[iterum]")) console.log(line.slice(line.indexOf("[iterum]")));
};
child.stdout.on("data", relay);
child.stderr.on("data", relay);
child.on("exit", (code) => process.exit(code ?? 1));

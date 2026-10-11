// Compares two golden runs (node scripts/mode.mjs golden <dir>) sample by sample.
//   node scripts/golden-diff.mjs <baselineDir> <candidateDir>
// Each .bin holds [smooth, distance] per pixel and sample (interior: [-1, sqrt(trap)],
// anomaly: [-2, 0]). Reports bit-identical share, class mismatches and value
// differences, plus the timing of both runs. Exit code 1 if any view is not
// bit-identical (informational: approximate optimisations are judged on the figures).
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const [a, b] = process.argv.slice(2);
if (!a || !b) {
  console.error("usage: node scripts/golden-diff.mjs <baselineDir> <candidateDir>");
  process.exit(2);
}
const ja = JSON.parse(readFileSync(join(a, "golden.json"), "utf8"));
const jb = JSON.parse(readFileSync(join(b, "golden.json"), "utf8"));
const cls = (s) => (s >= 0 ? "esc" : s > -1.5 ? "in" : "anom");
let allSame = true;
const rows = [];
for (const rb of jb.results) {
  const ra = ja.results.find((r) => r.name === rb.name);
  const fa = join(a, `${rb.name}.bin`);
  const fb = join(b, `${rb.name}.bin`);
  if (!ra || !existsSync(fa)) continue;
  const A = new Float32Array(readFileSync(fa).buffer.slice(0));
  const B = new Float32Array(readFileSync(fb).buffer.slice(0));
  const ua = new Uint32Array(A.buffer);
  const ub = new Uint32Array(B.buffer);
  let identical = 0, classDiff = 0, escBoth = 0, smoothOver = 0, maxSmooth = 0, maxDeRel = 0, maxTrap = 0;
  const n = Math.min(A.length, B.length) / 2;
  for (let i = 0; i < n; i++) {
    const sa = A[2 * i], sb = B[2 * i];
    if (ua[2 * i] === ub[2 * i] && ua[2 * i + 1] === ub[2 * i + 1]) { identical++; continue; }
    const ca = cls(sa), cb = cls(sb);
    if (ca !== cb) { classDiff++; continue; }
    if (ca === "esc") {
      escBoth++;
      const d = Math.abs(sa - sb);
      maxSmooth = Math.max(maxSmooth, d);
      if (d > 0.02) smoothOver++;
      const da = A[2 * i + 1], db = B[2 * i + 1];
      maxDeRel = Math.max(maxDeRel, Math.abs(da - db) / Math.max(Math.abs(da), 1e-30));
    } else if (ca === "in") {
      maxTrap = Math.max(maxTrap, Math.abs(A[2 * i + 1] - B[2 * i + 1]));
    }
  }
  if (identical !== n) allSame = false;
  rows.push({
    view: rb.name,
    identical: `${((identical / n) * 100).toFixed(4)}%`,
    classDiff,
    smoothOver0_02: smoothOver,
    maxSmooth: +maxSmooth.toExponential(2),
    maxDeRel: +maxDeRel.toExponential(2),
    maxTrap: +maxTrap.toExponential(2),
    wallS: `${ra.wallS} → ${rb.wallS} (×${(ra.wallS / rb.wallS).toFixed(2)})`,
    gpuMs: `${ra.gpuEscapeMs} → ${rb.gpuEscapeMs}`,
    maxIter: `${ra.maxIter} → ${rb.maxIter}`,
    iterations: `${(ra.iterations / 1e9).toFixed(2)}G → ${(rb.iterations / 1e9).toFixed(2)}G`,
  });
}
console.log(`baseline ${ja.commit} (${ja.date})  vs  candidate ${jb.commit} (${jb.date})`);
console.table(rows);
process.exit(allSame ? 0 : 1);

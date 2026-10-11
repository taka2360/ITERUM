// Number formatting for the progress panel (Japanese units).

/** 12345678 → "1,234.6 万" style, using 万 / 億 / 兆 for large counts. */
export function bigCount(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0";
  const units: [number, string][] = [[1e12, "兆"], [1e8, "億"], [1e4, "万"]];
  for (const [v, u] of units) if (n >= v) return `${(n / v).toFixed(n / v < 10 ? 2 : n / v < 100 ? 1 : 0)} ${u}`;
  return Math.round(n).toLocaleString("ja-JP");
}

export function int(n: number): string {
  return Math.round(n).toLocaleString("ja-JP");
}

export function seconds(s: number): string {
  if (!Number.isFinite(s)) return "—";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 60) return `${s.toFixed(1)} 秒`;
  const m = Math.floor(s / 60);
  return `${m} 分 ${Math.round(s - m * 60)} 秒`;
}

export function percent(f: number): string {
  return `${(Math.max(0, Math.min(1, f)) * 100).toFixed(f > 0.995 && f < 1 ? 1 : 0)}%`;
}

/** 10^-n for a log2 value, e.g. -332.2 → "10⁻¹⁰⁰". */
export function pow10FromLog2(log2: number): string {
  const e = Math.round(log2 * Math.LOG10E * Math.LN2);
  const sup = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  return `10${e < 0 ? "⁻" : ""}${String(Math.abs(e)).replace(/\d/g, (d) => sup[Number(d)])}`;
}

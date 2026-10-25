// Progress-panel rows shared by the escape-time phenomena (Mandelbrot, Julia):
// how far the current view is, what the pixels turned out to be, and how the
// iteration work is going.

import type { ProgressRow } from "../../core/phenomenon";
import { bigCount, int, percent, seconds } from "../../ui/format";
import type { EscapeEngine, EscapeProgress } from "./engine";

/** "12,345（3.2%）" for a pixel count. */
function share(n: number, pixels: number): string {
  return `${int(n)}（${((n / Math.max(1, pixels)) * 100).toFixed(1)}%）`;
}

/** Completion of the view: steps of the current sample, samples, time. */
export function stepRows(e: EscapeProgress, fineOnly = false): ProgressRow[] {
  const group = "進み具合";
  const rows: ProgressRow[] = [{ group, label: "全体", value: percent(e.overall), bar: e.overall }];
  if (!fineOnly) {
    rows.push(
      { group, label: "粗い下描き（1/16 の画素）", value: percent(e.coarse), bar: e.coarse },
      { group, label: "中間（1/4 の画素）", value: percent(e.mid), bar: e.mid },
    );
  }
  rows.push(
    { group, label: "すべての画素", value: percent(e.fine), bar: e.fine },
    { group, label: "重ね描き（アンチエイリアス）", value: `${Math.min(e.samplesDone + (e.converged ? 0 : 1), e.sampleTarget)} / ${e.sampleTarget} 枚目` },
    { group, detail: true, label: "この 1 枚の完了率", value: percent(e.sample), bar: e.sample },
    {
      group,
      label: e.converged ? "完成までの時間" : "経過 / 残り（推定）",
      value: e.converged ? seconds(e.elapsedS) : `${seconds(e.elapsedS)} / ${e.etaS !== undefined ? "約 " + seconds(e.etaS) : "—"}`,
    },
  );
  return rows;
}

/** What the pixels of the current sample turned out to be, and how the work is dispatched. */
export function pixelRows(engine: EscapeEngine, e: EscapeProgress): ProgressRow[] {
  const group = "画素の内訳（いま描いている 1 枚）";
  const s = e.converged ? (engine.lastComplete ?? engine.stats()) : engine.stats();
  const px = e.pixels;
  const unfinished = e.converged ? 0 : s.unfinished;
  const rows: ProgressRow[] = [
    { group, detail: true, label: "発散（集合の外）", value: share(s.escaped, px), bar: s.escaped / px },
    { group, detail: true, label: "内部と判定（集合の中）", value: share(s.interior, px), bar: s.interior / px },
    {
      group, detail: true, label: "　内部の判定方法",
      value: `周期を検出 ${int(s.cycling)} · 式で証明 ${int(s.proven)} · 上限まで逃げず ${int(Math.max(0, s.interior - s.cycling - s.proven))}`,
    },
    {
      group, detail: true, label: "計算中",
      value: unfinished < 0 ? "集計待ち" : share(unfinished, px),
      bar: unfinished < 0 ? undefined : unfinished / px,
    },
    { group, detail: true, label: "上限の後半で発散", value: `${int(s.late)} 画素（多いと反復上限を引き上げる）` },
    {
      group, detail: true, label: "処理の仕方",
      value: `${s.listMode ? "残りの画素だけを処理（リスト方式）" : "全画素を走査"} · 1 フレームあたり最大 ${int(s.budget)} 反復/画素`,
    },
  ];
  return rows;
}

/** Iteration limit, escape counts and throughput. */
export function iterationRows(engine: EscapeEngine, e: EscapeProgress): ProgressRow[] {
  const group = "反復";
  const s = e.converged ? (engine.lastComplete ?? engine.stats()) : engine.stats();
  const rows: ProgressRow[] = [
    { group, label: "反復", value: `上限 ${int(e.maxIter)} · 逃げるまでの最大 ${int(e.maxEscaped)}` },
    {
      group, detail: true, label: "発散までの反復回数",
      value: s.escaped > 0 ? `最少 ${int(s.minEscapedN)} 〜 最多 ${int(s.maxEscapedN)}` : "—",
    },
    {
      group, detail: true, label: "上限の範囲",
      value: `${int(engine.minIter)} 〜 ${int(engine.maxIterCap)}（足りなければ自動で 2 倍に）`,
    },
  ];
  const last = engine.raiseLog[engine.raiseLog.length - 1];
  rows.push({
    group, detail: true, label: "上限の引き上げ（起動から）",
    value: engine.restarts > 0 ? `${int(engine.restarts)} 回${last ? ` · 直近 ${int(last.from)} → ${int(last.to)}` : ""}` : "なし",
  });
  if (!e.converged) {
    rows.push({ group, label: "処理速度", value: `${bigCount(e.iterationsPerSecond)} 反復/秒` });
    if (e.stepsPerSecond > 0) {
      rows.push({
        group, detail: true, label: "実際のループ回数",
        value: `${bigCount(e.stepsPerSecond)} 回/秒（BLA による短縮 ×${(e.iterationsPerSecond / e.stepsPerSecond).toFixed(1)}）`,
      });
    }
  }
  rows.push({ group, detail: true, label: "起動からの総反復", value: `${bigCount(engine.iterationsTotal)} 反復 · ${int(engine.computeFrames)} フレーム` });
  return rows;
}

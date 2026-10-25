// Every shortcut in one table: main.ts dispatches keys from it and the help
// panel is drawn from it, so the two cannot drift apart.

export type KeyAction =
  | "step" | "number"
  | "play" | "reset"
  | "palette" | "quality" | "fullscreen" | "hideUi"
  | "info" | "location" | "progress" | "debug" | "help" | "close";

export interface HelpItem {
  /** How the key or gesture is written in the help panel. */
  label: string;
  text: string;
  /** KeyboardEvent.key values that trigger the action (none for mouse gestures). */
  keys?: string[];
  action?: KeyAction;
  /** Held keys repeat the action. */
  repeat?: boolean;
}

export interface HelpGroup {
  title: string;
  items: HelpItem[];
}

export const KEY_GROUPS: HelpGroup[] = [
  {
    title: "眺める",
    items: [
      { label: "ドラッグ", text: "移動（ローレンツ・アトラクタでは回転）" },
      { label: "ホイール", text: "拡大・縮小（カーソルの位置を中心に）" },
      { label: "Space", text: "再生 ／ 停止（マンデルブロ集合では Voyage）", keys: [" "], action: "play" },
      { label: "R", text: "視点を最初に戻す", keys: ["r", "R"], action: "reset" },
    ],
  },
  {
    title: "現象を選ぶ",
    items: [
      { label: "← →", text: "前 ／ 次の現象へ", keys: ["ArrowLeft", "ArrowRight"], action: "step", repeat: true },
      // The label is filled in with the number of phenomena (overlay.ts).
      { label: "1–9", text: "番号で選ぶ", action: "number" },
    ],
  },
  {
    title: "見た目",
    items: [
      { label: "C", text: "配色を変える", keys: ["c", "C"], action: "palette" },
      { label: "Q", text: "品質（自動 → 低 → 中 → 高）", keys: ["q", "Q"], action: "quality" },
      { label: "F ／ F11", text: "全画面", keys: ["f", "F", "F11"], action: "fullscreen" },
      { label: "H", text: "文字とボタンをすべて隠す（もう一度で戻す）", keys: ["h", "H"], action: "hideUi" },
    ],
  },
  {
    title: "パネル",
    items: [
      { label: "I", text: "解説", keys: ["i", "I"], action: "info" },
      { label: "L", text: "現在地 — 座標と拡大率の表示・コピー・移動", keys: ["l", "L"], action: "location" },
      { label: "P", text: "計算の様子 — 押すたびに 概要 → 詳細 → 閉じる", keys: ["p", "P"], action: "progress" },
      { label: "D", text: "計測情報（開発者向け）", keys: ["d", "D"], action: "debug" },
      { label: "?", text: "この操作一覧", keys: ["?", "/"], action: "help" },
      { label: "Esc", text: "開いているパネルを閉じる ／ 全画面を抜ける", keys: ["Escape"], action: "close" },
    ],
  },
];

/** The binding a key triggers, if any ("1".."9" map to "number"). */
export function keyBinding(key: string): HelpItem | undefined {
  for (const g of KEY_GROUPS) for (const it of g.items) if (it.keys?.includes(key)) return it;
  if (/^[1-9]$/.test(key)) return KEY_GROUPS[1].items[1];
  return undefined;
}

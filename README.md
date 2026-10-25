# ITERUM

単純な数学的ルールから生まれる構造を、静かに眺めるためのデスクトップアプリ。
マンデルブロ集合（倍率上限のない深いズーム）、ジュリア集合、ニュートン・フラクタル、ローレンツ・アトラクタを、GPU で実際に計算しながら描画します。

## 操作

キー一覧は `?` でいつでも表示できます（定義は `src/ui/keys.ts` に一本化）。

| 眺める | |
|---|---|
| ドラッグ | 移動（ローレンツ・アトラクタでは回転） |
| ホイール | 拡大・縮小（カーソル位置基準） |
| Space / ▶ ボタン | 再生・停止（マンデルブロ集合では Voyage の開始・停止） |
| R / ⟲ ボタン | 視点を最初に戻す |

| 現象を選ぶ | |
|---|---|
| ← → | 前 / 次の現象へ |
| 1–4 | 番号で選ぶ |

| 見た目 | |
|---|---|
| C | 配色 |
| Q | 品質（自動 → 低 → 中 → 高） |
| F / F11 | 全画面 |
| H | 文字とボタンをすべて隠す |

| パネル | |
|---|---|
| I | 解説 |
| L / ⌖ ボタン | 現在地（座標と拡大率の表示・コピー・移動） |
| P / ▮ ボタン | 計算の様子。押すたびに 概要 → 詳細 → 閉じる。詳細では進み具合・位置と精度・基準軌道・反復・画素の内訳・描画（GPU パス別時間など）を項目別に表示 |
| D | 計測情報（開発者向け） |
| ? | 操作一覧 |
| Esc | 開いているパネルを閉じる / 全画面を抜ける |

| 現象ごとの操作 | |
|---|---|
| ダブルクリック | その点へ近づく（マンデルブロ・ジュリア・ローレンツ） |
| V | Voyage：深い minibrot へ自動で潜り続ける（マンデルブロ集合） |
| 右ドラッグ | パラメータ c を動かす（ジュリア集合） |
| N | 根の数を変える（ニュートン・フラクタル） |

画面右下のボタンは、マウスを動かすと現れ、しばらく操作しないと静かに消えます。時間のかかる計算の間は、画面下端に細い進捗線が表示されます。

## 開発

```sh
npm install
npm run app          # 開発版を起動（Tauri + Vite）
npm run app:build    # インストーラと実行ファイルを作成
npm test             # Vitest（カメラ、floatexp、パレット、品質ガバナー、RK4、パケット）
npm run test:rust    # cargo test（固定小数点、基準軌道 vs astro-float、BLA、形式、核の探索）
npm run test:gpu     # 実機 GPU の正当性テスト（アプリを起動し、結果を表示して終了）
npm run bench        # ベンチマーク（artifacts/bench.json）
npm run shots        # 確認用スクリーンショット（artifacts/*.png）
npm run golden -- artifacts/golden-a   # 固定視点の計算結果と所要時間を保存（速度改善の前後比較用）
node scripts/golden-diff.mjs artifacts/golden-a artifacts/golden-b   # 2 回分をビット単位で比較
cargo run --release --manifest-path src-tauri/Cargo.toml --example orbit_bench   # 基準軌道の計算速度
ITERUM_ONLY=boundary-p2 ITERUM_CAP=4194304 ITERUM_TIMING=0 npm run golden -- artifacts/x   # 視点・上限を絞った実験（ITERUM_PROVEN / ITERUM_PATCHES / ITERUM_TRAP / ITERUM_SHARE も）
ITERUM_DUMP_LIMIT=65536 npm run golden -- artifacts/cudadump   # CUDA 比較用の書き出し（src-tauri/examples/cuda/README.md）
npm run voyages      # Voyage の目的地を再生成
```

必要なもの: Node.js 20+、Rust（MSVC）、WebView2（WebGPU 対応版）。

## 構成

```
src/
  main.ts                 フレームループ、切替、デバイス喪失からの復帰
  gpu/                    WebGPU 初期化、GPU 計時、品質ガバナー、最終合成
  core/                   現象インターフェース、登録、入力、カメラ（BigInt）、パレット、floatexp の CPU モデル
  phenomena/
    escape/               マンデルブロ・ジュリア共通の逐次・段階的な反復エンジン（WGSL）
    mandelbrot/           深いズーム（基準軌道の管理、モード選択、Voyage）
    julia/ newton/ lorenz/
  shaders/floatexp.wgsl   指数拡張浮動小数（Rust と共有する形式）
  testing/                GPU テスト、ベンチ、撮影（ITERUM_MODE で起動）
src-tauri/src/deepzoom/   任意精度の基準軌道、BLA、バイナリ形式、minibrot 探索
```

### 新しい現象の追加

1. `src/phenomena/<名前>/index.ts` に `Phenomenon`（`src/core/phenomenon.ts`）を実装する。
   `render()` は自分の出力テクスチャ（rgba16float、内部解像度）を返す。合成・切替・品質調整はアプリ側が行う。
2. `src/core/registry.ts` に 1 行追加する。

### 深いズームの仕組み

- 視点の中心は BigInt の固定小数点で保持し、倍率は log2 で持つため、カメラ自体に深さの上限はない。
- 1 画素の幅が中心座標の f32 精度に近づくと、Rust 側で中心の基準軌道を任意精度（固定小数点）で計算し、GPU は各画素の差分 δ だけを反復する（摂動法）。
- |Z+δ| < |δ| または軌道の終端で δ を全体値に置き換える rebasing により、1 本の基準軌道で画像全体を破綻なく描く。
- 画素幅や軌道の最小値が f32 の指数範囲を外れると、δ・軌道・BLA 係数をすべて floatexp（f32 仮数＋i32 指数）で扱うモードに切り替える。
- BLA（双線形近似）で多数の反復をまとめて飛ばす。
- 主カージオイド・周期 2 円板の内部は閉じた式で、その他の双曲成分の内部は CPU 上の Taylor モデルと Krawczyk 法で、誤差上限つきで証明し、証明できた画素は反復上限まで回さない（`src/phenomena/escape/PROOF.md`）。
- 基準軌道の延長は前回の続きから計算し、増えた部分だけを送る。反復上限の引き上げは GPU 内で判定する。
- 基準軌道の計算中は直前の画像を再投影して表示し、誤った画像を完成扱いにしない。

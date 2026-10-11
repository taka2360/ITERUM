# ITERUM

単純な数学的ルールから生まれる構造を、静かに眺めるためのデスクトップアプリ。
マンデルブロ集合（倍率上限のない深いズーム）、ジュリア集合、ニュートン・フラクタル、ローレンツ・アトラクタを、GPU で実際に計算しながら描画します。

## 操作

| キー / 操作 | |
|---|---|
| ← → / 1–4 | 現象を切り替える |
| ドラッグ | 移動（3D は回転） |
| ホイール | 拡大・縮小（カーソル位置基準） |
| ダブルクリック | その点へ近づく |
| 右ドラッグ | ジュリア集合のパラメータ c を動かす |
| V | Voyage：深い minibrot へ自動で潜り続ける（マンデルブロ集合） |
| N | 根の数を変える（ニュートン・フラクタル） |
| Space / ▶ ボタン | 再生・停止（マンデルブロ集合では Voyage の開始・停止） |
| C | 配色 |
| F / F11 | 全画面 |
| I / ? / D / H | 解説 / 操作一覧 / 計測情報 / すべて隠す |
| Q | 品質（自動・低・中・高） |
| R / ⟲ ボタン | 視点を戻す（なめらかに切り替わる） |

画面右下の ▶ と ⟲ は、マウスを動かすと現れ、しばらく操作しないと静かに消えます。

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
- 基準軌道の計算中は直前の画像を再投影して表示し、誤った画像を完成扱いにしない。

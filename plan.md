# 開発計画

SimpleVisualSLAM の現状整理と、今後の実装・検証を進めるための詳細計画です。
この文書は「何を作るか」だけでなく、「すでに何が入っていて、次にどこを詰めるべきか」を共有するための運用用メモも兼ねます。

## 1. 現在地

このリポジトリは、最小構成の単眼 Visual SLAM から出発しつつ、以下の要素まで実装が進んでいます。

- 単眼トラッキング
- 2-view 初期化 + Depth ベース単フレーム初期化
- 局所マッピング + Depth 補助ランドマーク生成
- 局所 BA（Depth Prior + Gravity Prior 付き）
- ループ検出とループ補正（Metric Depth 対応 Sim3）
- マップ保存 / 読み込み
- EuRoC / TUM RGB-D の入力サポート
- 深度センサー統合（TUM RGB-D depth.txt）
- 加速度計統合（重力推定・アライメント・静止検知）
- Deep Learning Depth 推定（Depth Anything v2 via ONNX Runtime）
- 軌跡出力および評価レポート生成スクリプト

## 2. 評価結果

### TUM RGB-D ベンチマーク（ATE mean [m]、Sim3 alignment）

| Dataset | Mono | Depth | Depth+Accel | DL Depth | Target |
|---------|------|-------|-------------|----------|--------|
| fr1_xyz | 0.025 | 0.017 | **0.011** | 0.051 | <0.05 |
| fr1_room | 0.816 | 0.291 | **0.214** | — | <0.30 |

**Key observations:**
- Depth sensor: スケール不定性を解消、初期化を単フレームで完了
- Accelerometer: 重力アライメントで座標系安定化、BA内の gravity prior で roll/pitch 拘束
- DL Depth (Depth Anything v2 Small): sensor depth なしでも fr1_xyz で 0.051m（CPU推論、5フレームごと）
- Loop closing: metric depth モードでは Sim3 スケール範囲を 0.85–1.15 に制限

## 3. 開発方針

このプロジェクトでは、広いリファクタよりも「小さく作って、実データで確認し、必要な点だけを補強する」方針を取ります。

- 既存のモジュール境界を活かし、Tracking / LocalMapping / LoopClosing / IO を中心に拡張する。
- 新機能追加よりも、まずは既存機能の破綻条件を減らす。
- 動画入力で見た目が動くことより、データセットで軌跡を比較できることを優先する。
- 数学的に重要な変数名や記法は維持し、可読性を崩す変更は避ける。

## 4. フェーズ別計画

### Phase 1: 基本構造とデータ型 (完了)

- [x] `Camera`, `Frame`, `Keyframe`, `Landmark`, `Map` の定義
- [x] 単眼入力を扱うための最小アプリケーション `apps/run_mono.cc` の作成
- [x] ORB 特徴抽出の統合
- [x] Sophus ベースの姿勢表現導入
- [x] `MapIO` の基本枠組み作成

### Phase 2: トラッキング (完了)

- [x] 等速運動モデルによる初期姿勢予測
- [x] 参照キーフレームに対する特徴マッチング
- [x] 局所地図ランドマークの投影による追跡補強
- [x] `SolvePnPRansac` を使った姿勢推定
- [x] キーフレーム挿入判定の基本実装

### Phase 3: 単眼初期化 (完了)

- [x] 2 フレーム間のホモグラフィ / 基本行列推定
- [x] モデル選択と相対運動 `(R, t)` の復元
- [x] 三角測量による初期ランドマーク生成
- [x] 初期 2 キーフレームと Map の構築

### Phase 4: 局所マッピング (完了)

- [x] キーフレーム挿入 + 共視関係の更新
- [x] 新規ランドマークの三角測量
- [x] 品質の低いマップポイントの間引き
- [x] Ceres を用いた局所 BA

### Phase 5: ループクロージャ (完了)

- [x] DBoW2 統合
- [x] 候補キーフレーム検索 + 幾何検証
- [x] ループ拘束の生成 + Sim3 ポーズグラフ最適化
- [x] Metric depth 対応（スケール範囲制限 + スケール重み強化）

### Phase 6: 永続化 (完了)

- [x] Keyframe / Landmark / Graph を含むマップ保存
- [x] マップ読み込み

### Phase 7: データセット対応と評価基盤 (完了)

- [x] EuRoC / TUM RGB-D データセット入力
- [x] オンライン軌跡 / キーフレーム軌跡の TUM 形式出力
- [x] TUM 用 HTML レポート + 3D マップビューア生成スクリプト
- [x] `scripts/eval_all.sh` 全モード一括評価スクリプト

### Phase 8: Depth + IMU 統合 (完了)

**Phase 8.0: データ基盤**
- [x] TUM depth.txt / accelerometer.txt パーサー
- [x] タイムスタンプベース depth-RGB アソシエーション（±30ms tolerance）
- [x] `--depth` / `--accel` CLI フラグ

**Phase 8.1: Depth 統合**
- [x] 単フレーム depth back-projection 初期化（`initializeWithDepth`）
- [x] Depth 補助ランドマーク生成（`createLandmarksFromDepth`）
- [x] DepthPriorError コスト関数（sensor: σ=0.02m, DL: σ=0.2m）
- [x] Metric depth 対応ループクロージャ

**Phase 8.2: Accelerometer 統合**
- [x] 重力推定 + Rodrigues アライメント
- [x] 静止検知による motion model override
- [x] GravityPriorError コスト関数（BA 内 roll/pitch 拘束）
- [x] Keyframe ごとの gravity_in_camera_ 計算

**Phase 8.3: DL Depth (ONNX Runtime)**
- [x] DepthEstimator 抽象基底クラス
- [x] OnnxDepthEstimator（Depth Anything v2, 518×518 入力, ImageNet 正規化）
- [x] `--depth-model <path.onnx>` CLI フラグ
- [x] CMake `USE_DEPTH_DL` オプション + FetchContent auto-download
- [x] フレームスキップ（5フレーム間隔）で CPU 推論コスト削減
- [x] Sensor depth 優先、DL depth フォールバック

## 5. アーキテクチャ概要

```
apps/run_mono.cc          # エントリポイント、CLI パース、メインループ
src/core/                 # Frame, Keyframe, Landmark, Map, Camera
src/tracking/             # Tracking (motion model, reference KF, local map)
src/backend/              # LocalMapping, Optimizer (BA, PoseGraph, GlobalBA)
src/loop_closing/         # LoopClosing (DBoW2, Sim3, pose correction)
src/io/                   # TumRgbdDataset, EurocDataset, MapIO
src/sensors/              # AccelerometerProcessor
src/depth/                # DepthEstimator, OnnxDepthEstimator
scripts/                  # 評価・レポート生成スクリプト
```

## 6. ビルドと実行

```bash
# 基本ビルド
cd build && cmake .. && make -j$(nproc)

# DL Depth 付きビルド
cmake .. -DUSE_DEPTH_DL=ON && make -j$(nproc)

# 実行例
./run_mono --tum <dataset_dir> --depth --accel --no-viz
./run_mono --tum <dataset_dir> --depth-model models/depth_anything_v2_small.onnx --no-viz

# 全モード一括評価
bash ../scripts/eval_all.sh
```

## 7. 今後の方向性

- [ ] 追加データセット（fr2_desk, fr3_long_office）での検証
- [ ] EuRoC での評価パイプライン整備
- [ ] DL Depth の GPU 推論（CUDA provider）
- [ ] 閾値群の設定ファイル化
- [ ] 長尺シーケンスでのメモリ・スレッド安定性確認

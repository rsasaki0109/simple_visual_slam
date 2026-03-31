# SimpleVisualSLAM 開発計画

## 1. ビジョン

**「読めるSLAM」を目指す。**

50k行のORB-SLAM3は正確だが読めない。DROID-SLAMはPyTorchの学習パイプラインを理解する前提がある。
SimpleVisualSLAMは5k行のC++で、特徴点ベースSLAM + 深度センサー + DL深度推定 + 加速度計統合を
一本のリポジトリで完結させる。ターゲットはSLAMを学びたいエンジニアと、
既存OSSのGPLライセンスを避けたいプロダクト開発者。

**ライセンス: BSD-2-Clause（予定）**

## 2. 競合ポジション

### 2.1 Feature-based SLAM

| Project | License | Approach | Strengths | Weaknesses |
|---------|---------|----------|-----------|------------|
| ORB-SLAM3 | GPL-3.0 | Feature | 最高精度、論文付き | GPL、50k行、読めない |
| stella_vslam | BSD-2 | Feature | BSD、コミュニティ維持 | DL未統合、精度やや劣る |
| **SimpleVisualSLAM** | **BSD-2** | **Feature+DL** | **DL Depth統合、5k行、教材向き** | **精度バラつき、データセット少** |

### 2.2 Direct / Deep SLAM（非GPL）

| Project | License | Approach | Notes |
|---------|---------|----------|-------|
| DROID-SLAM | BSD-3 | Deep (dense flow) | 最高精度級、Python/PyTorch、重い |
| DPVO | MIT | Deep (patch VO) | DROID-SLAMより3x高速 |
| SplaTAM | BSD-3 | 3DGS SLAM | 高品質rendering、0.3-0.5 FPS |
| Gaussian-SLAM | MIT | 3DGS sub-map | DROID-SLAMでtracking |
| NICE-SLAM | Apache-2.0 | NeRF implicit | 先駆的、後続に抜かれた |
| GO-SLAM | Apache-2.0 | NeRF + global opt | RGB/RGBD対応 |
| Point-SLAM | Apache-2.0 | Neural point cloud | rendering + tracking両立 |
| Basalt | BSD-3 | Direct VIO | EuRoC/TUM-VI向き |
| Kimera-VIO | BSD-2 | Feature VIO + mesh | MIT SPARK Lab |

### 2.3 差別化ポイント

1. **C++ネイティブ + DL深度**: Python系Deep SLAMと違い、C++でONNX Runtimeを使うので組込み系にも展開可
2. **Depth Anything v2統合済み**: センサーdepthなしでも動く単眼DL-SLAMはOSSでほぼない
3. **段階的に構築したgit履歴**: コミット履歴がそのままチュートリアル
4. **5k行**: ORB-SLAM3の1/10、stella_vslamの1/6

## 3. 現在の実装状態

### 3.1 完了済み機能

- [x] ORB特徴点抽出・マッチング
- [x] 2フレーム初期化 (H/F行列)
- [x] 等速モデル + 参照KFトラッキング + 局所マップトラッキング
- [x] 局所BA (Ceres, Depth Prior, Gravity Prior)
- [x] DBoW2ループ検出 + Sim3ポーズグラフ最適化
- [x] マップ保存/読み込み
- [x] EuRoC / TUM RGB-D入力
- [x] 深度センサー統合（単フレーム初期化、depth-assisted mapping）
- [x] 加速度計統合（重力推定・アライメント、静止検知）
- [x] DL深度推定（Depth Anything v2, ONNX Runtime）
- [x] ループ補正スレッド安全性（KF/LMスナップショット方式）
- [x] 評価スクリプト（eval_all.sh, HTML軌跡レポート, 3Dマップビューア）

### 3.2 精度実績

| Dataset | Mono | +Depth | +Depth+Accel | +DL Depth |
|---------|------|--------|--------------|-----------|
| Seq A (small motion) | 0.023 | 0.011 | 0.011 | 0.051 |
| Seq B (room-scale) | 0.845 | 0.227 | 0.235 | — |

※ ATE mean [m], Sim3 alignment。非決定的要因あり（ループ検出タイミング依存）

### 3.3 アーキテクチャ

```
apps/run_mono.cc              メインループ、CLI
src/core/                     Frame, Keyframe, Landmark, Map, Camera
src/tracking/                 Tracking (motion model, ref KF, local map)
src/backend/                  LocalMapping, Optimizer (BA, PoseGraph)
src/loop_closing/             LoopClosing (DBoW2, Sim3, fuse, correct)
src/io/                       TumRgbdDataset, EurocDataset, MapIO
src/sensors/                  AccelerometerProcessor
src/depth/                    DepthEstimator, OnnxDepthEstimator
scripts/                      評価・レポート生成
```

## 4. ロードマップ

### Phase A: stella_vslam同等の品質基盤（最優先）

**目標: 安定性と再現性でstella_vslamに並ぶ**

- [ ] A-1: 精度安定化
  - ループ検出の決定論的シード導入（ORB抽出のランダム性排除）
  - BA収束条件の厳密化
  - trackLocalMap中のoutlier rejection強化
  - 目標: 同一入力で±15%以内のATE変動

- [ ] A-2: 追加データセット検証（最低5シーケンス）
  - 既存: Seq A (small), Seq B (room)
  - 追加: 大規模環境、高速移動、低テクスチャの各パターン
  - 各モード (mono/depth/depth+accel/DL depth) で全シーケンス評価

- [ ] A-3: テスト・CI
  - ユニットテスト: Frame/Keyframe/Landmark/Map/Camera
  - 統合テスト: 合成データでトラッキング精度確認
  - GitHub Actions: ビルド確認 + 基本テスト（データセット不要分）

- [ ] A-4: README + 使い方ドキュメント
  - 英語README (日本語は別ファイル)
  - ビルド手順、依存関係、実行例
  - 結果テーブル + アーキテクチャ図
  - stella_vslamのREADMEと同等の見栄えを目標

### Phase B: DL深度の差別化強化

**目標: 「Depth Anything統合SLAM」として唯一のOSSになる**

- [ ] B-1: DL Depth精度改善
  - Metric3D v2 / UniDepth v2 対応（metric scaleの直接推定）
  - depth_is_metric_ = true で動作するDLモデル → sensor depthと同等の精度
  - モデル切替: Depth Anything v2 (relative) / Metric3D (metric) を選択可

- [ ] B-2: DL Depth推論高速化
  - ONNX Runtime CUDA ExecutionProvider 対応
  - TensorRT対応 (optional)
  - キーフレームのみ推論 → 初期化時のみフル解像度、以降は1/2解像度

- [ ] B-3: DL Depth単体モードの精度向上
  - DL depthのconfidence map活用（低信頼領域をBA対象から除外）
  - 時間的consistency: 前フレームのdepthとの整合性チェック
  - 目標: sensor depthの2倍以内の精度

### Phase C: 堅牢性の本格強化

**目標: 長時間実行で落ちないSLAM**

- [ ] C-1: スレッド安全性の体系的改善
  - Landmark::observations_ のロック戦略統一
  - Map操作のread-write lock導入
  - LocalMapping ↔ LoopClosing の排他制御見直し

- [ ] C-2: 再ローカライズの改善
  - BoW-based place recognition による失敗回復
  - マップの部分再利用

- [ ] C-3: メモリ管理
  - 古いキーフレーム/ランドマークの間引き
  - 長尺シーケンスでのメモリ使用量制限

### Phase D: 拡張機能

**目標: stella_vslamを超える機能セット**

- [ ] D-1: Stereo入力対応
  - ステレオペアからのdepth生成
  - 左右画像の特徴マッチング

- [ ] D-2: IMU tight coupling（オプション）
  - 加速度計/ジャイロのpre-integration
  - BA内でのIMU residual
  - VIO (Visual-Inertial Odometry) モード

- [ ] D-3: 3D Gaussian Splatting マッピング（実験的）
  - キーフレーム + depth → 3DGSシーン構築
  - SplaTAM/Gaussian-SLAM的なアプローチをC++で
  - rendering品質の可視化

- [ ] D-4: ROS 2ノード化
  - sensor_msgs/Image, sensor_msgs/Imu 対応
  - tf2によるpose publish
  - rqiz可視化

### Phase E: コミュニティ・ドキュメント

- [ ] E-1: チュートリアル記事
  - 「5000行で作る Visual SLAM」シリーズ
  - Phase別に解説（初期化 → トラッキング → BA → ループ → DL Depth）
  - git commitを追えば段階的に理解できる構成

- [ ] E-2: API文書
  - Doxygenコメント追加
  - 主要クラスのインタフェース仕様

- [ ] E-3: Contributing guide
  - コーディング規約
  - PRテンプレート
  - Issue テンプレート

## 5. 依存関係とライセンス

| 依存 | License | 用途 | 必須/オプション |
|------|---------|------|----------------|
| OpenCV 4.5+ | Apache-2.0 | 画像処理、特徴抽出 | 必須 |
| Ceres Solver | BSD-3 | BA、ポーズグラフ最適化 | 必須 |
| Sophus | MIT | SE3/Sim3 Lie群 | 必須 |
| Eigen3 | MPL-2.0 | 線形代数 | 必須 |
| DBoW2 | BSD (modified) | BoWループ検出 | オプション |
| ONNX Runtime | MIT | DL深度推定 | オプション |
| fmt | MIT | ログ出力 (将来) | オプション |

**全依存がBSD/MIT/Apache互換。GPL汚染なし。**

## 6. ビルドと実行

```bash
# 基本ビルド
mkdir build && cd build
cmake .. && make -j$(nproc)

# DL Depth付きビルド
cmake .. -DUSE_DEPTH_DL=ON && make -j$(nproc)

# 実行例
./run_mono --tum <dataset_dir> --depth --accel --no-viz
./run_mono --tum <dataset_dir> --depth-model models/depth_anything_v2_small.onnx --no-viz

# 全モード一括評価
bash ../scripts/eval_all.sh
```

## 7. 優先順位の考え方

```
安定性 > 精度 > 機能数 > 速度 > 見栄え
```

理由:
- 不安定なSLAMは使われない
- 精度が出なければ存在意義がない
- 機能が少なくても正確なら使い道がある
- 速度は後から最適化できる
- 見栄えはREADMEとデモで十分

## 8. 非目標

現時点では以下を優先しない:

- リアルタイムAR/VR向けの低レイテンシ最適化
- 商用グレードの堅牢化
- Web UI / GUI可視化ツール
- LiDAR, Event Camera対応
- 大規模環境（数km規模）のマッピング

## 9. この文書の運用

以下のタイミングで更新する:

- Phaseの完了時
- 方針変更時
- 新しい競合OSSの出現時
- ベンチマーク結果の大幅な変化時

目的: 「次に何をすべきか」を迷わない状態を維持すること。

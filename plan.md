# SimpleVisualSLAM 開発計画

> **この文書はAIコーディングエージェント（Codex等）への引き継ぎ資料を兼ねる。**
> 初めて触るエージェントがこの文書だけでコードベースを理解し、
> 次のタスクに着手できることを目標に書いている。

---

## 1. ビジョン

**「読めるSLAM」を目指す。**

50k行のORB-SLAM3は正確だが読めない。DROID-SLAMはPyTorchの学習パイプラインを理解する前提がある。
SimpleVisualSLAMは6k行のC++で、特徴点ベースSLAM + 深度センサー + DL深度推定 + 加速度計統合を
一本のリポジトリで完結させる。

**ターゲットユーザ:**
- SLAMを学びたいエンジニア（git履歴がチュートリアル）
- GPLを避けたいプロダクト開発者（全依存がBSD/MIT/Apache互換）
- 組込み系でDL深度推定を使いたい人（C++ + ONNX Runtime）

**ライセンス: BSD-2-Clause（予定）**

---

## 2. 競合ポジション

### 2.1 Feature-based SLAM

| Project | License | LOC | Strengths | Weaknesses |
|---------|---------|-----|-----------|------------|
| ORB-SLAM3 | GPL-3.0 | ~50k | 最高精度、Mono/Stereo/RGBD/IMU | GPL、巨大、読めない |
| stella_vslam | BSD-2 | ~30k | BSD、Mono/Stereo/RGBD、コミュニティ | DL未統合、精度やや劣る |
| **SimpleVisualSLAM** | **BSD-2** | **~6k** | **DL Depth統合、教材向き** | **精度バラつき、データセット少** |

### 2.2 Direct / Deep SLAM（非GPL）

| Project | License | Language | Approach | Notes |
|---------|---------|----------|----------|-------|
| DROID-SLAM | BSD-3 | Python/CUDA | Deep dense flow | 最高精度級、GPU必須、重い |
| DPVO | MIT | Python/CUDA | Deep patch VO | DROID-SLAMより3x高速 |
| SplaTAM | BSD-3 | Python/CUDA | 3DGS SLAM | 高品質rendering、0.3-0.5 FPS |
| Gaussian-SLAM | MIT | Python/CUDA | 3DGS sub-map | DROID-SLAMベースtracking |
| NICE-SLAM | Apache-2.0 | Python | NeRF implicit | 先駆的、後続に抜かれた |
| GO-SLAM | Apache-2.0 | Python | NeRF + global opt | RGB/RGBD両対応 |
| Point-SLAM | Apache-2.0 | Python | Neural point cloud | rendering + tracking両立 |
| Basalt | BSD-3 | C++ | Direct VIO | EuRoC/TUM-VI向き、堅牢 |
| Kimera-VIO | BSD-2 | C++ | Feature VIO + mesh | MIT SPARK Lab |

### 2.3 差別化ポイント

1. **C++ネイティブ + DL深度**: Python系Deep SLAMと違い、C++でONNX Runtimeを使うので組込み系にも展開可能
2. **Depth Anything v2統合済み**: センサーdepthなしでも動く単眼DL-SLAMはOSSでほぼ存在しない
3. **段階的に構築したgit履歴**: コミット履歴がそのままチュートリアル（14コミットで全機能が段階的に追加）
4. **6k行**: ORB-SLAM3の1/8、stella_vslamの1/5。全コードを1日で読める
5. **GPL汚染なし**: 全依存ライブラリがBSD/MIT/Apache互換

---

## 3. 現在の実装状態

### 3.1 完了済み機能

- [x] ORB特徴点抽出・マッチング（cv::ORB, BFMatcher HAMMING）
- [x] 2フレーム初期化（Homography / Fundamental行列、モデル選択、三角測量）
- [x] 単フレームDepth初期化（depth back-projection、100点以上で成功判定）
- [x] 等速モデル + 参照KFトラッキング + 局所マップトラッキング
- [x] PnP (EPNP+RANSAC) によるポーズ推定
- [x] 局所BA (Ceres Solver, ReprojectionError + DepthPriorError + GravityPriorError)
- [x] DBoW2ループ検出 + Sim3幾何検証 + ポーズグラフ最適化
- [x] マップ保存/読み込み（独自バイナリ形式 "SVSLAM"ヘッダ）
- [x] EuRoC / TUM RGB-D データセット入力
- [x] 深度センサー統合（TUM depth.txt、CV_16UC1 scale=5000 / CV_32FC1）
- [x] 加速度計統合（重力推定、Rodriguesアライメント、静止検知）
- [x] DL深度推定（Depth Anything v2 Small, ONNX Runtime, 518x518入力）
- [x] ループ補正スレッド安全性（KF/LMスナップショット + atomic flag）
- [x] 評価スクリプト（eval_all.sh, HTML軌跡レポート, 3Dマップビューア）

### 3.2 精度実績

| Dataset | Mono | +Depth | +Depth+Accel | +DL Depth | Target |
|---------|------|--------|--------------|-----------|--------|
| Seq A (small motion) | 0.023 | 0.011 | **0.011** | 0.051 | <0.05 |
| Seq B (room-scale) | 0.845 | **0.227** | 0.235 | — | <0.30 |

- 単位: ATE mean [m], Sim3 alignment
- 非決定的要因あり: ループ検出タイミング依存でSeq Bは0.19〜0.49mの幅がある
- DL Depthはセンサーdepthなし/CPU推論/5フレーム間隔で0.051m（Seq Aのみ測定）

### 3.3 ファイル構成と各ファイルの役割

```
simple_visual_slam/
├── CMakeLists.txt                  # ビルド設定。FetchContent: Sophus, Ceres 2.1, DBoW2, ONNX Runtime
├── plan.md                         # この文書
├── .gitignore
│
├── apps/
│   └── run_mono.cc                 # [392行] エントリポイント。CLI解析、メインループ、軌跡出力
│                                   #   --tum/--euroc/--depth/--accel/--depth-model/--no-viz
│                                   #   TUM形式 trajectory.txt (timestamp tx ty tz qx qy qz qw) 出力
│
├── src/core/
│   ├── common.h                    # [30行] 共通型定義: Vec2/Vec3/Mat33/SE3/Sim3, Eigen/Sophus include
│   ├── camera.h / camera.cc        # [50行] ピンホールカメラモデル: fx_, fy_, cx_, cy_, project(), unproject()
│   ├── frame.h / frame.cc          # [108行] フレーム: id_, timestamp_, T_cw_, keypoints_, descriptors_,
│   │                               #   depth_image_, depth_is_metric_, extractORB(), getDepth(), backprojectWithDepth()
│   ├── keyframe.h / keyframe.cc    # [150行] Keyframe extends Frame相当: covisibility graph, landmarks_,
│   │                               #   gravity_in_camera_, has_gravity_, updateConnections()
│   ├── landmark.h / landmark.cc    # [40行] 3Dランドマーク: pos_w_, observations_ (weak_ptr<KF>→idx),
│   │                               #   mutex_付きaddObservation/removeObservation/setPos/getPos
│   └── map.h / map.cc             # [60行] Map: keyframes_, landmarks_, mutex_, loop_correcting_ (atomic<bool>)
│
├── src/tracking/
│   ├── tracking.h / tracking.cc    # [1658行] メインのトラッキングモジュール。最大のファイル。
│   │   # 状態: NO_IMAGES_YET → NOT_INITIALIZED → OK / LOST
│   │   # addFrame() → initialize() or track()
│   │   # initialize(): initializeWithDepth() → 失敗時 2フレーム初期化
│   │   # track(): motion model → trackReferenceKeyframe() → trackLocalMap() → needNewKeyframe()
│   │   # trackReferenceKeyframe(): last_frame_とのORBマッチ、landmark伝播、PnP
│   │   # trackLocalMap(): ref KF + covisible KFのlandmarksを投影、PnP refinement
│   │   # relocalize(): 全KFとのマッチング試行（LOST時）
│   │   # reinitialize(): 長期LOST時に新規マップセグメント作成
│   │   # createLandmarksFromDepth(): depth付きKFの未マッチkpからlandmark生成
│   │   # setKeyframeGravity(): accel_buffer_からKF近傍の重力方向を推定しKFに保存
│   │   # onBACompleted(): BA完了通知でcurrent frameのポーズを再計算
│   │   #
│   │   # 既知の問題:
│   │   #   - loop_correcting_フラグ中はtrackLocalMapをスキップ（精度劣化の原因）
│   │   #   - trackReferenceKeyframeはフラグ中でも実行される（frame-local landmarks参照のみでsafe）
│   │   #   - needNewKeyframe()の閾値がハードコード（frames_since_last_kf_ >= 20等）
│   │
│   └── initializer.h / initializer.cc  # [378行] 2フレーム初期化: H/F推定、モデル選択、R|t復元、三角測量
│
├── src/backend/
│   ├── local_mapping.h / local_mapping.cc  # [481行] LocalMapping:
│   │   # insertKeyframe → processNewKeyframe → createNewMapPoints → mapPointCulling → optimization(BA)
│   │   # 別スレッドで動作。on_ba_completed_コールバックでTrackingに通知。
│   │   # setLoopClosing()でLoopClosingにKFを渡す接続。
│   │
│   └── optimizer.h / optimizer.cc  # [720行] Ceres最適化:
│       # ReprojectionError: 2残差 (pixel reprojection)
│       # DepthPriorError: 1残差 (z_camera - z_observed), σ=0.02m(sensor)/0.2m(DL)
│       # GravityPriorError: 3残差 (R_cw * g_world - g_camera_observed), weight=5.0
│       # PoseGraphError: 7残差 (translation + rotation + scale)
│       #
│       # bundleAdjustment(): local KFs + landmarks、Huber loss
│       # poseGraphOptimization(): Sim3 pose graph、KF/LMスナップショットで安全化済み
│       # globalBundleAdjustment(): 全KF/LM（LM数上限2000）
│       # poseOptimization(): stub（未実装）
│       #
│       # 重要な実装詳細:
│       #   - Ceres QuaternionManifold使用（[w,x,y,z]順）
│       #   - pose param: double[7] = [tx,ty,tz, qw,qx,qy,qz]（BA）
│       #                 double[8] = [tx,ty,tz, qw,qx,qy,qz, log_scale]（PoseGraph）
│       #   - landmark position_bound = 100.0m（発散防止）
│       #   - poseGraph: KF/LMをstd::mapにコピーしてからイテレート（スレッド安全性）
│
├── src/loop_closing/
│   └── loop_closing.h / loop_closing.cc  # [640行] LoopClosing:
│       # 別スレッド。insertKeyframe → processNewKeyframe:
│       #   1. detectLoop(): DBoW2でloop候補検索 (min_loop_score_=0.01, min_loop_interval_kf_=30)
│       #   2. computeSim3(): 3D-3D対応からSim3推定 (RANSAC 200iter, scale 0.85-1.15 for metric)
│       #   3. correctLoop(): poseGraphOptimization → fuseLoopLandmarks → updateConnections
│       #
│       # スレッド安全性:
│       #   - correctLoop()でmap_->loop_correcting_ = true にしてからfuse/update実行
│       #   - poseGraphOptimization前に20ms sleep（tracking側がフラグを見る猶予）
│       #   - poseGraphOptimization内部はKF/LMスナップショットで安全
│       #   - Tracking側はtrackLocalMap()でフラグチェック、trueならスキップ
│       #
│       # Metric Depth対応:
│       #   - has_metric_depth_フラグでSim3 scale範囲を0.85-1.15に制限
│       #   - scale_weight = 1000.0（通常15.0）でスケール変動を抑制
│       #
│       # 既知の問題:
│       #   - ループ検出は非決定的（DBoW2スコアのバラつき）
│       #   - fuseLoopLandmarks中のkf->landmarks_[]書き換えはmutexなし
│       #     （loop_correcting_フラグでtrackLocalMapはスキップされるので実害なし）
│       #   - mergeLandmarks()はsource->mutex_をロックするがtarget側はロックなし
│
├── src/io/
│   ├── tum_dataset.h / tum_dataset.cc  # [316行] TUM RGB-D:
│   │   # rgb.txt, depth.txt, accelerometer.txt パーサー
│   │   # nextWithDepth(): RGB + depth画像をタイムスタンプ同期（±30ms tolerance）で返す
│   │   # depth読み込み: CV_16UC1 → float/5000.0, CV_32FC1はそのまま
│   │   # K(): TUM freiburg1のカメラ内部パラメータ (fx=517.3, fy=516.5, cx=318.6, cy=255.3)
│   │
│   ├── euroc_dataset.h / euroc_dataset.cc  # [189行] EuRoC MAV: cam0/data/ + data.csv
│   │
│   └── map_io.h / map_io.cc      # [269行] バイナリマップ保存/読み込み
│       # ヘッダ "SVSLAM" + camera + keyframes + landmarks + covisibility
│       # 互換性バージョニングなし（既知の制限）
│
├── src/depth/
│   ├── depth_estimator.h          # [15行] 抽象基底: virtual estimate(cv::Mat) → cv::Mat
│   ├── onnx_depth_estimator.h     # [30行] #ifdef USE_DEPTH_DL
│   └── onnx_depth_estimator.cc    # [181行] Depth Anything v2推論:
│       # preprocess(): BGR→RGB, resize 518x518, float/255, ImageNet正規化, HWC→CHW
│       # postprocess(): disparity→depth (1/d), median-based scaling (1.5m), clamp [0.1, 20.0]
│       # estimate(): tensor作成→Session::Run→postprocess
│       # 入力: NCHW [1,3,518,518] float32
│       # 出力: 相対depth（非metric）、depth_is_metric_ = false
│
├── src/sensors/
│   └── accelerometer.h            # [80行] header-only:
│       # estimateGravity(): 加速度平均→正規化（magnitude 8.0-12.0 sanity check）
│       # isStationary(): 分散 < threshold（default 0.5）
│       # computeGravityAlignment(): gravity→[0,0,-1]へのRodrigues回転
│
├── scripts/
│   ├── eval_all.sh                # 全データセット × 全モード一括評価（evo_ape使用）
│   ├── eval_ate.sh                # 単一軌跡のATE評価
│   ├── extract_keyframe_trajectory.py  # map.binからKF軌跡をTUM形式で抽出
│   ├── generate_map_report.py     # map.bin → 3Dマップビューア HTML (Three.js)
│   └── generate_tum_report.py     # online + corrected軌跡のSVG比較レポート HTML
│
├── data/
│   ├── ORBvoc.txt                 # DBoW2語彙ファイル（gitignore対象、手動配置）
│   └── tum/                       # TUMデータセット（gitignore対象）
│       ├── rgbd_dataset_freiburg1_xyz/
│       └── rgbd_dataset_freiburg1_room/
│
└── models/
    └── depth_anything_v2_small.onnx  # DLモデル（gitignore対象、HuggingFace DL）
```

### 3.4 スレッドモデル

```
Main Thread (run_mono.cc)
  └── while(true): read image → create Frame → tracker->addFrame() → save trajectory

LocalMapping Thread (local_mapping->run())
  └── while(!stop): wait for KF → processNewKeyframe → createNewMapPoints → culling → BA
  └── BA完了時: on_ba_completed_ callback → Tracking::onBACompleted()

LoopClosing Thread (loop_closing->run())
  └── while(!stop): wait for KF → detectLoop → computeSim3 → correctLoop
  └── correctLoop: poseGraphOpt → set loop_correcting_ → fuse → updateConnections → clear flag
```

**スレッド間共有データと保護:**

| データ | 書き手 | 読み手 | 保護方法 |
|--------|--------|--------|----------|
| Map::keyframes_ | LocalMapping, LoopClosing | Tracking, Optimizer | poseGraphでスナップショットコピー |
| Map::landmarks_ | LocalMapping, LoopClosing | Tracking, Optimizer | poseGraphでスナップショットコピー |
| Landmark::observations_ | LocalMapping (addObservation) | Optimizer (iteration) | Landmark::mutex_ |
| Landmark::pos_w_ | Optimizer (setPos) | Tracking (getPos) | Landmark::mutex_ (setPos側のみ) |
| Keyframe::T_cw_ | Optimizer (write-back) | Tracking (read) | なし（SE3代入は実質atomic） |
| Keyframe::landmarks_[] | LoopClosing (fuse) | Tracking (trackLocalMap) | loop_correcting_ flag |
| Keyframe::connected_keyframes_ | LoopClosing (updateConn) | Tracking (getBestCovisibility) | loop_correcting_ flag |

### 3.5 CMakeビルドシステム

```cmake
# 必須依存
find_package(OpenCV 4.5 REQUIRED)
find_package(Eigen3 REQUIRED)
# Ceres 2.1: system版が2.0の場合はFetchContentで2.1.0取得
find_package(Ceres 2.1 QUIET)  # ← 2.1必須（manifold API使用）

# オプション
option(USE_DBOW2 "Enable DBoW2" ON)      # FetchContent: dorian3d/DBoW2
option(USE_DEPTH_DL "Enable DL depth" OFF) # FetchContent: ONNX Runtime 1.17.1 pre-built

# ターゲット
add_library(svslam_core STATIC ...)       # 全src/
add_executable(run_mono apps/run_mono.cc) # リンク: svslam_core
```

### 3.6 定数・閾値一覧（ハードコード）

| 場所 | 定数 | 値 | 意味 |
|------|------|-----|------|
| run_mono.cc | ORB features | 2000 | 1フレームあたりの特徴点数 |
| tracking.cc | min_init_depth_points | 100 | depth初期化に必要な最小3D点数 |
| tracking.cc | max_depth_for_init | 10.0m | depth初期化時の有効depth上限 |
| tracking.cc | kf_frames_threshold | 20 | 最低KF挿入間隔（フレーム数） |
| tracking.cc | kf_tracking_ratio_threshold | 0.75 | KF挿入判定の追跡率閾値 |
| tracking.cc | lowe_ratio | 0.75 | ORBマッチのLowe ratio test |
| tracking.cc | max_lost_frames_ | 30 | LOST状態の最大許容フレーム |
| tracking.cc | reinit_trigger_frames_ | 20 | 再初期化開始までのLOSTフレーム |
| optimizer.cc | position_bound | 100.0m | landmark位置のclamp範囲 |
| optimizer.h | depth_sigma_sensor | 0.02m | sensor depth prior σ |
| optimizer.h | depth_sigma_dl | 0.2m | DL depth prior σ |
| optimizer.h | gravity_weight | 5.0 | gravity prior weight |
| loop_closing.cc | min_loop_interval_kf_ | 30 | ループ検出の最低KF間隔 |
| loop_closing.cc | min_loop_inliers_ | 30 | Sim3検証の最低inlier数 |
| loop_closing.cc | min/max_sim3_scale_ | 0.85/1.15 | metric depthのSim3スケール範囲 |
| loop_closing.cc | scale_weight (metric) | 1000.0 | metric depthのPoseGraphスケール重み |
| onnx_depth_estimator | kInputW/H | 518 | DL推論入力サイズ |
| onnx_depth_estimator | median_target | 1.5m | relative depthのスケーリング基準 |
| run_mono.cc | dl_frame_skip | 5 | DL depth推論間隔 |

---

## 4. 既知の問題と技術的負債

### 4.1 精度のバラつき（最重要）

**症状:** 同じデータセットで実行ごとにATE meanが0.19m〜0.49m（Seq B）
**原因:**
1. ORB特徴抽出にランダム性がある（cv::ORBのFAST閾値適応）
2. ループ検出タイミングが非決定的（DBoW2スコアが微妙に変動）
3. BA結果がイテレーション数制限で微妙に異なる

**対策案:**
- `cv::ORB::create(2000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20)` のように固定パラメータ
- ORB抽出後のキーポイント座標をソートして決定論的順序にする
- Ceresの収束条件を厳密化（function_tolerance, gradient_tolerance）

### 4.2 trackLocalMapスキップによる精度劣化

**症状:** ループ補正中の数フレームでtrackLocalMapがスキップされ、ポーズ精度が落ちる
**原因:** `map_->loop_correcting_`フラグ中はtrackLocalMapを飛ばすため
**影響:** 通常1-5フレーム分。スキップ後に次のtrackLocalMapで回復。

**対策案:**
- LoopClosing側でcorrectLoop()を2段階に分ける:
  1. poseGraphOptimization → KFポーズのみ更新（trackLocalMapは継続可能）
  2. fuseLoopLandmarks → ここだけフラグでブロック（1-2フレーム分）
- 現在はposeGraph後にfuse+updateが走るので、フラグ期間はfuse+update分のみ

### 4.3 mergeLandmarksの不完全なロック

**症状:** 理論上、target landmarkのobservations_に同時アクセス可能
**場所:** `loop_closing.cc:mergeLandmarks()` — source側はロックするがtarget側しない
**対策:** `target->mutex_`もロックする（デッドロック回避のためID順でロック）

### 4.4 マップ保存形式にバージョニングなし

**症状:** map.binの形式を変更すると過去のファイルが読めなくなる
**対策:** ヘッダにバージョン番号を追加、マイグレーション機能

### 4.5 TUM以外のデータセットでの検証不足

**症状:** 2シーケンスでしかテストしていない
**対策:** Phase A-2で5シーケンス以上に拡大

### 4.6 DL Depth推論がCPUで遅い

**症状:** Depth Anything v2 Small で1フレーム約0.7秒（CPU）
**現状の緩和:** 5フレーム間隔でのみ推論
**対策:** ONNX Runtime CUDAProvider / TensorRT

---

## 5. ロードマップ

### Phase A: stella_vslam同等の品質基盤（最優先）

**目標: 安定性と再現性でstella_vslamに並ぶ。OSSとして公開できる最低品質。**

#### A-1: 精度安定化

**目的:** 同一入力で±15%以内のATE変動に抑える

**タスク:**
1. ORBの決定論化
   - `tracking.cc`: `cv::ORB::create()` にseed的パラメータを固定
   - 特徴点を座標値でソートしてからdescriptor計算
   - 効果測定: 同一入力5回実行でATE stdを計算

2. ループ検出の安定化
   - `loop_closing.cc`: DBoW2スコア閾値を動的に調整（近傍KFの平均スコアの0.8倍等）
   - 検出候補の選択をスコア降順で決定論的に

3. BA収束条件
   - `optimizer.cc`: `options.function_tolerance = 1e-6`（現在Ceresデフォルト1e-6）
   - `options.parameter_tolerance = 1e-8`
   - iterations上限を20に増加（現在10）

4. outlier rejection
   - `tracking.cc:trackLocalMap()`: PnP後のreproject errorでoutlier除外
   - 現在はPnP inlierをそのまま使用。2パスPnP（粗→精）を検討。

**検証:** Seq B depth+accelで5回実行、ATE mean/std/min/maxを記録

#### A-2: 追加データセット検証

**目的:** 最低5シーケンスで全モード評価

**タスク:**
1. データセット追加（計5シーケンス）
   - 既存2: Seq A (small motion), Seq B (room-scale)
   - 追加3: large-scale, fast-motion, low-texture の各パターン

2. eval_all.sh を拡張
   - 複数回実行モード（`--repeat N`）: N回実行してmean/stdを出力
   - 結果をMarkdownテーブルで自動生成

3. 結果テーブルを plan.md とREADMEに反映

**検証:** 5シーケンス × 4モード = 20パターン全て完走

#### A-3: テスト・CI

**目的:** ビルド破壊を検知できるCI環境

**タスク:**
1. ユニットテスト（Google Test）
   - `tests/test_camera.cc`: project → unproject の往復テスト
   - `tests/test_frame.cc`: ORB抽出、depth取得
   - `tests/test_landmark.cc`: addObservation/removeObservation のスレッド安全性
   - `tests/test_map.cc`: KF/LM追加・削除

2. 統合テスト
   - `tests/test_tracking_synthetic.cc`: 合成直線運動データでATE < 0.01mを確認
   - 実データセット不要（カメラモデル + 合成3D点 + 投影で入力生成）

3. GitHub Actions
   ```yaml
   # .github/workflows/ci.yml
   - Ubuntu 22.04
   - apt install: libopencv-dev libeigen3-dev
   - cmake .. -DUSE_DEPTH_DL=OFF && make -j
   - ctest --output-on-failure
   ```

4. CMakeにテスト追加
   ```cmake
   enable_testing()
   add_executable(tests tests/test_*.cc)
   target_link_libraries(tests svslam_core GTest::gtest_main)
   add_test(NAME unit_tests COMMAND tests)
   ```

#### A-4: README + ドキュメント

**目的:** stella_vslamのREADMEと同等の見栄えと情報量

**タスク:**
1. README.md（英語）
   - Overview + feature list
   - Architecture diagram（Mermaid）
   - Build instructions (Ubuntu 22.04/24.04)
   - Usage examples（全モード）
   - Results table（ベンチマーク名は伏せる、"Indoor Sequence A/B/C..."等）
   - License section
   - Acknowledgements

2. README_ja.md（日本語）
   - README.mdの翻訳 + 補足

3. ARCHITECTURE.md
   - モジュール間の依存関係図
   - データフロー図
   - スレッドモデル図

---

### Phase B: DL深度の差別化強化

**目標: 「Depth Anything統合SLAM」として唯一のOSSになる**

#### B-1: Metric DL Depth対応

**目的:** DL depth推定でmetric scaleを直接出力 → sensor depth同等の精度

**背景:**
- 現在のDepth Anything v2は relative depth（スケール不定）
- Metric3D v2 / UniDepth v2 は metric depth を直接推定可能
- metric depthなら `depth_is_metric_ = true` で動作 → 初期化・BA精度が大幅向上

**タスク:**
1. `src/depth/metric_depth_estimator.h/cc` 新規作成
   - Metric3D v2 Small のONNXモデル対応
   - `isMetric()` override → true
   - 入力サイズ・正規化パラメータはモデル依存（ヘッダで定数定義）

2. `apps/run_mono.cc` でモデル種別の自動判定 or CLI指定
   - `--depth-model path.onnx --depth-metric` フラグ追加
   - あるいはモデルメタデータから自動判定

3. `CMakeLists.txt` にMetric3D用のプリプロセス設定

**検証:** DL metric depth + Seq Aで ATE < 0.03m（sensor depthの3倍以内）

#### B-2: GPU推論対応

**目的:** DL depth推論を10x高速化（0.7s → 0.07s/frame）

**タスク:**
1. `onnx_depth_estimator.cc` に CUDA ExecutionProvider 追加
   ```cpp
   if (use_cuda) {
       OrtCUDAProviderOptions cuda_opts;
       session_options.AppendExecutionProvider_CUDA(cuda_opts);
   }
   ```

2. `CMakeLists.txt` に `USE_CUDA` オプション
   - ONNX Runtime GPU版のFetchContent URL切替

3. CLI: `--depth-gpu` フラグ

4. フレームスキップの動的制御
   - GPU推論なら毎フレーム、CPUなら5フレーム間隔

**検証:** GPU推論で全フレームdepth付与、Seq AでATE改善確認

#### B-3: DL Depth品質向上

**目的:** DL depth単体モードの精度をsensor depthの2倍以内に

**タスク:**
1. Confidence map活用
   - disparity出力の値域・分散からconfidence推定
   - 低confidence領域のdepthをBA対象から除外（depth prior weight → 0）

2. 時間的整合性
   - 前フレームのdepth mapをwarpして現フレームと比較
   - 大きく乖離する領域は信頼度を下げる

3. マルチスケール推論（オプション）
   - 518x518 + 256x256 の2スケール推論
   - 一致度が高い領域のdepthのみ使用

---

### Phase C: 堅牢性の本格強化

**目標: 長時間実行で落ちないSLAM**

#### C-1: スレッド安全性の体系的改善

**現状の問題:**
- Landmark::observations_ のロックが不統一（addObservation側はロック、iteration側は一部のみ）
- Keyframe::connected_keyframes_ のロックなし
- Map操作（addKeyframe/addLandmark）のロックなし

**タスク:**
1. `std::shared_mutex` (read-write lock) 導入
   - `Map::rw_mutex_`: addKeyframe/addLandmark は write lock、getAllKeyframes/getAllLandmarks は read lock
   - 既存の `Map::mutex_` は廃止

2. Landmark observations ロック統一
   - optimizer.cc 内の全 `lm->observations_` イテレーションに `std::lock_guard<std::mutex>(lm->mutex_)` 追加
   - bundleAdjustment (行193, 250, 279) とposeGraphOptimization (行536)

3. Keyframe mutex活用
   - `kf->mutex_` を `updateConnections()`, `getBestCovisibilityKeyframes()`, `landmarks_` アクセスで使用

4. loop_correcting_ フラグの廃止
   - read-write lockで置き換え。correctLoop()はwrite lock取得。

#### C-2: 再ローカライズ改善

**現状:** ORBマッチングベースの全KFスキャン（遅い、精度低い）

**タスク:**
1. DBoW2を使ったplace recognition
   - LOST時にBoWデータベースに現フレームを問い合わせ
   - 上位N候補に対してPnP検証

2. 段階的回復
   - まずBoW → PnP → 成功なら OK に遷移
   - 失敗が続けば reinitialize

#### C-3: メモリ管理

**タスク:**
1. KF/LM間引き
   - `LocalMapping::keyframeCulling()`: 90%以上のlandmarksが他KFでも観測されるKFを削除
   - landmark culling の基準強化

2. メモリ使用量モニタ
   - 定期的にKF/LM数をログ出力
   - 上限設定（例: max 500 KFs, max 100k LMs）

---

### Phase D: 拡張機能

**目標: stella_vslamを機能面で超える**

#### D-1: Stereo入力対応

**タスク:**
1. `src/io/stereo_dataset.h/cc`: ステレオ画像ペア読み込み
2. `Frame` にright image追加、ステレオマッチングでdepth生成
3. `apps/run_stereo.cc` or `--stereo` フラグ

#### D-2: IMU tight coupling

**タスク:**
1. `src/sensors/imu_preintegration.h/cc`: IMU pre-integration factor
2. `Optimizer` にIMU residual追加（Ceres AutoDiff）
3. `Frame` にIMU measurement追加
4. `--imu` フラグ

#### D-3: 3D Gaussian Splatting マッピング（実験的）

**背景:** SplaTAM/Gaussian-SLAM的なアプローチをC++でやるのは差別化になりうる

**タスク:**
1. KF + depth → 3DGS point cloud初期化
2. differentiable rendering はCeresのAutoDiffで近似可能か調査
3. まずは可視化のみ（rendering品質はPhase E向け）

#### D-4: ROS 2ノード化

**タスク:**
1. `ros2_ws/src/simple_visual_slam_ros/`: ROS 2パッケージ
2. `sensor_msgs/Image` subscribe → Frame生成
3. `geometry_msgs/PoseStamped` publish
4. `tf2` broadcast (camera → world)
5. `visualization_msgs/MarkerArray` でランドマーク可視化

---

### Phase E: コミュニティ・ドキュメント

#### E-1: チュートリアル記事

**「5000行で作る Visual SLAM」シリーズ:**

1. 第1回: カメラモデルとフレーム（Camera, Frame）
2. 第2回: 特徴点マッチングとポーズ推定（ORB, PnP）
3. 第3回: 2フレーム初期化と三角測量
4. 第4回: トラッキングとキーフレーム選択
5. 第5回: 局所マッピングとバンドル調整（Ceres入門）
6. 第6回: ループ検出とポーズグラフ最適化（DBoW2, Sim3）
7. 第7回: 深度センサー統合（TUM RGB-D）
8. 第8回: DL深度推定の統合（Depth Anything v2, ONNX Runtime）
9. 第9回: 加速度計と重力アライメント
10. 第10回: マルチスレッド設計とスレッド安全性

各回はgitコミットに対応。読者はそのコミットをcheckoutして手を動かせる構成。

#### E-2: API文書

- Doxygenコメント追加（公開APIのみ）
- `docs/` ディレクトリに生成
- GitHub Pagesで公開

#### E-3: Contributing guide

- CONTRIBUTING.md
- コーディング規約（clang-format設定含む）
- PR/Issueテンプレート
- CLA不要（BSD-2なので）

---

## 6. 依存関係とライセンス

| 依存 | License | 用途 | 必須/オプション | FetchContent |
|------|---------|------|----------------|--------------|
| OpenCV 4.5+ | Apache-2.0 | 画像処理、特徴抽出 | 必須 | system |
| Eigen3 | MPL-2.0 | 線形代数 | 必須 | system |
| Ceres Solver 2.1+ | BSD-3 | BA、ポーズグラフ最適化 | 必須 | system or FetchContent |
| Sophus | MIT | SE3/Sim3 Lie群 | 必須 | FetchContent |
| DBoW2 | BSD (modified) | BoWループ検出 | オプション | FetchContent |
| ONNX Runtime 1.17+ | MIT | DL深度推定 | オプション | FetchContent (pre-built) |
| Google Test | BSD-3 | テスト（将来） | オプション | FetchContent |

**全依存がBSD/MIT/Apache互換。GPL汚染なし。**

---

## 7. ビルドと実行

```bash
# 依存インストール (Ubuntu 22.04)
sudo apt install -y libopencv-dev libeigen3-dev libgoogle-glog-dev libgflags-dev

# 基本ビルド
mkdir build && cd build
cmake .. && make -j$(nproc)

# DL Depth付きビルド
cmake .. -DUSE_DEPTH_DL=ON && make -j$(nproc)

# DL モデルダウンロード
mkdir -p ../models
wget -O ../models/depth_anything_v2_small.onnx \
  https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx

# 実行例
./run_mono --tum <dataset_dir> --no-viz                              # mono
./run_mono --tum <dataset_dir> --depth --no-viz                      # depth
./run_mono --tum <dataset_dir> --depth --accel --no-viz              # depth + accel
./run_mono --tum <dataset_dir> --depth-model ../models/depth_anything_v2_small.onnx --no-viz  # DL depth

# 全モード一括評価
cd .. && bash scripts/eval_all.sh

# 出力ファイル
#   trajectory.txt          - 全フレームのTUM形式軌跡
#   trajectory_keyframes.txt - キーフレームのみの軌跡
#   map.bin                 - バイナリマップファイル
```

---

## 8. 優先順位の考え方

```
安定性 > 精度 > 機能数 > 速度 > 見栄え
```

- 不安定なSLAMは使われない（クラッシュ = 即離脱）
- 精度が出なければ存在意義がない（stella_vslam以下なら意味なし）
- 機能が少なくても正確なら使い道がある（DL Depth統合だけで差別化可能）
- 速度は後から最適化できる
- 見栄えはREADMEとデモ動画で十分

---

## 9. 非目標

現時点では以下を優先しない:

- リアルタイムAR/VR向けの低レイテンシ最適化
- 商用グレードの堅牢化（24/7運用）
- Web UI / GUI可視化ツール
- LiDAR, Event Camera, ToF Camera対応
- 大規模環境（数km規模）のマッピング
- 学習ベースのend-to-end SLAM（DROID-SLAM的アプローチ）

---

## 10. コーディング規約

（未整備。Phase E-3で正式化予定。現状の暗黙ルール:）

- C++17
- Eigen, Sophus ベースの数学表現
- `T_cw_` = Transform from World to Camera（ORB-SLAM系の慣例）
- `SE3` for rigid transform, `Sim3` for similarity
- `Vec3 = Eigen::Vector3d`, `Mat33 = Eigen::Matrix3d`
- `using Ptr = std::shared_ptr<ClassName>`
- ファイル命名: `snake_case.h/cc`
- クラス命名: `PascalCase`
- メンバ変数: `snake_case_`（末尾アンダースコア）
- namespace: `svslam`
- include guard: `#pragma once`
- コメント: 英語（plan.mdとREADMEは日本語可）

---

## 11. この文書の運用

以下のタイミングで更新する:

- Phaseの完了時
- 方針変更時
- 新しい競合OSSの出現時
- ベンチマーク結果の大幅な変化時
- AIコーディングエージェントへの引き継ぎ時

**目的:** 初めてこのリポジトリに触る人間またはAIが、
この文書だけで「今何ができて、次に何をすべきか」を判断できること。

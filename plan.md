# SimpleVisualSLAM 開発計画

> **この文書はAIコーディングエージェント（Codex等）への完全な引き継ぎ資料である。**
> この文書だけでコードベースの全体像を理解し、次のタスクに着手できることを目標に書いている。
> plan.mdの内容とコードの実態が矛盾する場合は、コードが正（plan.mdを修正せよ）。

---

## 1. ビジョン

**「読めるSLAM」— 6k行で動くVisual SLAM + DL深度推定。**

- ORB-SLAM3 (50k行, GPL) は読めない
- DROID-SLAM はPython/PyTorch前提で重い
- stella_vslam (30k行, BSD) はDL深度を統合していない

SimpleVisualSLAMは6k行のC++17で、特徴点ベースSLAM + 深度センサー + DL深度推定 + 加速度計統合を1リポジトリで完結させる。

**ライセンス:** BSD-2-Clause（予定）。全依存がGPL非汚染。

**直近の競合ターゲット:** stella_vslam に精度・安定性で並び、DL深度統合で差別化する。

### 1.1 2026-04-06 現在地

- `master` は `6435a99` まで到達しており、reference-keyframe policy 実験基盤は `PR #1` として mainline に統合済み。
- GitHub repository は public 化済み: `https://github.com/rsasaki0109/simple_visual_slam`
- GitHub Pages も公開済み: `https://rsasaki0109.github.io/simple_visual_slam/`
- 以後の開発方針は「正しい抽象を先に固定する」ではなく、「比較可能な複数実装を先に作り、repeat replay で収束させる」。
- この文書は Codex だけでなく Claude への handoff も意図して更新している。特に後半の「Claudeへの引き継ぎ」は作業開始前に必読。

---

## 2. コードベース全体像

### 2.1 ファイル構成（全ファイル、行数、役割）

```
simple_visual_slam/           # 6393行（テスト含む）
│
├── CMakeLists.txt            # FetchContent: Sophus, Ceres 2.1, DBoW2, ONNX Runtime 1.17, Google Test 1.14
│                             # オプション: USE_DBOW2(ON), USE_DEPTH_DL(OFF), BUILD_TESTS(ON)
├── plan.md                   # この文書
├── README.md                 # 英語README（Mermaidアーキテクチャ図、結果テーブル付き）
├── .gitignore                # *.bin, *.onnx, models/, eval_results/, trajectory*.txt, *.html
│
├── apps/
│   └── run_mono.cc           # [392行] エントリポイント
│       # CLI: --tum/--euroc/--depth/--accel/--depth-model <path>/--no-viz
│       # メインループ: 画像読み込み → Frame生成 → ORB抽出 → tracker->addFrame() → 軌跡保存
│       # DL depth: frame_id <= 1 || frame_id % 5 == 0 の時のみ推論（CPU高速化）
│       # 出力: trajectory.txt, trajectory_online.txt, trajectory_keyframes.txt, map.bin
│       # ORB: cv::ORB::create(2000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20)
│
├── src/core/
│   ├── common.h              # [30行] Vec2/Vec3/Mat33/SE3/Sim3 型エイリアス
│   │                         # Eigen, Sophus include。全ファイルがこれをinclude。
│   │
│   ├── camera.h / .cc        # [50行] ピンホールカメラ: fx_, fy_, cx_, cy_
│   │                         # project(Vec3→Vec2), unproject(Vec2→Vec3), K()→cv::Mat
│   │
│   ├── frame.h / .cc         # [108行] フレーム
│   │   # id_, timestamp_, T_cw_ (SE3), camera_, image_
│   │   # keypoints_, descriptors_ (ORB)
│   │   # landmarks_ (vector<Landmark::Ptr>, 特徴点と同サイズ)
│   │   # depth_image_ (CV_16UC1 or CV_32FC1), depth_is_metric_ (bool)
│   │   # extractORB(orb): 抽出後にresponse降順→座標順でソート（決定論化済み）
│   │   # getDepth(u,v): bilinear風、CV_16UC1は/5000.0(TUM), CV_32FC1はそのまま
│   │   # backprojectWithDepth(kp, depth): unproject * depth → T_wc変換で世界座標
│   │
│   ├── keyframe.h / .cc      # [157行] Keyframe extends Frame相当
│   │   # T_cw_ (SE3), depth_image_, depth_is_metric_
│   │   # gravity_in_camera_ (Vec3), has_gravity_ (bool) — 加速度計から推定した重力方向
│   │   # landmarks_ (vector<Landmark::Ptr>)
│   │   # connected_keyframes_ (map<KF::Ptr, int>) — covisibility graph（共有landmark数）
│   │   # updateConnections(): landmarks_の共有数を数えてconnected_keyframes_更新
│   │   # getBestCovisibilityKeyframes(N): weight降順でN個返す
│   │   # getDepth(u,v): Frame::getDepthと同じ
│   │
│   ├── landmark.h / .cc      # [40行] 3Dランドマーク
│   │   # id_, pos_w_ (Vec3)
│   │   # observations_ (map<weak_ptr<KF>, size_t>) — KF→特徴点index
│   │   # descriptor_ (cv::Mat)
│   │   # is_bad_ (bool)
│   │   # mutable mutex_ — setPos, getPos, addObservation, removeObservation で使用
│   │   # ★getPos()もmutexロック済み（Worker C-1で修正）
│   │
│   └── map.h / .cc           # [60行] マップ
│       # keyframes_ (map<id, KF::Ptr>), landmarks_ (map<id, LM::Ptr>)
│       # mutex_ — 汎用mutex（現在は一部のみ使用）
│       # loop_correcting_ (atomic<bool>) — ループ補正中フラグ
│       # addKeyframe/addLandmark/removeKeyframe/removeLandmark
│       # getAllKeyframes/getAllLandmarks — const参照を返す（コピーではない！）
│
├── src/tracking/
│   ├── tracking.h / .cc      # [1658行] ★最大のファイル。トラッキング全体を管理。
│   │   #
│   │   # === 状態遷移 ===
│   │   # NO_IMAGES_YET → NOT_INITIALIZED → OK ↔ LOST
│   │   #
│   │   # === 主要メソッド ===
│   │   # addFrame(frame): state分岐 → initialize() or track()
│   │   #
│   │   # initialize():
│   │   #   1. initializeWithDepth(): depth画像があれば単フレーム初期化
│   │   #      - 100+点をback-project、gravity alignmentを適用
│   │   #      - 成功 → OK
│   │   #   2. 失敗時: 2フレーム初期化（Initializer使用）
│   │   #      - H/F推定 → R|t復元 → 三角測量 → 2KF + landmarks作成
│   │   #
│   │   # track():
│   │   #   1. loop_correcting_チェック（trueなら即return true — スキップなし、後述）
│   │   #   2. Motion model prediction (等速 or 静止検知)
│   │   #   3. trackReferenceKeyframe(): last_frame_とORBマッチ → landmark伝播 → PnP
│   │   #   4. trackLocalMap(): ref KF + covisible KFのlandmarks投影 → PnP
│   │   #      ★loop_correcting_ true時はスキップ（return true）
│   │   #      ★2パスPnP実装済み（Worker A-1）: 1st PnP → reproj error > 5px除外 → 2nd PnP
│   │   #   5. needNewKeyframe(): frames閾値(20) or tracking ratio閾値(0.75)
│   │   #   6. KF作成 → setKeyframeGravity → addObservation → createLandmarksFromDepth
│   │   #
│   │   # trackReferenceKeyframe():
│   │   #   - current vs last_frame_ のORB knnMatch (Lowe ratio 0.75, distance < 65)
│   │   #   - last_frame_->landmarks_ からlandmark伝播
│   │   #   - PnP (EPNP+RANSAC) でポーズ推定
│   │   #
│   │   # trackLocalMap():
│   │   #   - reference_keyframe_ + covisible 15KFのlandmarksを収集
│   │   #   - 現フレームへ投影、descriptor距離でマッチ
│   │   #   - pose-gated matching (reproj error閾値)
│   │   #   - PnP → 2パスPnP refinement
│   │   #
│   │   # relocalize(): 全KFとマッチング試行（LOST時）
│   │   # reinitialize(): 長期LOST(20frame)後に新マップセグメント作成
│   │   # createLandmarksFromDepth(kf): depth付きKFの未マッチkpからlandmark生成
│   │   # setKeyframeGravity(kf): accel_buffer_からKF近傍±50msの重力方向を推定
│   │   # onBACompleted(): BA完了通知でcurrent frameのポーズを再計算
│   │   #
│   │   # === スレッド安全性 ===
│   │   # - trackLocalMapのみloop_correcting_チェック（KF landmarks直接イテレート）
│   │   # - trackReferenceKeyframeはチェックなし（frame-local landmarks参照のみ）
│   │   # - track()冒頭にはチェックなし（motion modelは安全）
│   │   #
│   │   # === 既知の問題 ===
│   │   # - needNewKeyframe()の閾値がハードコード
│   │   # - loop_correcting_中のtrackLocalMapスキップで1-5フレーム精度劣化
│   │   # - reinitialize()で新マップセグメントと旧マップの接続がない
│   │
│   └── initializer.h / .cc   # [378行] 2フレーム初期化
│       # H/F推定(RANSAC) → モデル選択(スコア比) → R|t復元 → 三角測量
│       # 三角測量: cv::triangulatePoints → 正depth/reproj error/max distance フィルタ
│
├── src/backend/
│   ├── local_mapping.h / .cc  # [481行] 局所マッピング（別スレッド）
│   │   # insertKeyframe → processNewKeyframe → createNewMapPoints → mapPointCulling → optimization
│   │   # processNewKeyframe: KFをmap追加、covisibility更新、LoopClosingへ転送
│   │   # createNewMapPoints: covisible KFペアで三角測量
│   │   #   - depth付きKFの場合: 先にdepth back-projectionでlandmark作成
│   │   # mapPointCulling: 観測率の低いlandmarkを除去
│   │   # optimization: bundleAdjustment呼び出し → on_ba_completed_コールバック
│   │   # on_ba_completed_: Tracking::onBACompleted()を呼ぶ（ポーズ再計算）
│   │
│   └── optimizer.h / .cc     # [721行] Ceres最適化
│       #
│       # === コスト関数 ===
│       # ReprojectionError: 2残差 [px] — (predicted_uv - observed_uv)
│       # DepthPriorError: 1残差 [m] — weight * (z_camera - z_observed)
│       #   weight = 1/σ, σ=0.02m(sensor depth), σ=0.2m(DL depth)
│       # GravityPriorError: 3残差 — weight * (R_cw * g_world - g_camera_observed)
│       #   weight=5.0, g_world=[0,0,-1]
│       # PoseGraphError: 7残差 — translation(3) + rotation(3) + scale(1)
│       #
│       # === ポーズ表現 ===
│       # BA: double[7] = [tx,ty,tz, qw,qx,qy,qz] + QuaternionManifold
│       # PoseGraph: double[8] = [tx,ty,tz, qw,qx,qy,qz, log_scale]
│       # Ceres QuaternionManifold: [w,x,y,z]順（Eigenと同じ）
│       #
│       # === bundleAdjustment() ===
│       # local KFs(可変) + fixed KFs(隣接、定数) + landmarks
│       # Huber loss(1.0), position_bound=100m, iterations=20
│       # function_tolerance=1e-6, parameter_tolerance=1e-8 (Worker A-1で強化)
│       # observations_イテレーション時にlm->mutex_ロック済み (Worker C-1で追加)
│       #
│       # === poseGraphOptimization() ===
│       # ★KF/LMをstd::mapコピーしてからイテレート（スレッド安全性のため）
│       # Sim3ポーズグラフ: covisibility edges + loop edges
│       # SPARSE_NORMAL_CHOLESKY, iterations=60
│       # 書き戻し: kf->T_cw_ = SE3(q, t/scale), lm->setPos(delta * pos)
│       #
│       # === globalBundleAdjustment() ===
│       # 全KF + 全LM（上限2000 LM、observation数降順で選択）
│       # 現在はcorrectLoop()内で呼ばれない（コメントアウト）
│
├── src/loop_closing/
│   └── loop_closing.h / .cc  # [543行] ループ検出・補正（別スレッド）
│       #
│       # === パイプライン ===
│       # insertKeyframe → processNewKeyframe:
│       #   1. detectLoop(): DBoW2で候補検索 (min_score=0.01, interval≥30KF)
│       #   2. computeSim3(): 3D-3D対応 → Sim3 RANSAC (200iter)
│       #      - metric depth: scale 0.85-1.15 に制限
│       #      - mono: scale 0.7-1.4
│       #   3. correctLoop():
│       #      - poseGraphOptimization(60iter)
│       #      - map_->loop_correcting_ = true + sleep 20ms
│       #      - fuseLoopLandmarks()
│       #      - updateConnections() 全KF
│       #      - map_->loop_correcting_ = false
│       #
│       # === mergeLandmarks() ===
│       # deadlock-free double lock (ID順) — target + source 両方のmutexをロック
│       # source observations → target へ移動、kf->landmarks_[]更新
│       # source→setBad() + map_->removeLandmark()
│       #
│       # === metric depth対応 ===
│       # has_metric_depth_: setMetricDepth(true) で有効化
│       # PoseGraph edge: scale_weight = 1000.0 (通常15.0)
│       # Sim3 scale range: 0.85-1.15 (通常0.7-1.4)
│       #
│       # === 既知の問題 ===
│       # - loop検出は非決定的（DBoW2スコアのバラつき）
│       # - fuseLoopLandmarks中のkf->landmarks_[]書き換えはkf->mutex_でロックするが、
│       #   同時にtrackingが同じkfのlandmarks_を読む可能性（loop_correcting_で緩和）
│       # - cooldown: loop_cooldown_kf_ = 120（前回ループ成功から120KF以内は再検出しない）
│
├── src/io/
│   ├── tum_dataset.h / .cc   # [316行] TUM RGB-D データセット読み込み
│   │   # AccelEntry: timestamp_sec, ax, ay, az
│   │   # DepthEntry: timestamp_sec, depth_path
│   │   # rgb.txt / depth.txt / accelerometer.txt パーサー
│   │   # nextWithDepth(): RGB + depth を±30msでアソシエーション（binary search）
│   │   # depth読み込み: CV_16UC1→float/5000.0, CV_32FC1→そのまま
│   │   # K(): TUM freiburg1 固定値 (fx=517.3, fy=516.5, cx=318.6, cy=255.3)
│   │   # allAccel(): 全accelerometerデータを返す
│   │
│   ├── euroc_dataset.h / .cc # [189行] EuRoC MAV: cam0/data/ + data.csv
│   │   # K(): EuRoC固定値
│   │
│   └── map_io.h / .cc        # [269行] バイナリマップ保存/読み込み
│       # ヘッダ "SVSLAM" + camera params + KFs + LMs + covisibility edges
│       # バージョニングなし（既知の制限）
│
├── src/depth/
│   ├── depth_estimator.h      # [15行] 抽象基底: virtual estimate(cv::Mat) → cv::Mat
│   │                          # virtual isMetric() → false
│   ├── onnx_depth_estimator.h # [30行] #ifdef USE_DEPTH_DL でガード
│   └── onnx_depth_estimator.cc # [181行] Depth Anything v2 推論
│       # preprocess: BGR→RGB, resize 518x518, float/255, ImageNet正規化, HWC→CHW
│       # postprocess: disparity→depth(1/d), median-based scaling(1.5m), clamp[0.1,20]
│       # estimate: tensor作成 → Session::Run → postprocess
│       # 入力: NCHW [1,3,518,518] float32
│       # 出力: relative depth（非metric）
│       # OnnxDepthEstimator::isMetric() → false
│
├── src/sensors/
│   └── accelerometer.h        # [80行] header-only
│       # estimateGravity(): 加速度平均→正規化（mag 8-12 sanity check）
│       # isStationary(): 分散 < threshold
│       # computeGravityAlignment(): gravity→[0,0,-1] のRodrigues回転
│
├── scripts/
│   ├── eval_all.sh            # 全データセット×全モード一括評価（evo_ape使用）
│   │   # build dirにcd → 各モード実行 → trajectory.txt → evo_ape → summary出力
│   │   # ★trajectory.txtをrm -fしてから実行（stale data防止）
│   │   # ★timeout 600秒、segfault許容（|| true）
│   ├── eval_ate.sh            # 単一軌跡のATE評価
│   ├── extract_keyframe_trajectory.py   # map.bin → KF軌跡TUM形式
│   ├── generate_map_report.py           # map.bin → 3Dマップビューア HTML (Three.js)
│   └── generate_tum_report.py           # online+corrected軌跡比較HTML
│
├── tests/
│   ├── test_camera.cc         # [59行] project→unproject roundtrip, principal point, K matrix
│   ├── test_landmark.cc       # [80行] addObservation, setPos/getPos, thread safety, isBad
│   ├── test_map.cc            # [96行] add/remove KF/LM, count, concurrent add
│   └── test_optimizer.cc      # [90行] 2KF+5LM BA: noise→BA→誤差減少確認
│
├── data/
│   ├── ORBvoc.txt             # DBoW2語彙（gitignore、手動配置）
│   └── tum/                   # TUMデータセット（gitignore）
│       ├── rgbd_dataset_freiburg1_xyz/
│       └── rgbd_dataset_freiburg1_room/
│
└── models/
    └── depth_anything_v2_small.onnx   # DL depth model（gitignore、HuggingFace DL）
        # DL URL: https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx
```

### 2.2 スレッドモデル

```
Main Thread (run_mono.cc)
  └── while(true): read image → Frame → ORB → tracker->addFrame() → save trajectory
      addFrame内でneedNewKeyframe判定 → KF生成 → local_mapping->insertKeyframe()

LocalMapping Thread (local_mapping->run())
  └── while(!stop): wait(cv) → processNewKeyframe → createNewMapPoints → culling → BA
      BA完了 → on_ba_completed_ → Tracking::onBACompleted()
      KFをloop_closing->insertKeyframe()へ転送

LoopClosing Thread (loop_closing->run())
  └── while(!stop): wait(cv) → detectLoop → computeSim3 → correctLoop
      correctLoop: poseGraphOpt → set loop_correcting_ → fuse → updateConn → clear flag
```

### 2.3 スレッド間共有データと保護

| データ | 書き手 | 読み手 | 保護方法 | 安全性 |
|--------|--------|--------|----------|--------|
| Map::keyframes_ | LocalMapping(add) | Tracking, PoseGraph | poseGraphでstd::mapコピー | ○ |
| Map::landmarks_ | LocalMapping(add) | Tracking, PoseGraph | poseGraphでstd::mapコピー | ○ |
| Landmark::pos_w_ | Optimizer(setPos) | Tracking(getPos) | Landmark::mutex_ (両方ロック) | ○ |
| Landmark::observations_ | LocalMapping(add) | Optimizer(iterate) | Landmark::mutex_ | ○ |
| Keyframe::T_cw_ | PoseGraph(write) | Tracking(read) | なし（SE3代入、実質atomic） | △ |
| Keyframe::landmarks_[] | LoopClosing(fuse) | Tracking(trackLocalMap) | loop_correcting_ flag | ○ |
| Keyframe::connected_kfs_ | LoopClosing(update) | Tracking(getBest) | loop_correcting_ flag | ○ |

### 2.4 定数・閾値一覧

| 場所 | 定数 | 値 | 意味 |
|------|------|-----|------|
| run_mono.cc | ORB features | 2000 | 1フレームの特徴点数 |
| run_mono.cc | dl_frame_skip | 5 | DL depth推論間隔 |
| tracking.cc | min_init_depth_points | 100 | depth初期化の最小3D点数 |
| tracking.cc | max_depth_for_init | 10.0m | depth初期化の有効depth上限 |
| tracking.cc | kf_frames_threshold | 20 | 最低KF挿入間隔 |
| tracking.cc | kf_tracking_ratio | 0.75 | KF挿入のtracking率閾値 |
| tracking.cc | lowe_ratio | 0.75 | ORBマッチのLowe ratio |
| tracking.cc | max_lost_frames_ | 30 | LOST最大許容フレーム |
| tracking.cc | reinit_trigger_ | 20 | 再初期化開始LOSTフレーム |
| tracking.cc | 2nd_pnp_reproj | 5.0px | 2パスPnPの除外閾値 |
| tracking.cc | gravity_window | 0.05s | KF gravity推定のaccel窓 |
| tracking.cc | stationary_threshold | 5.0 | isStationary閾値 |
| optimizer.h | depth_sigma_sensor | 0.02m | sensor depth prior σ |
| optimizer.h | depth_sigma_dl | 0.2m | DL depth prior σ |
| optimizer.h | gravity_weight | 5.0 | gravity prior weight |
| optimizer.h | BA_iterations | 20 | BA最大イテレーション |
| optimizer.cc | position_bound | 100.0m | landmark clamp範囲 |
| optimizer.cc | func_tolerance | 1e-6 | BA収束条件 |
| optimizer.cc | param_tolerance | 1e-8 | BA収束条件 |
| loop_closing | min_loop_interval | 30 KF | ループ検出最低間隔 |
| loop_closing | min_loop_inliers | 30 | Sim3検証最低inlier |
| loop_closing | sim3_scale (metric) | 0.85-1.15 | metric depthのscale範囲 |
| loop_closing | sim3_scale (mono) | 0.7-1.4 | monocularのscale範囲 |
| loop_closing | scale_weight (metric) | 1000.0 | PoseGraph scale weight |
| loop_closing | loop_cooldown | 120 KF | ループ成功後のcooldown |
| onnx_depth | kInputW/H | 518 | DL推論入力サイズ |
| onnx_depth | median_target | 1.5m | relative depthのscaling |

### 2.5 2026-04-06 時点の増分（旧構成表の補足）

旧来の「2.1 ファイル構成」は骨格として有効だが、現時点では以下が重要な増分である。

- `apps/run_mono.cc`
  - `--reference-policy <heuristic|score|pipeline>` を追加済み
  - `--skip-frames N`, `--max-frames N` で bounded replay が可能
  - `--repro-eval` で local mapping を同期実行し、loop closing を止めた再現性重視モードへ入る
- `src/core/reference_keyframe_policy.h`
  - reference-keyframe 採用判断を切り出した最小契約
  - 現在の surviving fields は `tracked_features`, `detected_keypoints`, `candidate_landmarks`, `frames_since_reference`, `lost_frames`, `has_depth`, `has_accel`
- `src/core/heuristic_reference_keyframe_policy.{h,cc}`
  - runtime default。既存挙動を切り出して `core` 側に置いたもの
- `src/experiments/reference_keyframe/`
  - `score_reference_keyframe_policy.{h,cc}`
  - `pipeline_reference_keyframe_policy.{h,cc}`
  - どちらも discardable な experiment 実装であり、まだ core へ昇格していない
- `tools/reference_policy_experiments.cc`
  - curated scenario corpus を同一 input / 同一 interface / 同一指標で比較する小さな実験バイナリ
- `tests/test_reference_keyframe_policy.cc`
  - policy seam 専用のテスト
  - `depth_accel` 系の `has_accel` 分岐もカバー済み
- `scripts/eval_reference_policies.sh`
  - bounded real-trace replay を policy × corpus × repeat で回す主ハーネス
  - `--mode`, `--policy`, `--repeat`, `--corpus`, `--output`, `--no-repro` を持つ
- `scripts/update_reference_policy_docs.py`
  - 実験 CSV を読み、`docs/index.md`, `docs/decisions.md`, `docs/experiments.md`, `docs/interfaces.md` を自動生成する
- `experiments/reference_keyframe/`
  - `scenarios.csv`: curated corpus
  - `real_trace_corpus.tsv`: full bounded replay corpus
  - `room_focus_corpus.tsv`: `room` hotspot 専用 follow-up corpus
- `docs/`
  - `index.md`: GitHub / Pages 向け landing page
  - `decisions.md`: 採用/不採用の現時点判断
  - `experiments.md`: 詳細比較表
  - `interfaces.md`: surviving minimal interface の記録

この増分こそが現在の「実験 → 収束」開発の中心である。古い記述と矛盾する場合、こちらを優先する。

---

## 3. 現在の評価の真実

### 3.1 README / `eval_all.sh` 側の公開スナップショット

README に載せている public-facing な SLAM 精度表は現時点でも以下である。

| Dataset | Mono | +Depth | +Depth+Accel |
|---------|------|--------|--------------|
| Seq A (small motion) | 0.023 | **0.011** | **0.011** |
| Seq B (room-scale) | 0.845 | **0.227** | 0.235 |

- これは `scripts/eval_all.sh` 系の high-level snapshot であり、今後も README 用の要約として使う。
- ただし **reference-keyframe policy の採用判断はこの表だけでは行わない**。
- `room` 系には run-to-run の揺れが残るため、policy 比較は必ず `--repro-eval` と repeat gate で見る。

### 3.2 Reference-Keyframe Policy 実験の現状

現時点の source of truth は `docs/index.md` / `docs/decisions.md` / `docs/experiments.md` である。要約は以下。

#### Curated corpus

- `score` と `pipeline` が accuracy `0.929` で同率首位
- `heuristic` は core baseline として維持
- `heuristic` の curated counters は `fp=2`, `fn=0`
- `score` は conservative、`pipeline` は latency 寄り

#### Bounded real-trace replay (`--repro-eval`, single-run)

- `heuristic = 0.099`
- `score = 0.070`
- `pipeline = 0.107`
- mode winner は `depth=heuristic`, `depth_accel=score`, `mono=score`

#### Full repeat-2 replay (`--repro-eval`)

- `heuristic = 0.078 ± 0.085`
- `score = 0.074 ± 0.064`
- `pipeline = 0.083 ± 0.088`
- mode winner は `depth=score`, `depth_accel=heuristic`, `mono=score`
- つまり `score` は overall best だが、**全 mode を単独支配していない**

#### Room hotspot repeat-2 (`--repro-eval`)

- `heuristic = 0.146 ± 0.088`
- `score = 0.164 ± 0.132`
- `pipeline = 0.137 ± 0.081`
- room-only mode winner は `depth_accel=score`, `mono=pipeline`
- `pipeline` は局所 hotspot では強いが、global default 昇格を正当化するほどではない

#### 結論

- runtime default は **まだ `heuristic`**
- `score` と `pipeline` は **捨てられる実験実装** として `src/experiments/` に残す
- `has_accel` は minimal interface に昇格済み
- 今後の policy 昇格条件は「curated + single-run replay + repeat replay」で勝ち切ること

### 3.3 Mono 安定性について

mono は依然として揺れやすい。特に `room_mono_head` が hotspot である。

- mono repeat-2 mean/std:
  - `heuristic = 0.177 ± 0.184`
  - `score = 0.163 ± 0.139`
  - `pipeline = 0.125 ± 0.080`
- mono repeat-2 with `--repro-eval`:
  - `heuristic = 0.130 ± 0.078`
  - `score = 0.142 ± 0.105`
  - `pipeline = 0.113 ± 0.061`

`--repro-eval` で async scheduling ノイズはかなり減るが、mono では still repeat comparison が前提である。

### 3.4 直近の確認コマンド

少なくとも以下は recent green path とみなしてよい。

```bash
cmake -S . -B build
cmake --build build -j4 --target run_mono reference_policy_experiments svslam_tests
ctest --test-dir build -R 'ReferenceKeyframePolicyTest' --output-on-failure

bash scripts/eval_reference_policies.sh --repeat 1
bash scripts/eval_reference_policies.sh --repeat 2 \
  --output eval_results/reference_keyframe_policy/real_trace_metrics_repeat2.csv
bash scripts/eval_reference_policies.sh --repeat 2 \
  --corpus experiments/reference_keyframe/room_focus_corpus.tsv \
  --output eval_results/reference_keyframe_policy/room_focus_repeat2.csv

./scripts/update_reference_policy_docs.py
```

### 3.5 テストの見方

- `ReferenceKeyframePolicyTest` は通る前提
- `test_camera`, `test_landmark`, `test_map`, `test_optimizer` も通常は green
- full `ctest` では Sophus 側の external test (`test_cartesian2`, `test_so2`) がノイズになり得る
- したがって、policy 変更の最低 gate は `ReferenceKeyframePolicyTest` + replay scripts + docs regen

---

## 4. 既知の問題と技術的負債

### 4.1 [最重要] universal default がまだ決まっていない

今は「比較可能な実験面」はできたが、「単一の勝者」はまだいない。

- `score` は overall repeat gate に強い
- `pipeline` は room mono hotspot に強い
- `heuristic` は depth_accel repeat gate で still 勝つケースがある

したがって、**default migration を焦ってはいけない**。mode-specific dispatch を導入するなら、それ自体を新しい experiment として扱うこと。

### 4.2 [重要] room / mono 系の残留 non-determinism

`--repro-eval` で local mapping / loop closing の非決定性はかなり減ったが、mono の replay variance はまだ残る。

残候補:
1. loop closing を切っても残る tracking 側の離散的分岐
2. relocalization の candidate 選択
3. データ窓の狭さによる順位逆転

次の一手は policy を増やすことではなく、**`room mono` corpus を厚くすること**。

### 4.3 [重要] `Map::getAllKeyframes()` / `getAllLandmarks()` が const 参照を返す

これは旧来から残る設計負債で、長期的には `shared_mutex` 化か snapshot API 化が必要。現状の experiment track では直接 blocker ではないが、広い refactor を始める前にここを整理した方がよい。

### 4.4 [中] `loop_correcting_` はまだ完全には消えていない

`repro-eval` では loop closing 自体を止めて比較しているが、runtime path では `loop_correcting_` に依存した読み書き回避が残る。`loop_correcting_` を根本的に外すなら、snapshot 化または `Map` の read/write discipline を強化する必要がある。

### 4.5 [中] docs 更新は自動生成だが、CI gate ではない

今は人手で以下を回している。

```bash
./scripts/update_reference_policy_docs.py
```

これを忘れると `docs/` が stale になる。GitHub Actions で replay までは重くても、少なくとも docs generation と policy tests の CI は欲しい。

### 4.6 [低] map.bin にバージョニングがない

旧来どおり。将来 map IO を公開 feature として押し出すなら `version` を入れるべき。

### 4.7 [低] camera parameter の一般化不足

TUM / EuRoC 固定で、設定ファイル読み込みはまだない。実験基盤の整備が優先されたため後回し。

---

## 5. ロードマップ

### Phase A: 品質基盤

| Task | Status | 内容 |
|------|--------|------|
| A-1 | ✅完了 | ORB決定論化、BA収束強化、2パスPnP |
| A-2 | 進行中 | `eval_all.sh --repeat N` と README 反映までは完了。追加シーケンス拡充は未完 |
| A-3 | ✅完了 | Google Test導入 |
| A-4 | ✅完了 | 英語 README と public-facing 結果表 |

### Phase F: Experiment-to-Convergence Workflow [現在の主戦場]

| Task | Status | 内容 |
|------|--------|------|
| F-1 | ✅完了 | reference-keyframe decision seam 抽出 |
| F-2 | ✅完了 | `heuristic` / `score` / `pipeline` の3系統を同一 interface で比較 |
| F-3 | ✅完了 | curated corpus, bounded real-trace corpus, room hotspot corpus 整備 |
| F-4 | ✅完了 | `--repro-eval` による async noise 切り分け |
| F-5 | ✅完了 | GitHub-friendly docs (`docs/index.md` 等) を自動生成 |
| F-6 | 進行中 | universal default の判定。まだ勝者不在 |
| F-7 | 未着手 | mode-specific dispatch を experiment として追加するか検討 |
| F-8 | 未着手 | `room mono` corpus 拡張と repeat gate 強化 |

#### F-6: universal default 判定

**現在の判断:**

- runtime default は `heuristic`
- `score` は overall candidate
- `pipeline` は hotspot candidate
- どれも universal migration には未達

**昇格条件:**

1. curated corpus で劣化しない
2. real-trace single-run で勝つ
3. repeat replay でも勝つ
4. room hotspot で catastrophic regression を起こさない

#### F-8: `room mono` corpus 拡張

Claude に最も引き継ぎたい next step はこれである。

候補:
1. `room_mono_mid`, `room_mono_late`, `room_mono_recovery` 相当の replay 窓を追加
2. repeat を `2` から `5` へ上げ、mean/std の順位が維持されるか確認
3. 必要なら `mode=mono` だけの別 decision table を docs に持つ

### Phase B: DL 深度の差別化強化

| Task | Status | 内容 |
|------|--------|------|
| B-1 | 未着手 | Metric DL Depth（Metric3D v2 / UniDepth対応） |
| B-2 | 未着手 | GPU推論（CUDA ExecutionProvider） |
| B-3 | 未着手 | DL Depth品質向上（confidence map、temporal consistency） |

### Phase C: 堅牢性強化

| Task | Status | 内容 |
|------|--------|------|
| C-1 | ✅完了 | mergeLandmarksダブルロック、observations_ロック統一、getPos() mutex |
| C-2 | 未着手 | `Map` の read/write discipline 明確化 |
| C-3 | 未着手 | relocalization 改善 |
| C-4 | 未着手 | KF/LM 間引きとメモリ管理 |

### Phase D: 機能拡張

| Task | Status | 内容 |
|------|--------|------|
| D-1 | 未着手 | Stereo 入力対応 |
| D-2 | 未着手 | IMU tight coupling |
| D-3 | 未着手 | 3D Gaussian Splatting マッピング |
| D-4 | 未着手 | ROS 2 ノード化 |

### Phase E: コミュニティ / 公開整備

| Task | Status | 内容 |
|------|--------|------|
| E-1 | 未着手 | チュートリアル記事 |
| E-2 | 未着手 | GitHub Actions CI |
| E-3 | 未着手 | Contributing guide |
| E-4 | 未着手 | `LICENSE` 作成（BSD-2-Clause） |

---

## 6. 依存関係とライセンス

| 依存 | License | 必須 | FetchContent |
|------|---------|------|--------------|
| OpenCV 4.5+ | Apache-2.0 | ✅ | system |
| Eigen3 | MPL-2.0 | ✅ | system |
| Ceres Solver 2.1+ | BSD-3 | ✅ | system or FC |
| Sophus | MIT | ✅ | FC |
| DBoW2 | BSD (modified) | opt | FC |
| ONNX Runtime 1.17+ | MIT | opt | FC (pre-built) |
| Google Test 1.14 | BSD-3 | opt | FC |

全依存は BSD / MIT / Apache 互換で、GPL 汚染はない。

---

## 7. ビルド・実行・評価

```bash
# 依存 (Ubuntu 22.04)
sudo apt install -y libopencv-dev libeigen3-dev libgoogle-glog-dev libgflags-dev

# 標準ビルド
cmake -S . -B build
cmake --build build -j$(nproc)

# DL depth を有効化
cmake -S . -B build -DUSE_DEPTH_DL=ON
cmake --build build -j$(nproc)

# テスト
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc) --target svslam_tests
ctest --test-dir build --output-on-failure

# 通常実行
./build/run_mono --tum <dir> --no-viz
./build/run_mono --tum <dir> --depth --no-viz
./build/run_mono --tum <dir> --depth --accel --no-viz
./build/run_mono --tum <dir> --depth-model models/depth_anything_v2_small.onnx --no-viz

# experiment seam を切り替えて bounded replay
./build/run_mono --tum <dir> --reference-policy heuristic --skip-frames 0 --max-frames 200 --repro-eval --no-viz
./build/run_mono --tum <dir> --reference-policy score --skip-frames 0 --max-frames 200 --repro-eval --no-viz
./build/run_mono --tum <dir> --reference-policy pipeline --skip-frames 0 --max-frames 200 --repro-eval --no-viz

# high-level SLAM evaluation
bash scripts/eval_all.sh
bash scripts/eval_all.sh --repeat 3

# reference policy evaluation
bash scripts/eval_reference_policies.sh --repeat 1
bash scripts/eval_reference_policies.sh --repeat 2 \
  --output eval_results/reference_keyframe_policy/real_trace_metrics_repeat2.csv
bash scripts/eval_reference_policies.sh --repeat 2 \
  --corpus experiments/reference_keyframe/room_focus_corpus.tsv \
  --output eval_results/reference_keyframe_policy/room_focus_repeat2.csv

# docs refresh
./scripts/update_reference_policy_docs.py
```

補足:

- `scripts/eval_reference_policies.sh` は build 済み `run_mono` と `evo_ape` を前提にする
- policy 比較時は基本 `--repro-eval` を使う
- `--no-repro` は async scheduling を含めた挙動を見たいときだけ使う

---

## 8. コーディング / 実験規約

- C++17, namespace `svslam`
- `T_cw_` = Transform Camera←World
- `SE3` は rigid、`Sim3` は similarity
- ファイル名は `snake_case.*`、クラスは `PascalCase`、メンバは `snake_case_`
- コメントは英語を優先
- broad abstract refactor は禁止。比較可能な seams だけを切り出す
- 新しい policy を足すなら、必ず以下を一緒に更新する:
  1. `src/core/reference_keyframe_policy.h`
  2. `src/tracking/tracking.cc` の input population
  3. `experiments/reference_keyframe/scenarios.csv`
  4. `tools/reference_policy_experiments.cc`
  5. `tests/test_reference_keyframe_policy.cc`
  6. `scripts/update_reference_policy_docs.py` が読める評価 CSV
  7. `docs/*.md` の再生成

---

## 9. 優先順位

```text
比較可能性 > 安定性 > 精度 > 機能数 > 速度 > 見栄え
```

このリポジトリでは今、綺麗な抽象より「同条件で比較できる複数実装」が優先される。

---

## 10. 非目標

- 今すぐ universal な美しい抽象を作ること
- 1つの policy 実装を急いで正解扱いすること
- Web UI / GUI を main feature にすること
- LiDAR / Event Camera / ToF への横展開を先にやること
- end-to-end Deep SLAM へ寄せること

---

## 11. git 履歴（重要マイルストーン）

```text
6435a99 Merge pull request #1 from rsasaki0109/codex/reference-policy-experiments
b8ddbdd add reference policy experiment workflow
73eda32 Final plan.md update for complete Codex handoff
f4fc265 Merge worker results: precision stabilization, thread safety, README, unit tests
5639305 Comprehensive plan update for Codex handoff: architecture, threading, constants, known issues, detailed roadmap
e2a3680 Update plan with comprehensive OSS roadmap and competitive analysis
66d5813 Fix loop closing thread safety and improve eval script
1fb546c Fix loop closing thread safety (atomic flag)
b88dac3 Add evaluation scripts, DL depth frame skip
5c48e38 Add gravity constraint in BA
fffb3e6 Add deep learning depth estimation via ONNX Runtime
6647223 Integrate depth sensor + accelerometer
```

`6435a99` が「experiment surface + public docs + GitHub Pages」まで含む現在の起点である。

---

## 12. Claude への引き継ぎ

### 12.1 最初に読む順番

Claude が作業を始めるなら、読む順番はこれでよい。

1. `plan.md` のこの章
2. `docs/index.md`
3. `docs/decisions.md`
4. `docs/interfaces.md`
5. `scripts/eval_reference_policies.sh`
6. `src/core/reference_keyframe_policy.h`
7. `src/tracking/tracking.cc` の policy input 生成箇所

その後に必要なら `docs/experiments.md` と `tools/reference_policy_experiments.cc` を読む。

### 12.2 Claude に期待する役割

Claude に期待するのは「abstract architect」ではなく、「比較面を壊さずに探索空間を広げる worker」である。

やってよいこと:

- corpus の追加
- repeat gate の強化
- mode-specific dispatch を **新しい実験** として追加
- `has_accel` のように、本当に surviving した field だけ interface に昇格

やってはいけないこと:

- `score` / `pipeline` を早計に core へ移す
- policy から `Frame*`, `Keyframe*`, `Map*` を見せて比較可能性を壊す
- docs regen を飛ばして `docs/` を stale にする
- `room mono` の variance を無視して single-run の勝敗だけで判断する

### 12.3 まずやるべき次タスク

最優先はこれ。

1. `experiments/reference_keyframe/room_focus_corpus.tsv` を拡張し、`room mono` の中盤・後半・回復局面を増やす
2. `bash scripts/eval_reference_policies.sh --repeat 5 --mode mono ...` を回す
3. `score` と `pipeline` の順位が repeat-5 でも維持されるかを見る
4. 必要なら mode-specific dispatch を experiment 実装として追加する
5. 結果を docs regen で GitHub / Pages に反映する

### 12.4 policy seam を触るときの実務ルール

policy 入力を増減させるなら、必ず「なぜその field が surviving したか」を説明できなければならない。

チェックリスト:

- curated corpus で使うか
- real-trace replay でも使うか
- implementation ごとの差ではなく input として比較可能か
- field を消したときに policy 間比較がむしろ明快になるか

説明できない field は足さない。抽象は **後から発見** する。

### 12.5 public repo としての注意

repo は public、Pages も public である。

- GitHub repo: `https://github.com/rsasaki0109/simple_visual_slam`
- Pages: `https://rsasaki0109.github.io/simple_visual_slam/`

したがって、`docs/` の内容は external-facing artifact でもある。内部メモのつもりで壊れた表や stale な数字を push しないこと。

### 12.6 触らないもの

少なくとも引き継ぎ開始時点では、以下は今回の PR scope 外だったため勝手に巻き込まない。

- `slam_result.jpg`
- `.claude/`
- `AGENTS.md`
- `data/`
- `scripts/__pycache__/`

### 12.7 終了条件

Claude の 1 ターンの終了条件は「抽象が美しいこと」ではない。以下のどれかを満たしたら十分である。

- corpus / repeat gate を1段前進させた
- docs / decisions を fresh な結果で更新した
- mode-specific dispatch の可否を比較可能な形で追加した
- universal default を据え置く理由を、より明快な証拠で補強した

---

## 13. Public URLs / 公開物

- Repository: `https://github.com/rsasaki0109/simple_visual_slam`
- GitHub Pages: `https://rsasaki0109.github.io/simple_visual_slam/`
- Landing page: `docs/index.md`
- Decision record: `docs/decisions.md`
- Experiment tables: `docs/experiments.md`
- Minimal interface: `docs/interfaces.md`

`plan.md` は内部 handoff 文書、`docs/` は public digest、`README.md` は入口、という役割分担で考えること。

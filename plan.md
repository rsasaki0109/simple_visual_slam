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

### 1.1 2026-04-06 現在地（2026-04-06 Claude Code session で更新）

- `master` は `3b11008` まで到達。前回 `6435a99` からの差分:
  - `402b196` expand Claude handoff plan
  - `3b11008` expand room corpus and clean up README
- GitHub repository は public 化済み: `https://github.com/rsasaki0109/simple_visual_slam`
- GitHub Pages も公開済み: `https://rsasaki0109.github.io/simple_visual_slam/`
- 以後の開発方針は「正しい抽象を先に固定する」ではなく、「比較可能な複数実装を先に作り、repeat replay で収束させる」。
- この文書は Codex / Claude Code / Cursor 等の AI コーディングエージェントへの handoff を意図して更新している。後半の「引き継ぎ」セクションは作業開始前に必読。
- **今回の Claude Code session で実施したこと:**
  1. `room_focus_corpus.tsv` を 6→10 行に拡張（`tail`, `recovery` 窓を追加）
  2. `real_trace_corpus.tsv` を 10→13 行に拡張（`mid` 窓を追加）
  3. `slam_result.jpg` を README / repo から削除（デバッグ画像で品質不十分）
  4. `update_reference_policy_docs.py` の room focus 説明文を動的化
  5. repeat-5 room focus 評価 + repeat-2 real trace 評価を実行（結果は Section 3 に記載）
  6. CMake ビルドキャッシュを修復（パス変更対応）

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
├── .gitignore                # *.bin, *.jpg, *.onnx, models/, eval_results/, trajectory*.txt, *.html
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

#### 2026-04-06 Claude Code session での変更

- `slam_result.jpg` 削除済み（README からの参照も除去）
- `.gitignore` に `*.jpg` 追加、`!slam_result.jpg` の例外ルール除去
- `experiments/reference_keyframe/room_focus_corpus.tsv`:
  - 追加: `room_mono_tail` (750-250), `room_mono_recovery` (350-300), `room_depth_accel_tail` (750-250), `room_depth_accel_recovery` (350-300)
- `experiments/reference_keyframe/real_trace_corpus.tsv`:
  - 追加: `room_mono_mid` (250-250), `room_depth_mid` (250-250), `room_depth_accel_head` (0-250)
- `scripts/update_reference_policy_docs.py`: room focus 説明文をハードコードから `describe_repro_mode()` による動的生成に変更

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

source of truth は `docs/index.md` / `docs/decisions.md` / `docs/experiments.md` だが、以下に 2026-04-06 時点の最新スナップショットを記載する。

#### Curated corpus（前回から変更なし）

- `score` と `pipeline` が accuracy `0.929` で同率首位
- `heuristic` は core baseline として維持
- `heuristic` の curated counters は `fp=2`, `fn=0`
- `score` は conservative、`pipeline` は latency 寄り

#### Real-trace repeat-2 replay (`--repro-eval`) — 2026-04-06 再評価

corpus を拡張（13 cases）した上で repeat-2 で再評価。

| Policy | Mean APE | Std APE | N |
|--------|----------|---------|---|
| heuristic | 0.098 | 0.080 | 24 |
| score | 0.100 | 0.082 | 23 |
| pipeline | 0.115 | 0.121 | 24 |

mode 別:

| Mode | heuristic | score | pipeline |
|------|-----------|-------|----------|
| depth | 0.053 ± 0.037 | 0.056 ± 0.035 | 0.053 ± 0.040 |
| depth_accel | 0.081 ± 0.030 | 0.080 ± 0.026 | 0.082 ± 0.034 |
| mono | 0.151 ± 0.096 | 0.148 ± 0.102 | 0.191 ± 0.156 |

- `heuristic` が overall best（前回は `score` だった）
- depth / depth_accel は3ポリシーほぼ横並び
- mono で `pipeline` が劣化（0.191）、`heuristic` と `score` はほぼ同等
- corpus 拡張により前回の `score` 優位は消失した — **3ポリシーは実質同等**

#### Room hotspot repeat-5 (`--repro-eval`) — 2026-04-06 新規

corpus を 10 cases に拡張し、repeat-5 で評価。これは今回新規に実施した最も厚い gate。

| Policy | Mean APE | Std APE | N |
|--------|----------|---------|---|
| heuristic | 0.187 | 0.106 | 38 |
| score | 0.198 | 0.099 | 35 |
| pipeline | 0.198 | 0.106 | 34 |

mode 別:

| Mode | heuristic | score | pipeline |
|------|-----------|-------|----------|
| depth_accel | 0.084 ± 0.021 | 0.084 ± 0.015 | 0.089 ± 0.018 |
| mono | 0.247 ± 0.086 | 0.244 ± 0.078 | 0.243 ± 0.094 |

case 別 mono 注目:

| Case | heuristic | score | pipeline |
|------|-----------|-------|----------|
| room_mono_head | 0.358 ± 0.081 | 0.316 ± 0.071 | 0.352 ± 0.120 |
| room_mono_mid | 0.213 ± 0.022 | 0.229 ± 0.031 | 0.225 ± 0.011 |
| room_mono_late | 0.147 ± 0.031 | 0.139 ± 0.014 | 0.147 ± 0.040 |
| room_mono_tail | 0.251 ± 0.046 | 0.256 ± 0.073 | 0.246 ± 0.046 |
| room_mono_recovery | 0.267 ± 0.060 | 0.279 ± 0.053 | 0.242 ± 0.075 |

- overall: `heuristic` がわずかに best だが、誤差範囲で3ポリシーは同等
- mono: 3ポリシーとも 0.243-0.247 でほぼ同一
- `score` は head / late で微小優位、`pipeline` は recovery で微小優位、`heuristic` は mid で微小優位
- **どのポリシーも他を一貫して支配していない**

#### 結論（2026-04-06 更新）

- runtime default は **引き続き `heuristic`** — 変更を正当化するデータがない
- corpus を拡張し repeat gate を厚くした結果、**3ポリシーの差はさらに縮まった**
- `score` / `pipeline` は **捨てられる実験実装** として `src/experiments/` に維持
- `has_accel` は minimal interface に昇格済み
- **次の前進方向は policy 間比較ではなく、SLAM コア品質の改善**（Section 5 参照）

### 3.3 Mono 安定性について（2026-04-06 更新）

mono は依然として揺れやすい。repeat-5 で確認した結果:

- room mono repeat-5 with `--repro-eval`:
  - `heuristic = 0.247 ± 0.086`
  - `score = 0.244 ± 0.078`
  - `pipeline = 0.243 ± 0.094`
- `room_mono_head` が最も誤差が大きい（0.316-0.358）
- `room_mono_recovery` (350-300) は新設窓だが、3ポリシーとも 0.242-0.279 で安定

repeat-5 にしても std は 0.078-0.094 で、mono の run-to-run variance は構造的。
`--repro-eval` で async scheduling ノイズはかなり減るが、mono では repeat comparison が必須。

**重要な発見:** repeat gate を厚くするほど3ポリシーの差が消える。これは policy 差より SLAM コア（tracking / local mapping）の non-determinism が支配的であることを意味する。

### 3.4 直近の確認コマンド（2026-04-06 更新）

少なくとも以下は recent green path とみなしてよい。

```bash
# ビルド（BUILD_TESTS=ON で全ターゲット）
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

# テスト（20/20 pass 確認済み 2026-04-06）
ctest --test-dir build --output-on-failure

# real trace repeat-2（13 cases × 3 policies × 2 = 78 runs, 約20分）
bash scripts/eval_reference_policies.sh --repeat 2 \
  --output eval_results/reference_keyframe_policy/real_trace_metrics_repeat2.csv

# room focus repeat-5（10 cases × 3 policies × 5 = 150 runs, 約50分）
bash scripts/eval_reference_policies.sh --repeat 5 \
  --corpus experiments/reference_keyframe/room_focus_corpus.tsv \
  --output eval_results/reference_keyframe_policy/room_focus_repeat5.csv

# docs 再生成
./scripts/update_reference_policy_docs.py

# ローカル回帰ゲート（TUM xyz+room、mono/depth/depth+accel 計5シナリオ、evo_ape 推奨。全ゲートで ~10 分級）
python3 scripts/check_regression_gate.py --all-gates --quiet
```

注意: CMake キャッシュが別パスで作られていた場合は `rm -rf build` してから再構成する必要がある（2026-04-06 にこの問題に遭遇）。Ceres のスレッド数は未設定時 **1**（再現性優先）。`SVSLAM_CERES_NUM_THREADS` で上書き可（`README.md` 参照）。

### 3.5 テストの見方

- `ReferenceKeyframePolicyTest` は通る前提
- `test_camera`, `test_landmark`, `test_map`, `test_optimizer` も通常は green
- full `ctest` では Sophus 側の external test (`test_cartesian2`, `test_so2`) がノイズになり得る
- したがって、policy 変更の最低 gate は `ReferenceKeyframePolicyTest` + replay scripts + docs regen

---

## 4. 既知の問題と技術的負債

### 4.1 [最重要→結論に近い] universal default は決まらなかった（意図的）

repeat-5 room focus + repeat-2 real trace の結果、**3ポリシーは実質同等**と判明。

- 差が最も大きい `room_mono_head` でも mean 差は 0.04m（0.316 vs 0.358）
- overall では 0.01m 以下の差で、std の範囲内
- corpus を厚くするほど差が消えるのは、policy 差より SLAM コアの non-determinism が支配的であることの証拠

**結論:** policy 間比較はこれ以上深追いしない。`heuristic` を default として確定し、今後の改善は SLAM コア品質（tracking 精度、loop closing 安定性、mono 初期化）に向ける。

`score` / `pipeline` は historical experiment として `src/experiments/` に残すが、active development の対象ではない。

### 4.2 [重要→昇格: 最重要] room / mono 系の残留 non-determinism

`--repro-eval` で local mapping / loop closing の非決定性はかなり減ったが、mono の replay variance はまだ残る。

残候補:
1. loop closing を切っても残る tracking 側の離散的分岐
2. relocalization の candidate 選択
3. データ窓の狭さによる順位逆転

corpus は厚くした（repeat-5 完了）。次の一手は **SLAM コアの non-determinism 源を特定・削減すること**。
候補: ORB マッチングの tie-break、PnP RANSAC の乱数 seed、relocalization 候補選択。

#### 2026-04-06 追記: `--repro-eval` の bitwise determinism を達成（進捗）

本 repo の repeat comparison を成立させるため、`--repro-eval` で **同一入力→同一出力（trajectory bitwise一致）** を達成した。

確認済み事実（ローカル検証）:

- 入力: `data/tum/rgbd_dataset_freiburg1_room`
- 実行: `run_mono --tum <seq> --reference-policy heuristic --skip-frames 0 --max-frames 200 --repro-eval --no-viz`
- 結果: 同一条件で 2 回実行した `trajectory.txt` が **sha256一致**（bitwise identical）

実施した対策（原因→対処）:

1. **OpenCV RANSAC の乱数を固定**
   - `run_mono` 起動時に常に `cv::setRNGSeed(0)` を適用（トラッキングスレッド上の RANSAC の run-to-run 揺れ低減。以前は `--repro-eval` 時のみ）。
   - `solvePnPRansac` / `findFundamentalMat(FM_RANSAC)` / `findHomography(RANSAC)` 等の内部RNG由来の揺れを抑制。
2. **tracking の候補ソート tie-break**
   - `std::sort` を距離のみ→距離+indexで安定化（同距離時の順序未定義を除去）。
3. **Local BA 入力の順序を決定論化（主要因）**
   - `LocalMapping::optimization()` が `std::set<std::shared_ptr<Landmark>>` を使用しており、
     **ポインタ値順（run-to-runで変わる）**により BA の入力順が揺れていた。
   - `Landmark::id_` ソート+uniqueに変更し、観測数での選別も同点を `id_` で tie-break。

注意:

- `--no-repro`（async local mapping / loop closing thread有効）では別要因が残り得る。
- CI が GitHub への到達性（DNS/ネットワーク）に依存するため、現環境では FetchContent による `googletest` / `DBoW2` 取得が失敗する場合がある。
  その場合は `BUILD_TESTS=OFF` / `USE_DBOW2=OFF` でコンパイルだけ先に確認し、疎通がある環境で full gate を回す。

### 4.3 [重要] `Map::getAllKeyframes()` / `getAllLandmarks()` が const 参照を返す

これは旧来から残る設計負債で、長期的には `shared_mutex` 化か snapshot API 化が必要。現状の experiment track では直接 blocker ではないが、広い refactor を始める前にここを整理した方がよい。

### 4.4 [中] `loop_correcting_` はまだ完全には消えていない

`repro-eval` では loop closing 自体を止めて比較しているが、runtime path では `loop_correcting_` に依存した読み書き回避が残る。`loop_correcting_` を根本的に外すなら、snapshot 化または `Map` の read/write discipline を強化する必要がある。

### 4.5 [中] docs 更新は自動生成だが、CI gate ではない

今は人手で以下を回している。

```bash
./scripts/update_reference_policy_docs.py
```

これを忘れると `docs/` が stale になる。`.github/workflows/ci.yml` で **`cmake` ビルド + `ctest`（ユニットテスト）** は CI 化済み。重い replay 評価や上記 docs 自動生成はまだ gate に含めていない。

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

### Phase F: Experiment-to-Convergence Workflow [収束完了]

| Task | Status | 内容 |
|------|--------|------|
| F-1 | ✅完了 | reference-keyframe decision seam 抽出 |
| F-2 | ✅完了 | `heuristic` / `score` / `pipeline` の3系統を同一 interface で比較 |
| F-3 | ✅完了 | curated corpus, bounded real-trace corpus, room hotspot corpus 整備 |
| F-4 | ✅完了 | `--repro-eval` による async noise 切り分け |
| F-5 | ✅完了 | GitHub-friendly docs (`docs/index.md` 等) を自動生成 |
| F-6 | ✅完了 | universal default の判定 → **`heuristic` を確定**（3ポリシー実質同等のため変更不要） |
| F-7 | スキップ | mode-specific dispatch → 不要（3ポリシーの差が消えたため意味がない） |
| F-8 | ✅完了 | `room mono` corpus 拡張（tail/recovery 追加）+ repeat-5 gate 実施 |

#### F-6: universal default 判定（確定）

**結論:** `heuristic` を runtime default として確定。

- repeat-5 room focus: 3ポリシーの overall mean 差は 0.011m（0.187 vs 0.198）
- repeat-2 real trace: 3ポリシーの overall mean 差は 0.017m（0.098 vs 0.115）
- mono mode では 3ポリシーとも std > mean 差であり、統計的に区別不能
- 従って policy migration は無意味。SLAM コア改善にリソースを向けるべき

#### F-8: `room mono` corpus 拡張（完了）

- `room_mono_tail` (750-250), `room_mono_recovery` (350-300) を追加
- `room_depth_accel_tail` (750-250), `room_depth_accel_recovery` (350-300) を追加
- repeat-5 で評価完了。結果は Section 3.2 に記載

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
| E-2 | ✅完了（最小） | GitHub Actions: Ubuntu で Ninja ビルド + `ctest`。`workflow_dispatch` 可。docs/replay は未 |
| E-3 | ✅完了 | `CONTRIBUTING.md`（ビルド・ゲート・コミット方針） |
| E-4 | ✅完了 | リポジトリ直下 `LICENSE`（BSD-2-Clause） |
| E-5 | ✅完了 | セマンティックバージョンは `CMakeLists.txt` の `project(VERSION)`。`run_mono --version`、`CHANGELOG.md`、`RELEASING.md` |

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

## 9. 優先順位（2026-04-06 更新）

```text
安定性 > 精度 > 比較可能性 > 機能数 > 速度 > 見栄え
```

Phase F の policy 比較は収束した。今後は SLAM コアの安定性と精度が最優先。

---

## 10. 非目標

- reference-keyframe policy 実験を再開すること（収束済み）
- Web UI / GUI を main feature にすること
- LiDAR / Event Camera / ToF への横展開を先にやること
- end-to-end Deep SLAM へ寄せること
- 不要な抽象レイヤーを増やすこと

---

## 11. git 履歴（重要マイルストーン）

```text
3b11008 expand room corpus and clean up README          ← 2026-04-06 Claude Code session
402b196 expand Claude handoff plan
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

`3b11008` が corpus 拡張 + README 整理を含む最新コミット。
`6435a99` が「experiment surface + public docs + GitHub Pages」の起点。

---

## 12. AI エージェントへの引き継ぎ（2026-04-06 更新）

### 12.1 最初に読む順番

作業を始めるなら、読む順番はこれでよい。

1. `plan.md` の Section 1.1（現在地）、Section 3（評価の真実）、Section 5（ロードマップ）
2. `src/tracking/tracking.cc` — 最大のファイル、改善の中心
3. `src/backend/optimizer.cc` — BA の実装
4. `src/loop_closing/loop_closing.cc` — ループ検出・補正
5. `docs/index.md` — public-facing な現状サマリ

policy 実験は **Phase F で収束済み**。`src/experiments/` や `tools/reference_policy_experiments.cc` は歴史的参考のみ。

### 12.2 期待する役割

Phase F（policy 実験）は完了した。今後期待するのは **SLAM コア品質の改善** である。

やってよいこと:

- tracking 精度の改善（マッチング品質、PnP の robustness）
- mono 初期化の安定性向上
- non-determinism の源の特定と削減
- loop closing の信頼性向上
- テストの追加（特に tracking / optimizer 周り）
- Phase B（DL 深度）、Phase D（機能拡張）の着手

やってはいけないこと:

- policy 実験を再開すること（収束済み）
- `docs/` を stale にすること（変更したら regen を回す）
- `--repro-eval` の仕組みを壊すこと
- public repo に壊れた表や stale な数字を push すること

### 12.3 まずやるべき次タスク

優先順位順:

1. **docs regen を回す** — 今回の評価結果を反映
   ```bash
   ./scripts/update_reference_policy_docs.py
   ```
   ※ `update_reference_policy_docs.py` が `room_focus_repeat5.csv` を読むよう `ROOM_FOCUS_STABILITY_FILE` の参照先を更新する必要があるかもしれない

2. **mono non-determinism の源を調査**
   - `--repro-eval` でも残る variance (std 0.078-0.094) の原因特定
   - 候補: ORB knnMatch の tie-break、PnP RANSAC seed、cv::solvePnPRansac の内部乱数
   - 調査方法: RANSAC seed を固定して repeat-5 の std が減るか確認

   2026-04-06 追記（進捗）:
   - `--repro-eval` での repeat 実行については **trajectory bitwise一致**まで改善できた。
     次は「`--no-repro` 側の揺れ」と「評価窓を広げたrepeat-5で std が実際に下がるか」を確認する。

3. **SLAM コア改善の candidate**
   - `room_mono_head` の ATE が 0.316-0.358 と大きい → 初期化品質の問題か tracking loss か
   - 2パス PnP の reproj 閾値 (5.0px) のチューニング
   - `needNewKeyframe()` のフレーム閾値 (20) / tracking ratio (0.75) の見直し

4. **Phase B-1: Metric DL Depth** — Depth Anything v2 は relative depth。Metric3D v2 / UniDepth で metric 化すれば depth sensor なしでも高精度化

5. **Phase E-2: GitHub Actions CI** — `ctest` は `.github/workflows/ci.yml` で実行済み。次の拡張候補: docs regen（`eval_results/` を repo に載せるか要設計）

   2026-04-06 追記（障害）:
   - 現環境で `github.com` の名前解決/到達性が不安定なことがあり、FetchContent で `googletest`/`DBoW2` が取れない場合がある。
     CI 導入時は runner 側のネットワーク前提を明確化し、必要なら依存を vendor する/ミラーを用意する。

### 12.4 public repo としての注意

repo は public、Pages も public である。

- GitHub repo: `https://github.com/rsasaki0109/simple_visual_slam`
- Pages: `https://rsasaki0109.github.io/simple_visual_slam/`

`docs/` は external-facing artifact。内部メモのつもりで壊れた表や stale な数字を push しないこと。

### 12.5 触らないもの

- `.claude/`
- `AGENTS.md`
- `data/` — ローカルデータセット。git 管理外
- `scripts/__pycache__/`

### 12.6 終了条件

1 ターンの終了条件は「抽象が美しいこと」ではない。以下のどれかを満たしたら十分。

- SLAM コアの品質を計測可能に改善した（ATE の mean または std が減った）
- non-determinism の源を1つ特定・修正した
- 新しい機能（Phase B/D）を動作する形で追加した
- テストカバレッジを拡大した
- docs を fresh な状態に更新した

---

## 13. Public URLs / 公開物

- Repository: `https://github.com/rsasaki0109/simple_visual_slam`
- GitHub Pages: `https://rsasaki0109.github.io/simple_visual_slam/`
- Landing page: `docs/index.md`
- Decision record: `docs/decisions.md`
- Experiment tables: `docs/experiments.md`
- Minimal interface: `docs/interfaces.md`

`plan.md` は内部 handoff 文書、`docs/` は public digest、`README.md` は入口、という役割分担で考えること。

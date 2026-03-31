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

---

## 3. 精度実績

### 3.1 eval_all.sh 自動評価結果（最新）

| Dataset | Mono | +Depth | +Depth+Accel |
|---------|------|--------|--------------|
| Seq A (small motion) | 0.023 | **0.011** | **0.011** |
| Seq B (room-scale) | 0.845 | **0.227** | 0.235 |

- ATE mean [m], Sim3 alignment, evo_ape --align --correct_scale
- eval_all.shで自動取得。手動実行時のベスト: Seq A 0.011m, Seq B 0.191m
- Seq B は非決定的: 0.19-0.49m幅（ループ検出タイミング依存）

### 3.2 DL Depth単体（sensor depthなし）

| Dataset | DL Depth | Target |
|---------|----------|--------|
| Seq A | 0.051 | <0.05 |

- CPU推論、5フレーム間隔、Depth Anything v2 Small (95MB)

### 3.3 テスト結果

```
ctest --output-on-failure
# test_camera: 3テスト PASSED
# test_landmark: 4テスト PASSED (thread safety含む)
# test_map: 6テスト PASSED (concurrent add含む)
# test_optimizer: 1テスト PASSED (BA noise→reduction)
# ※ Sophusの test_cartesian2, test_so2 が Not Run (実行ファイルパス問題、svslam側の問題ではない)
```

---

## 4. 既知の問題と技術的負債

### 4.1 [最重要] 精度の非決定的バラつき

**症状:** 同じデータでATE 0.19〜0.49m (Seq B)
**原因:**
1. ORBのFAST閾値適応にランダム性 → Worker A-1で緩和済み（ソート決定論化）
2. DBoW2スコアが微妙に変動 → ループ検出のyes/noが変わる
3. マルチスレッドの実行順序
**残作業:** ORB抽出自体のシード固定、ループ検出の決定論化

### 4.2 [重要] trackLocalMapスキップによる精度劣化

**症状:** ループ補正中1-5フレームでtrackLocalMapがスキップ → ポーズ精度低下
**原因:** loop_correcting_フラグ中のスキップ
**対策案:**
- fuseLoopLandmarks前にlandmarkのスナップショットを取り、trackLocalMapにはスナップショットを渡す
- あるいはfuse/updateConnections をアトミックに（所要時間短縮）

### 4.3 [中] Map::getAllKeyframes/getAllLandmarksがconst参照を返す

**症状:** 返り値の参照を保持したまま別スレッドがmapを変更するとUB
**対策:** コピーを返すか、read-write lock導入

### 4.4 [低] マップ保存形式にバージョニングなし

**対策:** "SVSLAM" ヘッダの後にuint32 version追加

### 4.5 [低] TUM以外のカメラパラメータ対応

**現状:** K()がTUM freiburg1 / EuRoC固定。設定ファイル読み込み未実装。

---

## 5. ロードマップ

### Phase A: stella_vslam同等の品質基盤 [進行中]

| Task | Status | 内容 |
|------|--------|------|
| A-1 | ✅完了 | ORB決定論化、BA収束強化、2パスPnP |
| A-2 | 未着手 | 追加データセット検証（5+シーケンス） |
| A-3 | ✅完了 | Google Test導入、4テストファイル |
| A-4 | ✅完了 | 英語README（Mermaid図、結果テーブル） |

#### A-2: 追加データセット検証

**目的:** 5+シーケンスで全モード評価

**タスク:**
1. 追加データセットを入手（大規模環境、高速移動、低テクスチャ）
2. eval_all.sh に `--repeat N` オプション追加（N回実行してmean/std出力）
3. 結果テーブルをREADME.mdに反映

**注意:** データセット名はREADME/plan.mdで伏せる（"Seq A/B/C..."）

### Phase B: DL深度の差別化強化

| Task | Status | 内容 |
|------|--------|------|
| B-1 | 未着手 | Metric DL Depth（Metric3D v2 / UniDepth対応） |
| B-2 | 未着手 | GPU推論（CUDA ExecutionProvider） |
| B-3 | 未着手 | DL Depth品質向上（confidence map、temporal consistency） |

#### B-1: Metric DL Depth対応

**背景:** 現在のDepth Anything v2はrelative depth（スケール不定）。Metric3D v2/UniDepthはmetric scaleを直接出力。

**タスク:**
1. `src/depth/metric_depth_estimator.h/cc` 新規作成
   - Metric3D v2 Small ONNX対応
   - `isMetric()` → true
2. `apps/run_mono.cc`: `--depth-metric` フラグ追加
3. depth_is_metric_ = true で動作 → sensor depth同等の初期化・BA精度

**検証:** DL metric depth + Seq Aで ATE < 0.03m

#### B-2: GPU推論

**タスク:**
1. `onnx_depth_estimator.cc`: CUDAProviderOptions追加
2. CMake: `USE_CUDA` オプション
3. CLI: `--depth-gpu`
4. GPU推論なら毎フレーム推論（5フレームスキップ不要）

### Phase C: 堅牢性強化

| Task | Status | 内容 |
|------|--------|------|
| C-1 | ✅完了 | mergeLandmarksダブルロック、observations_ロック統一、getPos() mutex |
| C-2 | 未着手 | Map read-write lock導入 |
| C-3 | 未着手 | 再ローカライズ改善（DBoW2 place recognition） |
| C-4 | 未着手 | メモリ管理（KF/LM間引き） |

#### C-2: Map read-write lock

**タスク:**
1. `Map::mutex_` → `std::shared_mutex rw_mutex_`
2. addKeyframe/addLandmark: `std::unique_lock` (write)
3. getAllKeyframes/getAllLandmarks: `std::shared_lock` (read)
4. loop_correcting_ フラグを廃止可能に

### Phase D: 拡張機能

| Task | Status | 内容 |
|------|--------|------|
| D-1 | 未着手 | Stereo入力対応 |
| D-2 | 未着手 | IMU tight coupling（pre-integration + BA IMU residual） |
| D-3 | 未着手 | 3D Gaussian Splatting マッピング（実験的） |
| D-4 | 未着手 | ROS 2ノード化 |

### Phase E: コミュニティ・ドキュメント

| Task | Status | 内容 |
|------|--------|------|
| E-1 | 未着手 | チュートリアル記事「5000行で作るVisual SLAM」 |
| E-2 | 未着手 | GitHub Actions CI |
| E-3 | 未着手 | Contributing guide、コーディング規約 |
| E-4 | 未着手 | LICENSE ファイル作成（BSD-2-Clause） |

#### E-2: GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get install -y libopencv-dev libeigen3-dev libgoogle-glog-dev libgflags-dev
      - run: mkdir build && cd build && cmake .. -DBUILD_TESTS=ON && make -j$(nproc)
      - run: cd build && ctest --output-on-failure
```

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

**全依存がBSD/MIT/Apache互換。GPL汚染なし。**

---

## 7. ビルドと実行

```bash
# 依存 (Ubuntu 22.04)
sudo apt install -y libopencv-dev libeigen3-dev libgoogle-glog-dev libgflags-dev

# ビルド
mkdir build && cd build
cmake .. && make -j$(nproc)

# DL Depth付き
cmake .. -DUSE_DEPTH_DL=ON && make -j$(nproc)

# テスト
cmake .. -DBUILD_TESTS=ON && make -j$(nproc) && ctest --output-on-failure

# 実行
./run_mono --tum <dir> --no-viz                    # mono
./run_mono --tum <dir> --depth --no-viz            # + depth sensor
./run_mono --tum <dir> --depth --accel --no-viz    # + accelerometer
./run_mono --tum <dir> --depth-model ../models/depth_anything_v2_small.onnx --no-viz  # DL depth

# DLモデル入手
wget -O models/depth_anything_v2_small.onnx \
  https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx

# 一括評価
bash scripts/eval_all.sh
```

---

## 8. コーディング規約

- C++17, namespace `svslam`
- `T_cw_` = Transform Camera←World（ORB-SLAM慣例）
- `SE3` for rigid, `Sim3` for similarity (Sophus)
- `Vec3 = Eigen::Vector3d`, `Mat33 = Eigen::Matrix3d`
- `using Ptr = std::shared_ptr<ClassName>`
- ファイル: `snake_case.h/.cc`, クラス: `PascalCase`, メンバ: `snake_case_`
- `#pragma once`, コメントは英語
- テスト: Google Test, `tests/test_*.cc`

---

## 9. 優先順位

```
安定性 > 精度 > 機能数 > 速度 > 見栄え
```

---

## 10. 非目標

- リアルタイムAR/VR低レイテンシ最適化
- LiDAR / Event Camera / ToF
- 大規模環境（km規模）
- end-to-end Deep SLAM（DROID-SLAM的）
- Web UI / GUI

---

## 11. git履歴（コミット順 = チュートリアル順）

```
f8b0e1e Initial implementation: Core classes, Tracking, Initialization, LocalMapping
2515eea Fix Sophus, Initialization, add PnP to Reference Tracking
b1a2e42 Improve tracking robustness
eada60d Map Persistence (Save/Load)
7ba00e3 Fix init triangulation; add TUM/EuRoC loaders
e91de1f Improve tracking stability: BA pose sync, matching quality, relocalization
51c5ee4 Tighten tracking matching thresholds
6647223 Integrate depth sensor + accelerometer
fffb3e6 Add DL depth estimation (ONNX Runtime, Depth Anything v2)
5c48e38 Add gravity constraint in BA (Phase 2.4)
b88dac3 Add evaluation scripts, DL depth frame skip
1fb546c Fix loop closing thread safety (atomic flag)
66d5813 Fix poseGraph snapshot (root cause of segfault)
e2a3680 OSS roadmap
5639305 Comprehensive plan for Codex handoff
f4fc265 Merge: precision stabilization, thread safety, README, unit tests
```

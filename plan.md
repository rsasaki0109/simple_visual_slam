# SimpleVisualSLAM 開発計画

> **この文書はAIコーディングエージェント（Claude / Codex / Cursor 等）への完全な引き継ぎ資料である。**
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

### 1.1 リポジトリの現状（2026-04-13 時点のサマリ）

- **Public OSS:** `https://github.com/rsasaki0109/simple_visual_slam` / Pages `https://rsasaki0109.github.io/simple_visual_slam/`
- **版:** CMake `project(VERSION 0.1.0)`、`./build/run_mono --version` と一致。**引用:** ルート `CITATION.cff`。
- **master HEAD:** `c5bcbe1`（Improve mono initialization, stabilize 600-frame loops, update README.）
- **2026-04-13 の作業:**
  - `6429d9b`: リファクタリング + テスト充実 (26→48)
  - `244bb56`: ループ安定化 + stella比較 + plan更新 (48→51)
  - `6d81697`: Metric Depth estimator + tracking改善 + ループ有効比較 (55 with DL ON)
  - `c5bcbe1`: Mono初期化改善 + 600-frame安定化 + README更新
- **テスト:** `ctest` **51/51 pass**（`USE_DEPTH_DL=ON` では **55/55**）。
- **回帰ゲート:** `python3 scripts/check_regression_gate.py --all-gates --quiet` **5/5 pass**。ベースライン締め付け済み。
- **リファクタ:** `tracking.cc` の helper 抽出、状態の `RecoveryState` / `LoopCorrectionState` / `ReinitializationState` への集約、`loop_closing` の internal helper 化、`optimizer` の cleanup を実施。
- **新規テスト:** `test_frame.cc`, `test_keyframe.cc`, `test_initializer.cc`, `test_loop_closing.cc`, `test_tracking.cc`, `test_tracking_pose_recompute.cc`, `test_synthetic_scene.h`, `test_metric_depth_estimator.cc` を追加。
- **Mono 初期化改善:** median parallax ベースの solution 選択により xyz_mono ATE **0.048→0.036** (-26%)。
- **600-frame 安定化:** loop cooldown 120→200 KF。ATE **0.109m / 0.124m** (旧 median 0.617m)。
- **Metric Depth:** `MetricDepthEstimator` 追加。`--metric-depth-model <path.onnx>` CLI 対応。
- **stella_vslam 比較:** 4シナリオ×repro-eval + loop-enabled の両方を実施。eval/stella_comparison_results.md に記録。
- **README 更新:** 比較表・テスト数・CLI を反映。
- **方針:** Reference-keyframe policy 実験は収束済み。**room 系の ATE 6-10x 差の根本対策**（Ceres スパースソルバー化、BA 窓拡大）が次の主戦場。
- **未解決の本丸:** room_depth は 250-frame で stella_vslam の 6.1x、room_mono は 9.8x。コア BA の構造的品質差（Ceres 密 vs g2o スパース）が主因と推定。

### 1.8 Next Phase: Closing the Room Gap（2026-04-13 設計）

**問題:** room 系で stella_vslam に 6-10x 負けている。ループ有効でも gap は縮まらない。

**根本原因の分析:**
1. **Ceres 密ソルバー vs g2o スパース:** poseGraphOptimization で `DENSE_QR` を使用。KF 数が増えると O(n³) でスケールしない。g2o は Schur complement + スパース構造を活用し O(n) 級
2. **Local BA 窓が狭い:** covisible KF 15 個。stella_vslam は 20+ を使う傾向
3. **Covisibility edge が均一重み:** 共有 landmark 数に応じた重み付けがない

**提案（優先順）:**

| # | 提案 | 期待効果 | リスク | 工数 |
|---|------|---------|--------|------|
| 1 | Ceres を `SPARSE_NORMAL_CHOLESKY` + SuiteSparse に切替 | pose graph の品質・速度改善。room 系で最大の改善見込み | SuiteSparse は既に依存。切替自体は低リスク | 小 |
| 2 | Local BA の covisible KF 窓を 15→25 に拡大 | BA 精度向上。xyz/room 両方に効く | BA 時間増加。ただし Ceres は iteration 上限で制約される | 小 |
| 3 | Pose graph edge に covisibility weight を導入 | 強い covisibility edge を優先。drift 低減 | edge 重み設計の試行錯誤が必要 | 中 |
| 4 | g2o への移行 | stella_vslam と同等の pose graph 品質 | 依存追加 (GPL の g2o vs BSD の Ceres)。ライセンス注意 | 大 |

**順序:** 1 → 2 → 3 の順。4 は最後の手段（GPL 汚染リスク）。

### 1.2 未コミット変更（2026-04-09 — ループ補正安定化の途中）

**9ファイル、約 +691/-178行。** ビルド・テスト通過済み（24/24 pass、`build_bench`）。2026-04-08 深夜から 2026-04-09 にかけて、`run_mono.cc` の callback 配線、`fuseLoopLandmarks()` の KF mutex 保護、`mergeLandmarks()` の生存側変更に加え、loop candidate を usable 3D correspondence 数で再選別する処理、`computeSim3()` 失敗理由の可視化、metric-depth 時の rigid (`scale=1`) 仮説生成、pose graph の metric-depth scale 固定、loop edge の confidence weighting、metric-depth final inlier 下限の引き上げ、pose-graph 入力の keyframe snapshot 化、さらに loop-correction handoff を tracking thread の safe point に移す処理まで含む。加えて、複数 loop constraint が残る metric-depth run に備えて、**古い loop edge を current map との整合性（translation/scale error）で再利用減衰する safety valve** を入れた。

| ファイル | 変更内容 |
|---------|---------|
| `apps/run_mono.cc` | `loop_closing->on_loop_corrected_` を `tracker->onLoopCorrected()` に接続。LoopClosing thread は callback 登録後に起動し、`on_ba_completed_` / `on_loop_corrected_` は `weak_ptr<Tracking>` capture にして循環参照を回避 |
| `src/backend/local_mapping.cc` | `processPendingWork()` 内で `map_->loop_correcting_` チェック追加。ループ補正中は `optimization()`（BA）をスキップ。**目的:** ループ補正中に Local BA が同時にポーズ/ランドマークを書き換えるデータレースを防止 |
| `src/backend/optimizer.h` | `poseGraphOptimization()` に `fix_scale` 引数を追加 |
| `src/backend/optimizer.cc` | metric-depth pose graph 用に各 KF の log-scale へ prior を追加し、covisibility edge の scale residual も強化。さらに solver 入力前に `T_cw_` と `connected_keyframes_` を mutex 下で snapshot して、pose graph に入る covisibility edge を同一時点の状態へ揃える。`PoseGraph: scale_stats min/max/mean` をログ出力して scale drift を可視化 |
| `src/loop_closing/loop_closing.h` | `on_loop_corrected_` コールバック（`std::function<void()>`）を public メンバとして追加 |
| `src/loop_closing/loop_closing.cc` | `correctLoop()` 末尾で `on_loop_corrected_()` を呼び出し。`computeSim3()` / `fuseLoopLandmarks()` で KF の landmark slot を mutex 下で snapshot / 更新し、landmark merge は current 側を生存側に変更。さらに loop closing 専用 descriptor match を Lowe ratio **0.8** に緩和し、BoW / fallback candidate は descriptor match 数に加えて usable 3D correspondence 数（`corr3d`）でも再選別。`computeSim3()` は `insufficient_ransac_inliers` などの failure reason に加え、relaxed residual (`0.35`) 時の inlier 数をログ出力。metric-depth では refinement seed を **15本** に緩和した上で rigid (`scale=1`) 仮説生成を使い、最終判定は **22 inlier + ratio guard** で実施。loop constraint は inlier 数と inlier ratio に応じた confidence weight を持つように変更。さらに、複数 loop constraint を pose graph に再投入する際は、古い edge を current map との translation/scale 整合で減衰する safety valve を追加 |
| `src/tracking/tracking.h` | `onLoopCorrected()` に加えて pending loop-correction state と safe-point helper (`applyPendingLoopCorrection()`) を追加 |
| `src/tracking/tracking.cc` | loop thread から `current_frame_` を直接触らず、`onLoopCorrected()` は pending flag だけ立てる形へ変更。tracking thread 側で `trackReferenceKeyframe()` 後・`trackLocalMap()` 後・`relocalize()` 成功直後の safe point に `applyPendingLoopCorrection()` を挿し、PnP が失敗した場合は pending を保持して同フレーム / 次フレームで再試行。pending loop correction の pose 再計算は **reprojection error が実際に改善した場合のみ採用**し、`err_after < 20` の absolute fallback は使わない。成功時にだけ velocity をリセット |

**検証状況:** `build_bench` で build + `ctest` 24/24 pass。初期の **0.0647〜0.0822 m** は `build_bench/trajectory.txt` を誤って評価した時期の値で、現在の authoritative な評価対象ではない。repo root で fresh `trajectory.txt` を作る rerun（`rm -f trajectory.txt trajectory_online.txt` 後に `build_bench/run_mono ...`）では、callback 配線と rigid metric-depth Sim3 までを含む retained baseline の ATE は **0.20401715 m** だった。loop correction 後に保存済み `trajectory.txt` の該当 frame を後追い更新し、worker thread join 後に軌跡を書き出す experiment は **0.20486272 m**、各 frame に `reference KF id + T_cr` を持たせて final keyframe pose から `trajectory.txt` を再構成する experiment は **0.28865224 m**、同 run の raw `trajectory_online.txt` でも **0.16064598 m** に留まったため、どちらも**不採用**。つまり repo root rerun では「保存済み軌跡の更新」や「final keyframe pose からの後段再構成」は gap の本命ではない。ここから pose graph 側へ移り、metric-depth pose graph に全 KF scale prior を追加して `PoseGraph: scale_stats min=0.99999.. max=1.00002` まで scale を固定したところ、ATE は **0.18181217** と **0.20063109** で、scale drift 自体は潰れたが改善は不安定だった。さらに loop constraint の translation/rotation weight を `inliers + inlier_ratio` ベースの confidence weight（例: `22 inliers / 0.392857 -> weight 4.94234`, `36 / 0.6 -> 6.79473`）にした rerun では、fresh `trajectory.txt` の ATE が **0.12247207** と **0.16569322** まで改善した。そこから metric-depth の final support を **22 inliers** に引き上げたところ、weak 20-inlier loop は reject されるようになり、fresh `trajectory.txt` rerun は **0.15802252** と **0.09035673** だった。さらに pose graph に入る `T_cw_` と covisibility edge を mutex 下で snapshot するようにした rerun では、fresh `trajectory.txt` の ATE が **0.13723747** と **0.07901260** まで改善した。ここに加えて、loop thread から `current_frame_` を直接触らず `onLoopCorrected()` は pending flag だけ立て、tracking thread 側の safe point で pose 再計算するよう変更した。`trackReferenceKeyframe()` 後だけで吸収する variant は **0.16352909** と **0.12553793** で、`pairs=0` callback miss は消えたものの初回試行が薄い frame に当たる run が残った。そこから pending を成功時まで保持し、`trackLocalMap()` 後にも同フレームで再試行する variant では、loop correction は `after local map` で `pairs=1041` / `1050` を使って吸収され、fresh `trajectory.txt` rerun は **0.12315088** と **0.12105041** だった。さらに `relocalize()` 成功直後も safe point 化した variant は、`after relocalization` で `pairs=1019` / `462` を使えた一方、reprojection が **悪化しても** pending recompute を採用してしまい、400-frame rerun は **0.18307474** と **0.22821251** で**不採用**。そこから pending loop correction に限って **`err_after < err_before` のときだけ採用**する strict accept に切り替えたところ、early handoff は `pairs=668/404/601/294` で reject され、`pairs=946` で初めて `1.53608 -> 1.53387` の改善が出た run は **0.07522716**、別 rerun は `pairs=1046` を reject して `pairs=589` で `1.56168 -> 1.55973` の改善が出て **0.08267831** だった。現時点ではこれが retained best state で、snapshot-only 時代の **0.07901260** に並ぶ帯まで戻せている。safe-point handoff を保持したまま `151->22` を通すために、metric-depth borderline accept と cooldown `140 KF` を組み合わせる experiment も試したが、1本目 **0.09094504** の後に rerun で `328->202` の second-loop が入り **0.41877315** まで崩れたため**不採用**。また、relaxed residual の inlier で Sim3 を再推定し `151->22` を strict `22 inliers` へ rescue する experiment も試したが、`280->121` の second-loop を防げず **0.23805287** に悪化したため**不採用**。metric-depth の loop cooldown を **180 KF** に伸ばして late second-loop を抑える experiment は、1本目こそ **0.07250512** だったものの rerun が **0.47003467** まで崩れたため**不採用**。constraint の prune/update を入れる experiment は 400-frame rerun が **0.37298953** と **0.36328082** まで崩れたため**不採用**。sleep ベース同期を coordinator に置き換える experiment も、tracking 全体を待つ版が **0.07198198** の後に **0.24692610** へ悪化、`trackLocalMap()` 限定版も **0.40753267** まで崩れたため**不採用**。local mapping 完了後に loop-closing queue へ handoff する experiment も **0.37072499** まで悪化したため**不採用**。`loop_correcting_` を `correctLoop()` 冒頭で早めに立て、local BA の開始直前に再チェックする experiment も **0.37356453** まで崩れたため**不採用**。従来の BoW false positive は descriptor match gating でかなり減り、実ループ候補では `desc_matches=50〜74` / `corr3d=32〜69` まで到達することを確認した。`residual_thresh=0.25` を **0.35** に緩めても `relaxed_inliers=7〜23` に留まる事実は変わらず、依然として本丸は correspondence quality / Sim3 仮説品質である。なお、2D 幾何（Fundamental RANSAC）で candidate を再選別する実験は決定打にならず、ATE が **0.08956200 m** まで悪化したため**不採用**。また、covisibility edge の相対測定値を保存して pose graph に使う構造変更は、1本目こそ **0.07266019 m** だったものの rerun で **0.18586552 m** まで悪化したため**不採用**。metric-depth の loop edge を support に応じて単純に弱める experiment（`trans/rot=4.60666`, `scale=589.286`）も **0.23616777 m** まで崩れたため**不採用**。さらに、reference keyframe の pose delta を tracking frame に直接適用する experiment は `trans=0.330026 / rot=0.658964 rad` の correction がそのまま世界ジャンプになり、frame 257 以降の pose が急崩壊したため**不採用**。loop 中の keyframe insertion defer を組み合わせても改善しなかった。loop correction 後の 2フレームだけ tracking gate / jump threshold を緩める recovery experiment は、1回目の loop 直後に `3D-2D correspondences=128 -> 419` と回復したものの、正しい出力先である repo root の `trajectory.txt` で評価すると ATE は **0.34410088 m** で**不採用**。同 experiment に加えて metric-depth で 2本目以降の loop correction を落とす one-shot experiment も、`trajectory.txt` 評価では改善しなかったため**不採用**。なお `build_bench/trajectory.txt` は古い run の残骸で、正しい評価対象ではない。

#### 1.2.1 2026-04-09 追記: 400-frame / 600-frame の現在地

2026-04-09 の追加検証では、**retained state 自体は変えず**に「400-frame では良いのに 600-frame で late-run が崩れる」理由を切り分けた。ここで重要なのは、今回の turn では新しい keep change は増えていないこと、および `plan.md` には retained / not-retained の境界を明示して残すこと、の2点である。

**まず retained state の再確認:**

- `build_bench` で再build、`ctest --test-dir build_bench --output-on-failure` は継続して **24/24 pass**
- retained codeのまま fresh `trajectory.txt` を作る **400-frame** rerun では **0.07351221**
- この run では `151->22` loop correction 後、pending loop correction は `pairs=4` から始まり、safe point で複数回 defer / failed recompute を挟んだ後に expire した
- その直後、`TrackLocalMap()` の local-map PnP が `support=184 / prior_support=409` で `trans=0.174787`, `rot=0.301117` の危険な update を出しており、実験 guard を入れた版ではここを reject できた。この reject 自体は**症状の可視化**として有益だった
- 400-frame の最終 ATE が `0.07351221` まで戻ったことから、少なくとも short run では retained state はまだ competitive で、`safe-point handoff + pending strict accept` の方向性は正しい

**一方で 600-frame retained baseline の実態:**

- 以前の retained baseline rerun は **0.17683183**
- この run では `151->22` に続いて `280->121` の second-loop も通り、`Reweighting stale loop edge ... decay=0.35` まで踏めた
- しかし second-loop 後の pending loop correction は `pairs=0` から始まり、`after reference tracking` / `after local map` で 6 回失敗して expire
- その後は late tail で `Lost -> Relocalization successful!` が繰り返され、local-map overwrite、relocalization、BA callback recompute の複合で姿勢が揺れる
- つまり 600-frame の現状課題は、「loop が通らない」ことではなく、**loop correction 後の long-tail tracking quality をどう安定化するか** に移っている

#### 1.2.2 2026-04-09 追記: `TrackLocalMap()` thin-support overwrite guard 実験の詳細

この turn では、late-run の主犯候補として `TrackLocalMap()` の local-map PnP が、すでに十分な support を持つ prior pose を弱い support で上書きしている点を疑い、**3段階の guard 実験**を行った。結論から言うと、**short run では効くが long run 全体では安定改善にならない**ため不採用である。

**実験の発想:**

- `trackReferenceKeyframe()` や pending loop correction で一度まともな pose が得られた後でも、同フレーム後段の `trackLocalMap()` が support の薄い PnP を受理して pose を飛ばすことがある
- とくに late run では、`support=119` や `support=110` 程度の local-map PnP が `0.08m` 級 update を出し、その次フレームで `TrackReferenceKeyframe: 3D-2D correspondences: 0` へ崩れるケースが見えた
- そのため、`prior pose` に対する update 量を `support` / `prior_support` / `used_global_fallback` に応じて制限する案を試した

**Variant A: 基本 guard**

- `prior pose` と `new_pose` の差分（translation / rotation）を診断ログ出力
- `prior_support >= 120` か `used_global_fallback` のときだけ、low-support な local-map update に stricter threshold をかける
- 400-frame rerun は **0.07351221**
- この run では `support=184 / prior_support=409` の危険 update を reject し、そのまま `Lost -> Relocalization successful!` へ落としても最終 ATE は悪化しなかった
- ただし 600-frame rerun は **0.19205861**。late tail では別の箇所の relocalization 連鎖へ failure が移っただけで、baseline 超えには届かなかった

**Variant B: fallback を常時 anchored とみなす版**

- `used_global_fallback` の update は、`prior_support` が 120 未満でも guard 対象とするよう変更
- 目的は `support=32 / prior_support=96 / fallback=1` のような late-tail false jump を止めること
- 実際に fallback 由来の大ジャンプは止められたが、600-frame rerun は **0.18324189**
- baseline `0.17683183` よりは近づいたものの、まだ**改善とは言えない**
- さらに late tail では `support=83 / prior_support=164` や `support=78 / prior_support=162` の non-fallback update が別途 reject され、そのたびに relocalization へ落ちる構図が残った

**Variant C: `support < 120` を 6cm / 0.06rad まで締めた版**

- `support=119` / `110` あたりの borderline local-map update を止めるため、non-fallback でも `support < 120` なら `max_update_trans=0.06`, `max_update_rot=0.06`
- 実際に `support=116` の `trans=0.0877587`, `rot=0.0668969` や `support=53` の `trans=0.118261`, `rot=0.0627472` は reject できた
- しかし 600-frame rerun は **0.19566213**
- これは guard が**効かなかった**のではなく、guard が効いた結果として早めに relocalization へ落ち、別の誤差経路で total ATE がむしろ悪化した形

**この実験から得た知見:**

1. `TrackLocalMap()` が weak-support pose で prior pose を壊す事象は実在する。ログで捕まえられた
2. ただしその局所的な overwrite を止めても、600-frame 全体では relocalization 連鎖や別経路の drift が残る
3. つまり主問題は `TrackLocalMap()` 1箇所の gate だけではなく、**post-loop / post-relocalization の数フレーム全体の handoff 設計**
4. このため thin-support overwrite guard 自体は、観測のための診断としては有益だが、現時点では keep しない

#### 1.2.3 2026-04-09 追記: 現時点の読み筋

今回の長めの rerun で、次に触るべきポイントはかなり絞れた。

- **候補 1: first-loop / second-loop 後の直近フレームで `TrackReferenceKeyframe()` が 0〜1 correspondence に落ちる件**
  - 例: 600-frame retained baseline の first-loop 直後では `Frame 258` で `TrackReferenceKeyframe: 3D-2D correspondences: 0`
  - 例: late-run でも `Frame 500` 直後に `TrackReferenceKeyframe: Propagated landmarks: 1`
  - ここは `TrackLocalMap()` overwrite より上流で、reference propagation / current_frame landmark handoff の質を見直すべき可能性が高い
  - **2026-04-09 実施:** first-loop 後の collapse を `room_depth` 400-frame rerun で再確認すると、loop correction handoff 成功フレームの次フレームで `Matches with last frame: 1191` に対し `Propagated landmarks: 12` / `3D-2D correspondences: 12` まで落ちていた。原因は `applyPendingLoopCorrection()` 成功時に `velocity_ = SE3()` へ戻しても、同フレーム末尾の `track()` 成功パスが **`velocity_ = current_frame_->getPose() * last_frame_->getPose().inverse()` で上書き**しており、loop-corrected current pose と pre-correction last pose の組で壊れた velocity を次フレームへ持ち越していた点
  - **fix:** `skip_velocity_update_once_` を追加し、pending loop correction を採用したフレームでは end-of-frame の通常 velocity 更新を 1 回だけスキップして identity を次フレームへ持ち越す
  - **結果:** 同じ 400-frame rerun で `Tracking: Preserving identity velocity after loop-correction handoff` を確認。loop correction 直後の次フレームは `Propagated landmarks: 540` / `3D-2D correspondences: 540`、さらに次フレームは `708` まで回復し、旧 run の `12` collapse は再現しなかった。一方、この 1-run の fresh `trajectory.txt` ATE は **0.20676216** で、post-loop propagation collapse の局所症状は潰せたが short-run 全体の数値改善はまだ未確定

- **候補 2: 通常 BA callback (`onBACompleted()`) 側の `recomputeCurrentPose()` はまだ absolute fallback を持っている件**
  - pending loop correction path は strict accept (`err_after < err_before`) だが、通常 BA callback は `err_after < 20.0` fallback により、reprojection が悪化しても pose を採用しうる
  - late-run ログでは `before=1.8447 after=2.15016` や `before=1.90384 after=1.9079` のようなケースでも pose update が通っている
  - これが long-tail の小さな不安定性を積み上げている可能性があり、次の有力候補
  - **2026-04-09 実施:** `recomputeCurrentPose()` の absolute fallback を削除し、BA callback も pending loop correction と同じ **strict accept (`err_after < err_before`)** に統一。`Tracking::shouldAcceptRecomputedPose()` を追加して unit test 化し、`BUILD=build_bench bash scripts/verify_comparison_benchmark.sh room_mono` では `before=2.24825 after=6.35709` の BA pose update が `"New pose rejected (no reprojection improvement)"` として落ちることを確認した

- **候補 3: second-loop 後の pending correction expire 後に何を reset すべきか**
  - 現状は expire 時に `velocity_` を reset するだけ
  - ただし second-loop 後の run では、それだけでは subsequent frame の reference tracking collapse を防げていない
  - `reference_keyframe_` 選び直し、`previous_reference_keyframe_` の扱い、または `current_frame_->landmarks_` の clean handoff まで見る価値がある
  - **2026-04-09 実施:** expire path 用に `force_keyframe_insertion_once_` / `force_reference_refresh_once_` を追加し、pending loop correction が最大 deferral に達したら **次フレームで keyframe insertion を強制し、その KF を reference に昇格、さらに `previous_reference_keyframe_` を捨てる** 形へ変更。狙いは、expire 後も stale reference を引きずらず現在位置の局所 map へ張り直すこと
  - **検証メモ:** 変更前の 600-frame rerun では `Frame 292` で `Pending loop correction expired after failed recomputes` が出ていたが、変更後 rerun では同じ 600-frame でも今回その expire 自体が再現せず、forced refresh marker は未発火。fresh `trajectory.txt` ATE は **0.22985886**。つまりコード上の fallback reset は入ったが、この path の効果はまだ run-to-run variability に埋もれており、引き続き second-loop / late-tail の複数 rerun が必要

### 1.3 stella_vslam との実測比較結果（2026-04-08 実施）

**条件:** TUM freiburg1（全フレーム）、`evo_ape tum ... --align --correct_scale --t_max_diff 0.05`

| Scenario | SimpleVisualSLAM (ループなし) | stella_vslam | 差 |
|----------|---------------------------|-------------|-----|
| **xyz_depth** | 0.011 m | **0.008 m** | 1.4x |
| **xyz_mono** | 0.025 m | **0.011 m** | 2.3x |
| **room_depth** | 0.161 m | **0.033 m** | **4.9x** |
| **room_mono** | 0.817 m | **0.026 m** | **31x** |

**注意:**
- stella_vslam はループクロージング有効（FBoW + g2o ポーズグラフ最適化）
- SimpleVisualSLAM はループクロージングを有効にすると room 系で ATE が**悪化**する（0.161→0.196〜0.341）
- stella_vslam の room_mono は 549/1362 フレームしか追跡成功（195→994 が LOST）。ATE は追跡成功区間のみ
- stella_vslam のビルド一式は `/media/sasaki/aiueo/ai_coding_ws/` 以下に残っている（g2o, stella_vslam, stella_vslam_examples, stella_eval）
- ORB 語彙: `data/ORBvoc.txt`（139MB、gitignore 対象）をダウンロード済み

### 1.4 コア改善の実測結果（2026-04-08 コミット `d3c81a7`）

**repro-eval（ループなし、決定論的）での比較:**

| Scenario | Before (`fcbde0a`) | After (`d3c81a7`) | 変化 |
|----------|-------------------|-------------------|------|
| xyz_depth | 0.0117 | **0.0105** | **-10%** |
| room_depth | 0.2748 | **0.2296** | **-16%** |

**実施した変更（`d3c81a7` でコミット済み）:**

1. `trackLocalMap()`: projection matching に **Lowe ratio test (0.7)** 追加 + search radius 120→100px
2. Local BA: iterations **5→10**（収束改善）、covisible KF **10→15**（BA 窓拡大）
3. `correctLoop()`: `loop_correcting_` フラグを `poseGraphOptimization()` の**前**に設定（旧: 後）、sleep 10→50ms

**ループあり実行では改善しなかった（むしろ悪化）**。ループ補正の安定化が未完了のため。

### 1.5 ループクロージングの根本問題（詳細分析結果）

2026-04-08 セッションで tracking.cc / optimizer.cc / loop_closing.cc の徹底分析を実施。以下が特定された**重要度順の問題リスト**:

| # | 問題 | 深刻度 | 状態 |
|---|------|--------|------|
| 1 | `poseGraphOptimization()` が全 `kf->T_cw_` を書き換えた後に `loop_correcting_` を設定していた → tracking がレース中に半更新のポーズを読む | Critical | **`d3c81a7` で修正済み**（フラグを先に設定） |
| 2 | `current_frame_` のポーズがループ補正後に更新されない → 次フレームの motion model が暴走 | Critical | **未コミット変更で修正済み**。`onLoopCorrected()` 追加 + `run_mono.cc` で callback 接続済み |
| 3 | Local mapping がループ補正中も BA を実行 → 同時書き込みレース | High | **未コミット変更で `loop_correcting_` チェック追加済み** |
| 4 | Covisibility edges がドリフト済みポーズから計算される（元の測定値を保存していない） | High | **未着手。** 根本的なデータ構造変更が必要 |
| 5 | `fuseLoopLandmarks()` で `kf->landmarks_[]` への書き込みが `kf->mutex_` なし | High | **未コミット変更で修正済み**。KF mutex 下で slot を snapshot / 更新 |
| 6 | 50ms sleep は tracking サイクル（33ms@30fps）に対して不十分な場合がある | Medium | `d3c81a7` で 10→50ms に延長したが根本解決ではない |
| 7 | `mergeLandmarks()` が candidate landmark の位置を保持（optimizer が移動させた current を破棄） | Medium | **未コミット変更で修正済み**。current 側 landmark を生存側に変更 |
| 8 | `loop_constraints_` が無制限に蓄積 | Low | **未着手** |

### 1.6 コミット履歴（2026-04-12 時点）

| コミット | 内容 |
|---------|------|
| `6429d9b` | **Refactor SLAM core, stabilize loop closing, and enrich test suite**: `tracking.cc` helper 抽出、`RecoveryState` / `LoopCorrectionState` / `ReinitializationState` への状態集約、`loop_closing` internal helper 化、`optimizer` cleanup、テスト **48/48**、回帰ゲート **5/5** |
| `d3c81a7` | **Improve tracking accuracy**: ratio test, BA iterations 5→10, covisible KF 10→15, loop correction ordering |
| `fcbde0a` | Add calibration override, run statistics, CLI help, eval tooling, citation, plan update |
| `52d3547` | Product release scaffolding: `--version`, `CHANGELOG.md`, `RELEASING.md` |
| `4288203` | xyz depth regression gate |
| `78b3b47` | room depth+accel regression gate + CONTRIBUTING.md |
| ... | （以前のコミットは `git log` 参照） |

### 1.7 過去セッション履歴

- **2026-04-06 Claude Code:** corpus 拡張、repeat 評価、CMake キャッシュ修復
- **2026-04-08 Claude Code (前半):** plan.md 大更新、キャリブ外部化、評価スクリプト共通化、学術引用
- **2026-04-08 Claude Code (後半):** stella_vslam 比較実測、コア改善、ループ補正安定化（途中）

---

## 2. コードベース全体像

### 2.1 ファイル構成（全ファイル、行数、役割）

```
simple_visual_slam/           # 8767行（テスト含む）
│
├── CMakeLists.txt            # FetchContent: Sophus, Ceres 2.1, DBoW2, ONNX Runtime 1.17, Google Test 1.14
│                             # オプション: USE_DBOW2(ON), USE_DEPTH_DL(OFF), BUILD_TESTS(ON)
├── plan.md                   # この文書
├── CITATION.cff              # 学術引用メタ（GitHub “Cite this repository”）
├── README.md                 # 英語README（Mermaidアーキテクチャ図、結果テーブル付き）
├── CONTRIBUTING.md / RELEASING.md / CHANGELOG.md / LICENSE
├── config/examples/
│   └── tum_pinhole_fr1.json  # --tum-camera-config 用サンプル
├── .gitignore                # *.bin, *.jpg, *.onnx, models/, eval_results/, trajectory*.txt, *.html
│
├── apps/
│   └── run_mono.cc           # [607行] エントリポイント
│       # CLI: --tum [--tum-camera-config <calib.json>] [--depth] [--accel] [--repro-eval]
│       #       [--reference-policy heuristic|score|pipeline] [--skip-frames N] [--max-frames N]
│       #       [--depth-model <model.onnx>] [--run-summary-json <path>] [--strict-exit]
│       #       [--no-viz] [--help/-h] [--version/-V]
│       #       --euroc <seq_dir> / <video_path>、[ORBvocab.txt]
│       # ★ --strict-exit: 終了時 OK でなければ exit 3
│       # ★ --run-summary-json: JSON 1行出力（schema: svslam.run_summary.v1）
│       # ★ vocab パス探索: --strict-exit / --run-summary-json / --tum-camera-config を正しくスキップ
│       # メインループ: 画像読み込み → Frame生成 → ORB抽出 → tracker->addFrame() → 軌跡保存
│       # DL depth: frame_id <= 1 || frame_id % 5 == 0 の時のみ推論（CPU高速化）
│       # 出力: trajectory.txt, trajectory_online.txt, trajectory_keyframes.txt, map.bin
│       # ORB: cv::ORB::create(2000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20)
│
├── src/core/
│   ├── common.h              # [30行] Vec2/Vec3/Mat33/SE3/Sim3 型エイリアス
│   │                         # Eigen, Sophus include。全ファイルがこれをinclude。
│   │
│   ├── camera.h / .cc        # [78行] ピンホールカメラ: fx_, fy_, cx_, cy_
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
│   ├── keyframe.h / .cc      # [168行] Keyframe extends Frame相当
│   │   # T_cw_ (SE3), depth_image_, depth_is_metric_
│   │   # gravity_in_camera_ (Vec3), has_gravity_ (bool) — 加速度計から推定した重力方向
│   │   # landmarks_ (vector<Landmark::Ptr>)
│   │   # connected_keyframes_ (map<KF::Ptr, int>) — covisibility graph（共有landmark数）
│   │   # updateConnections(): landmarks_の共有数を数えてconnected_keyframes_更新
│   │   # getBestCovisibilityKeyframes(N): weight降順でN個返す
│   │   # getDepth(u,v): Frame::getDepthと同じ
│   │
│   ├── landmark.h / .cc      # [67行] 3Dランドマーク
│   │   # id_, pos_w_ (Vec3)
│   │   # observations_ (map<weak_ptr<KF>, size_t>) — KF→特徴点index
│   │   # descriptor_ (cv::Mat)
│   │   # is_bad_ (bool)
│   │   # mutable mutex_ — setPos, getPos, addObservation, removeObservation で使用
│   │   # ★getPos()もmutexロック済み（Worker C-1で修正）
│   │
│   └── map.h / .cc           # [75行] マップ
│       # keyframes_ (map<id, KF::Ptr>), landmarks_ (map<id, LM::Ptr>)
│       # mutex_ — 汎用mutex（現在は一部のみ使用）
│       # loop_correcting_ (atomic<bool>) — ループ補正中フラグ
│       # addKeyframe/addLandmark/removeKeyframe/removeLandmark
│       # getAllKeyframes/getAllLandmarks — const参照を返す（コピーではない！）
│
├── src/tracking/
│   ├── tracking.h / .cc      # [2148行] ★最大のファイル。トラッキング全体を管理。
│   │   #
│   │   # === 状態遷移 ===
│   │   # NO_IMAGES_YET → NOT_INITIALIZED → OK ↔ LOST
│   │   # helper 抽出と `RecoveryState` / `LoopCorrectionState` / `ReinitializationState` への状態整理を実施済み
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
│   │   #
│   │   # === 運用カウンタ（TrackingRunStatistics run_stats_） ===
│   │   # reloc_attempts, reloc_successes, frames_tracking_lost, reinit_successes
│   │   # track() 内で LOST / relocalize / reinitialize の結果に応じてインクリメント
│   │   # runStatistics() で取得 → run_mono --run-summary-json で JSON 出力
│   │   #
│   │   # === 既知の問題 ===
│   │   # - needNewKeyframe()の閾値がハードコード
│   │   # - loop_correcting_中のtrackLocalMapスキップで1-5フレーム精度劣化
│   │   # - reinitialize()で新マップセグメントと旧マップの接続がない
│   │
│   └── initializer.h / .cc   # [486行] 2フレーム初期化
│       # H/F推定(RANSAC) → モデル選択(スコア比) → R|t復元 → 三角測量
│       # 三角測量: cv::triangulatePoints → 正depth/reproj error/max distance フィルタ
│
├── src/backend/
│   ├── local_mapping.h / .cc  # [502行] 局所マッピング（別スレッド）
│   │   # insertKeyframe → processNewKeyframe → createNewMapPoints → mapPointCulling → optimization
│   │   # processNewKeyframe: KFをmap追加、covisibility更新、LoopClosingへ転送
│   │   # createNewMapPoints: covisible KFペアで三角測量
│   │   #   - depth付きKFの場合: 先にdepth back-projectionでlandmark作成
│   │   # mapPointCulling: 観測率の低いlandmarkを除去
│   │   # optimization: bundleAdjustment呼び出し → on_ba_completed_コールバック
│   │   # on_ba_completed_: Tracking::onBACompleted()を呼ぶ（ポーズ再計算）
│   │
│   └── optimizer.h / .cc     # [889行] Ceres最適化
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
│   └── loop_closing.h / .cc  # [1004行] ループ検出・補正（別スレッド）
│       #
│       # === パイプライン ===
│       # insertKeyframe → processNewKeyframe:
│       #   1. detectLoop(): DBoW2で候補検索 (min_score=0.01, interval≥30KF)
│       #      - BoW候補は descriptor match 数でも再選別（ratio 0.8）
│       #      - 弱い候補しかなければ descriptor fallback scan に降格
│       #   2. computeSim3(): 3D-3D対応 → Sim3 RANSAC (200iter)
│       #      - metric depth: scale 0.85-1.15 に制限
│       #      - mono: scale 0.7-1.4
│       #      - failure reason をログ出力（insufficient_ransac_inliers 等）
│       #   3. correctLoop():
│       #      - poseGraphOptimization(60iter)
│       #      - map_->loop_correcting_ = true + sleep 20ms
│       #      - fuseLoopLandmarks()
│       #      - updateConnections() 全KF
│       #      - map_->loop_correcting_ = false
│       #      - on_loop_corrected_() で tracking に再計算通知
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
│       # - computeSim3()/fuseLoopLandmarks() の KF landmark access は mutex 化済みだが、
│       #   covisibility edge 自体はドリフト後ポーズ由来で、根本修正は未着手
│       # - cooldown: loop_cooldown_kf_ = 120（前回ループ成功から120KF以内は再検出しない）
│
├── src/io/
│   ├── tum_pinhole_calibration.h / .cc  # [27+124行] TUM 用ピンホールキャリブ
│   │   # TumPinholeCalibration: fx, fy, cx, cy, image_width, image_height, distortion[]
│   │   # fr1_default(): freiburg1 固定値（fx=517.3, fy=516.5, cx=318.6, cy=255.3 + 5係数）
│   │   # load_json_file(): 最小 JSON パーサ（外部ライブラリなし）
│   │   # 読み込み例: config/examples/tum_pinhole_fr1.json
│   │   # distortion が空なら undistort / remap をスキップ
│   │
│   ├── tum_dataset.h / .cc   # [329行] TUM RGB-D データセット読み込み
│   │   # AccelEntry: timestamp_sec, ax, ay, az
│   │   # DepthEntry: timestamp_sec, depth_path
│   │   # rgb.txt / depth.txt / accelerometer.txt パーサー
│   │   # nextWithDepth(): RGB + depth を±30msでアソシエーション（binary search）
│   │   # depth読み込み: CV_16UC1→float/5000.0, CV_32FC1→そのまま
│   │   # コンストラクタ: TumRgbdDataset(seq_dir) → fr1_default() 使用
│   │   #   TumRgbdDataset(seq_dir, TumPinholeCalibration) → 外部キャリブ
│   │   # K(): TumPinholeCalibration から構築（旧: freiburg1 ハードコード）
│   │   # allAccel(): 全accelerometerデータを返す
│   │
│   ├── euroc_dataset.h / .cc # [227行] EuRoC MAV: cam0/data/ + data.csv
│   │   # K(): EuRoC固定値
│   │
│   └── map_io.h / .cc        # [287行] バイナリマップ保存/読み込み
│       # ヘッダ "SVSLAM" + camera params + KFs + LMs + covisibility edges
│       # バージョニングなし（既知の制限）
│
├── src/depth/
│   ├── depth_estimator.h      # [19行] 抽象基底: virtual estimate(cv::Mat) → cv::Mat
│   │                          # virtual isMetric() → false
│   ├── onnx_depth_estimator.h # [38行] #ifdef USE_DEPTH_DL でガード
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
├── eval/
│   ├── regression_baselines.json   # 回帰ゲート: 5 gates (room_depth/depth_accel/mono, xyz_depth/mono)
│   │   # evo フラグ: --align --correct_scale --t_max_diff 0.05
│   │   # 各 gate: tum_sequence, skip/max_frames, repro_eval, use_depth/accel, max_mean_ape_m
│   ├── leaderboard_suite.json      # 研究用比較マトリクス: 2 seq (xyz_250, room_250) × 7 methods
│   │   # methods: mono/depth/depth_accel × heuristic, mono/depth × score/pipeline
│   └── comparison_protocol.md      # 外部OSSと並べるときの公平性ルール（94行）
│       # 同一 TUM 窓・モダリティ・evo_ape フラグでなければ比較不可
│       # Step 1: verify_comparison_benchmark.sh で自前基準を固定
│       # 推奨ピア: stella_vslam (BSD, 同クラス)
│
├── scripts/
│   ├── eval_lib.py                 # 共通ヘルパー（check_regression / leaderboard / print_ate_mean が import）
│   │   # sha256_file, clean_traj_artifacts, run_slam, evo_mean_ape
│   │   # ROOT = repo root
│   ├── check_regression_gate.py   # ビット一致2回 + ATE 上限（--all-gates 可）
│   │   # eval_lib を import（旧: 自前で sha256/run_slam/evo 実装を持っていた → 共通化済み）
│   ├── build_leaderboard.py       # methods×seq 排名（--dry-run で計画のみ、--json-out で raw）
│   │   # merge_run_gate(seq, method) で gate dict 合成
│   │   # 出力: Markdown (eval_results/leaderboard.md) + optional JSON
│   ├── print_ate_mean.py          # GT+trajectory → mean ATE（regression と同じ evo フラグ）
│   ├── verify_comparison_benchmark.sh  # 比較検証 Step 1
│   │   # プリセット: xyz_mono | xyz_depth | room_mono | room_depth
│   │   # 250フレ head, --repro-eval, heuristic, print_ate_mean.py 経由で mean ATE 出力
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
│   ├── test_frame.cc          # [90行] pose/depth/ORB accessor, backprojection
│   ├── test_initializer.cc    # [29行] 2-frame initializer smoke / triangulation gate
│   ├── test_keyframe.cc       # [74行] covisibility, depth access, best-KF selection
│   ├── test_landmark.cc       # [80行] addObservation, setPos/getPos, thread safety, isBad
│   ├── test_loop_closing.cc   # [74行] loop candidate selection, metric-depth / cooldown helpers
│   ├── test_map.cc            # [96行] add/remove KF/LM, count, concurrent add
│   ├── test_optimizer.cc      # [149行] BA / pose-graph helper coverage
│   ├── test_reference_keyframe_policy.cc  # [88行] policy seam contract / heuristic baseline
│   ├── test_tracking.cc       # [84行] tracking helper / recovery gate coverage
│   ├── test_tracking_pose_recompute.cc    # [39行] reprojection-improvement accept/reject
│   ├── test_tracking_run_statistics.cc   # [14行] TrackingRunStatistics 既定ゼロ確認
│   ├── test_tum_pinhole_calibration.cc  # [52行] JSON 読み込み・バリデーション（fr1 defaults, load, missing keys）
│   └── test_synthetic_scene.h  # [79行] synthetic scene fixture / test helper
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
| tracking.cc | recovery_stabilization_window_frames_ | 3 | relocalization / loop handoff 後に厳しめの pose update guard を維持するフレーム数 |
| tracking.cc | min_stable_support_ | 120 | local-map pose update を「stable support」とみなす最小対応数 |
| tracking.cc | recovery_max_change_strict_ | 0.12 | thin-support / support regression 時の pose update 最大変化量 |
| tracking.cc | recovery_max_change_relaxed_ | 0.18 | support が十分な recovery window 中の pose update 最大変化量 |
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

### 2.5 追加コンポーネント一覧（2026-04-08 時点）

**2.1 のファイルツリーに含まれないが重要な構成:**

#### run_mono CLI フラグ全容（597行、旧392行から大幅増）

```
--version / -V          semver 表示
--help / -h             全フラグ一覧（print_help()）
--tum <seq_dir>         TUM RGB-D モード
--tum-camera-config <json>  ピンホールキャリブ上書き（TumPinholeCalibration）
--depth                 depth.txt / sensor depth 使用
--accel                 accelerometer.txt 使用
--repro-eval            同期 mapping, loop closing 停止, 決定論モード
--reference-policy <heuristic|score|pipeline>
--skip-frames N         先頭 N フレームスキップ
--max-frames N          最大 N フレーム処理
--depth-model <onnx>    DL depth（USE_DEPTH_DL=ON ビルド時のみ）
--no-viz                OpenCV imshow 無効
--run-summary-json <path>  JSON 1行出力（schema: svslam.run_summary.v1）
--strict-exit           終了時 OK でなければ exit 3
--euroc <seq_dir>       EuRoC モード
<video_path>            動画ファイルモード
[ORBvocab.txt]          最終位置引数 or data/ORBvoc.txt 自動検索
```

#### Reference-keyframe policy（Phase F 収束済み、参考用）

- `src/core/reference_keyframe_policy.h` — 最小契約（`tracked_features`, `detected_keypoints`, `candidate_landmarks`, `frames_since_reference`, `lost_frames`, `has_depth`, `has_accel`）
- `src/core/heuristic_reference_keyframe_policy.{h,cc}` — runtime default
- `src/experiments/reference_keyframe/` — `score` / `pipeline`（discardable）
- `tools/reference_policy_experiments.cc` — curated corpus 比較バイナリ
- `tests/test_reference_keyframe_policy.cc` — policy seam テスト
- `scripts/eval_reference_policies.sh` — policy × corpus × repeat ハーネス（`--mode`, `--policy`, `--repeat`, `--corpus`, `--output`, `--no-repro`）
- `scripts/update_reference_policy_docs.py` — `docs/*.md` 自動生成
- `experiments/reference_keyframe/` — `scenarios.csv`, `real_trace_corpus.tsv`(13cases), `room_focus_corpus.tsv`(10cases)
- `docs/` — `index.md`(landing), `decisions.md`, `experiments.md`, `interfaces.md`

#### 過去の変更記録（2026-04-06 Claude Code session）

- `slam_result.jpg` 削除、`.gitignore` に `*.jpg` 追加
- `room_focus_corpus.tsv`: `room_mono_tail`, `room_mono_recovery`, `room_depth_accel_tail`, `room_depth_accel_recovery` 追加
- `real_trace_corpus.tsv`:
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

### 3.4 直近の確認コマンド（2026-04-08 更新）

少なくとも以下は recent green path とみなしてよい。

```bash
# ビルド（BUILD_TESTS=ON で全ターゲット + 新テスト2件含む）
cmake -S . -B build -G Ninja -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

# テスト（test_tum_pinhole_calibration + test_tracking_run_statistics 含む）
ctest --test-dir build --output-on-failure

# CLI ヘルプ（CI でも実行）
./build/run_mono --help

# ローカル回帰ゲート（TUM xyz+room、5シナリオ、evo_ape 必須。全ゲートで ~10 分級）
python3 scripts/check_regression_gate.py --all-gates --quiet

# 比較検証プリセット（data/tum/ + evo_ape 必須）
BUILD=build bash scripts/verify_comparison_benchmark.sh xyz_depth

# 研究用マトリクス（重い、ローカルのみ）
python3 scripts/build_leaderboard.py --build build --quiet

# mean ATE のみ
python3 scripts/print_ate_mean.py /path/to/groundtruth.txt /path/to/trajectory.txt

# real trace repeat-2（13 cases × 3 policies × 2 = 78 runs, 約20分）
bash scripts/eval_reference_policies.sh --repeat 2 \
  --output eval_results/reference_keyframe_policy/real_trace_metrics_repeat2.csv

# room focus repeat-5（10 cases × 3 policies × 5 = 150 runs, 約50分）
bash scripts/eval_reference_policies.sh --repeat 5 \
  --corpus experiments/reference_keyframe/room_focus_corpus.tsv \
  --output eval_results/reference_keyframe_policy/room_focus_repeat5.csv

# docs 再生成（policy 実験用、通常は不要）
./scripts/update_reference_policy_docs.py
```

注意:
- CMake キャッシュが別パスで作られていた場合は `rm -rf build` してから再構成（2026-04-06 にこの問題に遭遇）
- Ceres スレッド数は未設定時 **1**（再現性優先）。`SVSLAM_CERES_NUM_THREADS` で上書き可
- `build_nodbow2/`, `build_nodbow2_notests/`, `build_test_tumcal/` が作業ツリーに存在（gitignore 外の別ビルドディレクトリ）

### 3.5 テストの見方

- `ReferenceKeyframePolicyTest` は通る前提
- `test_camera`, `test_landmark`, `test_map`, `test_optimizer` も通常は green
- full `ctest` では Sophus 側の external test (`test_cartesian2`, `test_so2`) がノイズになり得る
- したがって、policy 変更の最低 gate は `ReferenceKeyframePolicyTest` + replay scripts + docs regen

---

## 4. 既知の問題と技術的負債

### 4.1 [最重要] ループクロージングの安定化（2026-04-08 分析済み、部分修正中）

ループクロージングを有効にすると room 系で ATE が **悪化** する（0.161→0.196〜0.341）。stella_vslam（g2o ベース、ループあり）は room_depth 0.033m を達成しており、**ループ補正の安定化が精度差の最大要因**。

**修正済み（`d3c81a7`）:**
- `correctLoop()` で `loop_correcting_` フラグを `poseGraphOptimization()` の前に設定
- sleep 10→50ms に延長

**未コミット（作業ツリーにある）:**
- `onLoopCorrected()`: loop thread では pending flag を立てるだけに変更。tracking thread 側で `after reference tracking` / `after local map` の safe point に `recomputeCurrentPose()` を実行し、成功時に velocity をリセット
- `processPendingWork()`: `loop_correcting_` 中は BA スキップ
- `run_mono.cc`: `on_loop_corrected_` callback 接続済み。BA / loop callback は `weak_ptr<Tracking>` capture で循環参照を回避
- `poseGraphOptimization()`: metric-depth 時は全 KF の log-scale へ prior を追加し、covisibility edge の scale residual も強化。さらに `T_cw_` と `connected_keyframes_` を mutex 下で snapshot してから solver に渡す。`PoseGraph: scale_stats` で min/max/mean を診断
- `detectLoop()`: BoW 候補を descriptor match 数と usable 3D correspondence 数（`corr3d`）で再選別。弱い候補は descriptor fallback に降格
- `matchLoopCandidate()`: loop closing 専用 descriptor match は Lowe ratio 0.8
- `fuseLoopLandmarks()`: KF landmark slot を mutex 下で snapshot / 更新
- `mergeLandmarks()`: current 側 landmark を生存側に変更
- `computeSim3()` / `correctLoop()`: failure reason をログ出力。`residual=0.35` 診断時の `relaxed_inliers` も出す。metric-depth では rigid (`scale=1`) 仮説生成 + `15 seed / 22 final / ratio guard` に変更し、loop constraint の translation/rotation weight は `inliers + inlier_ratio` ベースの confidence weight にした。pose-graph snapshot 化まで含む fresh `trajectory.txt` rerun では **0.13723747** / **0.07901260**
- `Tracking::track()` / `onLoopCorrected()`: loop callback を tracking thread の safe point に移し、初回 PnP が薄い frame に当たっても pending を保持して `trackLocalMap()` 後に同フレーム再試行するようにした。さらに pending loop correction は **`pairs >= 80`** のときだけ適用し、thin-support handoff は defer するように変更。`relocalize()` 成功直後も safe point に加えたが、それだけでは **0.18307474 / 0.22821251** と不安定だったため、pending recompute では **`err_after < err_before`** のときだけ採用する strict accept に変更。これにより early handoff を数フレーム reject して support が揃った所でだけ loop correction を吸収でき、fresh `trajectory.txt` rerun は 400-frame で **0.07522716** / **0.08267831**。strict 化前の retained `pairs>=80` 版は **0.08262759** / **0.18692874**、600-frame では second-loop + stale-edge reuse が入った run でも **0.14493733** まで改善（旧 `pairs=54` 即適用 run は **0.45154200**）。別 600-frame rerun は単発 loop で **0.10084369**
- `correctLoop()`: metric-depth で複数 loop constraint を pose graph に再利用する場合、古い edge を current map との translation/scale 整合で減衰する safety valve を追加。400-frame rerun は **0.15208171** / **0.07948407** で単発 loop のみだったが、600-frame rerun では `280->121` の second-loop 時に `Reweighting stale loop edge from_kf=151 to_kf=22 ... decay=0.35` を確認。`pairs<80` defer と組み合わせた 600-frame rerun は **0.14493733**、別 rerun は単発 loop で **0.10084369**
- `TrackLocalMap()` の thin-support overwrite guard も試した。`prior pose` に対する update 量を support / fallback ベースで絞る案は 400-frame では **0.07351221** まで改善し、`support=184` の危険な local-map 上書きも reject できたが、600-frame rerun は **0.19205861**, **0.18324189**, **0.19566213** で retained baseline を超えなかった。`support=119/110` や fallback `32` の update を止めても、別の箇所で relocalization 連鎖が出るだけで安定改善には繋がらなかったため**不採用**
- loop correction pending 中の frame で relocalization を 1 回遅らせる experiment も試したが、400-frame rerun は **0.16231753** で明確な改善証拠が取れず、bad run を確実に潰せる状態でもなかったため**不採用**。同様に `relocalize()` 成功直後に pending recompute を即適用するだけの variant も **0.18307474 / 0.22821251** で不採用。retained state は `pairs>=80` gate + pending strict accept まで

**未着手（次の改善候補）:**
- rigid metric-depth loop が入った後でも ATE が no-loop ベストを安定して超えない理由の解明。候補再選別と rigid 仮説生成で correction 自体は通るようになったので、次は **loop constraint の質 / pose graph 反映量** を詰める必要あり
- metric-depth の loop edge は **confidence weighting + final_min_inliers=22 + pose-graph snapshot + tracking-thread safe-point handoff** まで入れて、fresh `trajectory.txt` rerun が **0.12315088** / **0.12105041** に安定した。一方で best run は snapshot-only 時代の **0.07901260** なので、次にやるなら aging, schedule/timing, covisibility との相対バランスまで含めて設計する
- `151->22` を通すための borderline accept + cooldown `140 KF` や、relaxed residual での Sim3 再推定 rescue はどちらも rerun で second-loop に負けた。threshold 緩和方向で押すより、次は **2本目 loop の扱い** と **複数 constraint の設計** を見直す方が筋
- 古い loop edge の current-map consistency 減衰そのものは 600-frame で踏めるようになった。残る論点は **edge 再利用** より **loop correction handoff の support gate** と **late-run tracking quality** で、`pairs<80` defer は有効だったが、600-frame でも単発 loop run / second-loop run の揺れはまだ残る
- `loop_constraints_` の prune/update を素朴に入れるだけでも 400-frame rerun が **0.36〜0.37 m** に崩れたため、ここも schedule/timing 影響を含めてかなり慎重に扱う必要あり
- `sleep_for(50ms)` を coordinator / barrier で置き換える方向は筋が良いが、tracking 全体待ち版も `trackLocalMap()` 限定版も rerun で大崩れした。同期粒度を再設計しないまま入れるのは危険
- loop-closing queue への handoff を local mapping 完了後へ遅らせる案や、`loop_correcting_` を早めに立てる案も 400-frame rerun で **0.37 m** 級に崩れた。local mapping / loop closing のタイミングはかなり敏感
- Covisibility edges がドリフト済みポーズから計算される問題は依然あるが、元の相対測定値を保存する構造変更は rerun で **0.18586552 m** まで悪化したため、現時点では design を再考してから再挑戦
- `loop_constraints_` の無制限蓄積
- sleep ベースの同期を condition variable / barrier に置き換え

**分析の詳細は Section 1.5 を参照。**

### 4.2 [重要] stella_vslam との精度差（2026-04-08 実測）

Section 1.3 の実測結果を参照。主な差:
- **xyz:** 1.4〜2.3x 差 → コアの tracking/BA 品質で追える距離
- **room:** 5〜31x 差 → ループ補正なしのドリフト蓄積が支配的
- **構造的差異:** stella_vslam は g2o のグラフ最適化（Schur complement + スパース構造利用）、SimpleVisualSLAM は Ceres の密ソルバー。特にポーズグラフ最適化の品質が異なる

### 4.3 [結論済み] reference-keyframe policy は収束

3ポリシーは実質同等。`heuristic` が default。`score` / `pipeline` は `src/experiments/` に残すが active development 対象外。

### 4.4 [重要] room / mono 系の残留 non-determinism

`--repro-eval` で bitwise determinism を達成済み。async 実行（ループあり）では非決定性が残る。

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

### 4.7 [低→進行中] camera parameter の一般化

TUM は `--tum-camera-config <calib.json>` で外部キャリブ可能になった（`tum_pinhole_calibration.*`）。ただし:
- EuRoC は依然としてハードコード（`euroc_dataset.cc` の `K()`）
- 動画ファイルモードにはキャリブ指定手段がない
- 歪み係数が空の場合の undistort スキップは実装済み

---

## 5. ロードマップ

### Phase A: 品質基盤

| Task | Status | 内容 |
|------|--------|------|
| A-1 | ✅完了 | ORB決定論化、BA収束強化、2パスPnP |
| A-2 | 進行中 | `eval_all.sh --repeat N`、README 反映、回帰ゲート **5/5** までは完了。追加シーケンス拡充は未完 |
| A-3 | ✅完了 | Google Test suite 拡充（`ctest` **48/48**。`frame` / `keyframe` / `initializer` / `loop_closing` / `tracking` / `tracking_pose_recompute` 追加） |
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
| E-2 | ✅完了（拡張中） | GitHub Actions: Ubuntu Ninja ビルド + `ctest` + `run_mono --help` + `py_compile`(4スクリプト) + `build_leaderboard --dry-run`。`workflow_dispatch` 可。重い replay/ATE 計測は未 |
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

## 9. 優先順位（2026-04-08 更新）

```text
比較可能性（公平な数値の蓄積） > 安定性 > 精度 > 機能数 > 速度 > 見栄え
```

Phase F の policy 比較は収束。**外部OSSとの同一プロトコル比較**と **SLAM コア**を同列で進め、README で誇大なベンチ主張はしない。

**直近の優先事項（2026-04-08 時点）:**

1. **未コミット変更のコミット** — Section 1.2 / 1.4 参照。キャリブ外部化・運用フック・評価スクリプト共通化・学術引用等が作業ツリーに散在
2. **stella_vslam との同一プロトコル比較** — `comparison_protocol.md` Step 1 は `verify_comparison_benchmark.sh` で実行可。Step 2（外部OSS実行）は未着手
3. **SLAM コア品質** — `room_mono` 系の ATE 改善（tracking / 初期化 / 閾値チューニング）
4. **Metric DL Depth** — Phase B（Metric3D v2 / UniDepth）は未着手だが差別化の鍵

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
52d3547 Product release scaffolding: semver header, --version, CHANGELOG, RELEASING.  ← 2026-04 HEAD
4288203 Add xyz depth regression gate (freiburg1_xyz RGB-D).
78b3b47 Add room depth+accel regression gate and CONTRIBUTING.md.
9db75b4 Add BSD-2-Clause LICENSE and room depth regression gate.
6b32b5c Extend regression gate: xyz sequence + --all-gates.
31fbd1a Ceres threads via SVSLAM_CERES_NUM_THREADS; CI checks regression script.
785a923 Add local regression gate: repro SHA check + ATE ceiling vs baseline.
edd8805 Restore repro-eval bitwise determinism: BA ordering and covisibility ties.
45def39 Improve two-frame initialization: KNN ratio test and parallax gate.
fb46f82 Add GitHub Actions CI, stabilize OpenCV RNG and BA ordering, refresh docs.
4890411 update plan.md with repeat-5 evaluation results and phase shift
3b11008 expand room corpus and clean up README          ← 2026-04-06 Claude Code session
402b196 expand Claude handoff plan
6435a99 Merge pull request #1 from rsasaki0109/codex/reference-policy-experiments
b8ddbdd add reference policy experiment workflow
73eda32 Final plan.md update for complete Codex handoff
f4fc265 Merge worker results: precision stabilization, thread safety, README, unit tests
e2a3680 Update plan with comprehensive OSS roadmap and competitive analysis
66d5813 Fix loop closing thread safety and improve eval script
1fb546c Fix loop closing thread safety (atomic flag)
b88dac3 Add evaluation scripts, DL depth frame skip
5c48e38 Add gravity constraint in BA
fffb3e6 Add deep learning depth estimation via ONNX Runtime
6647223 Integrate depth sensor + accelerometer
```

**注意:** `52d3547` が現在の HEAD だが、その上に **未コミットの変更** が多数ある（Section 1.2 参照）。**最新の正確な履歴は常に `git log` + `git status`。**

---

## 12. AI エージェントへの引き継ぎ（2026-04-08 更新 — **Claude 向け**）

### 12.0 着手前チェック（最初の5分）

**必ず `git status` を確認**。この文書の執筆時点（2026-04-08）では **未コミット変更が多数ある**。

```bash
git status          # 13ファイル modified + 12 untracked
git diff --stat     # 変更量を確認
```

未コミット変更の詳細は **Section 1.2** に全リストがある。**コミットするかどうかは maintainer（ユーザ）の判断を仰ぐこと。** 勝手にコミットしない。

### 12.1 最初に読む順番（30分ルール）

1. **本書 Section 1.1–1.4** — 現状サマリ、未コミット変更一覧、コミット済み成果、分割案
2. **`eval/comparison_protocol.md`** — 外部OSSと数を並べるときの前提（窓・モダリティ・`evo_ape`）
3. **`README.md`** — 入口、CI、Citing、比較・回帰のコマンドへのポインタ
4. **`src/tracking/tracking.cc`** — 最大（1833行）・改善の主戦場
5. **`src/backend/optimizer.cc`**、**`src/loop_closing/loop_closing.cc`**
6. **`apps/run_mono.cc`** — CLI フラグの全容（597行）。`--help` で一覧表示
7. **`docs/index.md`** — public 向け要約（更新するなら `update_reference_policy_docs.py` 経由で整合）

**Phase F（reference-keyframe policy）は収束済み**。`src/experiments/`、`tools/reference_policy_experiments.cc` は参考。**policy 実験の再設計は非目標**（Section 10）。

### 12.2 期待する役割

**研究 OSS として**: 再現可能な数値・正直なスコープ・BSD のままの再利用。

**やってよいこと:**

| 区分 | 例 |
|------|-----|
| 比較・評価 | stella_vslam 等と **同一プロトコル**での ATE 記録；`verify_comparison_benchmark.sh` の拡張；`leaderboard_suite.json` の拡張 |
| コア | tracking（PnP、再局在、初期化）、BA、ループの信頼性向上 |
| 再現性 | `--repro-eval` を壊さず、`evo` / 乱数の分離の改善 |
| テスト | `tracking` / `merge_run_gate`（Python）等のカバレッジ |
| DL / 機能 | Phase B（metric depth）、Phase D — **計測フック付き**で |

**やってはいけないこと:**

- policy ablation の再開（収束済み）
- `docs/` に古い表・未検証の数字をそのまま push
- `--repro-eval` / 回帰 JSON の意味を変える変更を無言で入れる
- KITTI 公式リーダーボードと **同条件だと嘘をつく** README 記述（「研究用 TUM 窓」と言い切る）

### 12.3 推奨タスク優先度（**Claude が最初に着手するなら**）

**トラック A — 比較検証を完成させる（論文・競合示唆に直結）**

1. 自前4プリセットの欠損があれば埋める: `xyz_mono`, `room_mono`（`verify_comparison_benchmark.sh`）。
2. **stella_vslam**（または選定 OSS）を **同一 TUM ディレクトリ・250フレ方針・mono/depth** で実行し、`comparison_protocol.md` の表に列を追加。
3. 負けたセルだけ **原因仮説**（初期化 / スケール / ループ有無等）を1行メモ → トラック B へ。

**トラック B — SLAM コア（数値で勝つ側）**

1. `room_mono` 系の弱さ（歴史的に ATE 大）— `initializer.cc` / `trackLocalMap` / 閾値（Section 2.4、旧 12.3 の候補）を **1変更1ゲート**で。
2. `--repro-eval` 以外の async 実行の分散調査（OpenCV RANSAC 等）。
3. Metric DL depth（Depth Anything 以外の metric 化パイプ）。

**トラック C — メンテ・ドキュメント**

1. `scripts/update_reference_policy_docs.py` と評価 CSV のパス整合（`ROOM_FOCUS_STABILITY_FILE` 等）。
2. リリース時: `RELEASING.md` + `CITATION.cff` + `CHANGELOG` の三方同期。

**CI パイプライン（`.github/workflows/ci.yml`）:**

```
ubuntu-latest / 30min timeout
  → apt: build-essential cmake ninja-build git libopencv-dev libeigen3-dev libgoogle-glog-dev libgflags-dev libsuitesparse-dev
  → cmake -S . -B build -G Ninja -DBUILD_TESTS=ON
  → cmake --build build --parallel
  → ctest --test-dir build --output-on-failure
  → ./build/run_mono --help
  → py_compile: eval_lib.py, check_regression_gate.py, build_leaderboard.py, print_ate_mean.py
  → check_regression_gate.py --help
  → build_leaderboard.py --help + --dry-run
```

データセット（`data/tum/`）と `evo_ape` は CI に **ない** ため、回帰ゲート実行（ATE 計測）はローカルのみ。CI は syntax check + ビルド + ユニットテスト。

**FetchContent 注意:** ローカルで GitHub 到達性が悪い環境では `googletest`/`DBoW2` が取れないことがある。CI は `ubuntu-latest` 前提。

### 12.4 public repo としての注意

- Repo: `https://github.com/rsasaki0109/simple_visual_slam`
- Pages: `https://rsasaki0109.github.io/simple_visual_slam/`

`docs/` は external-facing。**壊れた表・未更新の性能表は push しない。** 内部の真実は `plan.md` + `git` + ログ。

### 12.5 触らないもの（メンテナポリシー）

- `.claude/`（ユーザ環境）
- **親ワークスペースの `AGENTS.md`** が別リポ用なら、そのルールをこのリポのコードに誤適用しない（この repo の規約は `CONTRIBUTING.md` / `plan.md`）。
- `data/` — データセットは git 管理外
- `scripts/__pycache__/`
- 巨大バイナリ・機密の混入

### 12.6 1ターンの終了条件（満たせば十分）

- ATE（mean または std）を **同一プロトコルで**改善した、と根拠付きで言える
- 再現性のバグを1つ潰した（または原因を特定した）
- 外部比較表の **新しい列・行**を埋め、プロトコルを文書化した
- テストまたは CI チェックを追加した
- `docs/` を意図的に更新し、再生成パスが通った

### 12.7 コマンド早見表（Claude 用コピペ）

```bash
# ビルド + ユニットテスト
cmake -S . -B build -G Ninja -DBUILD_TESTS=ON && cmake --build build && ctest --test-dir build --output-on-failure

# 回帰（ローカル data/tum + evo_ape 必須、十分時間）
python3 scripts/check_regression_gate.py --all-gates --quiet

# 比較検証プリセット（BUILD を適宜）
BUILD=build bash scripts/verify_comparison_benchmark.sh xyz_depth

# 研究用マトリクス（重い）
python3 scripts/build_leaderboard.py --build build --quiet

# mean ATE のみ（GT と trajectory のパス）
python3 scripts/print_ate_mean.py /path/to/groundtruth.txt /path/to/trajectory.txt

# CLI ヘルプ
./build/run_mono --help
```

### 12.8 ディレクトリ→目的（迷子防止）

| パス | 目的 |
|------|------|
| `eval/regression_baselines.json` | CI/ローカル回帰の **基準線**（5 gates, ATE 上限） |
| `eval/leaderboard_suite.json` | **内部** ablation 排名（外部 KITTI ではない） |
| `eval/comparison_protocol.md` | **外部 OSS** と並べる約束事（必読） |
| `scripts/eval_lib.py` | `run_mono` 起動 + `evo` 統計の **共通モジュール** |
| `scripts/check_regression_gate.py` | ビット一致 + ATE 回帰ゲート |
| `scripts/build_leaderboard.py` | methods×seq マトリクス（研究向け） |
| `scripts/verify_comparison_benchmark.sh` | 比較検証 Step 1（プリセット実行） |
| `scripts/print_ate_mean.py` | 単発 mean ATE 出力 |
| `apps/run_mono.cc` | 全入力モード・運用フラグの集合（`--help` で一覧） |
| `src/tracking/tracking.cc` | 追跡・初期化・再局在の中心（1833行） |
| `src/io/tum_pinhole_calibration.*` | TUM キャリブ JSON パーサ |
| `config/examples/` | `--tum-camera-config` サンプル |
| `CITATION.cff` / `RELEASING.md` | 学術引用・リリース手順 |
| `CHANGELOG.md` | Unreleased + 0.1.0 の変更記録 |

---

## 13. Public URLs / 公開物

- Repository: `https://github.com/rsasaki0109/simple_visual_slam`
- GitHub Pages: `https://rsasaki0109.github.io/simple_visual_slam/`
- Landing page: `docs/index.md`
- Decision record: `docs/decisions.md`
- Experiment tables: `docs/experiments.md`
- Minimal interface: `docs/interfaces.md`

`plan.md` は **内部 handoff（Claude / Codex / Cursor）**、`docs/` は public digest、`README.md` は入口、という役割分担。

---

## 14. 変更履歴メモ（plan.md 自身）

- **2026-04-12:** Section 1.1 を現行 HEAD `6429d9b` ベースへ更新。`d3c81a7` 後続コミットとして、`tracking.cc` helper 抽出、`RecoveryState` / `LoopCorrectionState` / `ReinitializationState` への状態集約、`loop_closing` internal helper 化、`optimizer` cleanup、loop stabilization の反映、`ctest` **48/48 pass**、回帰ゲート **5/5 pass**、新規テスト（`test_frame.cc`, `test_keyframe.cc`, `test_initializer.cc`, `test_loop_closing.cc`, `test_tracking.cc`, `test_tracking_pose_recompute.cc`, `test_synthetic_scene.h`）を追記。Section 1.6 のコミット表、Section 2.1 の行数と test 一覧、Section 2.4 の recovery 系定数、Section 5 Phase A の testing 状態も更新。
- **2026-04-09:** Section 1.1 を 2026-04-09 時点へ更新（未コミット差分量、retained best state、600-frame の未解決課題を反映）。Section 1.2 を拡張し、400-frame / 600-frame の rerun 結果、late-run での second-loop / relocalization 連鎖の現状、`TrackLocalMap()` thin-support overwrite guard 実験 3 variant（400-frame **0.07351221**、600-frame **0.19205861 / 0.18324189 / 0.19566213**、いずれも不採用）を長文で記録。次に触る候補として `TrackReferenceKeyframe()` の post-loop collapse、通常 BA callback 側 `recomputeCurrentPose()` の absolute fallback、pending correction expire 後の handoff 再設計を明記。
- **2026-04-09:** 候補 2 を着手済みに更新。`src/tracking/tracking.cc` で BA callback の pose recompute から `err_after < 20.0` absolute fallback を除去し、strict reprojection improvement のみ許可する形へ統一。`tests/test_tracking_pose_recompute.cc` を追加し、`build_bench` の `ctest` 26/26 pass、`verify_comparison_benchmark.sh room_mono` で worsened BA update の reject ログを確認。
- **2026-04-09:** 候補 1 の first-loop collapse を追加調査。`room_depth` 400-frame rerun で、loop correction handoff 成功後に end-of-frame velocity 更新が identity reset を上書きし、次フレーム `TrackReferenceKeyframe` が `12` correspondences まで落ちる再現を確認。`skip_velocity_update_once_` によって loop-corrected frame の velocity 更新を 1 回抑止したところ、同条件 rerun では loop 直後の次フレームが `540` correspondences、次が `708` まで回復し、局所 collapse は消失。fresh `trajectory.txt` ATE は **0.20676216** で、全体改善はまだ評価継続。
- **2026-04-09:** 候補 3 の expire-time reset も着手。pending loop correction が最大 deferral に達したら、次 frame で keyframe insertion を強制し、その keyframe を reference に昇格、`previous_reference_keyframe_` は clear する fallback を追加。`build_bench` の `ctest` は継続して 26/26 pass。変更後の 600-frame rerun では expire path 自体が今回は再現せず、forced refresh marker は未発火だったため、fresh `trajectory.txt` ATE **0.22985886** と合わせて「コードは入ったが効果は未確定」と記録。
- **2026-04-08:** Section 1 を大幅拡充（1.1 → 2026-04-08 時点に更新、1.2 未コミット変更の完全リスト追加、1.3 コミット済み成果テーブル化、1.4 コミット分割案新設、1.5 旧過去メモ）。Section 2.1 に `tum_pinhole_calibration`・詳細コメント追加、`run_mono.cc` 行数と全 CLI フラグ更新、テスト行数更新、`eval/` と `scripts/` に詳細注釈追加。Section 11 git 履歴を `52d3547` まで拡張。Section 12 に 12.0（着手前チェック）新設、12.1 読む順番改訂、CI パイプライン詳細追加。
- **2026-04（初版）:** Section 1 再構成（1.1 現状サマリ / 1.2 詳細差分 / 1.3 過去セッション）、Section 2.1 に `eval/`・`scripts/eval_*`・`config/examples`・新テストを反映、Section 12 を **Claude 引き継ぎ特化**で全面改稿（トラック A/B/C、12.7–12.8 追加）。

# SimpleVisualSLAM

C++17で実装された最小限の単眼Visual SLAM実装。

## ライセンス

MIT License.

## 依存関係

- **OpenCV** (必須)
- **Ceres Solver** (必須 - 最適化フェーズで有効化)
- **Sophus** (必須)
- **DBoW2** (必須 - Loop Closure用)

## ビルド

```bash
mkdir build
cd build
cmake ..
make
```

## 使用方法

```bash
./run_mono <path_to_video_or_camera_index>
```

## 機能 (計画)

- [x] 基本データ構造 (Frame, Keyframe, Map, Camera)
- [x] ORB特徴抽出
- [ ] トラッキング (等速運動モデル, 参照KF)
- [ ] 初期化 (単眼)
- [ ] 局所マッピング (Local BA)
- [ ] ループクロージャ (DBoW2 + Pose Graph)
- [ ] マップ保存/読み込み

詳細な開発計画は [plan.md](plan.md) を参照してください。

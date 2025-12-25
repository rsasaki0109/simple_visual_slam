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
# ビデオファイルの場合
./run_mono path/to/video.mp4

# カメラの場合 (ID指定)
./run_mono 0
```

### サンプル実行

リポジトリにはサンプル動画 `tree.avi` (OpenCV sample) を使用した実行結果が含まれています。

```bash
./run_mono tree.avi
```

実行後、`slam_result.jpg` が生成されます。

## 機能 (計画)

- [x] 基本データ構造 (Frame, Keyframe, Map, Camera)
- [x] ORB特徴抽出
- [ ] トラッキング (等速運動モデル, 参照KF)
- [ ] 初期化 (単眼)
- [ ] 局所マッピング (Local BA)
- [ ] ループクロージャ (DBoW2 + Pose Graph)
- [ ] マップ保存/読み込み

詳細な開発計画は [plan.md](plan.md) を参照してください。

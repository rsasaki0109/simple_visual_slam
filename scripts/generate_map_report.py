#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class KeyframePose:
    timestamp: float
    kf_id: int
    center: np.ndarray
    rotation: np.ndarray  # 3x3 rotation matrix (world <- camera)


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w],
        [2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w],
        [2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y],
    ])


def load_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps = []
    positions = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            timestamps.append(float(parts[0]))
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(timestamps), np.asarray(positions)


def associate(est_t: np.ndarray, gt_t: np.ndarray, max_diff: float) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    j = 0
    for i, t in enumerate(est_t):
        while j + 1 < len(gt_t) and abs(gt_t[j + 1] - t) <= abs(gt_t[j] - t):
            j += 1
        if abs(gt_t[j] - t) <= max_diff:
            pairs.append((i, j))
    return pairs


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=1, keepdims=True)
    dst_mean = dst.mean(axis=1, keepdims=True)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered @ src_centered.T) / src.shape[1]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        s[-1, -1] = -1.0
    rotation = u @ s @ vt
    variance = np.sum(src_centered ** 2) / src.shape[1]
    scale = np.trace(np.diag(d) @ s) / variance
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def load_map(map_path: Path) -> tuple[list[KeyframePose], np.ndarray]:
    keyframes: list[KeyframePose] = []
    landmarks: list[list[float]] = []

    with map_path.open("rb") as f:
        if f.read(6) != b"SVSLAM":
            raise RuntimeError("invalid map header")

        f.read(9 * 8)
        num_kfs = struct.unpack("Q", f.read(8))[0]
        ulong_size = struct.calcsize("L")

        for _ in range(num_kfs):
            kf_id = struct.unpack("L", f.read(ulong_size))[0]
            timestamp = struct.unpack("d", f.read(8))[0]
            t = np.frombuffer(f.read(8 * 3), dtype=np.float64).copy()
            q = np.frombuffer(f.read(8 * 4), dtype=np.float64).copy()

            # Tcw: rotation = R_cw, translation = t_cw
            R_cw = quaternion_to_rotation_matrix(q[0], q[1], q[2], q[3])
            center = -R_cw.T @ t      # camera center in world
            R_wc = R_cw.T             # world <- camera rotation

            num_kps = struct.unpack("Q", f.read(8))[0]
            f.read(num_kps * (4 + 4 + 4 + 4))
            rows, cols, _ = struct.unpack("iii", f.read(12))
            if rows > 0 and cols > 0:
                f.read(rows * cols)
            f.read(num_kps * 8)

            keyframes.append(KeyframePose(timestamp, kf_id, center, R_wc))

        num_lms = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_lms):
            f.read(ulong_size)
            pos = np.frombuffer(f.read(8 * 3), dtype=np.float64).copy()
            rows, cols, _ = struct.unpack("iii", f.read(12))
            if rows > 0 and cols > 0:
                f.read(rows * cols)
            if np.all(np.isfinite(pos)):
                landmarks.append([float(pos[0]), float(pos[1]), float(pos[2])])

    keyframes.sort(key=lambda item: (item.timestamp, item.kf_id))
    return keyframes, np.asarray(landmarks)


def align_to_gt(kf_centers: np.ndarray, gt_path: Path, max_diff: float, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Returns (aligned_centers, gt_matched, scale, rotation, translation)."""
    gt_t, gt_p = load_positions(gt_path)
    pairs = associate(timestamps, gt_t, max_diff)
    if len(pairs) < 3:
        raise RuntimeError("not enough associations to align map")

    est_idx = np.asarray([i for i, _ in pairs])
    gt_idx = np.asarray([j for _, j in pairs])

    scale, rotation, translation = umeyama_alignment(kf_centers[est_idx].T, gt_p[gt_idx].T)
    aligned = (scale * rotation @ kf_centers.T + translation).T
    matched_gt = gt_p[gt_idx]
    return aligned, matched_gt, scale, rotation, translation


def sample_indices(n: int, limit: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.linspace(0, n - 1, limit).astype(int)


def build_html(
    keyframes_aligned: np.ndarray,
    kf_rotations_aligned: list[list[list[float]]],
    landmarks: np.ndarray,
    gt_positions: np.ndarray,
    scale: float,
    num_kf_total: int,
    num_lm_total: int,
    scene_image_data: str | None,
) -> str:
    # Sample for performance
    lm_limit = min(8000, len(landmarks))
    landmark_sample = landmarks[sample_indices(len(landmarks), lm_limit)] if len(landmarks) else landmarks
    gt_sample = gt_positions[sample_indices(len(gt_positions), min(500, len(gt_positions)))]
    kf_sample_idx = sample_indices(len(keyframes_aligned), min(500, len(keyframes_aligned)))
    kf_sample = keyframes_aligned[kf_sample_idx]
    kf_rot_sample = [kf_rotations_aligned[i] for i in kf_sample_idx]

    # Convert to JSON
    lm_json = json.dumps(landmark_sample.tolist()) if len(landmark_sample) else "[]"
    kf_json = json.dumps(kf_sample.tolist()) if len(kf_sample) else "[]"
    kf_rot_json = json.dumps(kf_rot_sample)
    gt_json = json.dumps(gt_sample.tolist()) if len(gt_sample) else "[]"

    mins = landmarks.min(axis=0) if len(landmarks) else np.zeros(3)
    maxs = landmarks.max(axis=0) if len(landmarks) else np.zeros(3)

    scene_img_tag = ""
    if scene_image_data:
        scene_img_tag = f'<img id="sceneImg" src="data:image/jpeg;base64,{scene_image_data}" style="max-width:100%;border-radius:8px;">'

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SimpleVisualSLAM 3D Map Viewer</title>
  <style>
    :root {{
      --bg: #0d1117;
      --panel: rgba(22, 27, 34, 0.95);
      --ink: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --kf: #f0883e;
      --gt: #58a6ff;
      --cloud: #3fb950;
      --border: rgba(240, 246, 252, 0.1);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
      overflow: hidden;
      height: 100vh;
    }}
    #canvas3d {{
      width: 100vw;
      height: 100vh;
      display: block;
    }}
    .overlay {{
      position: fixed;
      top: 16px;
      left: 16px;
      z-index: 10;
      pointer-events: none;
    }}
    .overlay > * {{ pointer-events: auto; }}
    .info-panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 20px;
      backdrop-filter: blur(12px);
      min-width: 260px;
      max-width: 300px;
    }}
    .info-panel h1 {{
      font-size: 1.05rem;
      font-weight: 600;
      margin-bottom: 10px;
    }}
    .stat-row {{
      display: flex;
      justify-content: space-between;
      padding: 3px 0;
      font-size: 0.82rem;
    }}
    .stat-label {{ color: var(--muted); }}
    .stat-value {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
    .legend {{
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 5px;
      font-size: 0.82rem;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      flex-shrink: 0;
    }}
    .legend-frustum {{
      width: 10px;
      height: 10px;
      border: 2px solid var(--kf);
      flex-shrink: 0;
    }}
    .controls {{
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }}
    .controls label {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.8rem;
      color: var(--muted);
      cursor: pointer;
      padding: 2px 0;
    }}
    .controls input[type="checkbox"] {{
      accent-color: var(--accent);
    }}
    .controls input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .size-control {{
      margin-top: 6px;
    }}
    .size-control .size-label {{
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      color: var(--muted);
      margin-bottom: 2px;
    }}
    .scene-preview {{
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }}
    .scene-preview p {{
      font-size: 0.75rem;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .help {{
      position: fixed;
      bottom: 16px;
      left: 16px;
      z-index: 10;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 14px;
      backdrop-filter: blur(12px);
      font-size: 0.75rem;
      color: var(--muted);
      line-height: 1.7;
    }}
    .help kbd {{
      background: rgba(255,255,255,0.1);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 1px 5px;
      font-size: 0.72rem;
    }}
  </style>
</head>
<body>
  <canvas id="canvas3d"></canvas>

  <div class="overlay">
    <div class="info-panel">
      <h1>3D Map Viewer</h1>
      <div class="stat-row">
        <span class="stat-label">Keyframes</span>
        <span class="stat-value">{num_kf_total}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Landmarks</span>
        <span class="stat-value">{num_lm_total:,}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Sim3 Scale</span>
        <span class="stat-value">{scale:.4f}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">X</span>
        <span class="stat-value">{mins[0]:.2f} .. {maxs[0]:.2f}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Y</span>
        <span class="stat-value">{mins[1]:.2f} .. {maxs[1]:.2f}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Z</span>
        <span class="stat-value">{mins[2]:.2f} .. {maxs[2]:.2f}</span>
      </div>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background: #8be9fd;"></div> Landmarks (height color)</div>
        <div class="legend-item"><div class="legend-frustum"></div> Camera frustums</div>
        <div class="legend-item"><div class="legend-dot" style="background: var(--kf);"></div> Keyframe path</div>
        <div class="legend-item"><div class="legend-dot" style="background: var(--gt);"></div> Ground truth</div>
      </div>
      <div class="controls">
        <label><input type="checkbox" id="toggleLandmarks" checked> Landmarks</label>
        <label><input type="checkbox" id="toggleFrustums" checked> Camera frustums</label>
        <label><input type="checkbox" id="toggleKF" checked> Keyframe path</label>
        <label><input type="checkbox" id="toggleGT" checked> Ground truth</label>
        <label><input type="checkbox" id="toggleGrid" checked> Grid</label>
        <label><input type="checkbox" id="toggleAxes" checked> Axes</label>
        <div class="size-control">
          <div class="size-label"><span>Point size</span><span id="sizeValue">3.0</span></div>
          <input type="range" id="pointSize" min="1" max="12" step="0.5" value="3.0">
        </div>
        <div class="size-control">
          <div class="size-label"><span>Frustum size</span><span id="frustumValue">1.0</span></div>
          <input type="range" id="frustumSize" min="0.2" max="3" step="0.1" value="1.0">
        </div>
      </div>
      <div class="scene-preview">
        <p>Scene reference</p>
        {scene_img_tag}
      </div>
    </div>
  </div>

  <div class="help">
    <kbd>Left drag</kbd> Rotate &nbsp;
    <kbd>Right drag</kbd> Pan &nbsp;
    <kbd>Scroll</kbd> Zoom &nbsp;
    <kbd>R</kbd> Reset &nbsp;
    <kbd>1</kbd> Top &nbsp;
    <kbd>2</kbd> Front &nbsp;
    <kbd>3</kbd> Side
  </div>

  <script type="importmap">
  {{
    "imports": {{
      "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
    }}
  }}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

    const landmarkData = {lm_json};
    const kfData = {kf_json};
    const kfRotData = {kf_rot_json};
    const gtData = {gt_json};

    // --- Setup ---
    const canvas = document.getElementById('canvas3d');
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0d1117);

    const scene = new THREE.Scene();
    // Soft ambient + directional for depth cues
    scene.add(new THREE.AmbientLight(0x404050, 1.5));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight.position.set(5, 10, 5);
    scene.add(dirLight);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 500);

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 1.2;

    // --- Compute scene center and scale ---
    function computeBounds(arrays) {{
      let min = [Infinity, Infinity, Infinity];
      let max = [-Infinity, -Infinity, -Infinity];
      for (const arr of arrays) {{
        for (const p of arr) {{
          for (let i = 0; i < 3; i++) {{
            if (p[i] < min[i]) min[i] = p[i];
            if (p[i] > max[i]) max[i] = p[i];
          }}
        }}
      }}
      return {{ min, max }};
    }}

    const bounds = computeBounds([landmarkData, kfData, gtData]);
    const center = bounds.min.map((v, i) => (v + bounds.max[i]) / 2);
    const span = Math.max(...bounds.max.map((v, i) => v - bounds.min[i]), 0.1);

    // --- Color map (plasma-like) ---
    function plasmaColor(t) {{
      t = Math.max(0, Math.min(1, t));
      // simplified plasma colormap
      const r = Math.min(1, 0.05 + 1.3 * t + 0.3 * Math.sin(t * 6.28));
      const g = Math.min(1, Math.max(0, 0.85 * Math.sin(t * 3.14)));
      const b = Math.min(1, Math.max(0, 0.95 - 1.2 * t + 0.4 * Math.sin(t * 6.28 + 1)));
      return [r, g, b];
    }}

    // --- Landmarks (height-colored point cloud) ---
    const lmGeom = new THREE.BufferGeometry();
    const lmPositions = new Float32Array(landmarkData.length * 3);
    const lmColors = new Float32Array(landmarkData.length * 3);

    let yMin = Infinity, yMax = -Infinity;
    for (const p of landmarkData) {{
      if (p[1] < yMin) yMin = p[1];
      if (p[1] > yMax) yMax = p[1];
    }}
    const ySpan = Math.max(yMax - yMin, 1e-6);

    for (let i = 0; i < landmarkData.length; i++) {{
      lmPositions[i * 3 + 0] = landmarkData[i][0];
      lmPositions[i * 3 + 1] = landmarkData[i][1];
      lmPositions[i * 3 + 2] = landmarkData[i][2];
      const t = (landmarkData[i][1] - yMin) / ySpan;
      const [r, g, b] = plasmaColor(t);
      lmColors[i * 3 + 0] = r;
      lmColors[i * 3 + 1] = g;
      lmColors[i * 3 + 2] = b;
    }}

    lmGeom.setAttribute('position', new THREE.BufferAttribute(lmPositions, 3));
    lmGeom.setAttribute('color', new THREE.BufferAttribute(lmColors, 3));

    const lmMat = new THREE.PointsMaterial({{
      size: 0.022 * span,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
    }});
    const landmarkPoints = new THREE.Points(lmGeom, lmMat);
    scene.add(landmarkPoints);

    // --- Camera frustums ---
    const frustumGroup = new THREE.Group();
    let frustumScale = 0.06 * span;

    function buildFrustum(pos, rot, s) {{
      // Camera looks down -Z in camera frame
      // frustum corners in camera frame
      const hw = s * 0.6;  // half width
      const hh = s * 0.45; // half height
      const d = s;          // depth
      const corners = [
        [-hw, -hh, -d],
        [ hw, -hh, -d],
        [ hw,  hh, -d],
        [-hw,  hh, -d],
      ];
      // Transform to world
      const R = new THREE.Matrix3().fromArray([
        rot[0][0], rot[1][0], rot[2][0],
        rot[0][1], rot[1][1], rot[2][1],
        rot[0][2], rot[1][2], rot[2][2],
      ]);
      const o = new THREE.Vector3(pos[0], pos[1], pos[2]);

      const wCorners = corners.map(c => {{
        const v = new THREE.Vector3(c[0], c[1], c[2]).applyMatrix3(R);
        return v.add(o);
      }});

      const lines = [];
      // edges from origin to corners
      for (const c of wCorners) {{
        lines.push(o.x, o.y, o.z, c.x, c.y, c.z);
      }}
      // rectangle
      for (let i = 0; i < 4; i++) {{
        const a = wCorners[i];
        const b = wCorners[(i + 1) % 4];
        lines.push(a.x, a.y, a.z, b.x, b.y, b.z);
      }}

      return lines;
    }}

    function rebuildFrustums(s) {{
      frustumGroup.clear();
      const allLines = [];
      for (let i = 0; i < kfData.length; i++) {{
        const lines = buildFrustum(kfData[i], kfRotData[i], s);
        allLines.push(...lines);
      }}
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.Float32BufferAttribute(allLines, 3));
      const mat = new THREE.LineBasicMaterial({{ color: 0xf0883e, transparent: true, opacity: 0.7 }});
      frustumGroup.add(new THREE.LineSegments(geom, mat));

      // Near plane fill (semi-transparent)
      for (let i = 0; i < kfData.length; i++) {{
        const hw = s * 0.6;
        const hh = s * 0.45;
        const d = s;
        const corners = [[-hw, -hh, -d], [hw, -hh, -d], [hw, hh, -d], [-hw, hh, -d]];
        const R = new THREE.Matrix3().fromArray([
          kfRotData[i][0][0], kfRotData[i][1][0], kfRotData[i][2][0],
          kfRotData[i][0][1], kfRotData[i][1][1], kfRotData[i][2][1],
          kfRotData[i][0][2], kfRotData[i][1][2], kfRotData[i][2][2],
        ]);
        const o = new THREE.Vector3(kfData[i][0], kfData[i][1], kfData[i][2]);
        const wc = corners.map(c => new THREE.Vector3(c[0], c[1], c[2]).applyMatrix3(R).add(o));

        const planeGeom = new THREE.BufferGeometry();
        const verts = new Float32Array([
          wc[0].x, wc[0].y, wc[0].z,
          wc[1].x, wc[1].y, wc[1].z,
          wc[2].x, wc[2].y, wc[2].z,
          wc[0].x, wc[0].y, wc[0].z,
          wc[2].x, wc[2].y, wc[2].z,
          wc[3].x, wc[3].y, wc[3].z,
        ]);
        planeGeom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        const planeMat = new THREE.MeshBasicMaterial({{
          color: 0xf0883e,
          transparent: true,
          opacity: 0.12,
          side: THREE.DoubleSide,
          depthWrite: false,
        }});
        frustumGroup.add(new THREE.Mesh(planeGeom, planeMat));
      }}
    }}

    rebuildFrustums(frustumScale);
    scene.add(frustumGroup);

    // --- Keyframe trajectory ---
    function makeTrajectoryLine(data, color, linewidth) {{
      const positions = [];
      for (const p of data) positions.push(p[0], p[1], p[2]);
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      return new THREE.Line(geom, new THREE.LineBasicMaterial({{ color, linewidth }}));
    }}

    const kfLine = makeTrajectoryLine(kfData, 0xf0883e, 2);
    scene.add(kfLine);

    const gtLine = makeTrajectoryLine(gtData, 0x58a6ff, 2);
    scene.add(gtLine);

    // --- Grid ---
    const gridSize = Math.ceil(span * 1.5);
    const grid = new THREE.GridHelper(gridSize, Math.min(40, Math.ceil(gridSize / 0.1)), 0x30363d, 0x21262d);
    grid.position.set(center[0], bounds.min[1] - 0.02 * span, center[2]);
    scene.add(grid);

    // --- Axes ---
    const axesGroup = new THREE.Group();
    const axLen = span * 0.15;
    function makeAxis(dir, color) {{
      const geom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(center[0], center[1], center[2]),
        new THREE.Vector3(center[0] + dir[0] * axLen, center[1] + dir[1] * axLen, center[2] + dir[2] * axLen),
      ]);
      return new THREE.Line(geom, new THREE.LineBasicMaterial({{ color, linewidth: 2 }}));
    }}
    axesGroup.add(makeAxis([1, 0, 0], 0xff6666));
    axesGroup.add(makeAxis([0, 1, 0], 0x66ff66));
    axesGroup.add(makeAxis([0, 0, 1], 0x6688ff));
    scene.add(axesGroup);

    // --- Camera setup ---
    function resetView() {{
      camera.position.set(
        center[0] + span * 0.8,
        center[1] + span * 0.6,
        center[2] + span * 0.8
      );
      controls.target.set(center[0], center[1], center[2]);
      controls.update();
    }}
    resetView();

    // --- UI Controls ---
    document.getElementById('toggleLandmarks').addEventListener('change', e => {{
      landmarkPoints.visible = e.target.checked;
    }});
    document.getElementById('toggleFrustums').addEventListener('change', e => {{
      frustumGroup.visible = e.target.checked;
    }});
    document.getElementById('toggleKF').addEventListener('change', e => {{
      kfLine.visible = e.target.checked;
    }});
    document.getElementById('toggleGT').addEventListener('change', e => {{
      gtLine.visible = e.target.checked;
    }});
    document.getElementById('toggleGrid').addEventListener('change', e => {{
      grid.visible = e.target.checked;
    }});
    document.getElementById('toggleAxes').addEventListener('change', e => {{
      axesGroup.visible = e.target.checked;
    }});
    document.getElementById('pointSize').addEventListener('input', e => {{
      const val = parseFloat(e.target.value);
      document.getElementById('sizeValue').textContent = val.toFixed(1);
      lmMat.size = val * 0.0075 * span;
    }});
    document.getElementById('frustumSize').addEventListener('input', e => {{
      const val = parseFloat(e.target.value);
      document.getElementById('frustumValue').textContent = val.toFixed(1);
      frustumScale = val * 0.06 * span;
      rebuildFrustums(frustumScale);
    }});

    // --- Keyboard ---
    document.addEventListener('keydown', e => {{
      if (e.key === 'r' || e.key === 'R') resetView();
      if (e.key === '1') {{
        camera.position.set(center[0], center[1] + span * 1.5, center[2]);
        controls.target.set(center[0], center[1], center[2]);
        controls.update();
      }}
      if (e.key === '2') {{
        camera.position.set(center[0], center[1], center[2] + span * 1.5);
        controls.target.set(center[0], center[1], center[2]);
        controls.update();
      }}
      if (e.key === '3') {{
        camera.position.set(center[0] + span * 1.5, center[1], center[2]);
        controls.target.set(center[0], center[1], center[2]);
        controls.update();
      }}
    }});

    // --- Resize ---
    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    // --- Animate ---
    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, default=Path("map.bin"))
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tum_fr1_xyz_map_report.html"))
    parser.add_argument("--max-diff", type=float, default=0.02)
    parser.add_argument("--scene-image", type=Path, default=None, help="Reference image of the scene")
    args = parser.parse_args()

    keyframes, landmarks = load_map(args.map)
    kf_timestamps = np.asarray([kf.timestamp for kf in keyframes])
    kf_centers = np.asarray([kf.center for kf in keyframes])
    kf_rotations = [kf.rotation for kf in keyframes]

    aligned_kfs, gt_matched, scale, sim3_rot, sim3_trans = align_to_gt(
        kf_centers, args.groundtruth, args.max_diff, kf_timestamps
    )

    # Align landmarks
    aligned_landmarks = (scale * sim3_rot @ landmarks.T + sim3_trans).T if len(landmarks) else landmarks

    # Align KF rotations
    aligned_rotations = []
    for R_wc in kf_rotations:
        R_aligned = sim3_rot @ R_wc
        aligned_rotations.append(R_aligned.tolist())

    # Full GT trajectory
    gt_t, gt_p = load_positions(args.groundtruth)

    # Scene image (optional)
    scene_image_data = None
    if args.scene_image and args.scene_image.exists():
        import base64
        scene_image_data = base64.b64encode(args.scene_image.read_bytes()).decode()

    html_report = build_html(
        aligned_kfs, aligned_rotations, aligned_landmarks, gt_p,
        scale, len(keyframes), len(landmarks),
        scene_image_data,
    )
    args.output.write_text(html_report, encoding="utf-8")
    print(f"Written: {args.output} ({len(keyframes)} KFs, {len(landmarks)} landmarks)")


if __name__ == "__main__":
    main()

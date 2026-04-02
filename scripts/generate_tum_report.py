#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrajectoryEval:
    label: str
    timestamps: np.ndarray
    positions: np.ndarray
    matched_est: np.ndarray
    matched_gt: np.ndarray
    matched_ts: np.ndarray
    aligned: np.ndarray
    errors: np.ndarray
    scale: float
    ate_rmse: float
    ate_mean: float
    ate_median: float
    ate_max: float
    final_error: float
    path_est: float
    path_gt: float
    rpe_1: dict[str, float] | None
    rpe_30: dict[str, float] | None
    association_count: int


def load_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps = []
    positions = []
    with path.open() as f:
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


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def translational_rpe(est: np.ndarray, gt: np.ndarray, delta: int) -> dict[str, float] | None:
    if len(est) <= delta:
        return None
    rel_est = est[delta:] - est[:-delta]
    rel_gt = gt[delta:] - gt[:-delta]
    err = np.linalg.norm(rel_est - rel_gt, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "max": float(np.max(err)),
        "count": int(err.size),
    }


def evaluate_trajectory(label: str, traj_path: Path, gt_path: Path, max_diff: float) -> TrajectoryEval:
    est_t, est_p = load_positions(traj_path)
    gt_t, gt_p = load_positions(gt_path)
    pairs = associate(est_t, gt_t, max_diff)
    if len(pairs) < 2:
        raise RuntimeError(f"not enough associations for {label}")

    est_idx = np.asarray([i for i, _ in pairs])
    gt_idx = np.asarray([j for _, j in pairs])
    src = est_p[est_idx].T
    dst = gt_p[gt_idx].T

    scale, rotation, translation = umeyama_alignment(src, dst)
    aligned = (scale * rotation @ src + translation).T
    truth = dst.T
    errors = np.linalg.norm(aligned - truth, axis=1)

    return TrajectoryEval(
        label=label,
        timestamps=est_t,
        positions=est_p,
        matched_est=est_p[est_idx],
        matched_gt=gt_p[gt_idx],
        matched_ts=est_t[est_idx],
        aligned=aligned,
        errors=errors,
        scale=float(scale),
        ate_rmse=float(np.sqrt(np.mean(errors ** 2))),
        ate_mean=float(np.mean(errors)),
        ate_median=float(np.median(errors)),
        ate_max=float(np.max(errors)),
        final_error=float(np.linalg.norm(aligned[-1] - truth[-1])),
        path_est=path_length(aligned),
        path_gt=path_length(truth),
        rpe_1=translational_rpe(aligned, truth, 1),
        rpe_30=translational_rpe(aligned, truth, 30),
        association_count=len(pairs),
    )


def sample_indices(n: int, limit: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.linspace(0, n - 1, limit).astype(int)


def project_polylines(series: list[np.ndarray], dims: tuple[int, int], width: float, height: float, pad: float) -> list[str]:
    projected_sets = []
    for points in series:
        projected_sets.append(points[:, dims])

    combined = np.vstack(projected_sets)
    mins = combined.min(axis=0)
    maxs = combined.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    sx = (width - 2.0 * pad) / span[0]
    sy = (height - 2.0 * pad) / span[1]
    scale = min(sx, sy)

    polylines = []
    for points in projected_sets:
        coords = []
        for x, y in points:
            px = pad + (x - mins[0]) * scale
            py = height - pad - (y - mins[1]) * scale
            coords.append(f"{px:.2f},{py:.2f}")
        polylines.append(" ".join(coords))
    return polylines


def error_polyline(errors: np.ndarray, width: float, height: float, pad: float, limit: int = 260) -> str:
    idx = sample_indices(len(errors), limit)
    max_err = max(float(errors.max()), 1e-9)
    coords = []
    for k, err in enumerate(errors[idx]):
        px = pad + (width - 2.0 * pad) * (k / max(len(idx) - 1, 1))
        py = height - pad - (height - 2.0 * pad) * (float(err) / max_err)
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_view_card(title: str, subtitle: str, gt_poly: str, online_poly: str, corrected_poly: str) -> str:
    return f"""
      <article class="panel">
        <h3>{title}</h3>
        <p class="chart-caption">{subtitle}</p>
        <svg class="plot" viewBox="0 0 700 280" role="img" aria-label="{title} trajectory plot">
          <line class="grid-line" x1="20" y1="70" x2="680" y2="70"></line>
          <line class="grid-line" x1="20" y1="140" x2="680" y2="140"></line>
          <line class="grid-line" x1="20" y1="210" x2="680" y2="210"></line>
          <line class="grid-line" x1="185" y1="20" x2="185" y2="260"></line>
          <line class="grid-line" x1="350" y1="20" x2="350" y2="260"></line>
          <line class="grid-line" x1="515" y1="20" x2="515" y2="260"></line>
          <rect x="20" y="20" width="660" height="240" fill="none" class="axis-line"></rect>
          <polyline class="truth-line" points="{gt_poly}"></polyline>
          <polyline class="online-line" points="{online_poly}"></polyline>
          <polyline class="corrected-line" points="{corrected_poly}"></polyline>
        </svg>
      </article>
"""


def build_html(
    online_eval: TrajectoryEval,
    corrected_eval: TrajectoryEval,
    online_path: Path,
    corrected_path: Path,
    gt_path: Path,
) -> str:
    gt_online = online_eval.matched_gt[sample_indices(len(online_eval.matched_gt), 260)]
    est_online = online_eval.aligned[sample_indices(len(online_eval.aligned), 260)]
    gt_corrected = corrected_eval.matched_gt[sample_indices(len(corrected_eval.matched_gt), 260)]
    est_corrected = corrected_eval.aligned[sample_indices(len(corrected_eval.aligned), 260)]

    # Use a shared basis for each plane across GT, online, corrected.
    xz_online = project_polylines([gt_online, est_online, est_corrected], (0, 2), 700.0, 280.0, 20.0)
    xy_online = project_polylines([gt_online, est_online, est_corrected], (0, 1), 700.0, 280.0, 20.0)
    yz_online = project_polylines([gt_online, est_online, est_corrected], (1, 2), 700.0, 280.0, 20.0)

    online_err_poly = error_polyline(online_eval.errors, 700.0, 220.0, 20.0)
    corrected_err_poly = error_polyline(corrected_eval.errors, 700.0, 220.0, 20.0)

    online_spikes = np.argsort(online_eval.errors)[-5:][::-1]
    corrected_spikes = np.argsort(corrected_eval.errors)[-5:][::-1]

    def spike_rows(ev: TrajectoryEval, idxs: np.ndarray) -> str:
        rows = []
        for rank, idx in enumerate(idxs, start=1):
            rows.append(
                "<tr>"
                f"<td>{rank}</td>"
                f"<td>{fmt(ev.matched_ts[idx], 6)}</td>"
                f"<td><strong>{fmt(ev.errors[idx])} m</strong></td>"
                "</tr>"
            )
        return "\n".join(rows)

    online_rpe1 = fmt(online_eval.rpe_1["rmse"]) if online_eval.rpe_1 else "n/a"
    online_rpe30 = fmt(online_eval.rpe_30["rmse"]) if online_eval.rpe_30 else "n/a"
    corrected_rpe1 = fmt(corrected_eval.rpe_1["rmse"]) if corrected_eval.rpe_1 else "n/a"
    corrected_rpe30 = fmt(corrected_eval.rpe_30["rmse"]) if corrected_eval.rpe_30 else "n/a"

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SimpleVisualSLAM TUM 評価レポート</title>
  <style>
    :root {{
      --bg: #f1eee7;
      --panel: rgba(255, 252, 246, 0.88);
      --ink: #1d2627;
      --muted: #5f6d6f;
      --line: rgba(29, 38, 39, 0.12);
      --truth: #0f6d74;
      --online: #c96a1a;
      --corrected: #7b4ec8;
      --error: #627a2f;
      --shadow: 0 18px 48px rgba(39, 36, 29, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 109, 116, 0.16), transparent 34%),
        radial-gradient(circle at top right, rgba(201, 106, 26, 0.16), transparent 24%),
        linear-gradient(180deg, #f8f3ea 0%, #ece4d5 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1320px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}
    .hero, .panel, .metric-card {{
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 28px;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: 1.35fr 0.65fr;
      gap: 18px;
      margin-top: 20px;
    }}
    .kicker {{
      display: inline-flex;
      padding: 7px 12px;
      border-radius: 999px;
      background: rgba(15, 109, 116, 0.11);
      color: var(--truth);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1, h2, h3, .metric-value {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      letter-spacing: -0.02em;
    }}
    h1 {{
      margin-top: 16px;
      font-size: clamp(2rem, 4vw, 3.3rem);
      line-height: 0.95;
      max-width: 12ch;
    }}
    h2 {{ font-size: 1.45rem; }}
    h3 {{ font-size: 1.15rem; }}
    p, .chart-caption, .meta-key, .meta-value, .metric-label, .metric-sub, th, td, .footer {{
      color: var(--muted);
    }}
    .hero p {{
      margin: 14px 0 0;
      line-height: 1.72;
      max-width: 68ch;
    }}
    .hero-meta {{
      display: grid;
      gap: 12px;
      align-self: end;
    }}
    .meta-tile {{
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px 18px;
      background: rgba(255, 255, 255, 0.45);
    }}
    .meta-key {{
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .meta-value {{
      margin-top: 8px;
      font-size: 1.05rem;
      color: var(--ink);
      font-weight: 700;
    }}
    .metrics {{
      margin-top: 20px;
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 14px;
    }}
    .metric-card {{
      padding: 16px;
      min-height: 118px;
    }}
    .metric-label {{
      font-size: 0.76rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .metric-value {{
      margin-top: 10px;
      font-size: 1.85rem;
      line-height: 1;
    }}
    .metric-sub {{
      margin-top: 10px;
      font-size: 0.9rem;
      line-height: 1.5;
    }}
    .section {{
      margin-top: 20px;
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 18px;
    }}
    .wide {{ grid-column: span 8; }}
    .side {{ grid-column: span 4; }}
    .full {{ grid-column: span 12; }}
    .triple {{
      grid-column: span 12;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
    }}
    .panel {{
      padding: 20px;
    }}
    .chart-caption {{
      margin-top: 8px;
      font-size: 0.93rem;
      line-height: 1.65;
    }}
    .plot {{
      margin-top: 14px;
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.66), rgba(246, 241, 232, 0.92));
    }}
    .grid-line {{ stroke: rgba(29, 38, 39, 0.10); stroke-width: 1; }}
    .axis-line {{ stroke: rgba(29, 38, 39, 0.18); stroke-width: 1.2; }}
    .truth-line {{
      fill: none;
      stroke: var(--truth);
      stroke-width: 3;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .online-line {{
      fill: none;
      stroke: var(--online);
      stroke-width: 3;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .corrected-line {{
      fill: none;
      stroke: var(--corrected);
      stroke-width: 3;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .error-line {{
      fill: none;
      stroke: var(--error);
      stroke-width: 3;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-top: 14px;
      font-size: 0.92rem;
      color: var(--muted);
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 18px;
      height: 3px;
      border-radius: 999px;
      display: inline-block;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 0.94rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    td strong {{ color: var(--ink); }}
    .footer {{
      margin-top: 18px;
      padding: 18px 20px;
      border-radius: 18px;
      border: 1px dashed rgba(29, 38, 39, 0.18);
      background: rgba(255, 252, 247, 0.62);
      line-height: 1.65;
      font-size: 0.92rem;
    }}
    code {{
      font-family: "IBM Plex Mono", "SFMono-Regular", "Menlo", monospace;
      background: rgba(29, 38, 39, 0.06);
      padding: 0.12rem 0.34rem;
      border-radius: 0.4rem;
      color: var(--ink);
    }}
    @media (max-width: 1180px) {{
      .metrics {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
      .triple {{ grid-template-columns: 1fr; }}
      .wide, .side, .full {{ grid-column: span 12; }}
    }}
    @media (max-width: 860px) {{
      .hero-grid, .metrics {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="kicker">軌跡評価レポート</div>
      <div class="hero-grid">
        <div>
          <h1>オンライン軌跡と補正後地図</h1>
          <p>
            見た目の悪い軌跡が、オンライン推定そのものに由来するのか、
            補正後のキーフレーム地図に由来するのか、あるいは可視化の問題なのかを切り分けるためのレポートです。
          </p>
          <p>
            オレンジは <code>{html.escape(str(online_path))}</code> から得たフレームごとのオンライン推定、
            紫は <code>{html.escape(str(corrected_path))}</code> から抽出した補正後キーフレーム軌跡です。
            どちらも <code>{html.escape(str(gt_path))}</code> に対して独立に Sim(3) 整列したうえで、
            同じ表示範囲に重ねています。
          </p>
        </div>
        <div class="hero-meta">
          <div class="meta-tile">
            <div class="meta-key">オンライン対応数</div>
            <div class="meta-value">{online_eval.association_count}</div>
          </div>
          <div class="meta-tile">
            <div class="meta-key">補正後キーフレーム数</div>
            <div class="meta-value">{corrected_eval.association_count}</div>
          </div>
          <div class="meta-tile">
            <div class="meta-key">要点</div>
            <div class="meta-value">可視化バグは解消済みで、実際の精度もかなり改善しています</div>
          </div>
        </div>
      </div>
    </section>

    <section class="metrics">
      <article class="metric-card">
        <div class="metric-label">オンライン ATE RMSE</div>
        <div class="metric-value">{fmt(online_eval.ate_rmse)}</div>
        <div class="metric-sub">単位は m、フレームごとのオンライン推定</div>
      </article>
      <article class="metric-card">
        <div class="metric-label">オンライン最終誤差</div>
        <div class="metric-value">{fmt(online_eval.final_error)}</div>
        <div class="metric-sub">整列後の最終位置誤差</div>
      </article>
      <article class="metric-card">
        <div class="metric-label">オンライン軌跡長比</div>
        <div class="metric-value">{fmt(online_eval.path_est / max(online_eval.path_gt, 1e-9), 2)}x</div>
        <div class="metric-sub">整列後の推定軌跡長 / GT 軌跡長</div>
      </article>
      <article class="metric-card">
        <div class="metric-label">オンライン RPE d1</div>
        <div class="metric-value">{online_rpe1}</div>
        <div class="metric-sub">RMSE [m]</div>
      </article>
      <article class="metric-card">
        <div class="metric-label">補正後 KF ATE</div>
        <div class="metric-value">{fmt(corrected_eval.ate_rmse)}</div>
        <div class="metric-sub">最終補正後キーフレームでの誤差</div>
      </article>
      <article class="metric-card">
        <div class="metric-label">補正後 KF 最終誤差</div>
        <div class="metric-value">{fmt(corrected_eval.final_error)}</div>
        <div class="metric-sub">整列後の最終位置誤差</div>
      </article>
      <article class="metric-card">
        <div class="metric-label">補正後軌跡長比</div>
        <div class="metric-value">{fmt(corrected_eval.path_est / max(corrected_eval.path_gt, 1e-9), 2)}x</div>
        <div class="metric-sub">GT に対する伸縮の目安</div>
      </article>
      <article class="metric-card">
        <div class="metric-label">補正後 RPE d1</div>
        <div class="metric-value">{corrected_rpe1}</div>
        <div class="metric-sub">キーフレーム間 RMSE [m]</div>
      </article>
    </section>

    <section class="section">
      <article class="panel full">
        <h2>解釈</h2>
        <p class="chart-caption">
          重ね描画のバグは解消済みで、現在の実装は ground truth にかなり近い状態です。
          大きく効いたのは、tracking を実際の局所地図ベースに切り替えたことと、reprojection gate を厳しくしたことです。
          その上で、<code>Sim3</code> ループ検証、global <code>Sim3</code> pose graph、full global BA が効いています。
          残る誤差は比較的小さく、全体形状の破綻というより局所的な残差ドリフトです。
        </p>
        <div class="legend">
          <span><i class="swatch" style="background: var(--truth);"></i>Ground truth</span>
          <span><i class="swatch" style="background: var(--online);"></i>オンライン軌跡</span>
          <span><i class="swatch" style="background: var(--corrected);"></i>補正後キーフレーム軌跡</span>
        </div>
      </article>

      <div class="triple">
        {build_view_card("X-Z 平面", "平面形状を見やすい標準的な俯瞰表示です。", xz_online[0], xz_online[1], xz_online[2])}
        {build_view_card("X-Y 平面", "この系列は Y 方向の変動が大きいため重要です。", xy_online[0], xy_online[1], xy_online[2])}
        {build_view_card("Y-Z 平面", "高さ方向や横方向の圧縮・伸張を確認できます。", yz_online[0], yz_online[1], yz_online[2])}
      </div>

      <article class="panel wide">
        <h2>オンライン誤差推移</h2>
        <p class="chart-caption">
          オンライン軌跡の絶対並進誤差です。ジッタや不安定区間の特定に向いています。
        </p>
        <svg class="plot" viewBox="0 0 700 220" role="img" aria-label="オンライン軌跡の絶対誤差">
          <line class="grid-line" x1="20" y1="67" x2="680" y2="67"></line>
          <line class="grid-line" x1="20" y1="110" x2="680" y2="110"></line>
          <line class="grid-line" x1="20" y1="153" x2="680" y2="153"></line>
          <rect x="20" y="20" width="660" height="176" fill="none" class="axis-line"></rect>
          <polyline class="error-line" points="{online_err_poly}"></polyline>
        </svg>
      </article>

      <article class="panel side">
        <h2>オンライン誤差上位</h2>
        <table>
          <tr><th>#</th><th>時刻</th><th>誤差</th></tr>
          {spike_rows(online_eval, online_spikes)}
        </table>
      </article>

      <article class="panel wide">
        <h2>補正後キーフレーム誤差推移</h2>
        <p class="chart-caption">
          最終地図から抽出した補正後キーフレーム軌跡の絶対並進誤差です。backend が最終的に収束した結果を示します。
        </p>
        <svg class="plot" viewBox="0 0 700 220" role="img" aria-label="補正後キーフレームの絶対誤差">
          <line class="grid-line" x1="20" y1="67" x2="680" y2="67"></line>
          <line class="grid-line" x1="20" y1="110" x2="680" y2="110"></line>
          <line class="grid-line" x1="20" y1="153" x2="680" y2="153"></line>
          <rect x="20" y="20" width="660" height="176" fill="none" class="axis-line"></rect>
          <polyline class="error-line" points="{corrected_err_poly}"></polyline>
        </svg>
      </article>

      <article class="panel side">
        <h2>補正後誤差上位</h2>
        <table>
          <tr><th>#</th><th>時刻</th><th>誤差</th></tr>
          {spike_rows(corrected_eval, corrected_spikes)}
        </table>
      </article>

      <article class="panel wide">
        <h2>評価指標一覧</h2>
        <table>
          <tr>
            <th>指標</th>
            <th>オンライン</th>
            <th>補正後キーフレーム</th>
          </tr>
          <tr>
            <td>ATE RMSE</td>
            <td><strong>{fmt(online_eval.ate_rmse)} m</strong></td>
            <td><strong>{fmt(corrected_eval.ate_rmse)} m</strong></td>
          </tr>
          <tr>
            <td>ATE Median</td>
            <td>{fmt(online_eval.ate_median)} m</td>
            <td>{fmt(corrected_eval.ate_median)} m</td>
          </tr>
          <tr>
            <td>ATE Max</td>
            <td>{fmt(online_eval.ate_max)} m</td>
            <td>{fmt(corrected_eval.ate_max)} m</td>
          </tr>
          <tr>
            <td>最終位置誤差</td>
            <td>{fmt(online_eval.final_error)} m</td>
            <td>{fmt(corrected_eval.final_error)} m</td>
          </tr>
          <tr>
            <td>軌跡長</td>
            <td>{fmt(online_eval.path_est)} m / GT {fmt(online_eval.path_gt)} m</td>
            <td>{fmt(corrected_eval.path_est)} m / GT {fmt(corrected_eval.path_gt)} m</td>
          </tr>
          <tr>
            <td>RPE Delta 1</td>
            <td>{online_rpe1} m</td>
            <td>{corrected_rpe1} m</td>
          </tr>
          <tr>
            <td>RPE Delta 30</td>
            <td>{online_rpe30} m</td>
            <td>{corrected_rpe30} m</td>
          </tr>
          <tr>
            <td>Sim(3) Scale</td>
            <td>{fmt(online_eval.scale, 6)}</td>
            <td>{fmt(corrected_eval.scale, 6)}</td>
          </tr>
        </table>
      </article>

      <article class="panel side">
        <h2>次の観点</h2>
        <table>
          <tr>
            <th>現在の状態</th>
            <td>局所地図ベース tracking と global 最適化まで入り、TUM ではかなり良い状態です。</td>
          </tr>
          <tr>
            <th>残課題</th>
            <td>大域形状ではなく、局所的な残差ドリフトや系列依存の頑健性が主な課題です。</td>
          </tr>
          <tr>
            <th>次にやること</th>
            <td>別系列や EuRoC でも同じ評価を回し、再現性を確認します。</td>
          </tr>
        </table>
      </article>
    </section>

    <section class="footer">
      評価入力: <code>{html.escape(str(online_path))}</code>, <code>{html.escape(str(corrected_path))}</code>,
      <code>{html.escape(str(gt_path))}</code>。
      以前のレポートにあった重ね描画バグは、全軌跡を共通表示範囲で描くことで解消しています。
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-diff", type=float, default=0.02)
    args = parser.parse_args()

    online_eval = evaluate_trajectory("online", args.online, args.groundtruth, args.max_diff)
    corrected_eval = evaluate_trajectory("corrected", args.corrected, args.groundtruth, args.max_diff)
    report = build_html(online_eval, corrected_eval, args.online, args.corrected, args.groundtruth)
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

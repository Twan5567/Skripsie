"""
  python SlamCompareTest.py --boundary ../output/TrackMap_S3/TrackMap_S3_boundary.csv \\
      --map "../../../SLAM/Test 3 Slam/my_map_for_real.yaml" --out ../output/TrackMap_S3_SlamTest \\
      [--dense-dir ../output/TrackMap_S3]

"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree


def load_map(yaml_path: str, min_blob: int = 4) -> tuple[np.ndarray, dict]:
    
    meta = {}
    with open(yaml_path) as fh:
        for ln in fh:
            if ":" in ln:
                k, v = ln.split(":", 1)
                meta[k.strip()] = v.strip()
    res = float(meta["resolution"])
    ox, oy = [float(x) for x in meta["origin"].strip("[]").split(",")[:2]]
    img = cv2.imread(os.path.join(os.path.dirname(yaml_path), meta["image"]), cv2.IMREAD_GRAYSCALE)
    occ = (img < 0.35 * 255).astype(np.uint8)          # trinary: black = occupied
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_blob
    rows, cols = np.nonzero(keep[lab])
    H = img.shape[0]
    pts = np.c_[ox + (cols + 0.5) * res, oy + (H - 1 - rows + 0.5) * res]
    return pts, {"resolution": res, "origin": [ox, oy], "shape": list(img.shape),
                 "occupied_cells": int(occ.sum()), "kept_cells": len(pts)}


def load_boundary(path: str):
    
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            m = r.get("measured")
            meas = (m == "1") if m is not None else (r.get("source") == "measured")
            out.setdefault(r["loop"], []).append((float(r["x_mm"]) / 1000, float(r["y_mm"]) / 1000, meas))
    # Walls only: boundary.csv also carries the track midline, which is not a wall.
    return {k: (np.array([(x, y) for x, y, _ in v]), np.array([m for *_, m in v]))
            for k, v in out.items() if k in ("outer", "inner")}


def fit(src, dst, scale):

    ms, md = src.mean(0), dst.mean(0)
    P, Q = src - ms, dst - md
    U, S, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = (S * np.diag(D)).sum() / (P ** 2).sum() if scale else 1.0
    return s, R, md - s * R @ ms


def icp(src, tree, dst, R0, t0, scale, iters=60, trim=0.9):

    s, R, t = 1.0, R0, t0
    for _ in range(iters):
        cur = s * src @ R.T + t
        d, j = tree.query(cur)
        keep = d <= np.quantile(d, trim)
        s, R, t = fit(src[keep], dst[j[keep]], scale)
    d, _ = tree.query(s * src @ R.T + t)
    return s, R, t, d


def main():


    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dense-dir", default=None,
                    help="output/<NAME>/ of the model: also write the dense cloud with the SLAM walls in 3D")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.boundary))[0]

    slam, meta = load_map(args.map)
    tree = cKDTree(slam)
    loops = load_boundary(args.boundary)
    src = np.vstack([p[m] for p, m in loops.values()])
    print(f"SLAM map {os.path.basename(args.map)}: {meta['kept_cells']} wall cells "
          f"({meta['resolution'] * 100:.0f} cm grid)")
    print(f"boundary {os.path.basename(args.boundary)}: " + ", ".join(
        f"{k} {m.sum()} measured + {(~m).sum()} bridged" for k, (p, m) in loops.items()))

    
    best = None
    for mirror in (False, True):
        s0 = src * ([1, -1] if mirror else [1, 1])
        for a in np.radians(np.arange(0, 360, 5)):
            R0 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            t0 = slam.mean(0) - R0 @ s0.mean(0)
            s, R, t, d = icp(s0, tree, slam, R0, t0, False, iters=15)
            score = np.median(d)
            if best is None or score < best[0]:
                best = (score, mirror, R, t)
    _, mirror, R, t = best
    M = np.array([1, -1]) if mirror else np.array([1, 1])
    s0 = src * M
    _, Rr, tr, dr = icp(s0, tree, slam, R, t, False)
    ss, Rs, ts, ds = icp(s0, tree, slam, Rr, tr, True)
    rot = np.degrees(np.arctan2(Rr[1, 0], Rr[0, 0]))

    def stats(d):
        return {"median_mm": 1000 * float(np.median(d)), "rms_mm": 1000 * float(np.sqrt((d ** 2).mean())),
                "p95_mm": 1000 * float(np.percentile(d, 95)), "max_mm": 1000 * float(d.max()), "n": int(len(d))}

    rep = {"boundary": args.boundary, "map": args.map, "map_meta": meta, "mirrored": bool(mirror),
           "rotation_deg": float(rot),
           
           "transform": {"M": M.tolist(), "R": Rr.tolist(), "t": tr.tolist()},
           "rigid": stats(dr), "similarity": stats(ds),
           "similarity_scale": float(ss), "per_loop": {}}
    print(f"\nalignment: {'MIRRORED, ' if mirror else ''}rotation {rot:+.1f} deg")
    print(f"rigid fit      : boundary -> nearest SLAM wall cell  median {rep['rigid']['median_mm']:.0f} mm, "
          f"rms {rep['rigid']['rms_mm']:.0f}, p95 {rep['rigid']['p95_mm']:.0f}, max {rep['rigid']['max_mm']:.0f} mm")
    print(f"similarity fit : scale camera map / SLAM = {ss:.4f} ({100 * (ss - 1):+.2f}%), "
          f"median {rep['similarity']['median_mm']:.0f} mm, p95 {rep['similarity']['p95_mm']:.0f} mm")
    for k, (p, m) in loops.items():
        d, _ = tree.query((p * M) @ Rr.T + tr)
        rep["per_loop"][k] = {"measured": stats(d[m])}
        line = (f"  {k:6s} measured: median {1000 * np.median(d[m]):4.0f} mm, p95 "
                f"{1000 * np.percentile(d[m], 95):4.0f} mm  (n {m.sum()})")
        if (~m).any():
            rep["per_loop"][k]["bridged"] = stats(d[~m])
            line += (f"   bridged: median {1000 * np.median(d[~m]):4.0f} mm, p95 "
                     f"{1000 * np.percentile(d[~m], 95):4.0f} mm  (n {(~m).sum()})")
        print(line)
    with open(os.path.join(args.out, f"{name}_vs_slam.json"), "w") as fh:
        json.dump(rep, fh, indent=1)

    # overlay: SLAM walls grey, boundary coloured by distance (measured) / blue (bridged)
    allp = np.vstack([slam] + [(p * M) @ Rr.T + tr for p, _ in loops.values()])
    lo, hi = allp.min(0) - 0.3, allp.max(0) + 0.3
    px = 200                                            # pixels per metre
    W, Hh = (np.ceil((hi - lo) * px)).astype(int)
    img = np.full((Hh, W, 3), 255, np.uint8)
    to_px = lambda q: np.c_[(q[:, 0] - lo[0]) * px, (hi[1] - q[:, 1]) * px].astype(int)
    for x, y in to_px(slam):
        cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (175, 175, 175), -1)
    for p, m in loops.values():
        q = (p * M) @ Rr.T + tr
        d, _ = tree.query(q)
        for (x, y), di, mi in zip(to_px(q), d, m):
            if not mi:
                col = (255, 120, 0)
            else:
                f = min(di / 0.15, 1.0)                    # green 0 -> red 150 mm
                col = (0, int(200 * (1 - f)), int(255 * f))
            cv2.circle(img, (x, y), 3, col, -1)
    cv2.putText(img, f"grey: SLAM walls ({meta['resolution'] * 100:.0f} cm)   boundary: green 0 -> red >=150 mm,"
                f" blue = bridged", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"rigid: median {rep['rigid']['median_mm']:.0f} mm, p95 {rep['rigid']['p95_mm']:.0f} mm;"
                f"  scale {ss:.4f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    out_png = os.path.join(args.out, f"{name}_vs_slam.png")
    cv2.imwrite(out_png, img)
    print(f"\nwrote {out_png}")
    if args.dense_dir:
        write_dense_overlay(args.dense_dir, args.out, slam, M, Rr, tr, meta["resolution"], loops)


def write_dense_overlay(dense_dir, out, slam, M, R, t, res_m, loops):
    
    import sys
    from pathlib import Path
    import open3d as o3d
    from matplotlib.path import Path as MPath
    from scipy.spatial import ConvexHull
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from boundary import load_metric, resample

    O = Path(dense_dir)
    name = O.name
    _, P, cams = load_metric(str(O / f"{name}_dense.ply"), str(O / "sparse"), str(O / f"{name}_metric.json"))
    b = json.loads((O / f"{name}_boundary.json").read_text())
    inside = MPath(cams[ConvexHull(cams[:, :2]).vertices, :2]).contains_points(P[:, :2])
    RGB = np.asarray(o3d.io.read_point_cloud(str(O / f"{name}_dense.ply")).colors)[inside]
    P = P[inside]
    P[:, 2] -= b["floor"]["offset_mm"]                    # floor at z = 0, as boundary.py
    wall_top = float(b["wall_top_mm"])
    flip = np.array([1.0, -1.0, 1.0])                     # same view convention as the other _view.ply files
    keep = (P[:, 2] > -300) & (P[:, 2] < 2500)
    dense = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[keep] * flip))
    dense.colors = o3d.utility.Vector3dVector(RGB[keep])
    dense = dense.voxel_down_sample(4.0)

    # SLAM frame -> boundary world frame (mm): p = M R^T (q - t)
    xy = ((slam - t) @ R) * M * 1000.0
    half = res_m * 1000.0 / 2
    g = np.linspace(-half, half, 6)
    gx, gy = np.meshgrid(g, g)
    off = np.c_[gx.ravel(), gy.ravel()]
    hs = np.arange(0.0, wall_top + 1, 40.0)
    pts = (xy[:, None, None, :] + off[None, None, :, :]) * np.ones((1, len(hs), 1, 1))
    z = np.broadcast_to(hs[None, :, None, None], pts.shape[:3] + (1,))
    B = np.concatenate([pts, z], axis=-1).reshape(-1, 3)
    blocks = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(B * flip))
    blocks.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 1.0], (len(B), 1)))
    # the extracted boundary (world frame, mm): red lines at three heights
    L = []
    for p_m, _ in loops.values():
        q = resample(p_m * 1000.0, 2.0)
        for h in (0.0, wall_top / 2, wall_top):
            L.append(np.c_[q, np.full(len(q), h)])
    L = np.vstack(L)
    lines = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(L * flip))
    lines.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(L), 1)))
    path = os.path.join(out, f"{name}_dense_with_slam_view.ply")
    o3d.io.write_point_cloud(path, dense + blocks + lines)
    print(f"wrote {path} ({len(dense.points):,} dense + {len(B):,} SLAM-wall (magenta) "
          f"+ {len(L):,} boundary (red) points)")


if __name__ == "__main__":
    main()

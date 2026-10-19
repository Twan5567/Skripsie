"""
  python metric_scale.py --sparse output/<NAME>/sparse --images <frames> \
      --out output/<NAME>/<NAME>_metric.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.optimize import least_squares

# one detector definition for both pipelines
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline_B_tracking" / "src"))
from detector import build_detector  # noqa: E402

GCP_IDS = set(range(12))

_det = None


def _detect(path: str):
    
    global _det

    if _det is None:
        _det = build_detector()

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    c, ids, _ = _det.detectMarkers(gray)
    out = {}

    if ids is not None:

        for n, k in enumerate(np.array(ids).flatten()):

            if int(k) in GCP_IDS:

                out[int(k)] = np.array(c[n]).reshape(4, 2).astype(np.float64)


    return os.path.basename(path), out


def _dlt(Ps: list[np.ndarray], xs: list[np.ndarray]) -> np.ndarray:
   
    A = []
    for P, x in zip(Ps, xs):

        A.append(x[0] * P[2] - P[0])
        A.append(x[1] * P[2] - P[1])

    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def triangulate(obs, thresh_px: float, min_views: int, rng):
  
    n = len(obs)

    if n < min_views:
        return None

    Ps = np.array([o[0] for o in obs])
    pxs = np.array([o[2] for o in obs])
    cam = obs[0][3]                                

    def reproj(X, idx):

        idx = np.asarray(list(idx))
        Xc = Ps[idx, :, :3] @ X + Ps[idx, :, 3]
        e = np.full(len(idx), 1e6)
        ok = Xc[:, 2] > 0

        if ok.any():
            e[ok] = np.linalg.norm(cam.img_from_cam(Xc[ok]) - pxs[idx[ok]], axis=1)
        return e

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]


    if len(pairs) > 300:

        pairs = [pairs[k] for k in rng.choice(len(pairs), 300, replace=False)]


    best = None

    for i, j in pairs:

        X = _dlt([obs[i][0], obs[j][0]], [obs[i][1], obs[j][1]])
        inl = np.flatnonzero(reproj(X, range(n)) < thresh_px)

        if best is None or len(inl) > len(best):
            best = inl

    if best is None or len(best) < min_views:
        return None

    
    X = _dlt([obs[i][0] for i in best], [obs[i][1] for i in best])


    def resid(X):

        Xc = Ps[best, :, :3] @ X + Ps[best, :, 3]
        return (cam.img_from_cam(Xc) - pxs[best]).ravel()

    X = least_squares(resid, X, method="lm").x
    inl = np.flatnonzero(reproj(X, range(n)) < thresh_px)

    if len(inl) < min_views:
        return None

    
    rms = float(np.sqrt(np.mean(reproj(X, inl) ** 2)))
    return X, inl, rms


def sheet_lengths(c: np.ndarray) -> np.ndarray:

    """4 sides then 2 diagonals of a 4x3 corner array."""
    return np.array([np.linalg.norm(c[(j + 1) % 4] - c[j]) for j in range(4)] +
                    [np.linalg.norm(c[2] - c[0]), np.linalg.norm(c[3] - c[1])])


def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True, help="<NAME>_metric.json")
    ap.add_argument("--gcp-mm", type=float, default=150.0)
    ap.add_argument("--origin", type=int, default=1, help="GCP at the world origin")
    ap.add_argument("--x-to", type=int, default=5, help="GCP the +x axis points to")
    ap.add_argument("--thresh-px", type=float, default=2.0)
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8,
                    help="each holds one decoded 4K frame; 8 is ~1 GB")
    args = ap.parse_args()

    rec = pycolmap.Reconstruction(args.sparse)
    by_name = {im.name: im for im in rec.images.values()}
    paths = [os.path.join(args.images, n) for n in sorted(by_name)
             if os.path.exists(os.path.join(args.images, n))]

    
    if len(paths) < len(by_name):
        raise SystemExit(f"{len(by_name) - len(paths)} registered images missing from {args.images}")
    
    print(f"model: {len(by_name)} registered images, {len(rec.points3D):,} points")

    # --- 1. detect -----------------------------------------------------------

    pool = Pool(args.workers)
    dets = []
    for k, d in enumerate(pool.imap(_detect, paths, chunksize=4), 1):

        dets.append(d)

        if k % 50 == 0:
            print(f"  detect {k}/{len(paths)}", end="\r", flush=True)

    pool.close()
    pool.join()
    seen = {}
    for _, d in dets:

        for k in d:

            seen[k] = seen.get(k, 0) + 1

    print(f"  detect {len(paths)} frames; GCP sightings: "
          + ", ".join(f"{k}:{seen[k]}" for k in sorted(seen)))

    # --- 2. triangulate every corner -----------------------------------------

    obs: dict[tuple[int, int], list] = {}

    for name, d in dets:
        im = by_name[name]
        cam = im.camera
        P = im.cam_from_world().matrix() if callable(im.cam_from_world) else im.cam_from_world.matrix()

        for k, c in d.items():

            xn = cam.cam_from_img(c)

            for j in range(4):
                obs.setdefault((k, j), []).append((P, xn[j], c[j], cam))

    rng = np.random.default_rng(0)
    corners: dict[int, dict[int, np.ndarray]] = {}
    corner_info = []

    for (k, j) in sorted(obs):
        r = triangulate(obs[(k, j)], args.thresh_px, args.min_views, rng)
        if r is None:
            corner_info.append(dict(id=k, corner=j, views=len(obs[(k, j)]), inliers=0))
            continue
        X, inl, rms = r
        corners.setdefault(k, {})[j] = X
        corner_info.append(dict(id=k, corner=j, views=len(obs[(k, j)]),
                                inliers=int(len(inl)), rms_px=round(rms, 3), X_sfm=X.tolist()))
    sheets = {k: np.array([v[j] for j in range(4)]) for k, v in corners.items() if len(v) == 4}

    print(f"\ntriangulated {sum(len(v) for v in corners.values())} corners; "
          f"complete sheets: {sorted(sheets)}")
    
    for ci in corner_info:
        flag = "" if ci["inliers"] else "   <- FAILED"
        print(f"  GCP {ci['id']:>2} c{ci['corner']}  views {ci['views']:>4}  inliers {ci['inliers']:>4}"
              + (f"  rms {ci['rms_px']:.2f} px" if ci["inliers"] else "") + flag)
        
    if len(sheets) < 2:
        raise SystemExit("need at least 2 complete sheets for scale")

    # --- 3. scale -------------------------------------------------------------

    S = args.gcp_mm
    nominal = np.array([S] * 4 + [S * np.sqrt(2)] * 2)
    per_sheet = {k: float(np.median(nominal / sheet_lengths(c))) for k, c in sheets.items()}
    s = float(np.median(list(per_sheet.values())))
    spread = 100 * (max(per_sheet.values()) - min(per_sheet.values())) / s
    print(f"\n{'=' * 68}\nSCALE   {s:.3f} mm per SfM unit   (median of {len(sheets)} sheets)\n{'=' * 68}")
    print(f"{'id':>4} {'scale':>9} {'vs median':>10}   sides+diagonals at the median scale (mm)")

    for k in sorted(per_sheet):
        L = s * sheet_lengths(sheets[k])
        print(f"{k:>4} {per_sheet[k]:>9.3f} {100 * (per_sheet[k] / s - 1):>+9.2f}%   "
              + " ".join(f"{v:6.1f}" for v in L))
        
    print(f"per-sheet scale spread (max-min)/median: {spread:.2f}%")

   
    holdout = {}
    if len(sheets) >= 3:
        print(f"\nHOLD-OUT (scale from the other sheets, error of the held-out sheet's lengths)")

        for k in sorted(sheets):

            s_k = float(np.median([per_sheet[q] for q in per_sheet if q != k]))
            err = s_k * sheet_lengths(sheets[k]) - nominal
            holdout[k] = dict(scale=s_k, err_mm=err.round(2).tolist(),
                              rms_mm=float(np.sqrt(np.mean(err ** 2))),
                              max_mm=float(np.abs(err).max()))
            print(f"  GCP {k:>2}: rms {holdout[k]['rms_mm']:5.2f} mm   max {holdout[k]['max_mm']:5.2f} mm"
                  f"   ({100 * holdout[k]['rms_mm'] / S:.2f}% of {S:.0f} mm)")

    # --- 4. frame ---------------------------------------------------------------

    allc = np.vstack(list(sheets.values())) * s
    mu = allc.mean(axis=0)
    _, _, Vt = np.linalg.svd(allc - mu)
    n = Vt[2]
    Cs = np.array([im.projection_center() for im in rec.images.values()]) * s


    if np.median((Cs - mu) @ n) < 0:
        n = -n                                      
    plane_res = (allc - mu) @ n
   
    e = np.mean([np.cross(c[1] - c[0], c[3] - c[0]) for c in sheets.values()], axis=0)
    track_z = 1.0 if e @ n > 0 else -1.0 

    if args.origin not in sheets or args.x_to not in sheets:
        raise SystemExit(f"world frame needs GCPs {args.origin} and {args.x_to}; have {sorted(sheets)}")

    def inplane(p):
        return p - np.outer((p - mu) @ n, n) if p.ndim == 2 else p - ((p - mu) @ n) * n
    o = inplane(sheets[args.origin].mean(axis=0) * s)
    xh = inplane(sheets[args.x_to].mean(axis=0) * s) - o
    xh /= np.linalg.norm(xh)
    yh = np.cross(track_z * n, xh)
    R = np.stack([xh, yh, n])                     
    t = -R @ o

 
    print(f"\n{'=' * 68}\nWORLD FRAME   origin GCP {args.origin} centre, +x toward GCP {args.x_to}, z up\n{'=' * 68}")
    print(f"floor plane through {len(allc)} corners: rms {np.sqrt(np.mean(plane_res ** 2)):.1f} mm, "
          f"max {np.abs(plane_res).max():.1f} mm")
    print(f"tracker (x, y) seen from above is {'MIRRORED' if track_z < 0 else 'not mirrored'} "
          f"(det R = {np.linalg.det(R):+.0f})")
    cam_h = (Cs - o) @ n
    print(f"camera heights along the walk: median {np.median(cam_h):.0f} mm "
          f"(range {np.percentile(cam_h, 5):.0f}-{np.percentile(cam_h, 95):.0f})")
    print(f"\n{'id':>4} {'centre x':>9} {'centre y':>9} {'z':>7}")
    centres = {}


    for k in sorted(sheets):

        w = (R @ (sheets[k] * s).T).T + t
        centres[k] = w.mean(axis=0)
        print(f"{k:>4} {centres[k][0]:>9.0f} {centres[k][1]:>9.0f} {centres[k][2]:>7.1f}")

    out = dict(
        sparse=os.path.abspath(args.sparse), gcp_mm=S, origin_gcp=args.origin, x_gcp=args.x_to,
        scale_mm_per_unit=s, R=R.tolist(), t=t.tolist(),
        transform="world_mm = R @ (scale * X_sfm) + t;  x,y as the tracker, z = height above floor",
        tracker_frame_mirrored=bool(track_z < 0),
        per_sheet_scale=per_sheet, scale_spread_pct=spread, holdout=holdout,
        plane_rms_mm=float(np.sqrt(np.mean(plane_res ** 2))),
        plane_max_mm=float(np.abs(plane_res).max()),
        sheet_corners_mm={k: ((R @ (c * s).T).T + t).round(2).tolist() for k, c in sheets.items()},
        sheet_centres_mm={k: v.round(1).tolist() for k, v in centres.items()},
        camera_height_median_mm=float(np.median(cam_h)),
        corners=corner_info, gcp_sightings=seen,
    )
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

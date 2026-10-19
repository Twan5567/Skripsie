"""

python calibrate_charuco.py --camera A --out ../calib/S3_camA_linear.json \\
     "../../../Capture Test 3/Linear Cal.MP4" \\
     --check ../capture/GX011490.MP4 ../capture/GX011502.MP4

"""
from gpu import init_opencv_opencl

init_opencv_opencl()

import argparse
import json
import os
import re
import struct
import sys

import cv2
import numpy as np


# --- GoPro metadata ------------------------------------------------------------
def gopro_settings(video: str) -> dict:

    out, size = {}, os.path.getsize(video)
    with open(video, "rb") as fh:

        for off in (0, max(0, size - 8_000_000)):

            fh.seek(off)
            b = fh.read(8_000_000)

            for key in (b"VFOV", b"ZFOV", b"EISE", b"CASN"):

                k = key.decode()
                i = b.find(key)

                if i < 0 or k in out:
                    continue

                t, s = chr(b[i + 4]), b[i + 5]
                n = struct.unpack(">H", b[i + 6:i + 8])[0]
                d = b[i + 8:i + 8 + s * n]
                out[k] = round(struct.unpack(">f", d[:4])[0], 3) if t == "f" \
                    else d.decode("latin1").strip("\x00 ")
                
    if "CASN" not in out:   
        with open(video, "rb") as fh:
            b = fh.read(8_000_000)
        m = re.search(rb"C\d{13}", b)
        if m:
            out["CASN"] = m.group().decode()
    return out


MODES = {"L": "Linear", "W": "Wide", "S": "SuperView", "X": "HyperView", "N": "Narrow"}


# --- detection -----------------------------------------------------------------


def collect_views(video: str, det, board, stride: int, min_corners: int):
   
    cap = cv2.VideoCapture(video)
    W, Hh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    views, sampled, i, frame = [], 0, 0, None

    while cap.grab():

        if i % stride == 0:

            _, fr = cap.retrieve()

            sampled += 1

            if frame is None:

                frame = fr

            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            cc, ci, _, _ = det.detectBoard(g)

            if ci is not None and len(ci) >= min_corners:

                o, p = board.matchImagePoints(cc, ci)
                x, y, w, h = cv2.boundingRect(p.reshape(-1, 2).astype(np.float32))
                sharp = cv2.Laplacian(g[y:y + h, x:x + w], cv2.CV_64F).var()
                views.append((o.astype(np.float32), p.astype(np.float32), float(sharp)))
        i += 1
    cap.release()
    return views, sampled, (W, Hh), frame


def rough_K(W: int, Hh: int, zfov_deg: float | None) -> np.ndarray:
   
    f = np.hypot(W, Hh) / 2 / np.tan(np.radians(zfov_deg or 100.0) / 2)
    return np.array([[f, 0, W / 2], [0, f, Hh / 2], [0, 0, 1.0]])


def board_tilt(o, p, K) -> float:

    ok, rv, _ = cv2.solvePnP(o, p, K, None)
    if not ok:
        return np.nan
    R, _ = cv2.Rodrigues(rv)
    return float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))


def select_views(views, W, Hh, K, max_views: int):

    tilts = [board_tilt(o, p, K) for o, p, _ in views]
    bins = {}
    for j, (o, p, s) in enumerate(views):
        q = p.reshape(-1, 2)
        cx, cy = q.mean(axis=0)
        span = np.ptp(q[:, 0]) / W
        key = (min(3, int(4 * cx / W)), min(3, int(4 * cy / Hh)), span > 0.33,
               0 if tilts[j] < 15 else (1 if tilts[j] < 30 else 2))
        bins.setdefault(key, []).append(j)
    queues = [sorted(b, key=lambda j: -views[j][2]) for b in bins.values()]
    chosen, depth = [], 0
    while len(chosen) < max_views and any(depth < len(q) for q in queues):
        for q in queues:
            
            while depth < len(q) and any(abs(q[depth] - c) < 6 for c in q[:depth] if c in chosen):
                q.pop(depth)


            if depth < len(q) and len(chosen) < max_views:
                chosen.append(q[depth])


        depth += 1
    return sorted(chosen), np.array(tilts)


# --- fit ---------------------------------------------------------------------------
def fit(views, size, flags):

    
    O, P = [v[0] for v in views], [v[1] for v in views]
    rms, K, D, _, _, _, _, per = cv2.calibrateCameraExtended(O, P, size, None, None, flags=flags)
    keep = np.argsort(per.ravel())[:max(10, int(0.9 * len(O)))]
    O, P = [O[j] for j in keep], [P[j] for j in keep]
    rms, K, D, _, _, sd, _, _ = cv2.calibrateCameraExtended(O, P, size, None, None, flags=flags)
    return rms, K, D, sd.ravel(), len(O)


def reproj_errors(views, K, D):
    
    e = []
    for o, p, _ in views:
        ok, rv, tv = cv2.solvePnP(o, p, K, D)
        q, _ = cv2.projectPoints(o, rv, tv, K, D)
        e.append(float(np.sqrt(np.mean(np.sum((q.reshape(-1, 2) - p.reshape(-1, 2)) ** 2, 1)))))
    return np.array(e)


class Tee:

    def __init__(self, path):
        self.f, self.out = open(path, "w"), sys.stdout

    def write(self, s):
        self.f.write(s)
        self.out.write(s)

    def flush(self):
        self.f.flush()
        self.out.flush()


def main() -> None:

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="ChArUco clips to fit, same body + same lens mode")
    ap.add_argument("--camera", required=True, help="a name for this body, e.g. A or B")
    ap.add_argument("--out", required=True, help="JSON to write (.log and .jpg go next to it)")
    ap.add_argument("--check", nargs="*", default=[], help="held-out clips: measured, not fitted")
    ap.add_argument("--squares", default="8x5", help="board squares, columns x rows")
    ap.add_argument("--square-mm", type=float, default=25.0, help="measured square side")
    ap.add_argument("--marker-mm", type=float, default=None,
                    help="marker side (default: square x 18.5/25, as printed)")
    ap.add_argument("--dict", default="DICT_5X5_100")
    ap.add_argument("--stride", type=int, default=5, help="sample every Nth frame")
    ap.add_argument("--min-corners", type=int, default=15, help="of 28 on the A4 board")
    ap.add_argument("--max-views", type=int, default=120)
    ap.add_argument("--model", choices=["standard", "rational"], default="standard",
                    help="standard = k1 k2 p1 p2 k3; rational adds k4-k6 (Wide/fisheye-ish)")
    ap.add_argument("--force", action="store_true", help="fit even if the clips' settings differ")
    args = ap.parse_args()

    base = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    sys.stdout = Tee(base + ".log")

    sx, sy = (int(v) for v in args.squares.lower().split("x"))
    mk = args.marker_mm or args.square_mm * 18.5 / 25.0
    board = cv2.aruco.CharucoBoard((sx, sy), args.square_mm, mk,cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict)))

    det = cv2.aruco.CharucoDetector(board)
    n_corners = (sx - 1) * (sy - 1)
    print(f"board: {sx} x {sy} squares, square {args.square_mm:.2f} mm, marker {mk:.2f} mm, "f"{args.dict}, {n_corners} inner corners")

    # --- settings: one body, one mode ------------------------------------------

    print("\nclip settings (GoPro metadata):")
    settings = {}

    for v in args.videos + args.check:

        s = gopro_settings(v)
        settings[v] = s
        role = "fit  " if v in args.videos else "check"
        print(f"  {role} {os.path.basename(v):18s} serial {s.get('CASN', '?'):15s} "
              f"lens {MODES.get(s.get('VFOV'), s.get('VFOV', '?')):9s} FOV {s.get('ZFOV', '?')}  "
              f"stabilisation {'ON' if s.get('EISE') == 'Y' else 'off' if s.get('EISE') == 'N' else '?'}")
    fit_set = {(settings[v].get("CASN"), settings[v].get("VFOV"), settings[v].get("ZFOV"))
               for v in args.videos}

    
    if len(fit_set) > 1 and not args.force:
        raise SystemExit("the fit clips are not one body in one lens mode (see above); "
                         "split them, or pass --force")

    
    if any(settings[v].get("EISE") == "Y" for v in args.videos):
        print("  WARNING: stabilisation is ON in a fit clip — every frame is warped "
              "differently and no single lens model fits")

        
    ref = settings[args.videos[0]]

    for v in args.check:
        s = settings[v]
        if (s.get("CASN"), s.get("VFOV"), s.get("ZFOV")) != (ref.get("CASN"), ref.get("VFOV"), ref.get("ZFOV")):
            print(f"  note: check clip {os.path.basename(v)} is a different body or mode — "
                  f"expect a large error there")

    # --- detect --------------------------------------------------------------------
    print("\ndetection:")
    views, size, frame = [], None, None

    for v in args.videos:
        vv, sampled, sz, fr = collect_views(v, det, board, args.stride, args.min_corners)

        if size is None:
            size, frame = sz, fr

        elif sz != size:
            raise SystemExit(f"{v} is {sz}, others are {size}")
        
        views += vv

        print(f"  {os.path.basename(v):18s} {sampled} frames sampled, {len(vv)} with "
              f">= {args.min_corners} corners ({100 * len(vv) / max(sampled, 1):.0f}%)")

        
    if len(views) < 15:
        raise SystemExit(f"only {len(views)} usable views; need at least 15")
    W, Hh = size

    K0 = rough_K(W, Hh, ref.get("ZFOV"))
    chosen, tilts = select_views(views, W, Hh, K0, args.max_views)
    sel = [views[j] for j in chosen]

    # --- coverage --------------------------------------------------------------
    pts = np.vstack([v[1].reshape(-1, 2) for v in sel])
    r = np.hypot(pts[:, 0] - W / 2, pts[:, 1] - Hh / 2) / np.hypot(W / 2, Hh / 2)
    grid = np.zeros((3, 3), int)


    for x, y in pts:
        grid[min(2, int(3 * y / Hh)), min(2, int(3 * x / W))] += 1
    t = tilts[chosen]
    print(f"\ncoverage ({len(sel)} views chosen from {len(views)}):")
    print(f"  corners reach x {pts[:, 0].min():.0f}-{pts[:, 0].max():.0f}, "
          f"y {pts[:, 1].min():.0f}-{pts[:, 1].max():.0f} of {W}x{Hh}; "
          f"{100 * r.max():.0f}% of the way to the frame corners")
    print("  corners per 3x3 cell:  " + "   |   ".join(" ".join(f"{c:5d}" for c in row) for row in grid))
    print(f"  board tilt: median {np.nanmedian(t):.0f} deg, {np.sum(t > 25)} views past 25 deg")

    
    if np.sum(t > 25) < 10:
        print("  WARNING: few tilted views — focal length is weakly pinned down "
              "(see the half-split check below); film 30-45 deg tilts")

    # --- fit ---------------------------------------------------------------------

    flags = cv2.CALIB_RATIONAL_MODEL if args.model == "rational" else 0
    rms, K, D, sd, n_used = fit(sel, size, flags)
    print(f"\nfit ({args.model} model, {n_used} views after dropping the worst 10%):")
    print(f"  reprojection rms : {rms:.3f} px")
    print(f"  focal length     : fx {K[0, 0]:.1f} +- {sd[0]:.1f}   fy {K[1, 1]:.1f} +- {sd[1]:.1f} px")
    print(f"  principal point  : ({K[0, 2]:.1f} +- {sd[2]:.1f}, {K[1, 2]:.1f} +- {sd[3]:.1f})   "
          f"image centre ({W / 2:.0f}, {Hh / 2:.0f})")
    print(f"  distortion       : {np.array2string(D.ravel(), precision=5, suppress_small=True)}")

   
    halves = []
    for h in (0, 1):
        part = sel[h::2]
        if len(part) >= 12:
            halves.append(fit(part, size, flags)[1])
    if len(halves) == 2:
        f1, f2 = halves[0][0, 0], halves[1][0, 0]
        print(f"  half-split check : f {f1:.1f} / {f2:.1f} px  "
              f"(spread {100 * abs(f1 - f2) / K[0, 0]:.1f}% — the honest uncertainty)")

    e_train = reproj_errors(sel, K, D)
    print(f"  per-view error   : median {np.median(e_train):.2f} px, max {e_train.max():.2f} px")

    # --- held-out clips --------------------------------------------------------------
    checks = []
    if args.check:
        print("\nheld-out clips (lens fixed, only the board pose solved):")
    for v in args.check:
        vv, sampled, sz, _ = collect_views(v, det, board, args.stride * 4, args.min_corners)
        if sz != size or not vv:
            print(f"  {os.path.basename(v):18s} skipped ({'size ' + str(sz) if sz != size else 'no board'})")
            continue
        e = reproj_errors(vv, K, D)
        checks.append(dict(name=os.path.basename(v), views=len(vv),
                           median_px=float(np.median(e)), max_px=float(e.max())))
        print(f"  {os.path.basename(v):18s} {len(vv)} views, median {np.median(e):.2f} px, max {e.max():.2f} px")

    # --- pictures ----------------------------------------------------------------------
    img = frame.copy()
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)
    for k in (1, 2):
        cv2.line(img, (k * W // 3, 0), (k * W // 3, Hh), (0, 255, 0), 3)
        cv2.line(img, (0, k * Hh // 3), (W, k * Hh // 3), (0, 255, 0), 3)
    cv2.imwrite(base + "_coverage.jpg", cv2.resize(img, (W // 2, Hh // 2)))
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, size, 0.0)
    und = cv2.undistort(frame, K, D, None, newK)
    cv2.imwrite(base + "_undistorted.jpg",
                cv2.resize(np.hstack([frame, und]), (W, Hh // 2)))

    # --- JSON ------------------------------------------------------------------------
    with open(base + ".json", "w") as fh:
        json.dump(dict(
            model="opencv", dist_model=args.model, camera=args.camera,
            serial=ref.get("CASN"), lens_mode=MODES.get(ref.get("VFOV"), ref.get("VFOV")),
            gopro_fov=ref.get("ZFOV"), width=W, height=Hh,
            K=K.tolist(), dist=D.ravel().tolist(), rms_px=rms, views=n_used,
            focal_sd_px=float(sd[0]),
            focal_half_split_px=[float(h[0, 0]) for h in halves],
            board=dict(squares=[sx, sy], square_mm=args.square_mm, marker_mm=mk, dict=args.dict),
            clips=[os.path.basename(v) for v in args.videos], checks=checks), fh, indent=2)
    print(f"\nwrote {base}.json, .log, _coverage.jpg, _undistorted.jpg")
    print("(not yet usable with track_dual.py --lens-a: that reads calibrate_lens.py's format "
          "until the pipeline change, step 2 of the plan)")


if __name__ == "__main__":
    main()

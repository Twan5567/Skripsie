"""Vehicle trajectory from a fixed overhead camera, via a floor-GCP homography.

Run:
  python track_ground.py --video /path/to/clip.MP4   (or use ./run_tracking.sh)
"""
from gpu import init_opencv_opencl

init_opencv_opencl()

import argparse
import csv
import os

import cv2
import numpy as np

from aruco_config import ARUCO_DICT


# --------------------------------------------------------------------------
def build_detector() -> cv2.aruco.ArucoDetector:
    """Tuned for small, oblique markers on a 4K frame.

    The defaults miss the floor sheets at this scale: minMarkerPerimeterRate
    is relative to image size, so a 45 px marker in a 3840 px frame falls
    under the default 0.03 floor.
    """
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.minMarkerPerimeterRate = 0.008
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 45
    p.adaptiveThreshWinSizeStep = 4
    p.polygonalApproxAccuracyRate = 0.06
    return cv2.aruco.ArucoDetector(ARUCO_DICT, p)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def undistort(pts: np.ndarray, c: np.ndarray, s: float, k1: float, k2: float):
    """Division-model radial undistortion about image centre `c`.

    Works in pure image space, so it needs no focal length — which matters
    because the focal length is itself recovered later FROM the undistorted
    homography. `s` normalises radius so k1/k2 stay order-1.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    d = (p - c) / s
    r2 = (d ** 2).sum(axis=1)
    f = 1.0 + k1 * r2 + k2 * r2 ** 2
    return (c + (p - c) / f[:, None]).reshape(np.shape(pts))


def square_error(med_u: dict, anchor: int, S: float):
    """RMS mm error of every marker's sides and diagonals against a true square.

    The anchor is pinned to S by construction, so this measures how well the
    REST of the floor agrees — which is exactly what lens distortion breaks.
    Diagonals are included because they are scale-aware in a different way
    than the sides and stop a uniform-shrink degeneracy.
    """
    Hm, _ = cv2.findHomography(med_u[anchor],
                               np.array([[0, 0], [S, 0], [S, S], [0, S]], float))
    if Hm is None:
        return 1e9, {}, None
    errs, per = [], {}
    for k, q in med_u.items():
        w = cv2.perspectiveTransform(q.reshape(-1, 1, 2), Hm).reshape(4, 2)
        sides = [np.linalg.norm(w[j] - w[(j + 1) % 4]) for j in range(4)]
        diags = [np.linalg.norm(w[0] - w[2]), np.linalg.norm(w[1] - w[3])]
        per[k] = float(np.mean(sides))
        errs += [x - S for x in sides]
        errs += [x - S * np.sqrt(2) for x in diags]
    return float(np.sqrt(np.mean(np.square(errs)))), per, Hm


def camera_from_homography(Hf2i: np.ndarray, c: np.ndarray):
    """Recover focal length and camera centre from one floor->image homography.

    Assumes square pixels and the principal point at the image centre, which
    leaves focal length as the only unknown in K. Zhang's two orthonormality
    constraints on the rotation columns then each give an independent estimate
    of f; their agreement is a free check that the pinhole assumption holds.
    Returns (f, camera_xyz_in_floor_mm, f_from_each_constraint).
    """
    T = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]], float)
    Hc = T @ Hf2i                      # floor -> centred image
    h1, h2, h3 = Hc[:, 0], Hc[:, 1], Hc[:, 2]
    den1 = h1[2] * h2[2]
    f2a = -(h1[0] * h2[0] + h1[1] * h2[1]) / den1 if abs(den1) > 1e-12 else np.nan
    den2 = h2[2] ** 2 - h1[2] ** 2
    f2b = ((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / den2 \
        if abs(den2) > 1e-12 else np.nan
    cand = [x for x in (f2a, f2b) if np.isfinite(x) and x > 0]
    if not cand:
        return None, None, (f2a, f2b)
    f = float(np.sqrt(np.mean(cand)))
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]], float)
    A = np.linalg.inv(K) @ Hc
    # Sign of lambda is free in a homography; fix it so the floor lies in
    # front of the camera (positive depth), else the recovered centre comes
    # out mirrored through the plane and the height is negative.
    lam = 1.0 / np.linalg.norm(A[:, 0])
    r1, r2 = lam * A[:, 0], lam * A[:, 1]
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R)        # nearest true rotation
    R = U @ Vt
    t = lam * A[:, 2]
    C = -R.T @ t
    # The floor frame built from ArUco corner order (TL,TR,BR,BL) is
    # left-handed against a z-up world, because image y runs downward. That
    # makes r3 = r1 x r2 point INTO the floor and C_z come out negative with
    # the right magnitude. Height is the distance to the plane either way.
    C = np.array([C[0], C[1], abs(C[2])])
    return f, C, (np.sqrt(f2a) if f2a > 0 else np.nan,
                  np.sqrt(f2b) if f2b > 0 else np.nan)


def smooth_xy(arr: np.ndarray, ok: np.ndarray, win: int, poly: int = 2):
    """Savitzky-Golay smoothing applied per contiguous run of located frames.

    Filtering straight through a dropout would drag the fit across the gap and
    invent a shortcut, so each unbroken run is smoothed on its own. Savitzky-
    Golay rather than a moving average because it fits a local polynomial: it
    takes the jitter out of a corner without rounding the corner off, which a
    box filter does.
    """
    out = arr.copy()
    if win < 5:
        return out
    from scipy.signal import savgol_filter

    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out
    for run in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
        if run.size < 5:
            continue
        w = min(win, run.size)
        if w % 2 == 0:
            w -= 1
        if w < 5:
            continue
        for c in (0, 1):
            out[run, c] = savgol_filter(arr[run, c], w, min(poly, w - 1))
    return out


def parse_ids(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--outdir", default="../output")
ap.add_argument("--gcp-ids", default="0-11", help="IDs of the flat floor sheets")
ap.add_argument("--gcp-mm", type=float, default=150.0,
                help="black-square side of the floor sheets, mm")
ap.add_argument("--vehicle-ids", default="20,21")
ap.add_argument("--anchor-id", type=int, default=None,
                help="GCP that sets scale (default: best-detected, largest)")
ap.add_argument("--min-rate", type=float, default=0.60,
                help="detection rate a GCP needs to join the floor fit")
ap.add_argument("--stride", type=int, default=1, help="analyse every Nth frame")
ap.add_argument("--no-fallback", action="store_true",
                help="skip background-subtraction tracking")
ap.add_argument("--cam-height-mm", type=float, default=None,
                help="measured camera height, for cross-checking the recovered pose")
ap.add_argument("--tag-heights", default="20:57,21:67",
                help="height of each vehicle tag above the floor, mm (id:mm,...)")
ap.add_argument("--no-distortion", action="store_true",
                help="skip the radial-distortion fit (leave k1=k2=0)")
ap.add_argument("--max-step-px", type=float, default=260.0,
                help="largest plausible per-sample car motion, full-res pixels")
ap.add_argument("--smooth", type=int, default=15,
                help="Savitzky-Golay window in samples; 0 disables smoothing")
ap.add_argument("--no-overlay", action="store_true")
ap.add_argument("--overlay-width", type=int, default=1920)
args = ap.parse_args()

stem = os.path.splitext(os.path.basename(args.video))[0]
# One folder per capture keeps overlay/CSV/plots for a run together.
args.outdir = os.path.join(args.outdir, stem)
os.makedirs(args.outdir, exist_ok=True)
gcp_ids = set(parse_ids(args.gcp_ids))
veh_ids = parse_ids(args.vehicle_ids)
detector = build_detector()

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit(f"cannot open video: {args.video}")
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- pass 1: detect every marker, and accumulate a median background -------
# The camera is fixed, so a per-pixel median over frames sampled across the
# whole clip is the empty floor: the car occupies any given pixel for a small
# fraction of the run, so it never wins the median.
corners_by_frame: list[dict[int, np.ndarray]] = []
bg_samples: list[np.ndarray] = []
BG_N, BG_W = 40, 480

frame_i = 0
kept = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_i % args.stride == 0:
        c, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        d: dict[int, np.ndarray] = {}
        if ids is not None:
            for n, k in enumerate(np.array(ids).flatten()):
                d[int(k)] = np.array(c[n]).reshape(4, 2).astype(np.float64)
        corners_by_frame.append(d)
        kept += 1
        if kept % 200 == 0:
            print(f"  detect {kept}", end="\r", flush=True)
    frame_i += 1
cap.release()
n = len(corners_by_frame)
print(f"  detect {n} frames analysed" + " " * 20)

if n == 0:
    raise SystemExit("no frames decoded")

# --- which markers are the static floor sheets? ---------------------------
rates = {k: sum(k in d for d in corners_by_frame) / n
         for k in {k for d in corners_by_frame for k in d}}

# A floor sheet is glued down: its image position must barely move. This is
# what separates a real GCP from a spurious decode of floor tiling, which
# tends to appear in scattered places.
def spread_px(k: int) -> float:
    pts = np.array([d[k].mean(axis=0) for d in corners_by_frame if k in d])
    return float(np.hypot(*pts.std(axis=0))) if len(pts) > 1 else np.inf


static = {k: r for k, r in rates.items()
          if k in gcp_ids and r >= args.min_rate and spread_px(k) < 5.0}
if len(static) < 1:
    raise SystemExit(f"no stable floor GCPs found (rates: {rates})")

# Median corners are robust to the occasional bad decode.
med = {k: np.median(np.array([d[k] for d in corners_by_frame if k in d]), axis=0)
       for k in static}


def px_side(k: int) -> float:
    q = med[k]
    return float(np.mean([np.linalg.norm(q[j] - q[(j + 1) % 4]) for j in range(4)]))


anchor = args.anchor_id if args.anchor_id is not None else \
    max(static, key=lambda k: (rates[k] > 0.9, px_side(k)))
if anchor not in med:
    raise SystemExit(f"anchor {anchor} is not a stable floor GCP")

S = args.gcp_mm
square = np.array([[0, 0], [S, 0], [S, S], [0, S]], dtype=np.float64)
img_c = np.array([W / 2.0, Hh / 2.0])
norm_s = 0.5 * float(np.hypot(W, Hh))
tag_h = {int(a): float(b) for a, b in
         (t.split(":") for t in args.tag_heights.split(",") if t.strip())}


def apply_H(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    flat = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(flat, Hm).reshape(np.shape(pts))


# --- fit lens distortion from the markers themselves ----------------------
# A homography can only rectify a plane imaged by a pinhole camera. A GoPro is
# nowhere near pinhole, and the residual shows up as sheets that measure the
# wrong size the further they sit from the anchor. Every sheet is a known
# square, so the distortion that makes all of them square at once can be
# solved for directly — no ChArUco board required for this particular job.
e_before, _, _ = square_error(med, anchor, S)
k1 = k2 = 0.0
if not args.no_distortion and len(med) >= 3:
    from scipy.optimize import minimize

    def obj(kk):
        mu = {k: undistort(med[k], img_c, norm_s, kk[0], kk[1]) for k in med}
        return square_error(mu, anchor, S)[0]

    best = min((minimize(obj, s0, method="Nelder-Mead",
                         options=dict(xatol=1e-7, fatol=1e-5, maxiter=800))
                for s0 in ([0.0, 0.0], [-0.1, 0.0], [0.1, 0.0])),
               key=lambda r: r.fun)
    k1, k2 = float(best.x[0]), float(best.x[1])

med_u = {k: undistort(med[k], img_c, norm_s, k1, k2) for k in med}
e_after, per_side, _ = square_error(med_u, anchor, S)

H0, _ = cv2.findHomography(med_u[anchor], square)
if H0 is None:
    raise SystemExit("anchor homography failed")

# --- refine: fit the homography to EVERY stable GCP corner ----------------
# H0 rests on one marker's four corners, so its noise is that marker's noise.
# Rectifying the others through H0 gives their floor-plane layout for free;
# re-fitting against all of those corners at once averages the corner noise
# down without changing the scale, which still comes from the anchor alone.
floor_map = {k: apply_H(H0, med_u[k]) for k in med_u}
src = np.vstack([med_u[k] for k in sorted(med_u)])
dst = np.vstack([floor_map[k] for k in sorted(med_u)])
H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
if H is None:
    H = H0

# --- camera pose, and the check against the tape measure ------------------
f_px, C_cam, f_pair = camera_from_homography(np.linalg.inv(H), img_c)
nadir = C_cam[:2] if C_cam is not None else None
H_cam = float(C_cam[2]) if C_cam is not None else None

print(f"\n{'=' * 68}\nFLOOR PLANE FIT   anchor ID {anchor}, {S:.1f} mm assumed\n{'=' * 68}")
print(f"distortion fit    : k1 ={k1:+.5f}  k2 ={k2:+.5f}   "
      f"(RMS {e_before:.2f} -> {e_after:.2f} mm)")
if f_px:
    print(f"focal length      : {f_px:.0f} px   "
          f"(two constraints: {f_pair[0]:.0f} / {f_pair[1]:.0f} px)")
    print(f"camera height     : {H_cam:.0f} mm recovered from the homography", end="")
    if args.cam_height_mm:
        d = H_cam - args.cam_height_mm
        print(f"   vs {args.cam_height_mm:.0f} mm measured  ->  {d:+.0f} mm "
              f"({100 * d / args.cam_height_mm:+.1f}%)")
    else:
        print()
    print(f"nadir on floor    : x {nadir[0]:+.0f}, y {nadir[1]:+.0f} mm "
          f"(anchor ID {anchor} corner is the origin)")

print(f"\n{'id':>4} {'rate':>7} {'px':>7} {'mm side':>9} {'err':>8}")
check = {}
for k in sorted(med_u):
    w = apply_H(H, med_u[k])
    check[k] = float(np.mean([np.linalg.norm(w[j] - w[(j + 1) % 4])
                              for j in range(4)]))
    per_side[k] = check[k]
    print(f"{k:>4} {100 * rates[k]:>6.1f}% {px_side(k):>7.1f} {per_side[k]:>9.2f} "
          f"{per_side[k] - S:>+7.2f}")
for k in sorted(rates):
    if k in med_u or k in veh_ids or rates[k] < 0.5:
        continue
    pts = np.median(np.array([d[k] for d in corners_by_frame if k in d]), axis=0)
    q = apply_H(H, undistort(pts, img_c, norm_s, k1, k2))
    sz = float(np.mean([np.linalg.norm(q[j] - q[(j + 1) % 4]) for j in range(4)]))
    print(f"{k:>4} {100 * rates[k]:>6.1f}% {'':>7} {sz:>9.2f}   <- outside the GCP "
          f"set; this is its implied physical size")

resid = np.array([check[k] - S for k in check])
print(f"\nside-length residual: mean {resid.mean():+.2f} mm, "
      f"rms {np.sqrt((resid ** 2).mean()):.2f} mm over {len(check)} sheets")

# --- vehicle marker, where it fires --------------------------------------
# Each tag sits above the floor, so the floor homography puts it too far from
# the nadir. With the camera height known the correction is exact and needs no
# calibration: scale the radius from the nadir by (H_cam - h) / H_cam.
veh_px = np.full((n, 2), np.nan)
veh_mm = np.full((n, 2), np.nan)
veh_src = np.full(n, "", dtype=object)
par_shift = []
for i, d in enumerate(corners_by_frame):
    for vid in veh_ids:
        if vid in d:
            c_px = d[vid].mean(axis=0)
            raw = apply_H(H, undistort(c_px, img_c, norm_s, k1, k2)).reshape(2)
            if nadir is not None and H_cam:
                h = tag_h.get(vid, 0.0)
                cor = nadir + (raw - nadir) * (H_cam - h) / H_cam
                par_shift.append(float(np.linalg.norm(cor - raw)))
            else:
                cor = raw
            veh_px[i] = c_px
            veh_mm[i] = cor
            veh_src[i] = f"aruco{vid}"
            break
aruco_hits = int(np.isfinite(veh_px).all(axis=1).sum())
if par_shift:
    print(f"parallax correction : mean {np.mean(par_shift):.1f} mm, "
          f"max {np.max(par_shift):.1f} mm  (tag heights {tag_h})")

# --- fallback: median-background subtraction ------------------------------
blob_px = np.full((n, 2), np.nan)
if not args.no_fallback:
    cap = cv2.VideoCapture(args.video)
    idx = np.linspace(0, frame_i - 1, BG_N).astype(int)
    scale = BG_W / W
    want = set(idx.tolist())
    j = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if j in want:
            bg_samples.append(cv2.cvtColor(
                cv2.resize(fr, (BG_W, int(Hh * scale))), cv2.COLOR_BGR2GRAY))
        j += 1
    cap.release()
    bg = np.median(np.array(bg_samples), axis=0).astype(np.uint8)

    cap = cv2.VideoCapture(args.video)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Seed on the car rather than on whatever blob is biggest in frame 0.
    seed = np.flatnonzero(np.isfinite(veh_px).all(axis=1))
    prev = tuple(veh_px[seed[0]] * scale) if len(seed) else None
    fi = ki = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fi % args.stride == 0:
            small = cv2.cvtColor(cv2.resize(fr, (BG_W, int(Hh * scale))),
                                 cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(small, bg)
            _, m = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern, iterations=2)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cand = [c for c in cs if 30 < cv2.contourArea(c) < 4000]
            if cand:
                # Prefer the blob nearest the last known car position; on the
                # first frame, or after a long gap, take the largest instead.
                gate = args.max_step_px * scale
                if prev is not None:
                    def key(c):
                        M = cv2.moments(c)
                        if M["m00"] == 0:
                            return 1e9
                        return np.hypot(M["m10"] / M["m00"] - prev[0],
                                        M["m01"] / M["m00"] - prev[1])
                    c = min(cand, key=key)
                    # A blob that teleports is a different object — a person,
                    # a shadow, a barrier flexing. Drop it and coast; `prev`
                    # is kept so the track can re-acquire when the car returns.
                    if key(c) > gate:
                        c = None
                else:
                    c = max(cand, key=cv2.contourArea)
                M = cv2.moments(c) if c is not None else {"m00": 0}
                if M["m00"] > 0:
                    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                    prev = (cx, cy)
                    blob_px[ki] = (cx / scale, cy / scale)
            ki += 1
        fi += 1
    cap.release()

# --- combine: ArUco where trusted, blob to fill --------------------------
px = veh_px.copy()
fill = np.isnan(px).any(axis=1) & np.isfinite(blob_px).all(axis=1)
px[fill] = blob_px[fill]
veh_src[fill] = "blob"
have = np.isfinite(px).all(axis=1)
if not have.any():
    raise SystemExit("vehicle never located by either method")

mm = veh_mm.copy()
if fill.any():
    raw = apply_H(H, undistort(px[fill], img_c, norm_s, k1, k2)).reshape(-1, 2)
    # The silhouette centroid sits on the car body, so give it the mean tag
    # height rather than pretending it lies on the floor.
    h_blob = float(np.mean(list(tag_h.values()))) if tag_h else 0.0
    if nadir is not None and H_cam:
        raw = nadir + (raw - nadir) * (H_cam - h_blob) / H_cam
    mm[fill] = raw
mm[have] -= mm[have][0]

# Raw per-frame steps accumulate detector jitter AS DISTANCE: a car sitting
# still still racks up path length. Smoothing shrinks random jitter by about
# sqrt(window) while leaving real motion, which is smooth over that scale.
mm_raw = mm.copy()
px_raw = px.copy()
mm = smooth_xy(mm_raw, have, args.smooth)
px = smooth_xy(px_raw, have, args.smooth)

raw_path = float(np.linalg.norm(np.diff(mm_raw[have], axis=0), axis=1).sum())
path = float(np.linalg.norm(np.diff(mm[have], axis=0), axis=1).sum())
net = float(np.linalg.norm(mm[have][-1] - mm[have][0]))

# --- CSV ------------------------------------------------------------------
csv_path = os.path.join(args.outdir, f"{stem}_trajectory.csv")

with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "time_s", "source", "px_x", "px_y", "mm_x", "mm_y",
                "aruco_px_x", "aruco_px_y"])
    for i in range(n):
        fr = i * args.stride
        a = veh_px[i]
        w.writerow([fr, f"{fr / fps:.5f}", veh_src[i] or "",
                    f"{px[i, 0]:.2f}" if have[i] else "",
                    f"{px[i, 1]:.2f}" if have[i] else "",
                    f"{mm[i, 0]:.2f}" if have[i] else "",
                    f"{mm[i, 1]:.2f}" if have[i] else "",
                    f"{a[0]:.2f}" if np.isfinite(a).all() else "",
                    f"{a[1]:.2f}" if np.isfinite(a).all() else ""])
print(f"\n  wrote {csv_path}")

# # --- plots ----------------------------------------------------------------
# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# t = np.arange(n) * args.stride / fps

# is_ar = np.array([s.startswith("aruco") for s in veh_src]) & have


# fig, ax = plt.subplots(figsize=(9.5, 7))

# for k in sorted(static):
#     q = apply_H(H, med[k])
#     ax.add_patch(plt.Polygon(q, closed=True, fc="0.85", ec="0.45", lw=0.8, zorder=1))
#     ax.text(*q.mean(axis=0), str(k), ha="center", va="center",
#             fontsize=7, color="0.3", zorder=2)
# if args.smooth >= 5:
#     ax.plot(mm_raw[have, 0], mm_raw[have, 1], "-", lw=0.7, color="#9BB7BF",
#             label="raw", zorder=2.5)
# ax.plot(mm[have, 0], mm[have, 1], "-", lw=1.6, color="#0A6273",
#         label=f"smoothed (w={args.smooth})" if args.smooth >= 5
#         else f"trajectory ({int(have.sum())} pts)", zorder=3)
# if is_ar.any():
#     ax.plot(mm[is_ar, 0], mm[is_ar, 1], "o", ms=4, color="#B3400C",
#             label=f"vehicle ArUco fix ({int(is_ar.sum())})", zorder=4)
# ax.plot(*mm[have][0], "go", ms=9, label="start", zorder=5)
# ax.plot(*mm[have][-1], "rs", ms=9, label="end", zorder=5)
# ax.set_aspect("equal")
# ax.set_xlabel("x (mm)")
# ax.set_ylabel("y (mm)")
# ax.set_title(f"{stem} — vehicle trajectory, floor-plane coordinates")
# ax.grid(alpha=0.3)
# ax.legend(loc="best", fontsize=8)
# p1 = os.path.join(args.outdir, f"{stem}_trajectory.png")
# fig.tight_layout()
# fig.savefig(p1, dpi=150)
# plt.close(fig)

# dist = np.full(n, np.nan)
# dist[have] = np.linalg.norm(mm[have] - mm[have][0], axis=1)
# fig, ax = plt.subplots(figsize=(9.5, 4.5))
# ax.plot(t[have], dist[have], "-", lw=1.3, color="#0A6273")
# if is_ar.any():
#     ax.plot(t[is_ar], dist[is_ar], "o", ms=3.5, color="#B3400C",
#             label="vehicle ArUco fix")
#     ax.legend(fontsize=8)
# ax.set_xlabel("time (s)")
# ax.set_ylabel("distance from start (mm)")
# ax.set_title(f"{stem} — displacement over time")
# ax.grid(alpha=0.3)
# p2 = os.path.join(args.outdir, f"{stem}_distance.png")
# fig.tight_layout()
# fig.savefig(p2, dpi=150)
# plt.close(fig)
# print(f"  wrote {p1}\n  wrote {p2}")

# --- overlay --------------------------------------------------------------
if not args.no_overlay:
    ow = args.overlay_width
    oh = int(round(Hh * ow / W / 2) * 2)
    sc = ow / W
    out_path = os.path.join(args.outdir, f"{stem}_overlay.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.stride, (ow, oh))
    cap = cv2.VideoCapture(args.video)
    fi = ki = 0
    while True:
        ok, fr = cap.read()
        if not ok or ki >= n:
            break
        if fi % args.stride == 0:
            fr = cv2.resize(fr, (ow, oh), interpolation=cv2.INTER_AREA)
            for k in sorted(static):
                cv2.polylines(fr, [(med[k] * sc).astype(np.int32)], True,
                              (140, 140, 140), 2)
                cv2.putText(fr, str(k), tuple((med[k][0] * sc).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 2,
                            cv2.LINE_AA)
            trail = px[: ki + 1][have[: ki + 1]] * sc
            if len(trail) > 1:
                cv2.polylines(fr, [trail.astype(np.int32)], False, (0, 255, 255), 3)
            if have[ki]:
                col = (0, 0, 255) if veh_src[ki].startswith("aruco") else (0, 200, 0)
                cv2.circle(fr, tuple((px[ki] * sc).astype(int)), 9, col, -1)
            hud = [f"frame {fi:5d}   t {fi / fps:6.2f}s   src {veh_src[ki] or '--':>8}",
                   f"dist {dist[ki]:8.1f} mm" if have[ki] else "dist --"]
            for r, line in enumerate(hud):
                for colr, th in ((0, 0, 0), 6), ((255, 255, 255), 2):
                    cv2.putText(fr, line, (30, 46 + r * 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, colr, th, cv2.LINE_AA)
            writer.write(fr)
            ki += 1
            if ki % 200 == 0:
                print(f"  overlay {ki}/{n}", end="\r", flush=True)
        fi += 1
    cap.release()
    writer.release()
    print(f"  wrote {out_path}" + " " * 20)

# --- summary --------------------------------------------------------------
print(f"\n{'=' * 66}")
print(f"{stem}   {W}x{Hh} @ {fps:.2f} fps, {frame_i} frames "
      f"({n} analysed, stride {args.stride})")
print("=" * 66)
print(f"floor GCPs used     : {sorted(static)}  (anchor {anchor})")
print(f"vehicle ArUco fixes : {aruco_hits}/{n} ({100 * aruco_hits / n:.1f}%)")
print(f"blob-filled frames  : {int(fill.sum())}/{n} ({100 * fill.sum() / n:.1f}%)")
print(f"located total       : {int(have.sum())}/{n} ({100 * have.mean():.1f}%)")
print(f"path length (raw)   : {raw_path:.0f} mm ({raw_path / 1000:.2f} m)  "
      f"[jitter-inflated]")
print(f"path length (smooth): {path:.0f} mm ({path / 1000:.2f} m)  "
      f"[Savitzky-Golay w={args.smooth}]")
print(f"net displacement    : {net:.0f} mm")
print(f"scale check (rms)   : {np.sqrt((resid ** 2).mean()):.2f} mm on a {S:.0f} mm sheet "
      f"({100 * np.sqrt((resid ** 2).mean()) / S:.2f}%)")

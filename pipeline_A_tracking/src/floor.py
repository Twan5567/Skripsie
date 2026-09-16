"""Floor-plane fit from the static GCP sheets.

Picks the sheets that are really on the floor, fits lens distortion from
their squareness, builds the anchor homography and refines it on every GCP
corner, recovers the camera pose, and prints the fit report.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from geometry import apply_H, camera_from_homography, square_error, undistort


@dataclass
class FloorFit:
    rates: dict          # id -> detection rate over analysed frames
    static: dict         # id -> rate, for the sheets accepted as floor GCPs
    med: dict            # id -> median 4x2 corners (raw pixels)
    med_u: dict          # id -> median corners after undistortion
    anchor: int
    S: float             # sheet side, mm
    img_c: np.ndarray    # image centre
    norm_s: float        # radius normaliser for the distortion model
    k1: float
    k2: float
    H: np.ndarray        # image (undistorted) -> floor mm
    f_px: float | None
    nadir: np.ndarray | None
    H_cam: float | None
    resid: np.ndarray    # per-sheet side-length residual, mm
    per_side: dict = field(default_factory=dict)


def fit_floor(corners_by_frame: list, n: int, W: int, Hh: int,
              gcp_ids: set, veh_ids: list, min_rate: float, anchor_id: int | None,
              gcp_mm: float, no_distortion: bool, cam_height_mm: float | None) -> FloorFit:
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
              if k in gcp_ids and r >= min_rate and spread_px(k) < 5.0}
    if len(static) < 1:
        raise SystemExit(f"no stable floor GCPs found (rates: {rates})")

    # Median corners are robust to the occasional bad decode.
    med = {k: np.median(np.array([d[k] for d in corners_by_frame if k in d]), axis=0)
           for k in static}

    def px_side(k: int) -> float:
        q = med[k]
        return float(np.mean([np.linalg.norm(q[j] - q[(j + 1) % 4]) for j in range(4)]))

    anchor = anchor_id if anchor_id is not None else \
        max(static, key=lambda k: (rates[k] > 0.9, px_side(k)))
    if anchor not in med:
        raise SystemExit(f"anchor {anchor} is not a stable floor GCP")

    S = gcp_mm
    square = np.array([[0, 0], [S, 0], [S, S], [0, S]], dtype=np.float64)
    img_c = np.array([W / 2.0, Hh / 2.0])
    norm_s = 0.5 * float(np.hypot(W, Hh))

    # --- fit lens distortion from the markers themselves ----------------------
    # A homography can only rectify a plane imaged by a pinhole camera. A GoPro is
    # nowhere near pinhole, and the residual shows up as sheets that measure the
    # wrong size the further they sit from the anchor. Every sheet is a known
    # square, so the distortion that makes all of them square at once can be
    # solved for directly — no ChArUco board required for this particular job.
    e_before, _, _ = square_error(med, anchor, S)
    k1 = k2 = 0.0
    if not no_distortion and len(med) >= 3:
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
        if cam_height_mm:
            d = H_cam - cam_height_mm
            print(f"   vs {cam_height_mm:.0f} mm measured  ->  {d:+.0f} mm "
                  f"({100 * d / cam_height_mm:+.1f}%)")
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

    return FloorFit(rates=rates, static=static, med=med, med_u=med_u, anchor=anchor,
                    S=S, img_c=img_c, norm_s=norm_s, k1=k1, k2=k2, H=H, f_px=f_px,
                    nadir=nadir, H_cam=H_cam, resid=resid, per_side=per_side)

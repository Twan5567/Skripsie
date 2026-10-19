"""Floor-plane fit from the static GCP sheets.

Picks the sheets that are really on the floor, fits lens distortion from
their squareness, builds the anchor homography and refines it on every GCP
corner, recovers the camera pose, and prints the fit report.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from geometry import apply_H, camera_from_homography, cv_undistort, square_error, undistort


@dataclass
class FloorFit:
    rates: dict         
    static: dict         
    med: dict            
    med_u: dict         
    anchor: int
    S: float             
    img_c: np.ndarray   
    norm_s: float       
    k1: float
    k2: float
    H: np.ndarray        
    f_px: float | None
    nadir: np.ndarray | None
    H_cam: float | None
    resid: np.ndarray    
    per_side: dict = field(default_factory=dict)

    
    K: np.ndarray | None = None   
    D: np.ndarray | None = None   
    P: np.ndarray | None = None   
    cal: str | None = None


def fit_floor(corners_by_frame: list, n: int, W: int, Hh: int, gcp_ids: set, veh_ids: list, min_rate: float, anchor_id: int | None, gcp_mm: float, no_distortion: bool, cam_height_mm: float | None, lens: dict | None = None) -> FloorFit:
   
    # --- which markers are the static floor sheets? ---------------------------
    rates = {k: sum(k in d for d in corners_by_frame) / n
             for k in {k for d in corners_by_frame for k in d}}

    
    def spread_px(k: int) -> float:

        pts = np.array([d[k].mean(axis=0) for d in corners_by_frame if k in d])
        if len(pts) < 2:
            return np.inf
        
        return float(np.percentile(np.linalg.norm(pts - np.median(pts, axis=0), axis=1), 90))

    static = {k: r for k, r in rates.items()
              if k in gcp_ids and r >= min_rate and spread_px(k) < 5.0}
    
    if len(static) < 1:

        raise SystemExit(f"no stable floor GCPs found (rates: {rates})")

   
    med = {k: np.median(np.array([d[k] for d in corners_by_frame if k in d]), axis=0)
           for k in static}

    def px_side(k: int) -> float:

        q = med[k]
        return float(np.mean([np.linalg.norm(q[j] - q[(j + 1) % 4]) for j in range(4)]))

    S = gcp_mm
    
    square = np.array([[0, 0], [S, 0], [S, S], [0, S]], dtype=np.float64)
    img_c = np.array([W / 2.0, Hh / 2.0])
    norm_s = 0.5 * float(np.hypot(W, Hh))

    if lens is not None:

        if (lens.get("width"), lens.get("height")) != (W, Hh):

            raise SystemExit(f"lens file is for {lens.get('width')}x{lens.get('height')}, "
                             f"clip is {W}x{Hh}")

        
    cv_lens = lens is not None and lens.get("model") == "opencv"
    K = D = P = None

    if cv_lens:
        K = np.array(lens["K"], dtype=np.float64)
        D = np.array(lens["dist"], dtype=np.float64)
        f_sq = 0.5 * (K[0, 0] + K[1, 1])
        P = np.array([[f_sq, 0, K[0, 2]], [0, f_sq, K[1, 2]], [0, 0, 1.0]])
        img_c = K[:2, 2].copy()

    elif lens is not None:
        img_c = np.array([float(lens["cx"]), float(lens["cy"])])
        norm_s = float(lens["norm_s"])

    # --- fit lens distortion from the markers themselves ----------------------

    fit_k2 = len(med) >= 5
    r2_grid = np.linspace(0.0, 1.0, 11) ** 2

    def fit_distortion(anc: int) -> tuple[float, float, float]:
        """(k1, k2, residual after) for a given anchor sheet."""
        if cv_lens:
            mu = {k: cv_undistort(med[k], K, D, P) for k in med}
            return 0.0, 0.0, square_error(mu, anc, S)[0]
        if lens is not None:
            a, b = float(lens["k1"]), float(lens["k2"])
            mu = {k: undistort(med[k], img_c, norm_s, a, b) for k in med}
            return a, b, square_error(mu, anc, S)[0]
        if no_distortion or len(med) < 3:
            return 0.0, 0.0, square_error(med, anc, S)[0]
        from scipy.optimize import minimize

        def obj(kk):
            a, b = kk[0], (kk[1] if fit_k2 else 0.0)
            f = 1.0 + a * r2_grid + b * r2_grid ** 2
            if abs(a) > 0.5 or abs(b) > 0.25 or f.min() < 0.5 or f.max() > 1.5:
                return 1e6
            mu = {k: undistort(med[k], img_c, norm_s, a, b) for k in med}
            return square_error(mu, anc, S)[0]

        best = min((minimize(obj, s0, method="Nelder-Mead",
                             options=dict(xatol=1e-7, fatol=1e-5, maxiter=800))
                    for s0 in ([0.0, 0.0], [-0.1, 0.0], [0.1, 0.0])),
                   key=lambda r: r.fun)
        return float(best.x[0]), (float(best.x[1]) if fit_k2 else 0.0), float(best.fun)

    # --- choose the anchor ---------------------------------------------------

    if anchor_id is not None:

        anchor = anchor_id
        if anchor not in med:
            raise SystemExit(f"anchor {anchor} is not a stable floor GCP")
        k1, k2, e_after = fit_distortion(anchor)

    else:
        cands = [k for k in static if rates[k] > 0.9] or list(static)
        trials = {k: fit_distortion(k) for k in cands}
        anchor = min(trials, key=lambda k: trials[k][2])
        k1, k2, e_after = trials[anchor]
        print("anchor candidates : " + ", ".join(
            f"{k}: {trials[k][2]:.1f} mm" for k in sorted(trials)) + f"  -> {anchor}")
        
    e_before, _, _ = square_error(med, anchor, S)

    if cv_lens:
        med_u = {k: cv_undistort(med[k], K, D, P) for k in med}

    else:
        med_u = {k: undistort(med[k], img_c, norm_s, k1, k2) for k in med}


    e_after, per_side, _ = square_error(med_u, anchor, S)

    H0, _ = cv2.findHomography(med_u[anchor], square)
    if H0 is None:
        raise SystemExit("anchor homography failed")

    # --- refine: fit the homography to EVERY stable GCP corner ----------------
    
    floor_map = {k: apply_H(H0, med_u[k]) for k in med_u}
    src = np.vstack([med_u[k] for k in sorted(med_u)])
    dst = np.vstack([floor_map[k] for k in sorted(med_u)])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)

    if H is None:
        H = H0

    # --- camera pose, and the check against the tape measure ------------------


    f_px, C_cam, f_pair = camera_from_homography(np.linalg.inv(H), img_c, P[0, 0] if cv_lens else None)
    nadir = C_cam[:2] if C_cam is not None else None
    H_cam = float(C_cam[2]) if C_cam is not None else None

    print(f"\n{'=' * 68}\nFLOOR PLANE FIT   anchor ID {anchor}, {S:.1f} mm assumed\n{'=' * 68}")

    if cv_lens:
        print(f"lens from ChArUco : f {P[0, 0]:.0f} px, centre ({K[0, 2]:.0f}, {K[1, 2]:.0f}), "
              f"dist {np.array2string(D[:5], precision=4)}   (RMS {e_before:.2f} -> {e_after:.2f} mm)")
        
    else:
        print(f"distortion {'from lens file' if lens is not None else 'fit          '}: "
              f"k1 ={k1:+.5f}  k2 ={k2:+.5f}   (RMS {e_before:.2f} -> {e_after:.2f} mm)")
        
    if f_px:
        print(f"focal length      : {f_px:.0f} px{' (calibrated)' if cv_lens else ''}   "
              f"(from the floor, two constraints: {f_pair[0]:.0f} / {f_pair[1]:.0f} px)")
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
        q = apply_H(H, cv_undistort(pts, K, D, P) if cv_lens else undistort(pts, img_c, norm_s, k1, k2))
        sz = float(np.mean([np.linalg.norm(q[j] - q[(j + 1) % 4]) for j in range(4)]))
        print(f"{k:>4} {100 * rates[k]:>6.1f}% {'':>7} {sz:>9.2f}   <- outside the GCP "
              f"set; this is its implied physical size")

    resid = np.array([check[k] - S for k in check])
    print(f"\nside-length residual: mean {resid.mean():+.2f} mm, "
          f"rms {np.sqrt((resid ** 2).mean()):.2f} mm over {len(check)} sheets")

    return FloorFit(rates=rates, static=static, med=med, med_u=med_u, anchor=anchor,
                    S=S, img_c=img_c, norm_s=norm_s, k1=k1, k2=k2, H=H, f_px=f_px,
                    nadir=nadir, H_cam=H_cam, resid=resid, per_side=per_side, K=K, D=D, P=P)

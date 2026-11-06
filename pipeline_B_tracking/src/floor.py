# Floor plane for one camera: which markers are the static floor sheets, the lens
# distortion, the floor -> image homography and the camera pose.
from dataclasses import dataclass, field
from functools import cached_property

import cv2
import numpy as np
from scipy.optimize import minimize

from geometry import (apply_H, camera_from_homography, cv_distort, cv_undistort, distort,
                      side_lengths, square_error, undistort)
from util import banner


# Everything fitted for one camera.
#   med, med_u   median corners of each floor sheet in pixels, raw and undistorted
#   H            undistorted pixels -> floor mm (origin: a corner of the anchor sheet)
#   S            sheet side, mm
#   nadir, H_cam the floor point below the camera and the camera height, mm
#   resid        side length - S for each sheet through H (the scale check)
#   K, D, P      set when the lens came from a ChArUco file; cal when the fit was
#                loaded from a saved calibration
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

    def undistort(self, pts):
        if self.K is not None:
            return cv_undistort(pts, self.K, self.D, self.P)
        return undistort(pts, self.img_c, self.norm_s, self.k1, self.k2)

    def distort(self, pts):
        if self.K is not None:
            return cv_distort(pts, self.K, self.D, self.P)
        return distort(pts, self.img_c, self.norm_s, self.k1, self.k2)

    @cached_property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    def to_floor(self, px):
        return apply_H(self.H, self.undistort(px))

    def to_image(self, mm):
        return self.distort(apply_H(self.H_inv, mm))

    @property
    def has_pose(self) -> bool:
        return self.nadir is not None and bool(self.H_cam)

    # a tag h above the floor is seen where its ray hits the floor, H/(H-h) further from
    # the nadir than it really is; these two undo / redo that
    def floor_to_tag(self, p, h: float):
        return self.nadir + (p - self.nadir) * (self.H_cam - h) / self.H_cam

    def tag_to_floor(self, p, h: float):
        return self.nadir + (p - self.nadir) * self.H_cam / (self.H_cam - h)


# Fit the floor plane of one clip from its static floor sheets. Prints the fit report
# and returns a FloorFit.
def fit_floor(corners_by_frame: list, n: int, W: int, Hh: int, gcp_ids: set, veh_ids: list,
              min_rate: float, anchor_id: int | None, gcp_mm: float, no_distortion: bool,
              cam_height_mm: float | None, lens: dict | None = None) -> FloorFit:

    
    # --- which markers are the static floor sheets? ---------------------------
    rates = {k: sum(k in d for d in corners_by_frame) / n
             for k in {k for d in corners_by_frame for k in d}}

    def spread_px(k: int) -> float:
        pts = np.array([d[k].mean(axis=0) for d in corners_by_frame if k in d])

        if len(pts) < 2:
            return np.inf
        return float(np.percentile(np.linalg.norm(pts - np.median(pts, axis=0), axis=1), 90))

    # a floor sheet is seen in at least min_rate of the frames and its centre moves less
    # than 5 px (90th percentile)
    static = {k: r for k, r in rates.items()
              if k in gcp_ids and r >= min_rate and spread_px(k) < 5.0}
    if not static:
        raise SystemExit(f"no stable floor GCPs found (rates: {rates})")

    def median_corners(k: int) -> np.ndarray:
        return np.median(np.array([d[k] for d in corners_by_frame if k in d]), axis=0)

    # the median over all frames removes the detection jitter
    med = {k: median_corners(k) for k in static}
    S = gcp_mm
    square = np.array([[0, 0], [S, 0], [S, S], [0, S]], dtype=np.float64)
    img_c = np.array([W / 2.0, Hh / 2.0])
    norm_s = 0.5 * float(np.hypot(W, Hh))

    if lens is not None and (lens.get("width"), lens.get("height")) != (W, Hh):
        raise SystemExit(f"lens file is for {lens.get('width')}x{lens.get('height')}, "
                         f"clip is {W}x{Hh}")
    # a lens file holds either OpenCV's model (ChArUco) or the division model
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
    # k2 only with 5 or more sheets; fewer cannot pin down two terms
    fit_k2 = len(med) >= 5
    r2_grid = np.linspace(0.0, 1.0, 11) ** 2

    def undistort_all(a: float, b: float) -> dict:
        return {k: undistort(med[k], img_c, norm_s, a, b) for k in med}

    def fit_distortion(anc: int) -> tuple[float, float, float]:
        if cv_lens:
            return 0.0, 0.0, square_error({k: cv_undistort(med[k], K, D, P) for k in med}, anc, S)[0]
        if lens is not None:
            a, b = float(lens["k1"]), float(lens["k2"])
            return a, b, square_error(undistort_all(a, b), anc, S)[0]
        if no_distortion or len(med) < 3:
            return 0.0, 0.0, square_error(med, anc, S)[0]

        # square error for (k1, k2); distortion that rescales any radius by more than 50%
        # is rejected
        def obj(kk):
            a, b = kk[0], (kk[1] if fit_k2 else 0.0)
            f = 1.0 + a * r2_grid + b * r2_grid ** 2
            if abs(a) > 0.5 or abs(b) > 0.25 or f.min() < 0.5 or f.max() > 1.5:
                return 1e6
            return square_error(undistort_all(a, b), anc, S)[0]

        # three starting points, because Nelder-Mead stops in local minima
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
        # try each well-seen sheet as the anchor; keep the one that makes all sheets squarest
        cands = [k for k in static if rates[k] > 0.9] or list(static)
        trials = {k: fit_distortion(k) for k in cands}
        anchor = min(trials, key=lambda k: trials[k][2])
        k1, k2, e_after = trials[anchor]
        print("anchor candidates : " + ", ".join(
            f"{k}: {trials[k][2]:.1f} mm" for k in sorted(trials)) + f"  -> {anchor}")
    e_before, _, _ = square_error(med, anchor, S)

    def und(q):
        return cv_undistort(q, K, D, P) if cv_lens else undistort(q, img_c, norm_s, k1, k2)

    med_u = {k: und(med[k]) for k in med}
    e_after, per_side, _ = square_error(med_u, anchor, S)
    # first homography from the anchor's 4 corners alone
    H0, _ = cv2.findHomography(med_u[anchor], square)
    if H0 is None:
        raise SystemExit("anchor homography failed")

    # --- refine: fit the homography to EVERY stable GCP corner ----------------
    src = np.vstack([med_u[k] for k in sorted(med_u)])
    dst = np.vstack([apply_H(H0, med_u[k]) for k in sorted(med_u)])
    # the anchor-only fit sets the frame; refitting on all corners spreads the error
    # over every sheet
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if H is None:
        H = H0

    # --- camera pose, and the check against the tape measure ------------------
    f_px, C_cam, f_pair = camera_from_homography(np.linalg.inv(H), img_c,
                                                 P[0, 0] if cv_lens else None)
    nadir = C_cam[:2] if C_cam is not None else None
    H_cam = float(C_cam[2]) if C_cam is not None else None

    print(banner(f"FLOOR PLANE FIT   anchor ID {anchor}, {S:.1f} mm assumed"))
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
    # each sheet's side length through the final H
    check = {k: float(np.mean(side_lengths(apply_H(H, med_u[k])))) for k in sorted(med_u)}
    per_side.update(check)
    for k, side in check.items():
        print(f"{k:>4} {100 * rates[k]:>6.1f}% {np.mean(side_lengths(med[k])):>7.1f} {side:>9.2f} "
              f"{side - S:>+7.2f}")
    # markers that are not floor sheets (e.g. the scale bar), with their implied size
    for k in sorted(rates):
        if k in med_u or k in veh_ids or rates[k] < 0.5:
            continue
        sz = float(np.mean(side_lengths(apply_H(H, und(median_corners(k))))))
        print(f"{k:>4} {100 * rates[k]:>6.1f}% {'':>7} {sz:>9.2f}   <- outside the GCP "
              f"set; this is its implied physical size")

    resid = np.array([v - S for v in check.values()])
    print(f"\nside-length residual: mean {resid.mean():+.2f} mm, "
          f"rms {np.sqrt((resid ** 2).mean()):.2f} mm over {len(check)} sheets")

    return FloorFit(rates=rates, static=static, med=med, med_u=med_u, anchor=anchor,
                    S=S, img_c=img_c, norm_s=norm_s, k1=k1, k2=k2, H=H, f_px=f_px,
                    nadir=nadir, H_cam=H_cam, resid=resid, per_side=per_side, K=K, D=D, P=P)

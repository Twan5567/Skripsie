"""Vehicle localisation: ArUco fixes with parallax correction, the optional
median-background blob fallback, and combining + smoothing into one track."""
from dataclasses import dataclass

import cv2
import numpy as np

from floor import FloorFit
from geometry import apply_H, undistort

BG_N, BG_W = 40, 480


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


def locate_vehicle_aruco(corners_by_frame: list, n: int, veh_ids: list,
                         floor: FloorFit, tag_h: dict):
    """Vehicle position from its own ArUco tag, where it fires.

    Each tag sits above the floor, so the floor homography puts it too far from
    the nadir. With the camera height known the correction is exact and needs no
    calibration: scale the radius from the nadir by (H_cam - h) / H_cam.
    Returns (veh_px, veh_mm, veh_src, aruco_hits).
    """
    H, img_c, norm_s, k1, k2 = floor.H, floor.img_c, floor.norm_s, floor.k1, floor.k2
    nadir, H_cam = floor.nadir, floor.H_cam
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
    return veh_px, veh_mm, veh_src, aruco_hits


def blob_fallback(video: str, stride: int, frame_i: int, n: int, W: int, Hh: int,
                  veh_px: np.ndarray, max_step_px: float) -> np.ndarray:
    """Median-background subtraction, used to fill frames where the tag did not fire.

    The camera is fixed, so a per-pixel median over frames sampled across the
    whole clip is the empty floor: the car occupies any given pixel for a small
    fraction of the run, so it never wins the median.
    Returns blob_px, NaN where no blob was accepted.
    """
    blob_px = np.full((n, 2), np.nan)
    bg_samples: list[np.ndarray] = []

    cap = cv2.VideoCapture(video)
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

    cap = cv2.VideoCapture(video)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Seed on the car rather than on whatever blob is biggest in frame 0.
    seed = np.flatnonzero(np.isfinite(veh_px).all(axis=1))
    prev = tuple(veh_px[seed[0]] * scale) if len(seed) else None
    fi = ki = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fi % stride == 0:
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
                gate = max_step_px * scale
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
    return blob_px


@dataclass
class Track:
    px: np.ndarray        # smoothed, merged pixel track
    mm: np.ndarray        # smoothed, merged floor-plane track (origin = first located frame)
    px_raw: np.ndarray
    mm_raw: np.ndarray
    have: np.ndarray      # located mask
    fill: np.ndarray      # frames filled by the blob fallback
    veh_src: np.ndarray   # 'arucoNN' / 'blob' / ''
    dist: np.ndarray      # distance from start, NaN where not located
    raw_path: float
    path: float
    net: float


def combine_and_smooth(veh_px, veh_mm, veh_src, blob_px, floor: FloorFit,
                       tag_h: dict, smooth: int) -> Track:
    """ArUco where trusted, blob to fill; then shift the origin and smooth."""
    H, img_c, norm_s, k1, k2 = floor.H, floor.img_c, floor.norm_s, floor.k1, floor.k2
    nadir, H_cam = floor.nadir, floor.H_cam

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
    mm = smooth_xy(mm_raw, have, smooth)
    px = smooth_xy(px_raw, have, smooth)

    raw_path = float(np.linalg.norm(np.diff(mm_raw[have], axis=0), axis=1).sum())
    path = float(np.linalg.norm(np.diff(mm[have], axis=0), axis=1).sum())
    net = float(np.linalg.norm(mm[have][-1] - mm[have][0]))

    # Distance from start, used by the overlay HUD (and the plots).
    n = len(px)
    dist = np.full(n, np.nan)
    dist[have] = np.linalg.norm(mm[have] - mm[have][0], axis=1)

    return Track(px=px, mm=mm, px_raw=px_raw, mm_raw=mm_raw, have=have, fill=fill,
                 veh_src=veh_src, dist=dist, raw_path=raw_path, path=path, net=net)

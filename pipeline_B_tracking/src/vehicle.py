from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from floor import FloorFit
from util import runs


# Savitzky-Golay smoothing, applied to each unbroken run of fixes on its own, so it
# never smooths across a gap. win = window in samples; below 5 means no smoothing.
def smooth_xy(arr: np.ndarray, ok: np.ndarray, win: int, poly: int = 2):
    out = arr.copy()
    if win < 5:
        return out
    for run in runs(ok):
        w = min(win, run.size)
        if w % 2 == 0:                       # Savitzky-Golay needs an odd window
            w -= 1
        if w < 5:                            # also covers run.size < 5
            continue
        for c in (0, 1):
            out[run, c] = savgol_filter(arr[run, c], w, min(poly, w - 1))
    return out


# The car's floor position (mm) in every frame with a tag fix: tag centre ->
# undistort -> floor homography -> parallax correction for the tag height. Returns
# pixel and mm positions, the source of each fix and the number of fixes.
def locate_vehicle_aruco(corners_by_frame: list, n: int, veh_ids: list, floor: FloorFit, tag_h: dict):
    veh_px = np.full((n, 2), np.nan)
    veh_mm = np.full((n, 2), np.nan)
    veh_src = np.full(n, "", dtype=object)
    par_shift = []
    for i, d in enumerate(corners_by_frame):
        vid = next((v for v in veh_ids if v in d), None)    # first listed tag that was seen
        if vid is None:
            continue
        c_px = d[vid].mean(axis=0)
        raw = floor.to_floor(c_px).reshape(2)
        cor = floor.floor_to_tag(raw, tag_h.get(vid, 0.0)) if floor.has_pose else raw
        if floor.has_pose:
            par_shift.append(float(np.linalg.norm(cor - raw)))
        veh_px[i], veh_mm[i], veh_src[i] = c_px, cor, f"aruco{vid}"

    if par_shift:
        print(f"parallax correction : mean {np.mean(par_shift):.1f} mm, "
              f"max {np.max(par_shift):.1f} mm  (tag heights {tag_h})")
    return veh_px, veh_mm, veh_src, int(np.isfinite(veh_px).all(axis=1).sum())


# One camera's vehicle track. mm is relative to the first fix (origin holds that
# fix); *_raw are before smoothing; have marks the frames with a fix; dist is the
# distance from the start.
@dataclass
class Track:
    px: np.ndarray
    mm: np.ndarray
    origin: np.ndarray
    px_raw: np.ndarray
    mm_raw: np.ndarray
    have: np.ndarray
    veh_src: np.ndarray
    dist: np.ndarray
    raw_path: float
    path: float
    net: float


def path_length(xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


# Build the Track: shift to the first fix, smooth, measure the path lengths.
# allow_empty returns an empty track instead of stopping (with two cameras the other
# one may see the car).
def combine_and_smooth(veh_px, veh_mm, veh_src, smooth: int, allow_empty: bool = False) -> Track:
    have = np.isfinite(veh_px).all(axis=1)
    if not have.any():
        if not allow_empty:
            raise SystemExit("vehicle never located")
        print("vehicle never located in this camera — empty track (the other camera carries the run)")
        nan2 = np.full((len(veh_px), 2), np.nan)
        return Track(px=nan2, mm=nan2.copy(), origin=np.zeros(2), px_raw=nan2.copy(),
                     mm_raw=nan2.copy(), have=have, veh_src=veh_src,
                     dist=np.full(len(veh_px), np.nan), raw_path=0.0, path=0.0, net=0.0)

    mm_raw = veh_mm.copy()
    origin = mm_raw[have][0].copy()
    mm_raw[have] -= origin
    px_raw = veh_px.copy()
    mm = smooth_xy(mm_raw, have, smooth)
    px = smooth_xy(px_raw, have, smooth)
    dist = np.full(len(px), np.nan)
    dist[have] = np.linalg.norm(mm[have] - mm[have][0], axis=1)
    return Track(px=px, mm=mm, origin=origin, px_raw=px_raw, mm_raw=mm_raw, have=have,
                 veh_src=veh_src, dist=dist, raw_path=path_length(mm_raw[have]),
                 path=path_length(mm[have]), net=float(np.linalg.norm(mm[have][-1] - mm[have][0])))

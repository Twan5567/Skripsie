
from dataclasses import dataclass

import numpy as np

from floor import FloorFit
from geometry import apply_H, lens_undistort

def smooth_xy(arr: np.ndarray, ok: np.ndarray, win: int, poly: int = 2):

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


def locate_vehicle_aruco(corners_by_frame: list, n: int, veh_ids: list, floor: FloorFit, tag_h: dict):

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
                raw = apply_H(H, lens_undistort(floor, c_px)).reshape(2)
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


def combine_and_smooth(veh_px, veh_mm, veh_src, smooth: int, allow_empty: bool = False) -> Track:

    px = veh_px.copy()
    have = np.isfinite(px).all(axis=1)

    if not have.any():
        if not allow_empty:
            raise SystemExit("vehicle never located")
        print("vehicle never located in this camera — empty track (the other camera carries the run)")
        nan2 = np.full((len(px), 2), np.nan)
        return Track(px=nan2, mm=nan2.copy(), origin=np.zeros(2), px_raw=nan2.copy(),
                     mm_raw=nan2.copy(), have=have, veh_src=veh_src,
                     dist=np.full(len(px), np.nan), raw_path=0.0, path=0.0, net=0.0)

    mm = veh_mm.copy()
    origin = mm[have][0].copy()
    mm[have] -= origin


    mm_raw = mm.copy()
    px_raw = px.copy()
    mm = smooth_xy(mm_raw, have, smooth)
    px = smooth_xy(px_raw, have, smooth)

    raw_path = float(np.linalg.norm(np.diff(mm_raw[have], axis=0), axis=1).sum())
    path = float(np.linalg.norm(np.diff(mm[have], axis=0), axis=1).sum())
    net = float(np.linalg.norm(mm[have][-1] - mm[have][0]))

    n = len(px)
    dist = np.full(n, np.nan)
    dist[have] = np.linalg.norm(mm[have] - mm[have][0], axis=1)

    return Track(px=px, mm=mm, origin=origin, px_raw=px_raw, mm_raw=mm_raw, have=have,
                 veh_src=veh_src, dist=dist, raw_path=raw_path, path=path, net=net)

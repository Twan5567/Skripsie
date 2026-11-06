# Two-camera fusion: both cameras' fixes on one time grid, then a Kalman filter with
# an RTS smoother (or a tag-size-weighted mean), and the tag-height check.
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from geometry import side_lengths
from track_ground import ClipResult
from util import banner, runs
from vehicle import path_length, smooth_xy
from world import Rigid2D


# one camera's fixes on the common grid: world mm, tag size in px, and the clip frame
# each grid sample falls on (-1 = outside the clip)
@dataclass
class CamTrack:
    xy: np.ndarray
    tagpx: np.ndarray
    frame: np.ndarray


# The fused track. t = grid times; xy_raw = weighted mean; xy = smoothed; src = A, B,
# AB or K (K = bridged by the filter); disagree = A-B distance where both see the car.
@dataclass
class Fused:
    t: np.ndarray
    a: CamTrack
    b: CamTrack
    xy_raw: np.ndarray
    xy: np.ndarray
    src: np.ndarray
    have: np.ndarray
    disagree: np.ndarray
    tag_h: float


def _tag_px(res: ClipResult) -> np.ndarray:
    # apparent tag size (px) in each frame with a fix
    out = np.full(res.det.n, np.nan)
    for i, d in enumerate(res.det.corners_by_frame):
        s = res.track.veh_src[i]
        if isinstance(s, str) and s.startswith("aruco") and int(s[5:]) in d:
            out[i] = np.mean(side_lengths(d[int(s[5:])]))
    return out


def _anchor_mm(res: ClipResult, h_old: float, h_new: float) -> np.ndarray:
    # the fixes, re-corrected for parallax with tag height h_new instead of h_old
    mm = res.track.mm_raw + res.track.origin
    fl = res.floor
    if not fl.has_pose or h_new == h_old:
        return mm
    return fl.floor_to_tag(fl.tag_to_floor(mm, h_old), h_new)


def _on_grid(t_src: np.ndarray, vals: np.ndarray, have: np.ndarray, grid: np.ndarray,
             step: float) -> np.ndarray:
    # linear interpolation of a camera's samples onto the common time grid, only between
    # two valid samples
    out = np.full((len(grid),) + vals.shape[1:], np.nan)
    j = (grid - t_src[0]) / step
    j0 = np.floor(j).astype(int)
    frac = j - j0
    ok = (j0 >= 0) & (j0 < len(t_src) - 1)

    exact = ok & (np.abs(frac) < 1e-6)
    idx = j0[exact]
    hit = have[idx]
    out[np.flatnonzero(exact)[hit]] = vals[idx[hit]]

    between = ok & ~exact
    idx = j0[between]
    hit = have[idx] & have[idx + 1]
    f = frac[between][hit].reshape((-1,) + (1,) * (vals.ndim - 1))
    out[np.flatnonzero(between)[hit]] = (1 - f) * vals[idx[hit]] + f * vals[idx[hit] + 1]

    last = np.abs(j - (len(t_src) - 1)) < 1e-6
    if last.any() and have[-1]:
        out[last] = vals[-1]
    return out


def _cam_on_grid(res: ClipResult, to_world: Rigid2D, h_old: float, h_new: float,
                 grid: np.ndarray, step: float, fps: float) -> CamTrack:
    mm = _anchor_mm(res, h_old, h_new)
    have = res.track.have
    xy_w = np.full_like(mm, np.nan)
    xy_w[have] = to_world(mm[have])
    tagpx = _tag_px(res)
    t = res.t_global
    fr = np.rint((grid - t[0]) * fps).astype(int)
    fr[(fr < 0) | (fr >= res.det.frame_i)] = -1
    return CamTrack(xy=_on_grid(t, xy_w, have, grid, step),
                    tagpx=_on_grid(t, tagpx, have & np.isfinite(tagpx), grid, step), frame=fr)


def _merge(a: CamTrack, b: CamTrack):
    # tag-size-weighted mean where both cameras have a fix
    ha = np.isfinite(a.xy).all(axis=1)
    hb = np.isfinite(b.xy).all(axis=1)
    n = len(ha)

    def weight(c, h):
        ww = c.tagpx.copy()
        ww[h & ~np.isfinite(ww)] = np.nanmedian(ww) if np.isfinite(ww).any() else 1.0
        ww[~h] = 0.0
        return ww

    wa, wb = weight(a, ha), weight(b, hb)
    both = ha & hb
    xy = np.full((n, 2), np.nan)
    xy[both] = (wa[both, None] * a.xy[both] + wb[both, None] * b.xy[both]) / (wa[both] + wb[both])[:, None]
    xy[ha & ~hb] = a.xy[ha & ~hb]
    xy[hb & ~ha] = b.xy[hb & ~ha]
    src = np.full(n, "", dtype=object)
    src[ha & ~hb] = "A"
    src[hb & ~ha] = "B"
    src[both] = "AB"
    dis = np.full(n, np.nan)
    dis[both] = np.linalg.norm(a.xy[both] - b.xy[both], axis=1)
    return xy, src, ha | hb, dis


def kalman_smooth(a: CamTrack, b: CamTrack, step: float, max_gap: int, sigma_mm: float = 25.0,
                  ref_px: float = 100.0, accel_mm_s2: float = 4000.0):
    # constant-velocity model, state (x, y, vx, vy); forward filter, then RTS smoother
    n = len(a.xy)
    F = np.eye(4)
    F[0, 2] = F[1, 3] = step
    G = np.array([0.5 * step ** 2, 0.5 * step ** 2, step, step])
    Q = np.outer(G, G) * accel_mm_s2 ** 2 * np.kron(np.ones((2, 2)), np.eye(2))
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    # measurement noise grows as the tag shrinks in the image (further away): sigma_mm
    # at ref_px, never below half of it
    def meas(c: CamTrack, i: int):
        if not np.isfinite(c.xy[i]).all():
            return None
        px = c.tagpx[i] if np.isfinite(c.tagpx[i]) else np.nanmedian(c.tagpx)
        px = px if np.isfinite(px) else ref_px
        return c.xy[i], np.eye(2) * (sigma_mm * max(ref_px / px, 0.5)) ** 2

    has = np.isfinite(a.xy).all(axis=1) | np.isfinite(b.xy).all(axis=1)
    if not has.any():
        return np.full((n, 2), np.nan), np.zeros(n, bool), np.zeros(n, bool)
    first, last = np.flatnonzero(has)[[0, -1]]

    # forward pass
    xs, Ps, xps, Pps = [], [], [], []
    m0 = meas(a, first) or meas(b, first)
    x = np.r_[m0[0], 0.0, 0.0]
    P = np.diag([m0[1][0, 0], m0[1][1, 1], 3000.0 ** 2, 3000.0 ** 2])
    for i in range(first, last + 1):
        if i > first:
            x = F @ x
            P = F @ P @ F.T + Q
        xps.append(x.copy())
        Pps.append(P.copy())
        for z, R in filter(None, (meas(a, i), meas(b, i))):
            K = P @ Hm.T @ np.linalg.inv(Hm @ P @ Hm.T + R)
            x = x + K @ (z - Hm @ x)
            P = (np.eye(4) - K @ Hm) @ P
        xs.append(x.copy())
        Ps.append(P.copy())

    # backward (RTS) pass
    xs_s = xs.copy()
    for k in range(len(xs) - 2, -1, -1):
        C = Ps[k] @ F.T @ np.linalg.inv(Pps[k + 1])
        xs_s[k] = xs[k] + C @ (xs_s[k + 1] - xps[k + 1])

    xy = np.full((n, 2), np.nan)
    xy[first:last + 1] = np.array([v[:2] for v in xs_s])
    # gaps of up to max_gap samples are bridged by the smoother; longer gaps stay empty
    gap = ~has
    gap[:first] = gap[last + 1:] = False
    keep = np.zeros(n, bool)
    for run in runs(gap):
        if run.size and run.size <= max_gap:
            keep[run] = True
    xy[~(has | keep)] = np.nan
    return xy, keep, has | keep


# Fuse two clips. Both need a timecode. The grid runs at the analysed frame rate over
# both clips' time span, on A's frame times. tag_h_used = the tag height the fixes
# were computed with, tag_h = the one to fuse with.
def fuse(res_a: ClipResult, res_b: ClipResult, a_to_world: Rigid2D, b_to_world: Rigid2D,
         veh_ids: list[int], tag_h_used: float, tag_h: float, stride: int, smooth: int,
         method: str = "mean", max_gap: int = 15) -> Fused:
    if res_a.t_global is None or res_b.t_global is None:
        raise SystemExit("both clips need a timecode (tmcd) for two-camera fusion")
    fps = res_a.det.fps
    if abs(fps - res_b.det.fps) > 0.01:
        raise SystemExit(f"frame rates differ: {fps} vs {res_b.det.fps}")
    step = stride / fps
    ta, tb = res_a.t_global, res_b.t_global
    k0 = int(np.floor((min(ta[0], tb[0]) - ta[0]) / step))
    k1 = int(np.ceil((max(ta[-1], tb[-1]) - ta[0]) / step))
    grid = ta[0] + np.arange(k0, k1 + 1) * step

    a = _cam_on_grid(res_a, a_to_world, tag_h_used, tag_h, grid, step, fps)
    b = _cam_on_grid(res_b, b_to_world, tag_h_used, tag_h, grid, step, fps)
    xy_raw, src, have, dis = _merge(a, b)
    if method == "kalman":
        xy, filled, have = kalman_smooth(a, b, step, max_gap)
        src = src.copy()
        src[filled] = "K"
    else:
        xy = smooth_xy(xy_raw, have, smooth)
    return Fused(t=grid, a=a, b=b, xy_raw=xy_raw, xy=xy, src=src, have=have, disagree=dis, tag_h=tag_h)


def solve_tag_height(res_a, res_b, a_to_world, b_to_world, veh_ids, tag_h_used, stride,
                     lo: float = 0.0, hi: float = 400.0) -> tuple[float, float, float]:
    # the tag height at which the two cameras agree best
    def rms(h):
        d = fuse(res_a, res_b, a_to_world, b_to_world, veh_ids, tag_h_used, h, stride, 0).disagree
        d = d[np.isfinite(d)]
        return float(np.sqrt((d ** 2).mean())) if d.size else np.inf

    r = minimize_scalar(rms, bounds=(lo, hi), method="bounded", options=dict(xatol=0.5))
    return float(r.x), rms(tag_h_used), float(r.fun)


def print_fusion_summary(f: Fused, stem_a: str, stem_b: str) -> None:
    n = len(f.t)
    ha = np.isfinite(f.a.xy).all(axis=1)
    hb = np.isfinite(f.b.xy).all(axis=1)
    d = f.disagree[np.isfinite(f.disagree)]
    print(banner(f"TWO-CAMERA FUSION   A = {stem_a}, B = {stem_b}"))
    print(f"grid                : {n} samples at {np.median(np.diff(f.t)) * 1000:.2f} ms "
          f"({f.t[-1] - f.t[0]:.1f} s)")
    for label, h in (("A", ha), ("B", hb), ("both", ha & hb), ("either", ha | hb)):
        print(f"{'located by ' + label:<20}: {h.sum():5d} ({100 * h.mean():5.1f}%)")
    nk = int((f.src == "K").sum())
    if nk:
        print(f"bridged by Kalman   : {nk:5d} ({100 * nk / n:5.1f}%)  -> track covers "
              f"{f.have.sum()} ({100 * f.have.mean():.1f}%)")
    if d.size:
        print(f"A-B disagreement    : rms {np.sqrt((d ** 2).mean()):.1f} mm, "
              f"median {np.median(d):.1f} mm, 95% {np.percentile(d, 95):.1f} mm, "
              f"max {d.max():.1f} mm  (over {d.size} co-visible samples)")
    print(f"tag height used     : {f.tag_h:.0f} mm")
    path = path_length(f.xy[f.have])
    print(f"fused path length   : {path:.0f} mm ({path / 1000:.2f} m)")

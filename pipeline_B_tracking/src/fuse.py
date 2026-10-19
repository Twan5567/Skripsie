from dataclasses import dataclass

import numpy as np

from track_ground import ClipResult
from vehicle import smooth_xy
from world import Rigid2D


@dataclass
class CamTrack:
    
    xy: np.ndarray        
    tagpx: np.ndarray     
    frame: np.ndarray    


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


def _tag_px(res: ClipResult, veh_ids: list[int]) -> np.ndarray:
   
    out = np.full(res.det.n, np.nan)

    for i, d in enumerate(res.det.corners_by_frame):
        s = res.track.veh_src[i]


        if isinstance(s, str) and s.startswith("aruco"):

            k = int(s[5:])

            if k in d:
                q = d[k]
                out[i] = np.mean([np.linalg.norm(q[j] - q[(j + 1) % 4]) for j in range(4)])


    return out


def _anchor_mm(res: ClipResult, h_old: float, h_new: float) -> np.ndarray:

    mm = res.track.mm_raw + res.track.origin
    fl = res.floor
    if fl.nadir is None or not fl.H_cam or h_new == h_old:
        return mm
    raw = fl.nadir + (mm - fl.nadir) * fl.H_cam / (fl.H_cam - h_old)
    return fl.nadir + (raw - fl.nadir) * (fl.H_cam - h_new) / fl.H_cam


def _on_grid(t_src: np.ndarray, vals: np.ndarray, have: np.ndarray, grid: np.ndarray, step: float) -> np.ndarray:

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
    sel = np.flatnonzero(between)[hit]
    f = frac[between][hit]
    out[sel] = (1 - f)[:, None] * vals[idx[hit]] + f[:, None] * vals[idx[hit] + 1] \
        if vals.ndim == 2 else (1 - f) * vals[idx[hit]] + f * vals[idx[hit] + 1]
    
   
    last = (np.abs(j - (len(t_src) - 1)) < 1e-6)
    if last.any() and have[-1]:
        out[last] = vals[-1]
    return out


def _cam_on_grid(res: ClipResult, to_world: Rigid2D, veh_ids: list[int], h_old: float, h_new: float, grid: np.ndarray, step: float, fps: float) -> CamTrack:
    
    mm = _anchor_mm(res, h_old, h_new)
    have = res.track.have
    xy_w = np.full_like(mm, np.nan)
    xy_w[have] = to_world(mm[have])
    tagpx = _tag_px(res, veh_ids)
    t = res.t_global
    xy = _on_grid(t, xy_w, have, grid, step)
    tp = _on_grid(t, tagpx, have & np.isfinite(tagpx), grid, step)
    fr = np.rint((grid - t[0]) * fps).astype(int)
    fr[(fr < 0) | (fr >= res.det.frame_i)] = -1

    return CamTrack(xy=xy, tagpx=tp, frame=fr)


def _merge(a: CamTrack, b: CamTrack):
    ha = np.isfinite(a.xy).all(axis=1)
    hb = np.isfinite(b.xy).all(axis=1)
    n = len(ha)


    def w(c, h):
        ww = c.tagpx.copy()
        med = np.nanmedian(ww) if np.isfinite(ww).any() else 1.0
        ww[h & ~np.isfinite(ww)] = med
        ww[~h] = 0.0
        return ww

    
    wa, wb = w(a, ha), w(b, hb)
    xy = np.full((n, 2), np.nan)
    both = ha & hb
    xy[both] = (wa[both, None] * a.xy[both] + wb[both, None] * b.xy[both]) \
        / (wa[both] + wb[both])[:, None]
    xy[ha & ~hb] = a.xy[ha & ~hb]
    xy[hb & ~ha] = b.xy[hb & ~ha]
    src = np.full(n, "", dtype=object)
    src[ha & ~hb] = "A"
    src[hb & ~ha] = "B"
    src[both] = "AB"
    dis = np.full(n, np.nan)
    dis[both] = np.linalg.norm(a.xy[both] - b.xy[both], axis=1)
    return xy, src, ha | hb, dis


def kalman_smooth(a: CamTrack, b: CamTrack, step: float, max_gap: int, sigma_mm: float = 25.0, ref_px: float = 100.0, accel_mm_s2: float = 4000.0):

    n = len(a.xy)
    F = np.eye(4)
    F[0, 2] = F[1, 3] = step
    q = accel_mm_s2 ** 2
    G = np.array([0.5 * step ** 2, 0.5 * step ** 2, step, step])
    Q = np.outer(G, G) * q
    Q[0, 1] = Q[1, 0] = Q[0, 3] = Q[3, 0] = Q[1, 2] = Q[2, 1] = Q[2, 3] = Q[3, 2] = 0
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    def meas(c: CamTrack, i: int):
        if not np.isfinite(c.xy[i]).all():
            return None
        px = c.tagpx[i] if np.isfinite(c.tagpx[i]) else np.nanmedian(c.tagpx)
        px = px if np.isfinite(px) else ref_px
        sg = sigma_mm * max(ref_px / px, 0.5)
        return c.xy[i], np.eye(2) * sg ** 2

    has = np.array([meas(a, i) is not None or meas(b, i) is not None for i in range(n)])
    if not has.any():
        return np.full((n, 2), np.nan), np.zeros(n, bool), np.zeros(n, bool)
    first, last = np.flatnonzero(has)[[0, -1]]

    # forward pass
    xs, Ps, xps, Pps = [], [], [], []
    x = np.zeros(4)
    m0 = meas(a, first) or meas(b, first)
    x[:2] = m0[0]
    P = np.diag([m0[1][0, 0], m0[1][1, 1], 3000.0 ** 2, 3000.0 ** 2])
    for i in range(first, last + 1):
        if i > first:
            x = F @ x
            P = F @ P @ F.T + Q
        xps.append(x.copy()); Pps.append(P.copy())
        for c in (a, b):
            m = meas(c, i)
            if m is None:
                continue
            z, R = m
            S = Hm @ P @ Hm.T + R
            K = P @ Hm.T @ np.linalg.inv(S)
            x = x + K @ (z - Hm @ x)
            P = (np.eye(4) - K @ Hm) @ P
        xs.append(x.copy()); Ps.append(P.copy())

    # backward (RTS) pass
    m = len(xs)
    xs_s = [None] * m
    xs_s[-1] = xs[-1]
    Ps_s = Ps[-1]
    for k in range(m - 2, -1, -1):
        C = Ps[k] @ F.T @ np.linalg.inv(Pps[k + 1])
        xs_s[k] = xs[k] + C @ (xs_s[k + 1] - xps[k + 1])
        Ps_s = Ps[k] + C @ (Ps_s - Pps[k + 1]) @ C.T

    xy = np.full((n, 2), np.nan)
    xy[first:last + 1] = np.array([v[:2] for v in xs_s])
    
    idx = np.flatnonzero(~has[first:last + 1]) + first
    keep = np.zeros(n, bool)
    for run in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
        if run.size and run.size <= max_gap:
            keep[run] = True
    xy[~(has | keep)] = np.nan
    return xy, keep, has | keep


def fuse(res_a: ClipResult, res_b: ClipResult, a_to_world: Rigid2D, b_to_world: Rigid2D, veh_ids: list[int], tag_h_used: float, tag_h: float, stride: int, smooth: int, method: str = "mean", max_gap: int = 15) -> Fused:
    
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

    a = _cam_on_grid(res_a, a_to_world, veh_ids, tag_h_used, tag_h, grid, step, fps)
    b = _cam_on_grid(res_b, b_to_world, veh_ids, tag_h_used, tag_h, grid, step, fps)
    xy_raw, src, have, dis = _merge(a, b)


    if method == "kalman":
        xy, filled, have2 = kalman_smooth(a, b, step, max_gap)
        src = src.copy()
        src[filled] = "K"
        have = have2
    else:
        xy = smooth_xy(xy_raw, have, smooth)
    return Fused(t=grid, a=a, b=b, xy_raw=xy_raw, xy=xy, src=src, have=have,
                 disagree=dis, tag_h=tag_h)


def solve_tag_height(res_a, res_b, a_to_world, b_to_world, veh_ids, tag_h_used, stride, lo: float = 0.0, hi: float = 400.0) -> tuple[float, float, float]:

    from scipy.optimize import minimize_scalar

    def rms(h):
        f = fuse(res_a, res_b, a_to_world, b_to_world, veh_ids, tag_h_used, h, stride, 0)
        d = f.disagree[np.isfinite(f.disagree)]
        return float(np.sqrt((d ** 2).mean())) if d.size else np.inf

    r = minimize_scalar(rms, bounds=(lo, hi), method="bounded", options=dict(xatol=0.5))
    return float(r.x), rms(tag_h_used), float(r.fun)


def print_fusion_summary(f: Fused, stem_a: str, stem_b: str) -> None:


    n = len(f.t)
    ha = np.isfinite(f.a.xy).all(axis=1)
    hb = np.isfinite(f.b.xy).all(axis=1)
    d = f.disagree[np.isfinite(f.disagree)]
    
    print(f"\n{'=' * 68}\nTWO-CAMERA FUSION   A = {stem_a}, B = {stem_b}\n{'=' * 68}")
    print(f"grid                : {n} samples at {np.median(np.diff(f.t)) * 1000:.2f} ms "
          f"({f.t[-1] - f.t[0]:.1f} s)")
    print(f"located by A        : {ha.sum():5d} ({100 * ha.mean():5.1f}%)")
    print(f"located by B        : {hb.sum():5d} ({100 * hb.mean():5.1f}%)")
    print(f"located by both     : {(ha & hb).sum():5d} ({100 * (ha & hb).mean():5.1f}%)")
    print(f"located by either   : {(ha | hb).sum():5d} ({100 * (ha | hb).mean():5.1f}%)")
    nk = int((f.src == "K").sum())
    if nk:
        print(f"bridged by Kalman   : {nk:5d} ({100 * nk / n:5.1f}%)  -> track covers "
              f"{f.have.sum()} ({100 * f.have.mean():.1f}%)")
    if d.size:
        print(f"A-B disagreement    : rms {np.sqrt((d ** 2).mean()):.1f} mm, "
              f"median {np.median(d):.1f} mm, 95% {np.percentile(d, 95):.1f} mm, "
              f"max {d.max():.1f} mm  (over {d.size} co-visible samples)")
    print(f"tag height used     : {f.tag_h:.0f} mm")
    xy = f.xy[f.have]
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    print(f"fused path length   : {path:.0f} mm ({path / 1000:.2f} m)")

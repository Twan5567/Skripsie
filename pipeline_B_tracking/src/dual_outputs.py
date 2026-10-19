import csv
import os

import cv2
import numpy as np

from fuse import Fused
from geometry import apply_H, lens_distort
from timecode import fmt_tod
from track_ground import ClipResult
from world import WorldFit


def write_fused_csv(path: str, f: Fused) -> None:
    def cell(v, fmt="{:.2f}"):
        return fmt.format(v) if np.isfinite(v) else ""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_global_s", "source", "x_mm", "y_mm",
                    "xA_mm", "yA_mm", "xB_mm", "yB_mm", "disagree_mm",
                    "tagpx_A", "tagpx_B", "frame_A", "frame_B"])
        for i in range(len(f.t)):
            w.writerow([f"{f.t[i]:.3f}", f.src[i],
                        cell(f.xy[i, 0]) if f.have[i] else "",
                        cell(f.xy[i, 1]) if f.have[i] else "",
                        cell(f.a.xy[i, 0]), cell(f.a.xy[i, 1]),
                        cell(f.b.xy[i, 0]), cell(f.b.xy[i, 1]),
                        cell(f.disagree[i], "{:.1f}"),
                        cell(f.a.tagpx[i], "{:.1f}"), cell(f.b.tagpx[i], "{:.1f}"),
                        f.a.frame[i] if f.a.frame[i] >= 0 else "",
                        f.b.frame[i] if f.b.frame[i] >= 0 else ""])
    print(f"  wrote {path}")


def write_topdown(path: str, f: Fused, wf: WorldFit, title: str) -> None:
    
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    for k, q in sorted(wf.layout.items()):
        ax.add_patch(plt.Polygon(q, closed=True, fc="0.85", ec="0.45", lw=0.8, zorder=1))
        ax.text(*q.mean(axis=0), str(k), ha="center", va="center", fontsize=7,
                color="0.3", zorder=2)
    ha = np.isfinite(f.a.xy).all(axis=1)
    hb = np.isfinite(f.b.xy).all(axis=1)
    ax.plot(f.a.xy[ha, 0], f.a.xy[ha, 1], ".", ms=2.5, color="#B3400C", alpha=0.6,
            label=f"camera A fixes ({ha.sum()})", zorder=3)
    ax.plot(f.b.xy[hb, 0], f.b.xy[hb, 1], ".", ms=2.5, color="#1F5FBF", alpha=0.6,
            label=f"camera B fixes ({hb.sum()})", zorder=3)
    
    
    xy = f.xy.copy()
    xy[~f.have] = np.nan
    ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.4, color="#0A6273",
            label=f"fused, smoothed ({f.have.sum()})", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def _world_to_px(xy_w: np.ndarray, res: ClipResult, to_world) -> np.ndarray:
   
    fl = res.floor
    mm = to_world.inv()(xy_w)
    und = apply_H(np.linalg.inv(fl.H), mm)
    return lens_distort(fl, und)


class _Seq:
   
    def __init__(self, video: str):
        self.cap = cv2.VideoCapture(video)
        self.i = -1
        self.frame = None

    def get(self, want: int):
        if want < 0:
            return None
        while self.i < want:
            ok, fr = self.cap.read()
            if not ok:
                return None
            self.i += 1
            self.frame = fr
        return self.frame if self.i == want else None

    def release(self):
        self.cap.release()


def write_dual_overlay(path: str, f: Fused, res_a: ClipResult, res_b: ClipResult, wf: WorldFit, panel_w: int = 1920) -> None:

   
    pw = panel_w
    ph = int(round(res_a.det.Hh * pw / res_a.det.W / 2) * 2)
    fps_out = 1.0 / float(np.median(np.diff(f.t)))
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (2 * pw, ph))
    ra, rb = _Seq(res_a.video), _Seq(res_b.video)


    px_a = np.full_like(f.xy, np.nan)
    px_b = np.full_like(f.xy, np.nan)
    px_a[f.have] = _world_to_px(f.xy[f.have], res_a, wf.a_to_world)
    px_b[f.have] = _world_to_px(f.xy[f.have], res_b, wf.b_to_world)

    def panel(seq, res, want, own_xy_px, fused_px, k, label):
        sc = pw / res.det.W
        
        lim = 2.0 * max(res.det.W, res.det.Hh)
        drawable = lambda p: np.isfinite(p).all(axis=-1) & (np.abs(p) < lim).all(axis=-1)
        fr = seq.get(want)
        if fr is None:
            img = np.zeros((ph, pw, 3), np.uint8)
            cv2.putText(img, f"{label}: not recording", (40, ph // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (120, 120, 120), 3, cv2.LINE_AA)
            return img
        img = cv2.resize(fr, (pw, ph), interpolation=cv2.INTER_AREA)

        
        col = (0, 140, 255) if res.floor.cal else (140, 140, 140)
        for g in sorted(res.floor.static):
            cv2.polylines(img, [(res.floor.med[g] * sc).astype(np.int32)], True, col, 2)
            cv2.putText(img, f"{g} cal" if res.floor.cal else str(g),
                        tuple((res.floor.med[g][0] * sc).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

       
        ok = drawable(fused_px[: k + 1]) & f.have[: k + 1]
        idx = np.flatnonzero(ok)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
            if run.size > 1:
                cv2.polylines(img, [(fused_px[run] * sc).astype(np.int32)], False,
                              (0, 255, 255), 3)
        if f.have[k] and drawable(fused_px[k]):
            cv2.circle(img, tuple(int(v) for v in fused_px[k] * sc), 11, (0, 255, 255), 2)
        if drawable(own_xy_px):
            cv2.circle(img, tuple(int(v) for v in own_xy_px * sc), 8, (0, 0, 255), -1)
        cv2.putText(img, label, (30, ph - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return img

    n = len(f.t)
    
    for k in range(n):

        
        def own(res):
            fr = f.a.frame[k] if res is res_a else f.b.frame[k]
            if fr < 0:
                return np.array([np.nan, np.nan])
            i = int(round(fr / res.stride))
            if 0 <= i < res.det.n and res.track.have[i]:
                return res.track.px[i]
            return np.array([np.nan, np.nan])
        left = panel(ra, res_a, f.a.frame[k], own(res_a), px_a, k, f"A  {res_a.stem}")
        right = panel(rb, res_b, f.b.frame[k], own(res_b), px_b, k, f"B  {res_b.stem}")
        img = np.hstack([left, right])
        d = f.disagree[k]
        hud = [f"t {fmt_tod(f.t[k])}   src {f.src[k] or '--':>2}   "
               + (f"A-B {d:6.0f} mm" if np.isfinite(d) else "A-B     --")]
        for line_i, line in enumerate(hud):
            for colr, th in ((0, 0, 0), 6), ((255, 255, 255), 2):
                cv2.putText(img, line, (30, 46 + line_i * 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, colr, th, cv2.LINE_AA)
        writer.write(img)
        if (k + 1) % 200 == 0:
            print(f"  overlay {k + 1}/{n}", end="\r", flush=True)
    ra.release()
    rb.release()
    writer.release()
    print(f"  wrote {path}" + " " * 20)

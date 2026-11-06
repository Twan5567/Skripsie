import csv

import cv2
import matplotlib
import numpy as np

from fuse import Fused
from outputs import draw_gcps, draw_hud
from timecode import fmt_tod
from track_ground import ClipResult
from util import runs
from world import WorldFit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)


# one row per grid sample; empty cells where a value is missing
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
                        *((cell(v) for v in f.xy[i]) if f.have[i] else ("", "")),
                        *map(cell, f.a.xy[i]), *map(cell, f.b.xy[i]),
                        cell(f.disagree[i], "{:.1f}"),
                        cell(f.a.tagpx[i], "{:.1f}"), cell(f.b.tagpx[i], "{:.1f}"),
                        f.a.frame[i] if f.a.frame[i] >= 0 else "",
                        f.b.frame[i] if f.b.frame[i] >= 0 else ""])
    print(f"  wrote {path}")


# top-down plot in world mm: sheet layout, each camera's fixes, the fused track
def write_topdown(path: str, f: Fused, wf: WorldFit, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for k, q in sorted(wf.layout.items()):
        ax.add_patch(plt.Polygon(q, closed=True, fc="0.85", ec="0.45", lw=0.8, zorder=1))
        ax.text(*q.mean(axis=0), str(k), ha="center", va="center", fontsize=7, color="0.3", zorder=2)
    for cam, c, col in (("A", f.a, "#B3400C"), ("B", f.b, "#1F5FBF")):
        h = np.isfinite(c.xy).all(axis=1)
        ax.plot(c.xy[h, 0], c.xy[h, 1], ".", ms=2.5, color=col, alpha=0.6,
                label=f"camera {cam} fixes ({h.sum()})", zorder=3)
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


class _Seq:
    # forward-only frame access by frame number (the two clips advance at their own rate)
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


# Both views side by side at the grid rate: each camera's own fix (red dot) and the
# fused track projected into both views (yellow). Look at this to judge the tracking.
def write_dual_overlay(path: str, f: Fused, res_a: ClipResult, res_b: ClipResult, wf: WorldFit,
                       panel_w: int = 1920) -> None:
    pw = panel_w
    ph = int(round(res_a.det.Hh * pw / res_a.det.W / 2) * 2)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / float(np.median(np.diff(f.t))),
                             (2 * pw, ph))

    def fused_px(res, to_world):
        # the fused track projected into this camera (no parallax: it is drawn on the floor)
        px = np.full_like(f.xy, np.nan)
        px[f.have] = res.floor.to_image(to_world.inv()(f.xy[f.have]))
        return px

    def own(res, frames, k):
        # this camera's own fix at grid sample k, if it has one
        fr = frames[k]
        i = int(round(fr / res.stride)) if fr >= 0 else -1
        return res.track.px[i] if 0 <= i < res.det.n and res.track.have[i] else np.array([np.nan, np.nan])

    def panel(seq, res, want, own_px, fpx, k, label):
        sc = pw / res.det.W
        lim = 2.0 * max(res.det.W, res.det.Hh)

        def drawable(p):
            return np.isfinite(p).all(axis=-1) & (np.abs(p) < lim).all(axis=-1)

        fr = seq.get(want)
        if fr is None:
            img = np.zeros((ph, pw, 3), np.uint8)
            cv2.putText(img, f"{label}: not recording", (40, ph // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (120, 120, 120), 3, cv2.LINE_AA)
            return img
        img = cv2.resize(fr, (pw, ph), interpolation=cv2.INTER_AREA)
        draw_gcps(img, res.floor, sc)
        for run in runs(drawable(fpx[: k + 1]) & f.have[: k + 1]):
            if run.size > 1:
                cv2.polylines(img, [(fpx[run] * sc).astype(np.int32)], False, (0, 255, 255), 3)
        if f.have[k] and drawable(fpx[k]):
            cv2.circle(img, tuple(int(v) for v in fpx[k] * sc), 11, (0, 255, 255), 2)
        if drawable(own_px):
            cv2.circle(img, tuple(int(v) for v in own_px * sc), 8, (0, 0, 255), -1)
        cv2.putText(img, label, (30, ph - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    cams = [(_Seq(r.video), r, c.frame, fused_px(r, tw), f"{ab}  {r.stem}")
            for r, c, tw, ab in ((res_a, f.a, wf.a_to_world, "A"), (res_b, f.b, wf.b_to_world, "B"))]
    n = len(f.t)
    for k in range(n):
        img = np.hstack([panel(seq, r, frames[k], own(r, frames, k), fpx, k, label)
                         for seq, r, frames, fpx, label in cams])
        d = f.disagree[k]
        draw_hud(img, [f"t {fmt_tod(f.t[k])}   src {f.src[k] or '--':>2}   "
                       + (f"A-B {d:6.0f} mm" if np.isfinite(d) else "A-B     --")])
        writer.write(img)
        if (k + 1) % 200 == 0:
            print(f"  overlay {k + 1}/{n}", end="\r", flush=True)
    for seq, *_ in cams:
        seq.release()
    writer.release()
    print(f"  wrote {path}" + " " * 20)

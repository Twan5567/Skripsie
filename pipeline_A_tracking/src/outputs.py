"""Outputs: trajectory CSV, plots, overlay video, and the summary print."""
import csv
import os

import cv2
import numpy as np

from floor import FloorFit
from geometry import apply_H
from vehicle import Track


def write_csv(path: str, track: Track, veh_px: np.ndarray, stride: int, fps: float) -> None:
    px, mm, have, veh_src = track.px, track.mm, track.have, track.veh_src
    n = len(px)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "source", "px_x", "px_y", "mm_x", "mm_y",
                    "aruco_px_x", "aruco_px_y"])
        for i in range(n):
            fr = i * stride
            a = veh_px[i]
            w.writerow([fr, f"{fr / fps:.5f}", veh_src[i] or "",
                        f"{px[i, 0]:.2f}" if have[i] else "",
                        f"{px[i, 1]:.2f}" if have[i] else "",
                        f"{mm[i, 0]:.2f}" if have[i] else "",
                        f"{mm[i, 1]:.2f}" if have[i] else "",
                        f"{a[0]:.2f}" if np.isfinite(a).all() else "",
                        f"{a[1]:.2f}" if np.isfinite(a).all() else ""])
    print(f"\n  wrote {path}")


def write_plots(outdir: str, stem: str, track: Track, floor: FloorFit,
                stride: int, fps: float, smooth: int) -> None:
    """Trajectory and displacement plots.

    Currently DISABLED (body commented out, as in the original script) pending
    a rework. The aliases below map the original variable names onto the
    Track/FloorFit fields so the block can be uncommented and run as written.
    """
    mm, mm_raw, have, veh_src, dist = track.mm, track.mm_raw, track.have, track.veh_src, track.dist
    static, med, H = floor.static, floor.med, floor.H
    n = len(mm)
    args_smooth = smooth  # original code used args.smooth
    _ = (mm, mm_raw, have, veh_src, dist, static, med, H, n, args_smooth, apply_H, outdir, stem, stride, fps)

    # # --- plots ----------------------------------------------------------------
    # import matplotlib

    # matplotlib.use("Agg")
    # import matplotlib.pyplot as plt

    # t = np.arange(n) * args.stride / fps

    # is_ar = np.array([s.startswith("aruco") for s in veh_src]) & have


    # fig, ax = plt.subplots(figsize=(9.5, 7))

    # for k in sorted(static):
    #     q = apply_H(H, med[k])
    #     ax.add_patch(plt.Polygon(q, closed=True, fc="0.85", ec="0.45", lw=0.8, zorder=1))
    #     ax.text(*q.mean(axis=0), str(k), ha="center", va="center",
    #             fontsize=7, color="0.3", zorder=2)
    # if args.smooth >= 5:
    #     ax.plot(mm_raw[have, 0], mm_raw[have, 1], "-", lw=0.7, color="#9BB7BF",
    #             label="raw", zorder=2.5)
    # ax.plot(mm[have, 0], mm[have, 1], "-", lw=1.6, color="#0A6273",
    #         label=f"smoothed (w={args.smooth})" if args.smooth >= 5
    #         else f"trajectory ({int(have.sum())} pts)", zorder=3)
    # if is_ar.any():
    #     ax.plot(mm[is_ar, 0], mm[is_ar, 1], "o", ms=4, color="#B3400C",
    #             label=f"vehicle ArUco fix ({int(is_ar.sum())})", zorder=4)
    # ax.plot(*mm[have][0], "go", ms=9, label="start", zorder=5)
    # ax.plot(*mm[have][-1], "rs", ms=9, label="end", zorder=5)
    # ax.set_aspect("equal")
    # ax.set_xlabel("x (mm)")
    # ax.set_ylabel("y (mm)")
    # ax.set_title(f"{stem} — vehicle trajectory, floor-plane coordinates")
    # ax.grid(alpha=0.3)
    # ax.legend(loc="best", fontsize=8)
    # p1 = os.path.join(args.outdir, f"{stem}_trajectory.png")
    # fig.tight_layout()
    # fig.savefig(p1, dpi=150)
    # plt.close(fig)

    # dist = np.full(n, np.nan)
    # dist[have] = np.linalg.norm(mm[have] - mm[have][0], axis=1)
    # fig, ax = plt.subplots(figsize=(9.5, 4.5))
    # ax.plot(t[have], dist[have], "-", lw=1.3, color="#0A6273")
    # if is_ar.any():
    #     ax.plot(t[is_ar], dist[is_ar], "o", ms=3.5, color="#B3400C",
    #             label="vehicle ArUco fix")
    #     ax.legend(fontsize=8)
    # ax.set_xlabel("time (s)")
    # ax.set_ylabel("distance from start (mm)")
    # ax.set_title(f"{stem} — displacement over time")
    # ax.grid(alpha=0.3)
    # p2 = os.path.join(args.outdir, f"{stem}_distance.png")
    # fig.tight_layout()
    # fig.savefig(p2, dpi=150)
    # plt.close(fig)
    # print(f"  wrote {p1}\n  wrote {p2}")


def write_overlay(video: str, out_path: str, track: Track, floor: FloorFit,
                  stride: int, fps: float, W: int, Hh: int, overlay_width: int) -> None:
    px, have, veh_src, dist = track.px, track.have, track.veh_src, track.dist
    static, med = floor.static, floor.med
    n = len(px)

    ow = overlay_width
    oh = int(round(Hh * ow / W / 2) * 2)
    sc = ow / W
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / stride, (ow, oh))
    cap = cv2.VideoCapture(video)
    fi = ki = 0
    while True:
        ok, fr = cap.read()
        if not ok or ki >= n:
            break
        if fi % stride == 0:
            fr = cv2.resize(fr, (ow, oh), interpolation=cv2.INTER_AREA)
            for k in sorted(static):
                cv2.polylines(fr, [(med[k] * sc).astype(np.int32)], True,
                              (140, 140, 140), 2)
                cv2.putText(fr, str(k), tuple((med[k][0] * sc).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 2,
                            cv2.LINE_AA)
            trail = px[: ki + 1][have[: ki + 1]] * sc
            if len(trail) > 1:
                cv2.polylines(fr, [trail.astype(np.int32)], False, (0, 255, 255), 3)
            if have[ki]:
                col = (0, 0, 255) if veh_src[ki].startswith("aruco") else (0, 200, 0)
                cv2.circle(fr, tuple((px[ki] * sc).astype(int)), 9, col, -1)
            hud = [f"frame {fi:5d}   t {fi / fps:6.2f}s   src {veh_src[ki] or '--':>8}",
                   f"dist {dist[ki]:8.1f} mm" if have[ki] else "dist --"]
            for r, line in enumerate(hud):
                for colr, th in ((0, 0, 0), 6), ((255, 255, 255), 2):
                    cv2.putText(fr, line, (30, 46 + r * 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, colr, th, cv2.LINE_AA)
            writer.write(fr)
            ki += 1
            if ki % 200 == 0:
                print(f"  overlay {ki}/{n}", end="\r", flush=True)
        fi += 1
    cap.release()
    writer.release()
    print(f"  wrote {out_path}" + " " * 20)


def print_summary(stem: str, W: int, Hh: int, fps: float, frame_i: int, n: int,
                  stride: int, floor: FloorFit, track: Track, aruco_hits: int,
                  smooth: int) -> None:
    resid, S = floor.resid, floor.S
    print(f"\n{'=' * 66}")
    print(f"{stem}   {W}x{Hh} @ {fps:.2f} fps, {frame_i} frames "
          f"({n} analysed, stride {stride})")
    print("=" * 66)
    print(f"floor GCPs used     : {sorted(floor.static)}  (anchor {floor.anchor})")
    print(f"vehicle ArUco fixes : {aruco_hits}/{n} ({100 * aruco_hits / n:.1f}%)")
    print(f"blob-filled frames  : {int(track.fill.sum())}/{n} ({100 * track.fill.sum() / n:.1f}%)")
    print(f"located total       : {int(track.have.sum())}/{n} ({100 * track.have.mean():.1f}%)")
    print(f"path length (raw)   : {track.raw_path:.0f} mm ({track.raw_path / 1000:.2f} m)  "
          f"[jitter-inflated]")
    print(f"path length (smooth): {track.path:.0f} mm ({track.path / 1000:.2f} m)  "
          f"[Savitzky-Golay w={smooth}]")
    print(f"net displacement    : {track.net:.0f} mm")
    print(f"scale check (rms)   : {np.sqrt((resid ** 2).mean()):.2f} mm on a {S:.0f} mm sheet "
          f"({100 * np.sqrt((resid ** 2).mean()) / S:.2f}%)")

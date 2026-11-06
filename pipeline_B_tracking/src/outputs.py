import csv

import cv2
import numpy as np

from detector import Video
from floor import FloorFit
from vehicle import Track


# one row per analysed frame: frame, times, source, smoothed px and mm, raw tag px
def write_csv(path: str, track: Track, veh_px: np.ndarray, stride: int, fps: float,
              t_global: np.ndarray | None = None) -> None:
    def cells(vals, ok):
        return [f"{v:.2f}" if ok else "" for v in vals]

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "time_global_s", "source", "px_x", "px_y", "mm_x", "mm_y",
                    "aruco_px_x", "aruco_px_y"])
        for i in range(len(track.px)):
            fr = i * stride
            have = track.have[i]
            w.writerow([fr, f"{fr / fps:.5f}",
                        f"{t_global[i]:.3f}" if t_global is not None else "",
                        track.veh_src[i] or "",
                        *cells(track.px[i], have), *cells(track.mm[i], have),
                        *cells(veh_px[i], np.isfinite(veh_px[i]).all())])
    print(f"\n  wrote {path}")


def draw_gcps(img: np.ndarray, floor: FloorFit, sc: float) -> None:
    col = (0, 140, 255) if floor.cal else (140, 140, 140)
    for k in sorted(floor.static):
        cv2.polylines(img, [(floor.med[k] * sc).astype(np.int32)], True, col, 2)
        cv2.putText(img, f"{k} cal" if floor.cal else str(k), tuple((floor.med[k][0] * sc).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)


def draw_hud(img: np.ndarray, lines: list[str]) -> None:
    for r, line in enumerate(lines):
        for colr, th in ((0, 0, 0), 6), ((255, 255, 255), 2):
            cv2.putText(img, line, (30, 46 + r * 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colr, th,
                        cv2.LINE_AA)


# the clip with the floor sheets, the track so far and the current fix drawn on it;
# look at this to judge the tracking
def write_overlay(video: str, out_path: str, track: Track, floor: FloorFit, stride: int, fps: float,
                  W: int, Hh: int, overlay_width: int) -> None:
    px, have, veh_src, dist = track.px, track.have, track.veh_src, track.dist
    n = len(px)
    ow = overlay_width
    oh = int(round(Hh * ow / W / 2) * 2)
    sc = ow / W
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (ow, oh))
    with Video(video) as vid:
        for ki, fr in vid.frames(stride):
            if ki >= n:
                break
            fi = ki * stride
            fr = cv2.resize(fr, (ow, oh), interpolation=cv2.INTER_AREA)
            draw_gcps(fr, floor, sc)
            trail = px[: ki + 1][have[: ki + 1]] * sc
            if len(trail) > 1:
                cv2.polylines(fr, [trail.astype(np.int32)], False, (0, 255, 255), 3)
            if have[ki]:
                col = (0, 0, 255) if veh_src[ki].startswith("aruco") else (0, 200, 0)
                cv2.circle(fr, tuple((px[ki] * sc).astype(int)), 9, col, -1)
            draw_hud(fr, [f"frame {fi:5d}   t {fi / fps:6.2f}s   src {veh_src[ki] or '--':>8}",
                          f"dist {dist[ki]:8.1f} mm" if have[ki] else "dist --"])
            writer.write(fr)
            if (ki + 1) % 200 == 0:
                print(f"  overlay {ki + 1}/{n}", end="\r", flush=True)
    writer.release()
    print(f"  wrote {out_path}" + " " * 20)


# end-of-run summary for one camera
def print_summary(stem: str, W: int, Hh: int, fps: float, frame_i: int, n: int, stride: int,
                  floor: FloorFit, track: Track, aruco_hits: int, smooth: int) -> None:
    rms, S = np.sqrt((floor.resid ** 2).mean()), floor.S
    print(f"\n{'=' * 66}")
    print(f"{stem}   {W}x{Hh} @ {fps:.2f} fps, {frame_i} frames ({n} analysed, stride {stride})")
    print("=" * 66)
    print(f"floor GCPs used     : {sorted(floor.static)}  (anchor {floor.anchor})")
    print(f"vehicle ArUco fixes : {aruco_hits}/{n} ({100 * aruco_hits / n:.1f}%)")
    print(f"located total       : {int(track.have.sum())}/{n} ({100 * track.have.mean():.1f}%)")
    print(f"path length (raw)   : {track.raw_path:.0f} mm ({track.raw_path / 1000:.2f} m)  "
          f"[jitter-inflated]")
    print(f"path length (smooth): {track.path:.0f} mm ({track.path / 1000:.2f} m)  "
          f"[Savitzky-Golay w={smooth}]")
    print(f"net displacement    : {track.net:.0f} mm")
    print(f"scale check (rms)   : {rms:.2f} mm on a {S:.0f} mm sheet ({100 * rms / S:.2f}%)")

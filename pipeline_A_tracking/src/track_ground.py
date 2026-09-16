"""Vehicle trajectory from a fixed overhead camera, via a floor-GCP homography.

Run:
  python track_ground.py --video /path/to/clip.MP4   (or use ../run_tracking.sh)

This file is only the command line and the order of operations. The work is in:
  detector.py   tuned ArUco detector, ID parsing, pass 1 over the video
  geometry.py   undistortion, square-error metric, camera-from-homography
  floor.py      static-GCP selection, distortion fit, homography, camera pose
  vehicle.py    ArUco vehicle fixes + parallax, blob fallback, combine + smooth
  outputs.py    CSV, plots (currently disabled), overlay video, summary
"""
from gpu import init_opencv_opencl

init_opencv_opencl()   # must run before anything imports cv2

import argparse
import os

import numpy as np

from detector import build_detector, detect_all_frames, parse_ids
from floor import fit_floor
from outputs import print_summary, write_csv, write_overlay, write_plots
from vehicle import blob_fallback, combine_and_smooth, locate_vehicle_aruco


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", default="../output")
    ap.add_argument("--gcp-ids", default="0-11", help="IDs of the flat floor sheets")
    ap.add_argument("--gcp-mm", type=float, default=150.0,
                    help="black-square side of the floor sheets, mm")
    ap.add_argument("--vehicle-ids", default="20,21")
    ap.add_argument("--anchor-id", type=int, default=None,
                    help="GCP that sets scale (default: best-detected, largest)")
    ap.add_argument("--min-rate", type=float, default=0.60,
                    help="detection rate a GCP needs to join the floor fit")
    ap.add_argument("--stride", type=int, default=1, help="analyse every Nth frame")
    ap.add_argument("--no-fallback", action="store_true",
                    help="skip background-subtraction tracking")
    ap.add_argument("--cam-height-mm", type=float, default=None,
                    help="measured camera height, for cross-checking the recovered pose")
    ap.add_argument("--tag-heights", default="20:57,21:67",
                    help="height of each vehicle tag above the floor, mm (id:mm,...)")
    ap.add_argument("--no-distortion", action="store_true",
                    help="skip the radial-distortion fit (leave k1=k2=0)")
    ap.add_argument("--max-step-px", type=float, default=260.0,
                    help="largest plausible per-sample car motion, full-res pixels")
    ap.add_argument("--smooth", type=int, default=15,
                    help="Savitzky-Golay window in samples; 0 disables smoothing")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--overlay-width", type=int, default=1920)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    # One folder per capture keeps overlay/CSV/plots for a run together.
    args.outdir = os.path.join(args.outdir, stem)
    os.makedirs(args.outdir, exist_ok=True)
    gcp_ids = set(parse_ids(args.gcp_ids))
    veh_ids = parse_ids(args.vehicle_ids)
    tag_h = {int(a): float(b) for a, b in
             (t.split(":") for t in args.tag_heights.split(",") if t.strip())}

    # --- pass 1: detect every marker in every stride-th frame ---------------
    det = detect_all_frames(args.video, args.stride, build_detector())

    # --- floor plane from the static GCP sheets -----------------------------
    floor = fit_floor(det.corners_by_frame, det.n, det.W, det.Hh,
                      gcp_ids, veh_ids, args.min_rate, args.anchor_id,
                      args.gcp_mm, args.no_distortion, args.cam_height_mm)

    # --- vehicle: ArUco where it fires, blob to fill (unless disabled) ------
    veh_px, veh_mm, veh_src, aruco_hits = locate_vehicle_aruco(
        det.corners_by_frame, det.n, veh_ids, floor, tag_h)
    if args.no_fallback:
        blob_px = np.full((det.n, 2), np.nan)
    else:
        blob_px = blob_fallback(args.video, args.stride, det.frame_i, det.n,
                                det.W, det.Hh, veh_px, args.max_step_px)
    track = combine_and_smooth(veh_px, veh_mm, veh_src, blob_px, floor, tag_h, args.smooth)

    # --- outputs -------------------------------------------------------------
    write_csv(os.path.join(args.outdir, f"{stem}_trajectory.csv"),
              track, veh_px, args.stride, det.fps)
    write_plots(args.outdir, stem, track, floor, args.stride, det.fps, args.smooth)
    if not args.no_overlay:
        write_overlay(args.video, os.path.join(args.outdir, f"{stem}_overlay.mp4"),
                      track, floor, args.stride, det.fps, det.W, det.Hh, args.overlay_width)
    print_summary(stem, det.W, det.Hh, det.fps, det.frame_i, det.n, args.stride,
                  floor, track, aruco_hits, args.smooth)


if __name__ == "__main__":
    main()

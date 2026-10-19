"""
Run:
  python track_ground.py --video /path/to/clip.MP4   (or use ../run_tracking.sh)

"""
from gpu import init_opencv_opencl

init_opencv_opencl()   # must run before anything imports cv2

import argparse
import os
from dataclasses import dataclass

import numpy as np

from detector import Detections, build_detector, detect_all_frames, parse_ids
from floor import FloorFit, fit_floor
from outputs import print_summary, write_csv, write_overlay, write_plots
from timecode import clip_start_seconds, fmt_tod, frame_times
from calcache import (check_identity, clip_identity, floor_from_dict, load_cal,
                      move_check, save_cal)
from vehicle import Track, combine_and_smooth, locate_vehicle_aruco


def add_tracking_args(ap: argparse.ArgumentParser) -> None:

    ap.add_argument("--gcp-ids", default="0-11", help="IDs of the flat floor sheets")
    ap.add_argument("--gcp-mm", type=float, default=150.0,
                    help="black-square side of the floor sheets, mm")
    ap.add_argument("--vehicle-ids", default="20")
    ap.add_argument("--anchor-id", type=int, default=None,
                    help="GCP that sets scale (default: best-detected, largest)")
    ap.add_argument("--min-rate", type=float, default=0.60,
                    help="detection rate a GCP needs to join the floor fit")
    ap.add_argument("--stride", type=int, default=1, help="analyse every Nth frame")
    ap.add_argument("--cam-height-mm", type=float, default=None,
                    help="measured camera height, for cross-checking the recovered pose")
    ap.add_argument("--tag-heights", default="20:198",
                    help="height of each vehicle tag above the floor, mm (id:mm,...)")
    ap.add_argument("--lens", default=None,
                    help="lens JSON from calibrate_charuco.py for this camera body; fixes the "
                         "distortion instead of fitting it from this clip's sheets")
    ap.add_argument("--no-distortion", action="store_true",
                    help="skip the radial-distortion fit (leave k1=k2=0)")
    ap.add_argument("--max-step-px", type=float, default=260.0,
                    help="largest plausible per-sample car motion, full-res pixels")
    ap.add_argument("--smooth", type=int, default=15,
                    help="Savitzky-Golay window in samples; 0 disables smoothing")
    ap.add_argument("--no-window-search", action="store_true",
                    help="disable the windowed, upscaled second look for the vehicle tag")
    ap.add_argument("--start-s", type=float, default=0.0,
                    help="ignore vehicle fixes before this many seconds into the clip "
                         "(the floor fit still uses the whole clip)")
    ap.add_argument("--save-cal", default=None, metavar="NAME",
                    help="save this clip's floor fit (and, two cameras, the world fit) as "
                         "calib/NAME.json, for clips shot after the sheets were lifted")
    ap.add_argument("--load-cal", default=None, metavar="NAME",
                    help="use calib/NAME.json instead of fitting the floor on this clip "
                         "(calibrate, then lift)")
    ap.add_argument("--correct-motion", action="store_true",
                    help="with --load-cal: fold the measured camera movement since the "
                         "calibration clip into the floor fit (default: report it only)")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--overlay-width", type=int, default=1920)


def parse_args() -> argparse.Namespace:

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", default="../output")
    add_tracking_args(ap)
    return ap.parse_args()


def parse_tag_heights(spec: str) -> dict[int, float]:
    return {int(a): float(b) for a, b in
            (t.split(":") for t in spec.split(",") if t.strip())}


@dataclass
class ClipResult:
   
    stem: str
    video: str
    outdir: str
    stride: int
    det: Detections
    floor: FloorFit
    track: Track
    veh_px: np.ndarray          
    aruco_hits: int
    t_global: np.ndarray | None  
    ident: dict | None = None    


def process_clip(video: str, outdir: str, args: argparse.Namespace, write_outputs: bool = True) -> ClipResult:

    stem = os.path.splitext(os.path.basename(video))[0]

    if args.save_cal and args.load_cal:
        raise SystemExit("--save-cal and --load-cal together: calibrate on one clip, load on others")
    if args.load_cal and args.lens:
        raise SystemExit("--lens with --load-cal: the calibration already carries the lens "
                         "it was fitted with")
    if args.correct_motion and not args.load_cal:
        raise SystemExit("--correct-motion only applies with --load-cal")
    
  
    outdir = os.path.join(outdir, stem)
    os.makedirs(outdir, exist_ok=True)
    gcp_ids = set(parse_ids(args.gcp_ids))
    veh_ids = parse_ids(args.vehicle_ids)
    tag_h = parse_tag_heights(args.tag_heights)

    
    ident = clip_identity(video)
    cal = None
    if args.load_cal:
        cal = load_cal(args.load_cal)
        if ident["CASN"] not in cal["cameras"]:
            raise SystemExit(f"camera {ident['CASN']} ({stem}) is not in {cal['_path']} "
                             f"(it has {sorted(cal['cameras'])})")
        check_identity(cal["cameras"][ident["CASN"]]["identity"], ident, video)

    # --- pass 1: detect every marker in every stride-th frame ---------------
    det = detect_all_frames(video, args.stride, build_detector(),
                            veh_ids=None if args.no_window_search else veh_ids,
                            max_step_px=args.max_step_px)
    t0 = clip_start_seconds(video, det.fps)
    t_global = frame_times(t0, det.n, args.stride, det.fps)
    if t0 is not None:
        print(f"  timecode start {fmt_tod(t0)} (seconds of day {t0:.3f})")

    # --- floor plane: from a calibration clip, or from this clip's sheets ----
    if args.load_cal:
        floor = floor_from_cal(cal, args, video, ident, outdir, stem)
    else:
        floor = fit_floor_here(args, det, gcp_ids, veh_ids)

    # --- vehicle: ArUco fixes ------------------------------------------------
    veh_px, veh_mm, veh_src, aruco_hits = locate_vehicle_aruco(
        det.corners_by_frame, det.n, veh_ids, floor, tag_h)
    if args.start_s > 0:
    
        early = np.arange(det.n) * args.stride / det.fps < args.start_s
        veh_px[early] = np.nan
        veh_mm[early] = np.nan
        veh_src[early] = ""
        aruco_hits = int(np.isfinite(veh_px).all(axis=1).sum())
        print(f"  --start-s {args.start_s:g}: fixes in the first {int(early.sum())} "
              f"analysed frames discarded")
    track = combine_and_smooth(veh_px, veh_mm, veh_src, args.smooth,
                               allow_empty=getattr(args, "allow_no_vehicle", False))

    res = ClipResult(stem=stem, video=video, outdir=outdir, stride=args.stride,
                     det=det, floor=floor, track=track,
                     veh_px=veh_px, aruco_hits=aruco_hits, t_global=t_global, ident=ident)
    if write_outputs:
        write_clip_outputs(res, args)
    return res


def floor_from_cal(cal: dict, args: argparse.Namespace, video: str, ident: dict, outdir: str, stem: str) -> FloorFit:
    
   
    serial = ident["CASN"]
    c = cal["cameras"][serial]
    floor = floor_from_dict(c["floor"], cal["name"])
    print(f"\n{'=' * 68}\nFLOOR PLANE FROM CALIBRATION   {cal['name']} "
          f"(clip {c['clip']}, camera {serial})\n{'=' * 68}")
    print(f"sheets            : {sorted(floor.static)}  anchor {floor.anchor}, side rms "
          f"{np.sqrt((floor.resid ** 2).mean()):.2f} mm on the calibration clip")

    if floor.H_cam is not None:
        print(f"camera height     : {floor.H_cam:.0f} mm, focal {floor.f_px:.0f} px "
              f"(from the calibration clip)")
    return move_check(cal, serial, floor, video, outdir, stem, args.correct_motion)


def fit_floor_here(args: argparse.Namespace, det: Detections, gcp_ids: set, veh_ids: list) -> FloorFit:

    
    lens = None
    if args.lens:

        import json

        with open(args.lens) as fh:
            lens = json.load(fh)

        if lens.get("model") == "opencv":
            print(f"  lens: {args.lens} (ChArUco, camera {lens.get('camera')} "
                  f"{lens.get('lens_mode')}, {lens.get('views')} views, {lens.get('rms_px', 0):.2f} px)")
            
        else:
            print(f"  lens: {args.lens} (camera {lens.get('camera')}, {lens.get('poses')} poses)")

    return fit_floor(det.corners_by_frame, det.n, det.W, det.Hh,
                     gcp_ids, veh_ids, args.min_rate, args.anchor_id,
                     args.gcp_mm, args.no_distortion, args.cam_height_mm, lens)


def relocate_vehicle(res: ClipResult, args: argparse.Namespace) -> None:

    veh_ids = parse_ids(args.vehicle_ids)
    tag_h = parse_tag_heights(args.tag_heights)
    veh_px, veh_mm, veh_src, hits = locate_vehicle_aruco(
        res.det.corners_by_frame, res.det.n, veh_ids, res.floor, tag_h)
    
    if args.start_s > 0:
        early = np.arange(res.det.n) * args.stride / res.det.fps < args.start_s
        veh_px[early] = np.nan
        veh_mm[early] = np.nan
        veh_src[early] = ""
        hits = int(np.isfinite(veh_px).all(axis=1).sum())

    res.track = combine_and_smooth(veh_px, veh_mm, veh_src, args.smooth,
                                   allow_empty=getattr(args, "allow_no_vehicle", False))
    res.veh_px, res.aruco_hits = veh_px, hits


def write_clip_outputs(res: ClipResult, args: argparse.Namespace) -> None:
   
    stem, outdir, det, floor, track = res.stem, res.outdir, res.det, res.floor, res.track

    write_csv(os.path.join(outdir, f"{stem}_trajectory.csv"),
              track, res.veh_px, args.stride, det.fps, res.t_global)
    write_plots(outdir, stem, track, floor, args.stride, det.fps, args.smooth)

    if not args.no_overlay:
        write_overlay(res.video, os.path.join(outdir, f"{stem}_overlay.mp4"),
                      track, floor, args.stride, det.fps, det.W, det.Hh, args.overlay_width)
    print_summary(stem, det.W, det.Hh, det.fps, det.frame_i, det.n, args.stride,
                  floor, track, res.aruco_hits, args.smooth)


def main() -> None:
    
    args = parse_args()
    res = process_clip(args.video, args.outdir, args)
    if args.save_cal:
        save_cal(args.save_cal, [res])


if __name__ == "__main__":
    main()

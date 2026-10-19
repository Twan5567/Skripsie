"""
  python track_dual.py --video-a A.MP4 --video-b B.MP4   (or ../run_dual.sh A B)

Calibrate, then lift (floor sheets down only for the calibration clips):
  ../run_dual.sh A-1 B-1 -- --save-cal S3_cal1    # sheets down: fit, save calib/S3_cal1.json
  ../run_dual.sh A-2 B-2 -- --load-cal S3_cal1    # sheets lifted: reuse it
"""
from gpu import init_opencv_opencl

init_opencv_opencl()  

import argparse
import os

import numpy as np

from detector import parse_ids
from dual_outputs import write_dual_overlay, write_fused_csv, write_topdown
from fuse import fuse, print_fusion_summary, solve_tag_height
from timecode import fmt_tod
from crossguide import guide
from track_ground import (add_tracking_args, parse_tag_heights, process_clip,
                          relocate_vehicle, write_clip_outputs)
from calcache import load_cal, save_cal, world_from_cal
from world import fit_world, fit_world_map, load_map_corners


def parse_args() -> argparse.Namespace:
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-a", required=True)
    ap.add_argument("--video-b", required=True)
    ap.add_argument("--outdir", default="../output")
    ap.add_argument("--name", default=None, help="output folder name (default <A>_<B>)")
    ap.add_argument("--world-origin", type=int, default=1,
                    help="GCP whose centre is the world origin")
    ap.add_argument("--world-x", type=int, default=5,
                    help="GCP whose centre the world +x axis points at")
    ap.add_argument("--map-anchor", default=None,
                    help="mapping run NAME (pipeline_A_mapping/output/<NAME>/<NAME>_metric.json): "
                         "fit each camera to the map's sheet corners instead of joining the "
                         "cameras through the sheets they share")
    ap.add_argument("--start-a", type=float, default=0.0,
                    help="seconds into clip A before which fixes are ignored")
    ap.add_argument("--start-b", type=float, default=0.0,
                    help="seconds into clip B before which fixes are ignored")
    ap.add_argument("--lens-a", default=None, help="lens JSON for camera A")
    ap.add_argument("--lens-b", default=None, help="lens JSON for camera B")
    ap.add_argument("--fusion", choices=["kalman", "mean"], default="kalman",
                    help="kalman: constant-velocity filter + RTS smoother over both cameras' "
                         "fixes, bridging gaps up to --max-gap; mean: tag-size-weighted mean "
                         "then Savitzky-Golay")
    ap.add_argument("--max-gap", type=int, default=15,
                    help="longest gap (samples) the Kalman fusion bridges by prediction")
    ap.add_argument("--no-cross-guide", action="store_true",
                    help="skip the pass where each camera searches for the tag where "
                         "the other camera saw the car")
    ap.add_argument("--solve-tag-height", action="store_true",
                    help="use the tag height that best reconciles the two cameras "
                         "(it is always reported)")
    add_tracking_args(ap)
    return ap.parse_args()


def main() -> None:

    args = parse_args()
    stem_a = os.path.splitext(os.path.basename(args.video_a))[0]
    stem_b = os.path.splitext(os.path.basename(args.video_b))[0]

    name = args.name or f"{stem_a}_{stem_b}"
    outdir = os.path.join(args.outdir, name)

    os.makedirs(outdir, exist_ok=True)
    veh_ids = parse_ids(args.vehicle_ids)
    tag_h = parse_tag_heights(args.tag_heights)
    h_used = tag_h.get(veh_ids[0], 0.0)

    # --- the two single-camera runs, each into its own subfolder ---------------
    import copy
    args_a, args_b = copy.copy(args), copy.copy(args)
    args_a.start_s, args_b.start_s = args.start_a, args.start_b
    args_a.lens, args_b.lens = args.lens_a, args.lens_b
  
    args_a.allow_no_vehicle = args_b.allow_no_vehicle = True
    print(f"\n######## camera A: {stem_a} ########")
    res_a = process_clip(args.video_a, outdir, args_a, write_outputs=False)
    print(f"\n######## camera B: {stem_b} ########")
    res_b = process_clip(args.video_b, outdir, args_b, write_outputs=False)
    if res_a.t_global is not None and res_b.t_global is not None:
        print(f"\nclip starts         : A {fmt_tod(res_a.t_global[0])}   "
              f"B {fmt_tod(res_b.t_global[0])}   "
              f"(B - A = {res_b.t_global[0] - res_a.t_global[0]:+.3f} s)")

    if not res_a.track.have.any() and not res_b.track.have.any():
        raise SystemExit("vehicle never located by either camera")

    # --- one world frame -------------------------------------------------------
    if args.load_cal:

        wf = world_from_cal(load_cal(args.load_cal), res_a.ident["CASN"], res_b.ident["CASN"])
    else:
        if args.map_anchor:
            mpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                 "pipeline_A_mapping", "output", args.map_anchor,
                                 f"{args.map_anchor}_metric.json")
            if not os.path.exists(mpath):
                raise SystemExit(f"--map-anchor {args.map_anchor}: no {mpath}")
            corners, args.world_origin, args.world_x = load_map_corners(mpath)
            wf = fit_world_map(res_a.floor, res_b.floor, corners, args.map_anchor)
        else:
            wf = fit_world(res_a.floor, res_b.floor, args.world_origin, args.world_x)
        if args.save_cal:
            print()
            save_cal(args.save_cal, [res_a, res_b], wf, args.world_origin, args.world_x,
                     map_anchor=args.map_anchor)

    # --- each camera looks where the other one saw the car --------------------

    if not args.no_cross_guide:
        print()
        fa = res_a.aruco_hits
        fb = res_b.aruco_hits
        
        ga = guide(res_a, wf.a_to_world, res_b, wf.b_to_world, veh_ids, h_used, args.stride)
        gb = guide(res_b, wf.b_to_world, res_a, wf.a_to_world, veh_ids, h_used, args.stride)
        if ga:
            relocate_vehicle(res_a, args_a)
        if gb:
            relocate_vehicle(res_b, args_b)
        print(f"vehicle fixes       : A {fa} -> {res_a.aruco_hits}   B {fb} -> {res_b.aruco_hits}")
        
    print(f"\n######## outputs: {stem_a} ########")
    write_clip_outputs(res_a, args_a)
    print(f"\n######## outputs: {stem_b} ########")
    write_clip_outputs(res_b, args_b)

    # --- tag height: report the value the two cameras agree on -----------------
    h_best, rms_used, rms_best = solve_tag_height(
        res_a, res_b, wf.a_to_world, wf.b_to_world, veh_ids, h_used, args.stride)
    if not np.isfinite(rms_best):
        print("\ntag height solve    : not available (no samples seen by both cameras)")
    else:
        print(f"\ntag height solve    : {h_used:.0f} mm gives A-B rms {rms_used:.1f} mm; "
              f"best agreement at {h_best:.0f} mm (rms {rms_best:.1f} mm)"
              + ("   <- applied" if args.solve_tag_height else
                 "   (pass --solve-tag-height to use it)"))
    h_fuse = h_best if args.solve_tag_height and np.isfinite(rms_best) else h_used

    # --- fuse ----------------------------------------------------------------
    f = fuse(res_a, res_b, wf.a_to_world, wf.b_to_world, veh_ids, h_used, h_fuse,
             args.stride, args.smooth, args.fusion, args.max_gap)
    print_fusion_summary(f, stem_a, stem_b)

    # --- outputs ---------------------------------------------------------------
    print()
    write_fused_csv(os.path.join(outdir, f"{name}_fused.csv"), f)
    write_topdown(os.path.join(outdir, f"{name}_topdown.png"), f, wf,
                  f"{name} — fused trajectory, world frame "
                  f"(origin GCP {args.world_origin}, +x to GCP {args.world_x})")
    if not args.no_overlay:
        write_dual_overlay(os.path.join(outdir, f"{name}_overlay.mp4"), f, res_a, res_b,
                           wf, args.overlay_width)


if __name__ == "__main__":
    main()

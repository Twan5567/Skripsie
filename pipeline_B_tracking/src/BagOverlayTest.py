"""

  python src/BagOverlayTest.py --track "output/Test 4/A-4_B-4_Cal1mSave/A-4_B-4_Cal1mSave_fused.csv" \\
      --bag "../../Map and ROS bag TEST 4/antwan_bags/for_real_antwan_2026-10-17_12-10-55" \\
      --slam-fit ../pipeline_A_mapping/output/TrackMap_S4_L1_SlamTest/TrackMap_S4_L1_boundary_vs_slam.json \\
      --boundary ../pipeline_A_mapping/output/TrackMap_S4_L1/TrackMap_S4_L1_boundary.csv \\
      --date 2026-10-17 --out "output/Test 4/A-4_B-4_Cal1m_BagOverlay"

 +43.65 s, 

Writes <out>/bag_overlay.png, matched.csv, report.json.
"""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_track(path: str, sources: str | None) -> tuple[np.ndarray, np.ndarray]:
   
    t, xy = [], []
    with open(path) as fh:

        for r in csv.DictReader(fh):

            if not r["x_mm"] or (sources and r["source"] not in sources.split(",")):
                continue
            t.append(float(r["time_global_s"]))
            xy.append((float(r["x_mm"]) / 1000, float(r["y_mm"]) / 1000))

    return np.array(t), np.array(xy)


def yaw_of(q) -> float:

    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2)))


def load_topic(bag: str, topic: str, use_bag_time: bool) -> np.ndarray:

   
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    rows = []

    with AnyReader([Path(bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as rd:
        conns = [c for c in rd.connections if c.topic == topic]

        if not conns:
            raise SystemExit(f"topic {topic} not in bag")
        
        for c, t_bag, raw in rd.messages(connections=conns):

            m = rd.deserialize(raw, c.msgtype)
            st = m.header.stamp
            t = t_bag / 1e9 if use_bag_time or st.sec == 0 else st.sec + st.nanosec / 1e9

            if hasattr(m, "transform"):

                p, q = m.transform.translation, m.transform.rotation
            else:

                pose = m.pose.pose if hasattr(m.pose, "pose") else m.pose
                p, q = pose.position, pose.orientation

            rows.append((t, p.x, p.y, yaw_of(q)))
    return np.array(sorted(rows))



def slam_to_world(xy_slam: np.ndarray, fit: dict) -> np.ndarray:

    R = np.array(fit["transform"]["R"])
    t = np.array(fit["transform"]["t"])
    M = np.array(fit["transform"]["M"], dtype=float)
    return ((xy_slam - t) @ R) * M


def world_to_slam(xy_world: np.ndarray, fit: dict) -> np.ndarray:

    R = np.array(fit["transform"]["R"])
    t = np.array(fit["transform"]["t"])
    M = np.array(fit["transform"]["M"], dtype=float)
    return (xy_world * M) @ R.T + t


def solve_lever(cam_slam: np.ndarray, ref_xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
   
    c, s_ = np.cos(yaw), np.sin(yaw)
    A = np.zeros((2 * len(yaw), 2))
    A[0::2] = np.c_[c, -s_]
    A[1::2] = np.c_[s_, c]
    return np.linalg.lstsq(A, (cam_slam - ref_xy).reshape(-1), rcond=None)[0]


def apply_lever(ref_xy: np.ndarray, yaw: np.ndarray, lever: np.ndarray) -> np.ndarray:

    c, s_ = np.cos(yaw), np.sin(yaw)
    return ref_xy + np.c_[c * lever[0] - s_ * lever[1], s_ * lever[0] + c * lever[1]]


def load_walls(path: str) -> dict:

    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["loop"], []).append((float(r["x_mm"]), float(r["y_mm"])))
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:


    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, help="<run>_fused.csv")
    ap.add_argument("--bag", required=True)
    ap.add_argument("--topic", default="/pf/pose/odom")
    ap.add_argument("--slam-fit", required=True, help="<NAME>_boundary_vs_slam.json")
    ap.add_argument("--boundary", required=True, help="<NAME>_boundary.csv (walls drawn)")
    ap.add_argument("--date", required=True)
    ap.add_argument("--utc-offset", type=float, default=2.0)
    ap.add_argument("--time-offset", type=float, default=0.0,
                    help="s added to camera time to put it on the car's clock (S4: the Jetson ran "
                         "~43.6 s ahead of the GoPros)")
    ap.add_argument("--max-gap", type=float, default=0.25)
    lv = ap.add_mutually_exclusive_group()
    lv.add_argument("--lever", default=None, help="tag offset in base_link, mm: X,Y (x fwd, y left)")
    lv.add_argument("--solve-lever", action="store_true", help="estimate the tag offset on this run")
    ap.add_argument("--search-offset", type=float, default=1.0,
                    help="report the clock offset (s, +-) that minimises the median distance")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t_tod, cam = load_track(args.track, None)                     
    midnight = dt.datetime.fromisoformat(args.date).replace(
        tzinfo=dt.timezone(dt.timedelta(hours=args.utc_offset))).timestamp()
    t_cam = midnight + t_tod + args.time_offset
    ref = load_topic(args.bag, args.topic, False)                 
    fit = json.load(open(args.slam_fit))
    lever = np.zeros(2)
    if args.lever:
        lever = np.array([float(v) for v in args.lever.split(",")]) / 1000
    elif args.solve_lever:
        cam_s = world_to_slam(cam, fit)
        yaw_c = np.unwrap(ref[:, 3])
        okl = _interp(ref[:, 0], ref[:, 1:3], t_cam, args.max_gap)[1]
        xy_c, _ = _interp(ref[:, 0], ref[:, 1:3], t_cam, args.max_gap)
        yw, _ = _interp(ref[:, 0], yaw_c[:, None], t_cam, args.max_gap)
        lever = solve_lever(cam_s[okl], xy_c[okl], yw[okl, 0])
    ref_xy = apply_lever(ref[:, 1:3], ref[:, 3], lever)
    ref_w = slam_to_world(ref_xy, fit)
    print(f"lever arm (tag in base_link, x fwd / y left): {lever[0] * 1000:+.0f} / {lever[1] * 1000:+.0f} mm"
          + (" (solved on this run)" if args.solve_lever else ""))

    offs = np.arange(-args.search_offset, args.search_offset + 1e-9, 0.01)
    med = []
    for o in offs:
        xy, ok = _interp(ref[:, 0], ref_w, t_cam + o, args.max_gap)
        med.append(np.median(np.linalg.norm(xy[ok] - cam[ok], axis=1)) if ok.sum() > 10 else np.nan)
    med = np.array(med)
    best = float(offs[np.nanargmin(med)]) if np.isfinite(med).any() else 0.0

    xy0, ok0 = _interp(ref[:, 0], ref_w, t_cam, args.max_gap)
    if ok0.sum() == 0:
        raise SystemExit("no time overlap between the camera track and the bag")
    e0 = np.linalg.norm(xy0[ok0] - cam[ok0], axis=1) * 1000
    xyb, okb = _interp(ref[:, 0], ref_w, t_cam + best, args.max_gap)
    eb = np.linalg.norm(xyb[okb] - cam[okb], axis=1) * 1000

    st = lambda e: dict(n=int(len(e)), median_mm=float(np.median(e)), rms_mm=float(np.sqrt((e ** 2).mean())),
                        p95_mm=float(np.percentile(e, 95)), max_mm=float(e.max()))
    tz = dt.timezone(dt.timedelta(hours=args.utc_offset))
    hms = lambda u: dt.datetime.fromtimestamp(u, tz).strftime("%H:%M:%S")
    rep = dict(track=args.track, bag=args.bag, topic=args.topic, slam_fit=args.slam_fit,
               camera_span=[hms(t_cam[0]), hms(t_cam[-1])], bag_span=[hms(ref[0, 0]), hms(ref[-1, 0])],
               time_offset_s=args.time_offset, lever_mm=[float(lever[0] * 1000), float(lever[1] * 1000)],
               lever_solved=bool(args.solve_lever), overlap_s=float(t_cam[ok0][-1] - t_cam[ok0][0]),
               no_offset=st(e0), best_offset_s=best, at_best_offset=st(eb),
               note="no trajectory fit; onboard pose -> world through the wall/SLAM-map fit; "
                    "onboard point moved to the tag by lever_mm")
    
    json.dump(rep, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    with open(os.path.join(args.out, "matched.csv"), "w") as fh:
        fh.write("t_unix,cam_x_mm,cam_y_mm,bag_x_mm,bag_y_mm,dist_mm\n")
        for t, c, b in zip(t_cam[ok0], cam[ok0] * 1000, xy0[ok0] * 1000):
            fh.write(f"{t:.3f},{c[0]:.1f},{c[1]:.1f},{b[0]:.1f},{b[1]:.1f},{np.linalg.norm(c - b):.1f}\n")

    print(f"camera {rep['camera_span'][0]}-{rep['camera_span'][1]}   bag {rep['bag_span'][0]}-{rep['bag_span'][1]}"
          f"   overlap {rep['overlap_s']:.1f} s, {ok0.sum()} matched samples")
    for lab, s in (("no clock offset", rep["no_offset"]), (f"offset {best:+.2f} s", rep["at_best_offset"])):
        print(f"camera vs onboard ({lab:>15}): median {s['median_mm']:.0f} mm, rms {s['rms_mm']:.0f}, "
              f"p95 {s['p95_mm']:.0f}, max {s['max_mm']:.0f}  (n {s['n']})")

    # --- picture ----------------------------------------------------------------
    walls = load_walls(args.boundary)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw=dict(width_ratios=[2.3, 1]))
    for k, w in walls.items():
        if k in ("outer", "inner"):
            ax.plot(*np.r_[w, w[:1]].T, color="0.55", lw=1.5, label="walls (map)" if k == "outer" else None)
    ax.plot(*(ref_w * 1000).T, color="#9ecae1", lw=0.8, label="onboard pose, whole bag")
    win = (ref[:, 0] >= t_cam[0]) & (ref[:, 0] <= t_cam[-1])
    ax.plot(*(ref_w[win] * 1000).T, ".", color="#08519c", ms=3, label="onboard pose, during this clip")
    ax.plot(*(cam * 1000).T, ".", color="#d62728", ms=2, label="camera (fused)")
    idx = np.flatnonzero(ok0)[::30]                               # a tie line every ~1 s
    for i in idx:
        ax.plot([cam[i, 0] * 1000, xy0[i, 0] * 1000], [cam[i, 1] * 1000, xy0[i, 1] * 1000],
                color="k", lw=0.6, alpha=0.6)
    ax.plot([], [], color="k", lw=0.6, label="same instant, ~1 s apart")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (mm) — world frame: origin GCP 1, +x toward GCP 5"); ax.set_ylabel("y (mm)")
    s = rep["no_offset"]
    ax.set_title(f"{os.path.basename(args.track).replace('_fused.csv', '')} vs {os.path.basename(args.bag.rstrip('/'))}\n"
                 f"no trajectory fit: median {s['median_mm']:.0f} mm, p95 {s['p95_mm']:.0f} mm over "
                 f"{rep['overlap_s']:.0f} s overlap (lever {lever[0] * 1000:+.0f}/{lever[1] * 1000:+.0f} mm; camera clock +{args.time_offset:.2f} s)", fontsize=10)
    ax2.plot(t_cam[ok0] - t_cam[ok0][0], e0, color="#d62728", lw=1)
    ax2.set_xlabel(f"time in overlap (s); camera clock +{args.time_offset:.2f} s"); ax2.set_ylabel("camera - onboard distance (mm)")
    ax2.grid(alpha=0.3)
    ax2.set_title(f"clock-offset search: best {best:+.2f} s -> median {rep['at_best_offset']['median_mm']:.0f} mm",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "bag_overlay.png"), dpi=130)
    print(f"-> {args.out}")


def _interp(t_ref, xy_ref, t, max_gap):
    """Linear interpolation of xy_ref at t; False where t is outside the log or in a gap > max_gap."""
    j = np.searchsorted(t_ref, t)
    ok = (j > 0) & (j < len(t_ref))
    jj = np.clip(j, 1, len(t_ref) - 1)
    t0, t1 = t_ref[jj - 1], t_ref[jj]
    ok &= (t1 - t0) <= max_gap
    w = np.clip((t - t0) / np.maximum(t1 - t0, 1e-9), 0, 1)[:, None]
    return xy_ref[jj - 1] * (1 - w) + xy_ref[jj] * w, ok


if __name__ == "__main__":
    main()

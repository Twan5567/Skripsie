# Calibrate, then lift: save a clip's floor fit (and the two-camera world fit) while
# the floor sheets are down, load it for the clips shot after they were lifted, and
# check whether a camera moved in between.
import json
import os
from dataclasses import fields, replace
from datetime import date

import cv2
import numpy as np

from calibrate_charuco import gopro_settings
from floor import FloorFit
from geometry import apply_H
from util import banner
from world import Rigid2D, WorldFit

CALIB_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib"))


# a bare NAME means calib/NAME.json; a path is used as given
def cal_path(name: str) -> str:
    if os.sep in name or name.endswith(".json"):
        return name if name.endswith(".json") else name + ".json"
    return os.path.join(CALIB_DIR, name + ".json")


# --- which camera, in which setting ---------------------------------------------
IDENT_KEYS = ("CASN", "VFOV", "ZFOV", "EISE", "width", "height", "fps")


# which camera body, in which lens setting and resolution, shot this clip
def clip_identity(video: str) -> dict:
    cap = cv2.VideoCapture(video)
    W, Hh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    g = gopro_settings(video)
    return {**{k: g.get(k) for k in IDENT_KEYS[:4]}, "width": W, "height": Hh, "fps": round(fps, 3)}


# a calibration only applies to the same body in the same setting
def check_identity(cal_ident: dict, ident: dict, video: str) -> None:
    bad = [k for k in IDENT_KEYS if cal_ident.get(k) != ident.get(k)]
    if bad:
        raise SystemExit(f"{os.path.basename(video)} does not match the calibration: "
                         + ", ".join(f"{k} {ident.get(k)!r} vs {cal_ident.get(k)!r}" for k in bad))


# --- FloorFit <-> JSON ------------------------------------------------------------
def _to_json(v):
    if isinstance(v, dict):
        return {str(k): _to_json(x) for k, x in v.items()}
    return v.tolist() if isinstance(v, np.ndarray) else v


def _from_json(v):
    if isinstance(v, dict):
        return {int(k): _from_json(x) for k, x in v.items()}
    return np.array(v, dtype=np.float64) if isinstance(v, list) else v


def floor_to_dict(fl: FloorFit) -> dict:
    return {f.name: _to_json(getattr(fl, f.name)) for f in fields(fl) if f.name != "cal"}


def floor_from_dict(d: dict, cal_name: str) -> FloorFit:
    return FloorFit(**{k: _from_json(v) for k, v in d.items()}, cal=cal_name)


# --- background ---------------------------------------------------------------------
def median_background(video: str, n: int = 31) -> np.ndarray:
    # median of n frames spread over the clip: the car and passers-by average out
    cap = cv2.VideoCapture(video)
    frames = []
    for i in np.linspace(0, max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    if not frames:
        raise SystemExit(f"could not read frames from {video}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


# --- save / load ------------------------------------------------------------------------
# Write calib/NAME.json: per camera (keyed by serial number) its floor fit, its
# identity and a median background image; with two cameras also the world fit.
def save_cal(name: str, results: list, wf: WorldFit | None = None, origin_id: int | None = None,
             x_id: int | None = None, map_anchor: str | None = None) -> str:
    path = cal_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = os.path.splitext(path)[0]
    cams = {}
    for r in results:
        serial = r.ident["CASN"]
        if not serial:
            raise SystemExit(f"{r.stem}: no camera serial in the GoPro metadata; "
                             f"cannot key a calibration on it")
        bg_file = f"{os.path.basename(base)}_{serial}.png"
        print(f"  background of {r.stem} (median of 31 frames) ...", flush=True)
        cv2.imwrite(os.path.join(os.path.dirname(path), bg_file), median_background(r.video))
        cams[serial] = {"clip": r.stem, "video": os.path.realpath(r.video), "identity": r.ident,
                        "background": bg_file, "floor": floor_to_dict(r.floor)}

    out = {"name": os.path.basename(base), "created": date.today().isoformat(), "cameras": cams}
    if wf is not None:
        a, b = results[0].ident["CASN"], results[1].ident["CASN"]
        out["world"] = {
            "origin_id": origin_id, "x_id": x_id, "shared": wf.shared,
            "resid_mm": wf.resid_mm.tolist(), "scale_ab": wf.scale_ab,
            "layout": _to_json(wf.layout), "map_anchor": map_anchor,
            "to_world": {a: wf.a_to_world.to_json(), b: wf.b_to_world.to_json()},
        }
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  wrote calibration {path}")
    return path


# read a calibration; it remembers its own path, for error messages and for finding
# the background images
def load_cal(name: str) -> dict:
    path = cal_path(name)
    if not os.path.exists(path):
        raise SystemExit(f"no calibration {path} (make it with --save-cal on a sheets-down clip)")
    with open(path) as fh:
        cal = json.load(fh)
    cal["_path"] = path
    return cal


# the world fit saved with a two-camera calibration
def world_from_cal(cal: dict, serial_a: str, serial_b: str) -> WorldFit:
    w = cal.get("world")
    if w is None:
        raise SystemExit(f"{cal['_path']} has no two-camera world fit (it was saved "
                         f"from a single-camera run)")
    for s in (serial_a, serial_b):
        if s not in w["to_world"]:
            raise SystemExit(f"camera {s} is not in the world fit of {cal['_path']}")
    a_w, b_w = (Rigid2D.from_json(w["to_world"][s]) for s in (serial_a, serial_b))
    wf = WorldFit(shared=list(w["shared"]), a_to_b=a_w.then(b_w.inv()),
                  resid_mm=np.array(w["resid_mm"]), scale_ab=w["scale_ab"],
                  b_to_world=b_w, a_to_world=a_w,
                  layout={int(k): np.array(v) for k, v in w["layout"].items()})
    print(banner(f"WORLD FRAME   from calibration {cal['name']}: origin GCP "
                 f"{w['origin_id']} centre, +x toward GCP {w['x_id']}"))
    print(f"shared GCPs       : {wf.shared}   A->B residual rms "
          f"{np.sqrt((wf.resid_mm ** 2).mean()):.1f} mm, scale B/A {wf.scale_ab:.4f} "
          f"(as fitted on the calibration clips)")
    if w.get("map_anchor"):
        print(f"map anchor        : {w['map_anchor']}   scale map/camera "
              f"{a_w.s:.4f} (A), {b_w.s:.4f} (B)")
    return wf


# --- has the camera moved? ------------------------------------------------------------
# SIFT matches between two images (found at half size), as full-size pixel pairs
def _sift_pairs(img_a: np.ndarray, img_b: np.ndarray, scale: float = 0.5):
    def gray(img):
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    sift = cv2.SIFT_create(nfeatures=10000)
    ka, da = sift.detectAndCompute(gray(img_a), None)
    kb, db = sift.detectAndCompute(gray(img_b), None)
    if da is None or db is None:
        return np.empty((0, 2)), np.empty((0, 2))
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]      # Lowe's ratio test
    pa = np.array([ka[m.queryIdx].pt for m in good], dtype=np.float64) / scale
    pb = np.array([kb[m.trainIdx].pt for m in good], dtype=np.float64) / scale
    return pa, pb


# Has the camera moved since the calibration clip? Register this clip's median
# background onto the calibration's (SIFT + homography G, both undistorted), report
# the image shift and the floor error it would cause. correct=True folds G into the
# floor fit (H <- H G^-1). Writes <stem>_movecheck.jpg: red = calibration, cyan =
# this clip, grey where they agree.
def move_check(cal: dict, serial: str, floor: FloorFit, video: str, out_dir: str, stem: str,
               correct: bool) -> FloorFit:
    cam = cal["cameras"][serial]
    bg_cal = cv2.imread(os.path.join(os.path.dirname(cal["_path"]), cam["background"]))
    bg_run = median_background(video)
    sc = 1920 / bg_run.shape[1]
    ana = np.dstack([cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), None, fx=sc, fy=sc,
                                interpolation=cv2.INTER_AREA)
                     for x in (bg_run, bg_run, bg_cal)])          # red = calibration, cyan = run
    cv2.imwrite(os.path.join(out_dir, f"{stem}_movecheck.jpg"), ana)

    pa, pb = _sift_pairs(bg_cal, bg_run)
    print(banner(f"CAMERA-MOVED CHECK   {stem} vs calibration {cal['name']} ({cam['clip']})"))
    if len(pa) < 30:
        print(f"only {len(pa)} background matches — cannot tell whether the camera moved")
        return floor
    ua, ub = floor.undistort(pa), floor.undistort(pb)
    G, inl = cv2.findHomography(ua, ub, cv2.RANSAC, 2.0)
    if G is None:
        print("background registration failed — cannot tell whether the camera moved")
        return floor
    inl = inl.ravel().astype(bool)
    G, _ = cv2.findHomography(ua[inl], ub[inl], 0)
    res = np.linalg.norm(apply_H(G, ua[inl]) - ub[inl], axis=1)

    W, Hh = cam["identity"]["width"], cam["identity"]["height"]
    gx, gy = np.meshgrid(np.linspace(0, W, 9), np.linspace(0, Hh, 6))
    grid = floor.undistort(np.c_[gx.ravel(), gy.ravel()])
    shift = np.linalg.norm(apply_H(G, grid) - grid, axis=1)
    corners = np.vstack([floor.med_u[k] for k in sorted(floor.med_u)])
    err_mm = np.linalg.norm(apply_H(floor.H, apply_H(G, corners)) - apply_H(floor.H, corners), axis=1)
    print(f"background matches: {inl.sum()} inliers of {len(pa)}, residual "
          f"{np.sqrt((res ** 2).mean()):.2f} px rms")
    print(f"image shift       : median {np.median(shift):.2f} px, max {shift.max():.2f} px over the frame")
    print(f"floor error if not corrected: median {np.median(err_mm):.1f} mm, max "
          f"{err_mm.max():.1f} mm at the calibration sheets' corners")

    if np.median(shift) > 1.0:
        print("  WARNING: the camera has moved by more than 1 px since the calibration clip"
              + ("" if correct else " (pass --correct-motion to fold the shift into the fit)"))
    else:
        print("  -> camera has not moved measurably" + (" (correction applied anyway)" if correct else ""))
    if not correct:
        return floor

    med_u = {k: apply_H(G, v) for k, v in floor.med_u.items()}
    print("  --correct-motion: floor homography updated (H <- H G^-1)")
    return replace(floor, H=floor.H @ np.linalg.inv(G), med_u=med_u,
                   med={k: floor.distort(v) for k, v in med_u.items()})

# Cross-camera guide: where one camera missed the tag, look for it where the other
# camera saw the car at the same moment.
import numpy as np

from detector import build_detector, search_frames
from geometry import apply_H
from track_ground import ClipResult
from world import Rigid2D


# a camera's raw fixes in world mm, with their times
def _world_track(res: ClipResult, to_world: Rigid2D):
    have = res.track.have
    mm = res.track.mm_raw + res.track.origin
    xy = np.full_like(mm, np.nan)
    xy[have] = to_world(mm[have])
    return res.t_global, xy, have


# world point at tag height -> pixel in this camera (locate_vehicle_aruco backwards)
def _px_of_world(xy_w: np.ndarray, res: ClipResult, to_world: Rigid2D, tag_h: float):
    fl = res.floor
    mm = to_world.inv()(xy_w)
    if fl.has_pose:
        mm = fl.tag_to_floor(mm, tag_h)
    return fl.to_image(mm)


# how large a 150 mm tag looks at this pixel, from the floor homography
def _expected_side_px(px: np.ndarray, res: ClipResult, S: float = 150.0) -> float:
    fl = res.floor
    mm = apply_H(fl.H, px.reshape(1, 2)).reshape(2)
    q = apply_H(fl.H_inv, np.array([mm, mm + (S, 0), mm + (0, S)]))
    return float(0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[0])))


# Search target's frames that have no tag fix, at the pixel where source saw the car
# (interpolated to target's frame time). Adds what it finds to target's detections
# and returns how many it found.
def guide(target: ClipResult, target_to_world: Rigid2D, source: ClipResult, source_to_world: Rigid2D,
          veh_ids: list[int], tag_h: float, stride: int, gate_px: float = 220.0) -> int:
    if target.t_global is None or source.t_global is None:
        return 0
    ts, xy_s, have_s = _world_track(source, source_to_world)
    if not have_s.any():
        return 0

    # where the source camera saw the car, at each target frame without a tag
    step = stride / source.det.fps
    targets = {}
    for i in range(target.det.n):
        if any(k in target.det.corners_by_frame[i] for k in veh_ids):
            continue
        j = (target.t_global[i] - ts[0]) / step
        j0 = int(np.floor(j))
        if j0 < 0 or j0 >= len(ts) - 1 or not (have_s[j0] and have_s[j0 + 1]):
            continue
        f = j - j0
        xy_w = (1 - f) * xy_s[j0] + f * xy_s[j0 + 1]
        px = _px_of_world(xy_w.reshape(1, 2), target, target_to_world, tag_h).reshape(2)
        if 0 <= px[0] < target.det.W and 0 <= px[1] < target.det.Hh:
            targets[i] = (px, _expected_side_px(px, target))
    if not targets:
        return 0

    found = search_frames(target.video, stride, build_detector(), targets, veh_ids, gate_px)
    for i, (k, q) in found.items():
        target.det.corners_by_frame[i][k] = q
        target.det.recovered.add(i)
    print(f"cross-camera guide  : {target.stem} <- {source.stem}: {len(targets)} frames "
          f"searched, tag found in {len(found)}")
    return len(found)

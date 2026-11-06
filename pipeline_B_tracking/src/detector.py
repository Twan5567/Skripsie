from dataclasses import dataclass, field
from functools import cache

import cv2
import numpy as np

from aruco_config import ARUCO_DICT
from geometry import side_lengths


# ArUco detector tuned for small markers in 4K frames (the defaults miss 45 px
# markers): subpixel corners, a lower minimum size, more threshold window sizes.
def build_detector() -> cv2.aruco.ArucoDetector:
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.minMarkerPerimeterRate = 0.008
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 45
    p.adaptiveThreshWinSizeStep = 4
    p.polygonalApproxAccuracyRate = 0.06
    return cv2.aruco.ArucoDetector(ARUCO_DICT, p)


# the second-look detector for a partly hidden tag: accepts more bit errors
@cache
def _occlusion_detector() -> cv2.aruco.ArucoDetector:
    p = build_detector().getDetectorParameters()
    p.errorCorrectionRate = 1.0
    return cv2.aruco.ArucoDetector(ARUCO_DICT, p)


# "0-11,20" -> [0, 1, ..., 11, 20]
def parse_ids(spec: str) -> list[int]:
    out: list[int] = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


def marker_corners(corners, ids) -> dict[int, np.ndarray]:
    if ids is None:
        return {}
    return {int(k): np.array(c).reshape(4, 2).astype(np.float64)
            for c, k in zip(corners, np.array(ids).flatten())}


class Video:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Hh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.count = 0                       # frames grabbed so far

    def frames(self, stride: int = 1, only: set | None = None):
        # (k, frame) for every stride-th frame, k counting those; with `only`, just
        # those k, and stop after the last of them
        k, last = -1, max(only) if only else None
        while (last is None or k < last) and self.cap.grab():
            self.count += 1
            if (self.count - 1) % stride:
                continue
            k += 1
            if only is not None and k not in only:
                continue
            ok, frame = self.cap.retrieve()
            if ok:
                yield k, frame
            elif only is None:
                return

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cap.release()


# All markers found in one clip.
#   corners_by_frame[i]  {id: 4x2 corners, px} for the i-th analysed frame
#   frame_i              frames in the clip
#   recovered            analysed frames where the window search found the tag
#   rejected             tag decodes thrown out by the prediction gate
@dataclass
class Detections:
    corners_by_frame: list[dict[int, np.ndarray]] = field(default_factory=list)
    fps: float = 0.0
    W: int = 0
    Hh: int = 0
    frame_i: int = 0
    n: int = 0
    recovered: set = field(default_factory=set)
    rejected: int = 0


# Where the car's tag should be next: constant velocity from the last fixes,
# forgotten after `memory` frames without one. The gate (how far a decode may lie
# from the prediction) grows with the frames since the last fix.
class _VehiclePredictor:
    def __init__(self, max_step_px: float, memory: int = 30):
        self.max_step = max_step_px
        self.memory = memory
        self.pos = None
        self.vel = np.zeros(2)
        self.side = 40.0
        self.age = 10 ** 9

    def predict(self):
        if self.pos is None or self.age > self.memory:
            return None
        return self.pos + self.vel * min(self.age, 3)

    def gate_px(self):
        return self.max_step * min(self.age, 3) + 2 * self.side

    def update(self, corners: np.ndarray):
        c = corners.mean(axis=0)
        self.vel = (c - self.pos) / self.age if self.pos is not None and self.age <= 3 else np.zeros(2)
        self.pos = c
        self.side = float(np.mean(side_lengths(corners)))
        self.age = 0

    def tick(self):
        self.age += 1


# Second look for the car's tag near a predicted position. Crop around it, upscale
# until the tag is about target_px wide and detect again. If that fails, close small
# gaps (a cable or a hand across the tag) and retry with the lenient detector.
# Returns (id, corners in full-frame px) of the decode nearest the prediction, or None.
def _window_search(gray: np.ndarray, detector, pred: np.ndarray, side: float, gate: float, veh_ids: list[int], target_px: float = 110.0):

    
    H, W = gray.shape
    half = int(max(gate, 4 * side, 160))
    x0, y0 = int(max(pred[0] - half, 0)), int(max(pred[1] - half, 0))
    x1, y1 = int(min(pred[0] + half, W)), int(min(pred[1] + half, H))

    if x1 - x0 < 32 or y1 - y0 < 32:
        return None

    scale = float(np.clip(target_px / max(side, 8.0), 1.0, 4.0))
    crop = gray[y0:y1, x0:x1]

    if scale > 1.05:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    c, ids, _ = detector.detectMarkers(crop)

    if ids is None or not any(int(k) in veh_ids for k in np.array(ids).flatten()):

        k = 2 * int(round(1.25 * scale)) + 1
        closed = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        c, ids, _ = _occlusion_detector().detectMarkers(closed)

    if ids is None:
        return None

    best = None
    for n, k in enumerate(np.array(ids).flatten()):

        if int(k) not in veh_ids:
            continue


        q = np.array(c[n]).reshape(4, 2).astype(np.float64) / scale + (x0, y0)
        dist = float(np.linalg.norm(q.mean(axis=0) - pred))


        if dist <= gate and (best is None or dist < best[2]):
            best = (int(k), q, dist)
    return best[:2] if best else None


# Pass 1 over a clip: detect every marker in every stride-th frame. With veh_ids it
# also follows the car's tag: decodes far from the prediction are dropped, and the
# window search runs where the tag was missed.

def detect_all_frames(video: str, stride: int, detector: cv2.aruco.ArucoDetector, veh_ids: list[int] | None = None, max_step_px: float = 260.0) -> Detections:

    veh_ids = list(veh_ids or [])
    pred = _VehiclePredictor(max_step_px)
    with Video(video) as vid:
        det = Detections(fps=vid.fps, W=vid.W, Hh=vid.Hh)

        for kept, frame in vid.frames(stride):

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            d = marker_corners(*detector.detectMarkers(gray)[:2])

            if veh_ids:

                pred.tick()
                p = pred.predict()

                if p is not None:
                    # a vehicle-tag decode far from where the car can be is a false one
                    for k in [k for k in veh_ids if k in d]:
                        if np.linalg.norm(d[k].mean(axis=0) - p) > pred.gate_px():
                            del d[k]
                            det.rejected += 1
                seen = [k for k in veh_ids if k in d]

                if not seen and p is not None:
                    hit = _window_search(gray, detector, p, pred.side, pred.gate_px(), veh_ids)
                
                    if hit is not None:
                        d[hit[0]] = hit[1]
                        det.recovered.add(kept)
                        seen = [hit[0]]
                
                if seen:
                    pred.update(d[seen[0]])
            
            det.corners_by_frame.append(d)
            
            if (kept + 1) % 200 == 0:
                print(f"  detect {kept + 1}", end="\r", flush=True)
                
        det.frame_i = vid.count

    det.n = len(det.corners_by_frame)
    extra = (f"; vehicle tag recovered by windowed search in {len(det.recovered)}, "
             f"{det.rejected} decodes rejected by the prediction gate") if veh_ids else ""
    print(f"  detect {det.n} frames analysed{extra}" + " " * 10)
    if det.n == 0:
        raise SystemExit("no frames decoded")
    return det


# The window search in chosen frames only. targets = {analysed frame: (predicted tag
# centre px, expected tag side px)}. Used by the cross-camera guide.
def search_frames(video: str, stride: int, detector: cv2.aruco.ArucoDetector,
                  targets: dict[int, tuple[np.ndarray, float]], veh_ids: list[int],
                  gate: float) -> dict[int, tuple[int, np.ndarray]]:
    if not targets:
        return {}
    found = {}
    with Video(video) as vid:
        for kept, frame in vid.frames(stride, only=set(targets)):
            pred, side = targets[kept]
            hit = _window_search(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), detector, pred, side,
                                 gate, veh_ids)
            if hit is not None:
                found[kept] = hit
    return found

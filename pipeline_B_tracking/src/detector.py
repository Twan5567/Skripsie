
from dataclasses import dataclass, field

import cv2
import numpy as np

from aruco_config import ARUCO_DICT


def build_detector() -> cv2.aruco.ArucoDetector:

    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.minMarkerPerimeterRate = 0.008
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 45
    p.adaptiveThreshWinSizeStep = 4
    p.polygonalApproxAccuracyRate = 0.06
    return cv2.aruco.ArucoDetector(ARUCO_DICT, p)


_OCC_DET = None


def _occlusion_detector() -> cv2.aruco.ArucoDetector:
   
    global _OCC_DET
    if _OCC_DET is None:
        p = build_detector().getDetectorParameters()
        p.errorCorrectionRate = 1.0
        _OCC_DET = cv2.aruco.ArucoDetector(ARUCO_DICT, p)
    return _OCC_DET


def parse_ids(spec: str) -> list[int]:

    out: list[int] = []

    for part in spec.split(","):

        part = part.strip()

        if not part:
            continue

        if "-" in part:

            lo, hi = part.split("-", 1)
            out += list(range(int(lo), int(hi) + 1))

        else:
            out.append(int(part))


    return out


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

        if self.pos is not None and self.age <= 3:
            self.vel = (c - self.pos) / self.age

        else:
            self.vel = np.zeros(2)

        self.pos = c
        self.side = float(np.mean([np.linalg.norm(corners[j] - corners[(j + 1) % 4])
                                   for j in range(4)]))
        
        self.age = 0

    def tick(self):

        self.age += 1


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


def detect_all_frames(video: str, stride: int, detector: cv2.aruco.ArucoDetector, veh_ids: list[int] | None = None, max_step_px: float = 260.0) -> Detections:
    
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():

        raise SystemExit(f"cannot open video: {video}")
    
    det = Detections(fps=cap.get(cv2.CAP_PROP_FPS),
                     W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     Hh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    
    veh_ids = list(veh_ids or [])
    pred = _VehiclePredictor(max_step_px)

    frame_i = 0
    kept = 0
    while True:

        ok, frame = cap.read()

        if not ok:
            break


        if frame_i % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            c, ids, _ = detector.detectMarkers(gray)
            d: dict[int, np.ndarray] = {}

            if ids is not None:

                for n, k in enumerate(np.array(ids).flatten()):

                    d[int(k)] = np.array(c[n]).reshape(4, 2).astype(np.float64)

            if veh_ids:

                pred.tick()
                p = pred.predict()
                seen = [k for k in veh_ids if k in d]

                
                if p is not None:

                    for k in seen:

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
            kept += 1

            if kept % 200 == 0:

                print(f"  detect {kept}", end="\r", flush=True)
        frame_i += 1


    cap.release()
    det.frame_i = frame_i
    det.n = len(det.corners_by_frame)
    extra = ""

    if veh_ids:

        extra = (f"; vehicle tag recovered by windowed search in {len(det.recovered)}, "
                 f"{det.rejected} decodes rejected by the prediction gate")
        
    print(f"  detect {det.n} frames analysed{extra}" + " " * 10)

    if det.n == 0:
        raise SystemExit("no frames decoded")
    return det


def search_frames(video: str, stride: int, detector: cv2.aruco.ArucoDetector, targets: dict[int, tuple[np.ndarray, float]], veh_ids: list[int], gate: float) -> dict[int, tuple[int, np.ndarray]]:


    if not targets:
        return {}
    
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    
    found: dict[int, tuple[int, np.ndarray]] = {}
    last = max(targets)
    frame_i = 0
    kept = 0

    while kept <= last:

        ok = cap.grab()

        if not ok:
            break

        if frame_i % stride == 0:

            if kept in targets:

                ok, frame = cap.retrieve()

                if ok:

                    pred, side = targets[kept]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hit = _window_search(gray, detector, pred, side, gate, veh_ids)
                    if hit is not None:
                        found[kept] = hit

            kept += 1
        frame_i += 1

    cap.release()










    return found

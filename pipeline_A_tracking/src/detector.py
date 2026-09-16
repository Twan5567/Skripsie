"""ArUco detection: the tuned detector, ID-spec parsing, and pass 1 over the video."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from aruco_config import ARUCO_DICT


def build_detector() -> cv2.aruco.ArucoDetector:
    """Tuned for small, oblique markers on a 4K frame.

    The defaults miss the floor sheets at this scale: minMarkerPerimeterRate
    is relative to image size, so a 45 px marker in a 3840 px frame falls
    under the default 0.03 floor.
    """
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.minMarkerPerimeterRate = 0.008
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 45
    p.adaptiveThreshWinSizeStep = 4
    p.polygonalApproxAccuracyRate = 0.06
    return cv2.aruco.ArucoDetector(ARUCO_DICT, p)


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
    """Everything pass 1 learns about the clip."""
    corners_by_frame: list[dict[int, np.ndarray]] = field(default_factory=list)
    fps: float = 0.0
    W: int = 0
    Hh: int = 0
    frame_i: int = 0          # total frames decoded
    n: int = 0                # frames analysed (every `stride`-th)


def detect_all_frames(video: str, stride: int, detector: cv2.aruco.ArucoDetector) -> Detections:
    """Pass 1: detect every marker in every `stride`-th frame."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    det = Detections(fps=cap.get(cv2.CAP_PROP_FPS),
                     W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     Hh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    frame_i = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % stride == 0:
            c, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            d: dict[int, np.ndarray] = {}
            if ids is not None:
                for n, k in enumerate(np.array(ids).flatten()):
                    d[int(k)] = np.array(c[n]).reshape(4, 2).astype(np.float64)
            det.corners_by_frame.append(d)
            kept += 1
            if kept % 200 == 0:
                print(f"  detect {kept}", end="\r", flush=True)
        frame_i += 1
    cap.release()
    det.frame_i = frame_i
    det.n = len(det.corners_by_frame)
    print(f"  detect {det.n} frames analysed" + " " * 20)

    if det.n == 0:
        raise SystemExit("no frames decoded")
    return det

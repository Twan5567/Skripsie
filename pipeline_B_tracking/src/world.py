# One world frame for two cameras: 2-D rigid or similarity fits between the cameras'
# floor frames, or of each camera onto the map from pipeline A.
import json
from dataclasses import dataclass

import numpy as np

from floor import FloorFit
from geometry import apply_H
from util import banner


# each floor sheet's corners in this camera's floor frame, mm
def gcp_corners_mm(floor: FloorFit) -> dict[int, np.ndarray]:
    return {k: apply_H(floor.H, floor.med_u[k]) for k in floor.med_u}


# Least-squares rotation (and scale) plus translation taking src onto dst
# (Kabsch / Umeyama). D stops it returning a reflection. Returns (s, R, t).
def kabsch_2d(src: np.ndarray, dst: np.ndarray, scale: bool = False):
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    P, Q = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = float((sig * np.diag(D)).sum() / (P ** 2).sum()) if scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x ** 2).mean()))


# p -> s R p + t on 2-D points; inv() undoes it, a.then(b) is "a, then b"
@dataclass
class Rigid2D:
    R: np.ndarray
    t: np.ndarray
    s: float = 1.0

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64)
        return (self.s * p.reshape(-1, 2) @ self.R.T + self.t).reshape(p.shape)

    def inv(self) -> "Rigid2D":
        return Rigid2D(self.R.T, -self.R.T @ self.t / self.s, 1.0 / self.s)

    def then(self, other: "Rigid2D") -> "Rigid2D":
        return Rigid2D(other.R @ self.R, other.s * other.R @ self.t + other.t,
                       self.s * other.s)

    def to_json(self) -> dict:
        return {"R": self.R.tolist(), "t": self.t.tolist(), "s": self.s}

    @classmethod
    def from_json(cls, d: dict) -> "Rigid2D":
        return cls(np.array(d["R"]), np.array(d["t"]), float(d.get("s", 1.0)))


# The two cameras joined: each camera's floor frame -> world, the A -> B fit with
# its residuals on the shared sheet corners, and the sheet layout in world mm.
@dataclass
class WorldFit:
    shared: list[int]
    a_to_b: Rigid2D
    resid_mm: np.ndarray
    scale_ab: float
    b_to_world: Rigid2D
    a_to_world: Rigid2D
    layout: dict[int, np.ndarray]


# world frame: origin at the centre of sheet origin_id, +x toward the centre of x_id
def frame_from_layout(layout: dict[int, np.ndarray], origin_id: int, x_id: int) -> Rigid2D:
    o = layout[origin_id].mean(axis=0)
    xhat = layout[x_id].mean(axis=0) - o
    xhat /= np.linalg.norm(xhat)
    R = np.stack([xhat, [-xhat[1], xhat[0]]])
    return Rigid2D(R, -R @ o)


# Join the cameras through the sheets both see: rigid fit A -> B on the shared
# corners, then the world frame from the merged sheet layout.
def fit_world(floor_a: FloorFit, floor_b: FloorFit, origin_id: int, x_id: int) -> WorldFit:
    ca, cb = gcp_corners_mm(floor_a), gcp_corners_mm(floor_b)
    shared = sorted(set(ca) & set(cb))
    if len(shared) < 2:
        raise SystemExit(f"cameras share {len(shared)} floor GCP(s) "
                         f"(A {sorted(ca)}, B {sorted(cb)}); need at least 2")
    src = np.vstack([ca[k] for k in shared])
    dst = np.vstack([cb[k] for k in shared])
    _, R, t = kabsch_2d(src, dst)
    a_to_b = Rigid2D(R, t)
    resid = np.linalg.norm(a_to_b(src) - dst, axis=1)
    s_ab, _, _ = kabsch_2d(src, dst, scale=True)

    def in_b(k: int) -> np.ndarray:                  # average where both cameras saw it
        if k in shared:
            return 0.5 * (a_to_b(ca[k]) + cb[k])
        return cb[k] if k in cb else a_to_b(ca[k])

    layout_b = {k: in_b(k) for k in set(ca) | set(cb)}
    for k in (origin_id, x_id):
        if k not in layout_b:
            raise SystemExit(f"world frame needs GCP {k}, which neither camera accepted "
                             f"(have {sorted(layout_b)})")
    b_to_world = frame_from_layout(layout_b, origin_id, x_id)
    layout = {k: b_to_world(v) for k, v in layout_b.items()}

    print(banner(f"WORLD FRAME   origin GCP {origin_id} centre, +x toward GCP {x_id}"))
    print(f"shared GCPs       : {shared}  ({4 * len(shared)} corners)")
    print(f"A -> B rigid fit  : rotation {np.degrees(np.arctan2(R[1, 0], R[0, 0])):+.1f} deg, "
          f"residual rms {rms(resid):.1f} mm, max {resid.max():.1f} mm")
    print(f"scale B/A         : {s_ab:.4f}  ({100 * (s_ab - 1):+.2f}% — how far the two "
          f"cameras' metric scales disagree)")
    print(f"\n{'id':>4} {'seen by':>8} {'centre x':>9} {'centre y':>9} {'A-B rms':>8}")
    for k in sorted(layout):
        c = layout[k].mean(axis=0)
        who = "A+B" if k in shared else ("B" if k in cb else "A")
        i = shared.index(k) if k in shared else None
        r = rms(resid[4 * i:4 * i + 4]) if i is not None else np.nan
        print(f"{k:>4} {who:>8} {c[0]:>9.0f} {c[1]:>9.0f} {r:>8.1f}")

    return WorldFit(shared=shared, a_to_b=a_to_b, resid_mm=resid, scale_ab=s_ab,
                    b_to_world=b_to_world, a_to_world=a_to_b.then(b_to_world), layout=layout)


# the floor-sheet corners (world mm) measured by pipeline A's metric_scale.py, and
# the two sheets that define its frame
def load_map_corners(metric_json: str) -> tuple[dict[int, np.ndarray], int, int]:
    with open(metric_json) as fh:
        m = json.load(fh)
    corners = {int(k): np.array(v, dtype=np.float64)[:, :2]
               for k, v in m["sheet_corners_mm"].items()}
    return corners, int(m["origin_gcp"]), int(m["x_gcp"])


# Fit each camera on its own onto the map's sheet corners (similarity: the scale
# comes from the map). The cameras need not share a sheet; where they do, the A vs B
# residual is a check.
def fit_world_map(floor_a: FloorFit, floor_b: FloorFit, map_corners: dict[int, np.ndarray],
                  map_name: str) -> WorldFit:
    ca, cb = gcp_corners_mm(floor_a), gcp_corners_mm(floor_b)
    print(banner(f"WORLD FRAME   anchored to map {map_name} "
                 f"(its frame: origin GCP centre, +x toward the x GCP)"))
    fits = {}
    for cam, cc in (("A", ca), ("B", cb)):
        ids = sorted(k for k in cc if k in map_corners)
        if len(ids) < 2:
            raise SystemExit(f"camera {cam} has {len(ids)} sheet(s) in the map "
                             f"(camera {sorted(cc)}, map {sorted(map_corners)}); need at least 2")
        src = np.vstack([cc[k] for k in ids])
        dst = np.vstack([map_corners[k] for k in ids])
        s, R, t = kabsch_2d(src, dst, scale=True)
        fits[cam] = tw = Rigid2D(R, t, s)
        res = np.linalg.norm(tw(src) - dst, axis=1)
        print(f"camera {cam} -> map  : sheets {ids}, scale map/camera {tw.s:.4f} "
              f"({100 * (tw.s - 1):+.2f}%), residual rms {rms(res):.1f} mm, "
              f"max {res.max():.1f} mm")
        print("    per sheet rms mm: " + ", ".join(
            f"{k}: {rms(res[4 * i:4 * i + 4]):.0f}" for i, k in enumerate(ids)))
    a_w, b_w = fits["A"], fits["B"]

    shared = sorted(set(ca) & set(cb))
    if shared:
        resid = np.linalg.norm(np.vstack([a_w(ca[k]) for k in shared])
                               - np.vstack([b_w(cb[k]) for k in shared]), axis=1)
        print(f"shared GCPs       : {shared}  A vs B after anchoring: rms "
              f"{rms(resid):.1f} mm, max {resid.max():.1f} mm")
    else:
        resid = np.zeros(0)
        print("shared GCPs       : none (not needed with a map anchor)")
    print(f"scale B/A         : {b_w.s / a_w.s:.4f}  (relative scale correction "
          f"the map applied to the two cameras)")
    layout = {k: map_corners[k] for k in sorted(set(ca) | set(cb)) if k in map_corners}
    return WorldFit(shared=shared, a_to_b=a_w.then(b_w.inv()), resid_mm=resid,
                    scale_ab=b_w.s / a_w.s, b_to_world=b_w, a_to_world=a_w, layout=layout)

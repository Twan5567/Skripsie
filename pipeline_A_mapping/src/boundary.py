"""Track boundary from a metric dense cloud: the centreline of each barrier wall.

Input is a dense cloud already put in millimetres and in the tracker's world
frame by metric_scale.py. Every threshold below is therefore a physical
distance, set once, not tuned per cloud.

  1. crop      keep points inside the convex hull of the mapping walk (the walk
               circled the track, so this drops the room and the far floaters)
  2. floor     least-squares plane through the near-floor points; heights are
               measured from it (a check on metric_scale's sheet plane too)
  3. band      keep the wall: band_lo < z < wall_top + margin. The floor and the
               sheets are excluded; the top rim is kept, because on a thin
               vertical wall the rim IS the centreline, and on stretches seen at
               a glancing angle the rim and foot are all the cloud has
  4. raster    project the band to the floor, count points per cell
  5. DBSCAN    cluster the occupied cells; the two big loops are the outer wall
               and the inner island (outer = the one that encloses the other)
  6. centre    each loop's band -> the line equidistant from its two edges
               (distance transforms), then every vertex is pulled onto the
               median of the wall points along its normal, then a closed
               smoothing spline
  7. corridor  the same equidistant-line trick between the two walls gives the
               track midline, and twice the distance to a wall gives its width

  --bridge-gaps MM (off by default). When only the island closes because
  stretches of the outer wall are missing from the cloud (TrackMap_S3: the
  window side), the outer wall is rebuilt around the island: in each 1-degree
  sector the nearest wall cell outside the island, outliers against the
  neighbouring sectors dropped, empty sectors interpolated in angle. Gaps
  longer than MM are refused. Vertices with no wall point within 100 mm are
  INTERPOLATED, not measured: column `measured` = 0 in the CSV, dashed in the
  plot, and they are left out of every residual.

Outputs, output/<NAME>/:
  <NAME>_boundary.csv          loop, i, x_mm, y_mm   (outer, inner, midline)
  <NAME>_boundary.dxf          the three polylines, mm, for re-laying the track
  <NAME>_boundary.json         lengths, areas, widths, fit residuals, checks
  <NAME>_boundary_topdown.png  density + walls + midline + sheets (+ car tracks)
  <NAME>_width.png             track width along the midline
  <NAME>_walls3d.ply           the two walls extruded to the measured height
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import pycolmap
from matplotlib.path import Path as MPath
from scipy.interpolate import splev, splprep
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import DBSCAN

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_metric(dense: str, sparse: str, metric: str):

    m = json.loads(Path(metric).read_text())
    R, t, s = np.array(m["R"]), np.array(m["t"]), m["scale_mm_per_unit"]
    pc = o3d.io.read_point_cloud(dense)
    P = np.asarray(pc.points) * s @ R.T + t
    rec = pycolmap.Reconstruction(sparse)
    ims = sorted(rec.images.values(), key=lambda i: i.name)
    cams = np.array([im.projection_center() for im in ims]) * s @ R.T + t
    return m, P, cams


def fit_floor(P: np.ndarray, band_mm: float):

   
    sel = np.abs(P[:, 2]) < band_mm
    for tol in (band_mm, band_mm / 2):
        A = np.c_[P[sel, 0], P[sel, 1], np.ones(sel.sum())]
        coef, *_ = np.linalg.lstsq(A, P[sel, 2], rcond=None)
        r = P[:, 2] - np.c_[P[:, :2], np.ones(len(P))] @ coef
        sel = np.abs(r) < tol
    rms = float(np.sqrt(np.mean(r[sel] ** 2)))
    return coef, rms


class Grid:

    def __init__(self, xy: np.ndarray, cell: float, pad: float):
        self.cell = cell
        self.x0, self.y0 = xy.min(axis=0) - pad
        x1, y1 = xy.max(axis=0) + pad
        self.nx, self.ny = int((x1 - self.x0) / cell) + 1, int((y1 - self.y0) / cell) + 1

    def idx(self, xy):
        return (((xy[:, 1] - self.y0) / self.cell).astype(int),
                ((xy[:, 0] - self.x0) / self.cell).astype(int))

    def xy(self, rc):
        
        rc = np.asarray(rc, float)
        return np.c_[self.x0 + (rc[:, 1] + 0.5) * self.cell, self.y0 + (rc[:, 0] + 0.5) * self.cell]

    def count(self, xy):
        H = np.zeros((self.ny, self.nx), np.int32)
        r, c = self.idx(xy)
        np.add.at(H, (r, c), 1)
        return H


def equidistant_contour(outer_mask: np.ndarray, inner_mask: np.ndarray, grid: Grid):

    d_out = cv2.distanceTransform(outer_mask.astype(np.uint8), cv2.DIST_L2, 5)
    d_in = cv2.distanceTransform((~inner_mask).astype(np.uint8), cv2.DIST_L2, 5)
    region = (outer_mask & (d_in <= d_out)) | inner_mask
    cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2)          # (col, row)
    return grid.xy(np.c_[c[:, 1], c[:, 0]])


def resample(poly: np.ndarray, step: float) -> np.ndarray:

    p = np.vstack([poly, poly[:1]])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = np.arange(0, d[-1], step)
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]


def normals(poly: np.ndarray) -> np.ndarray:

    tg = np.roll(poly, -1, 0) - np.roll(poly, 1, 0)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    return np.c_[-tg[:, 1], tg[:, 0]]


def snap_to_wall(poly, pts, tree, half_width, reach):
  
    nrm = normals(poly)
    tg = np.c_[nrm[:, 1], -nrm[:, 0]]
    out = poly.copy()
    for i, (p, n, t_) in enumerate(zip(poly, nrm, tg)):
        idx = tree.query_ball_point(p, np.hypot(half_width, reach))
        if len(idx) < 5:
            continue
        d = pts[idx] - p
        a, b = d @ n, d @ t_
        k = (np.abs(a) <= reach) & (np.abs(b) <= half_width)
        if k.sum() >= 5:
            out[i] = p + np.median(a[k]) * n
    return out


def smooth_closed(poly: np.ndarray, step: float, smooth_mm: float) -> np.ndarray:

    p = resample(poly, step)
    tck, _ = splprep([p[:, 0], p[:, 1]], s=len(p) * smooth_mm ** 2, per=True)
    L = np.sum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1))
    u = np.linspace(0, 1, int(L / step), endpoint=False)
    return np.column_stack(splev(u, tck))


def seg_dist(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:

    dense = resample(poly, 2.0)
    return cKDTree(dense).query(pts)[0]


def bridge_outer(inner: dict, clusters: list, grid: Grid, max_gap: float, min_cells: int = 30):

    ctr = grid.xy(np.argwhere(inner["filled"])).mean(0)
    cand = [c for c in clusters if c is not inner and c["cells"] >= min_cells]
    rc = np.vstack([np.argwhere(c["mask"]) for c in cand])
    rc = rc[~inner["filled"][rc[:, 0], rc[:, 1]]]
    xy = grid.xy(rc)
    d = xy - ctr
    ang, rad = np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1])
    nb = 360
    b = ((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    r = np.full(nb, np.nan)

    for k in range(nb):
        rk = rad[b == k]
        if len(rk):
            r[k] = rk.min()


    for k in np.flatnonzero(~np.isnan(r)):
        nbr = r[[(k + j) % nb for j in range(-10, 11) if j]]
        if np.isfinite(nbr).sum() >= 4 and abs(r[k] - np.nanmedian(nbr)) > 400:
            r[k] = np.nan


    have = np.flatnonzero(~np.isnan(r))

    if len(have) < nb // 3:
        raise SystemExit(f"--bridge-gaps: wall found in only {len(have)} of {nb} sectors around the island")
    th = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
    rr = np.interp(np.arange(nb), have, r[have], period=nb)
    poly = ctr + np.c_[np.cos(th), np.sin(th)] * rr[:, None]
    gaps, spans = [], []


    for i, j in zip(have, np.roll(have, -1)):
        if (j - i) % nb > 1:
            a, z = poly[i], poly[j]
            gaps.append((float(np.hypot(*(z - a))), a.round(0).tolist(), z.round(0).tolist()))
            spans.append(poly[[(i + s) % nb for s in range((j - i) % nb + 1)]])

    if gaps and max(g_[0] for g_ in gaps) > max_gap:
        raise SystemExit(f"--bridge-gaps: largest gap {max(g_[0] for g_ in gaps):.0f} mm > {max_gap:.0f} mm")
    
   
    near = cKDTree(resample(poly, 10.0)).query(xy)[0] < 150
    real = np.zeros(inner["mask"].shape, bool)
    real[rc[near, 0], rc[near, 1]] = True

   
    mask = real.copy().astype(np.uint8)
    r_, c_ = grid.idx(poly)                     
    cv2.polylines(mask, [np.c_[c_, r_].astype(np.int32)], True, 1, 2)
    for sp in spans:
        r_, c_ = grid.idx(sp)
        cv2.polylines(mask, [np.c_[c_, r_].astype(np.int32)], False, 1, 5)
    ks = 2 * int(60 / grid.cell / 2) + 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cs, -1, 1, -1)
    mask, filled = mask.astype(bool), filled.astype(bool)
    return mask, filled, filled & ~mask, real, gaps


def polygon_length(p):
    return float(np.sum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)))


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def write_dxf(path: Path, polys: dict[str, np.ndarray]):

    L = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
         "0", "SECTION", "2", "ENTITIES"]
    for layer, p in polys.items():
        L += ["0", "POLYLINE", "8", layer, "66", "1", "70", "1"]
        for x, y in p:
            L += ["0", "VERTEX", "8", layer, "10", f"{x:.2f}", "20", f"{y:.2f}", "30", "0.0"]
        L += ["0", "SEQEND"]
    L += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(L) + "\n")


def write_walls_ply(path: Path, polys: list[np.ndarray], height: float):
    V, F = [], []
    for p in polys:
        o = len(V)
        n = len(p)
        for x, y in p:
            V += [(x, y, 0.0), (x, y, height)]
        for i in range(n):
            a, b = o + 2 * i, o + 2 * ((i + 1) % n)
            F += [(a, b, a + 1), (b, b + 1, a + 1)]
    hdr = (f"ply\nformat ascii 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\n"
           f"property float z\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    path.write_text(hdr + "".join(f"{x:.1f} {y:.1f} {z:.1f}\n" for x, y, z in V)
                    + "".join(f"3 {a} {b} {c}\n" for a, b, c in F))


def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--metric", required=True, help="<NAME>_metric.json from metric_scale.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cell", type=float, default=10.0, help="raster cell, mm")
    ap.add_argument("--band-lo", type=float, default=30.0, help="wall band starts this high, mm")
    ap.add_argument("--top-margin", type=float, default=20.0, help="band stops this far ABOVE the wall top")
    ap.add_argument("--close-mm", type=float, default=60.0,
                    help="morphological closing of the wall raster: bridges gaps up to ~this")
    ap.add_argument("--min-count", type=int, default=3, help="points for a cell to count as wall")
    ap.add_argument("--eps", type=float, default=50.0, help="DBSCAN eps, mm")
    ap.add_argument("--smooth", type=float, default=10.0, help="spline RMS allowance, mm")
    ap.add_argument("--snap-width", type=float, default=40.0,
                    help="half-width along the wall of the strip each vertex takes its median from, mm")
    ap.add_argument("--tracks", default="", help="glob of fused trajectory CSVs to check against")
    ap.add_argument("--bridge-gaps", type=float, default=0.0,
                    help="if only the island closes, rebuild the outer wall around it and bridge gaps up "
                         "to this long, mm (0 = off); bridged vertices are flagged measured=0")
    args = ap.parse_args()
    out = Path(args.out_dir)
    stem = out / args.name
    rep: dict = {}

    m, P, cams = load_metric(args.dense, args.sparse, args.metric)
    print(f"dense cloud: {len(P):,} points, scale {m['scale_mm_per_unit']:.2f} mm/unit")

    # ---  crop ---------------------------------------------------------------
    hull = MPath(cams[ConvexHull(cams[:, :2]).vertices, :2])
    P = P[hull.contains_points(P[:, :2])]
    print(f"inside the walk's hull: {len(P):,}")

    # ---  floor ----------------------------------------------------------------
    coef, floor_rms = fit_floor(P, 30.0)
    tilt = np.degrees(np.arctan(np.hypot(coef[0], coef[1])))
    P[:, 2] -= np.c_[P[:, :2], np.ones(len(P))] @ coef
    print(f"floor refit: offset {coef[2]:+.1f} mm, tilt {tilt:.3f} deg vs the sheet plane, "
          f"floor rms {floor_rms:.1f} mm")
    rep["floor"] = dict(offset_mm=float(coef[2]), tilt_deg=float(tilt), rms_mm=floor_rms)

    # ---  band -------------------------------------------------------------------
    z = P[:, 2]
    tall = z[(z > args.band_lo) & (z < 1000)]
    hist, edges = np.histogram(tall, bins=np.arange(args.band_lo, 1000, 10))
    
    peak = hist.max()
    top_i = np.flatnonzero(hist > 0.05 * peak).max()
    wall_top = float(edges[top_i + 1])
    band_hi = wall_top + args.top_margin
    band = P[(z > args.band_lo) & (z < band_hi), :2]
    print(f"wall top ~{wall_top:.0f} mm; band {args.band_lo:.0f}-{band_hi:.0f} mm: {len(band):,} points")
    rep["wall_top_mm"] = wall_top
    rep["band_mm"] = [args.band_lo, band_hi]

    # ---  raster -------------------------------------------------------------------
    g = Grid(band, args.cell, pad=300)
    H = g.count(band)
    occ = np.argwhere(H >= args.min_count)
    print(f"raster {g.nx}x{g.ny} @ {args.cell:.0f} mm, {len(occ):,} occupied cells")

    # ---  DBSCAN -------------------------------------------------------------------
    lab = DBSCAN(eps=args.eps / args.cell, min_samples=4).fit(occ).labels_
    clusters = []
    for L in set(lab) - {-1}:
        rc = occ[lab == L]
        mask = np.zeros(H.shape, bool)
        mask[rc[:, 0], rc[:, 1]] = True
        ks = 2 * int(args.close_mm / args.cell / 2) + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(mask, np.uint8)
        cv2.drawContours(filled, cs, -1, 1, -1)
        filled = filled.astype(bool)
        hole = filled & ~mask
        clusters.append(dict(label=int(L), cells=len(rc), mask=mask, filled=filled, hole=hole,
                             area=filled.sum() * args.cell ** 2))
    clusters.sort(key=lambda c: -c["area"])
    for c in clusters[:5]:
        print(f"  cluster {c['label']}: {c['cells']} cells, encloses {c['hole'].sum() * args.cell ** 2 / 1e6:.3f} m^2")
    loops = [c for c in clusters if c["hole"].sum() * args.cell ** 2 > 0.1e6]   # encloses > 0.1 m^2
    if len(loops) == 1 and args.bridge_gaps > 0:
        inner = loops[0]
        mask, filled, hole, real, gaps = bridge_outer(inner, clusters, g, args.bridge_gaps)
        outer = dict(label=-2, cells=int(real.sum()), mask=mask, filled=filled, hole=hole, real=real,
                     area=filled.sum() * args.cell ** 2)
        print(f"bridged the outer wall around the island: {len(gaps)} gap(s), "
              f"longest {max([g_[0] for g_ in gaps], default=0):.0f} mm, "
              f"total {sum(g_[0] for g_ in gaps):.0f} mm (straight-line lengths)")
        rep["bridged_gaps"] = [dict(length_mm=round(l_), start_mm=a, end_mm=z) for l_, a, z in gaps]
    elif len(loops) < 2:
        raise SystemExit(f"found {len(loops)} closed wall loop(s); expected outer + inner"
                         + ("" if args.bridge_gaps else " (try --bridge-gaps)"))
    else:
        outer = loops[0]
        inner = next((c for c in loops[1:] if (c["filled"] & outer["hole"]).sum() == c["filled"].sum()), None)
    if inner is None:
        raise SystemExit("no closed loop inside the outer wall")
    dropped = [c for c in clusters if c is not outer and c is not inner]
    print(f"DBSCAN: {len(clusters)} clusters; outer {outer['cells']} cells, inner {inner['cells']} cells; "
          f"dropped {len(dropped)} ({sum(c['cells'] for c in dropped)} cells)")
    rep["dropped_clusters"] = [dict(cells=c["cells"], area_m2=round(c["area"] / 1e6, 4)) for c in dropped]

    # --- wall centrelines ----------------------------------------------------------
    walls, measured = {}, {}
    for name, c in (("outer", outer), ("inner", inner)):
        sel = c.get("real", c["mask"])[g.idx(band)]
        pts = band[sel]
        tree = cKDTree(pts)
        poly = resample(equidistant_contour(c["filled"], c["hole"], g), 20.0)
        for _ in range(2):
            poly = snap_to_wall(poly, pts, tree, half_width=args.snap_width, reach=120.0)
        poly = smooth_closed(poly, 10.0, args.smooth)
        d = seg_dist(pts, poly)
        walls[name] = poly
        measured[name] = tree.query(poly)[0] <= 100.0
        seg = np.linalg.norm(np.roll(poly, -1, 0) - poly, axis=1)
        rep[name] = dict(
            length_mm=polygon_length(poly), area_m2=polygon_area(poly) / 1e6,
            interpolated_mm=float(seg[~measured[name]].sum()),
            interpolated_pct=float(100 * seg[~measured[name]].sum() / seg.sum()),
            points=int(len(pts)),
            resid_rms_mm=float(np.sqrt(np.mean(d[d < 150] ** 2))),
            resid_median_mm=float(np.median(d[d < 150])),
            resid_p95_mm=float(np.percentile(d[d < 150], 95)),
            outliers_gt150mm_pct=float(100 * (d >= 150).mean()),
            extent_mm=(poly.max(0) - poly.min(0)).round(0).tolist())
        print(f"{name:>5} wall: length {rep[name]['length_mm'] / 1000:.3f} m, "
              f"encloses {rep[name]['area_m2']:.3f} m^2, points-to-line rms {rep[name]['resid_rms_mm']:.1f} mm "
              f"(median {rep[name]['resid_median_mm']:.1f}, p95 {rep[name]['resid_p95_mm']:.1f}), "
              f"extent {rep[name]['extent_mm']}"
              + (f"; INTERPOLATED {rep[name]['interpolated_mm'] / 1000:.2f} m "
                 f"({rep[name]['interpolated_pct']:.0f}%)" if rep[name]["interpolated_mm"] else ""))

    # ---  corridor midline and width -----------------------------------------------------
    fine = Grid(np.vstack(list(walls.values())), 5.0, pad=100)

    def poly_mask(p):
        m_ = np.zeros((fine.ny, fine.nx), np.uint8)
        r, c = fine.idx(p)
        cv2.fillPoly(m_, [np.c_[c, r].astype(np.int32)], 1)
        return m_.astype(bool)
    out_m, in_m = poly_mask(walls["outer"]), poly_mask(walls["inner"])
    mid = smooth_closed(equidistant_contour(out_m, in_m, fine), 10.0, args.smooth)
    dist_o, dist_i = seg_dist(mid, walls["outer"]), seg_dist(mid, walls["inner"])
    width = dist_o + dist_i
    walls["midline"] = mid
    rep["midline"] = dict(length_mm=polygon_length(mid),
                          width_min_mm=float(width.min()), width_mean_mm=float(width.mean()),
                          width_max_mm=float(width.max()),
                          centring_rms_mm=float(np.sqrt(np.mean((dist_o - dist_i) ** 2))))
    rep["drivable_area_m2"] = rep["outer"]["area_m2"] - rep["inner"]["area_m2"]
    print(f"midline {rep['midline']['length_mm'] / 1000:.3f} m; width min {width.min():.0f} / "
          f"mean {width.mean():.0f} / max {width.max():.0f} mm (wall centre to wall centre); "
          f"drivable area {rep['drivable_area_m2']:.2f} m^2")

    # --- tracking cross-check --------------------------------------------------------------
    tracks = {}
    if args.tracks:
        po, pi = MPath(walls["outer"]), MPath(walls["inner"])
        rep["tracks"] = {}
        for f in sorted(glob.glob(args.tracks)):
            xy = []
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    if row.get("x_mm") and row.get("y_mm"):
                        xy.append((float(row["x_mm"]), float(row["y_mm"])))
            if not xy:
                continue
            xy = np.array(xy)
            inside = po.contains_points(xy) & ~pi.contains_points(xy)
            clear = np.minimum(seg_dist(xy, walls["outer"]), seg_dist(xy, walls["inner"]))
            clear[~inside] *= -1
            key = Path(f).stem.replace("_fused", "")
            tracks[key] = xy
            rep["tracks"][key] = dict(samples=len(xy), outside_pct=float(100 * (~inside).mean()),
                                      min_clearance_mm=float(clear.min()),
                                      p5_clearance_mm=float(np.percentile(clear, 5)),
                                      median_clearance_mm=float(np.median(clear)))
            print(f"  track {key:10s}: {len(xy):5d} samples, outside corridor {100 * (~inside).mean():5.2f}%, "
                  f"clearance min {clear.min():6.0f} / p5 {np.percentile(clear, 5):5.0f} / "
                  f"median {np.median(clear):4.0f} mm")

    # --- outputs ---------------------------------------------------------------------------
    with open(f"{stem}_boundary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["loop", "i", "x_mm", "y_mm", "measured"])
        for name, p in walls.items():
            ok = measured.get(name, np.ones(len(p), bool))
            for i, (x, y) in enumerate(p):
                w.writerow([name, i, f"{x:.1f}", f"{y:.1f}", int(ok[i])])
    write_dxf(Path(f"{stem}_boundary.dxf"), walls)
    write_walls_ply(Path(f"{stem}_walls3d.ply"), [walls["outer"], walls["inner"]], wall_top)

    fig, ax = plt.subplots(figsize=(11, 11))
    ext = [g.x0, g.x0 + g.nx * g.cell, g.y0, g.y0 + g.ny * g.cell]
    ax.imshow(np.log1p(H), origin="lower", extent=ext, cmap="Greys", alpha=0.8)
    cols = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for (k, xy), col in zip(tracks.items(), cols):
        ax.plot(xy[:, 0], xy[:, 1], ".", ms=0.6, color=col, alpha=0.5, label=f"car {k}")
    for name, col, ls in (("outer", "#d62728", "-"), ("inner", "#d62728", "-"), ("midline", "#ff7f0e", "--")):
        p = np.vstack([walls[name], walls[name][:1]])
        ax.plot(p[:, 0], p[:, 1], ls, color=col, lw=1.4,
                label={"outer": "wall centrelines", "inner": None, "midline": "track midline"}[name])
        if name in measured and not measured[name].all():
            q = walls[name].copy()
            q[measured[name]] = np.nan
            ax.plot(q[:, 0], q[:, 1], "-", color="#ffd400", lw=3.5, alpha=0.9,
                    label=f"{name} wall INTERPOLATED (no wall in the cloud)")
    for k, c in m["sheet_corners_mm"].items():
        c = np.array(c)
        ax.fill(c[:, 0], c[:, 1], color="k")
        ax.annotate(k, c[:, :2].mean(0), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)  — world frame: origin GCP 1, +x toward GCP 5")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{args.name} — track boundary (wall centrelines), metric\n"
                 f"outer {rep['outer']['length_mm'] / 1000:.2f} m, inner {rep['inner']['length_mm'] / 1000:.2f} m, "
                 f"width {rep['midline']['width_min_mm']:.0f}-{rep['midline']['width_max_mm']:.0f} mm, "
                 f"fit rms {rep['outer']['resid_rms_mm']:.0f}/{rep['inner']['resid_rms_mm']:.0f} mm")
    ax.legend(loc="upper right", fontsize=8, markerscale=12)
    ax.grid(alpha=0.2)
    lo, hi = walls["outer"].min(0) - 400, walls["outer"].max(0) + 400
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    fig.tight_layout()
    fig.savefig(f"{stem}_boundary_topdown.png", dpi=150)
    plt.close(fig)

    s_along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(mid, axis=0), axis=1))] / 1000
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(s_along, width, lw=1.2)
    ax.set_xlabel("distance along the midline (m), from its first vertex")
    ax.set_ylabel("width, wall centre to wall centre (mm)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}_width.png", dpi=130)
    plt.close(fig)

    Path(f"{stem}_boundary.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {stem}_boundary.{{csv,dxf,json}}, _boundary_topdown.png, _width.png, _walls3d.ply")


if __name__ == "__main__":
    main()

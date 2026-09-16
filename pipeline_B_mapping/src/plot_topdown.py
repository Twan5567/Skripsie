"""Top-down render of a COLMAP sparse model, projected onto its floor plane.

Two jobs. First, a sanity check you can read at a glance: if the track outline
and a closed camera path are visible, the reconstruction worked. Second, it is
a preview of the boundary-extraction stage — the RANSAC plane fit here is the
same one that pipeline needs, so the inlier fraction and the height spread tell
you in advance whether floor and barriers separate cleanly.

Axes are in SfM units, not metres; metric scale comes later from the ArUco GCPs.

  python plot_topdown.py --sparse colmap/<stem>/sparse/0 --out out.png
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pycolmap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fit_plane(P: np.ndarray, iters: int = 800, thresh_frac: float = 0.01,
              seed: int = 0):
    """RANSAC-fit the dominant plane. Returns (normal, d, inlier_mask).

    thresh_frac is relative to the cloud's largest extent, so it does not need
    retuning per scene. Keep it tight: a loose threshold calls almost every
    point an inlier and the "99% inliers" that results means nothing.
    """
    rng = np.random.default_rng(seed)
    thresh = thresh_frac * np.ptp(P, axis=0).max()
    best_mask, best = None, None
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        d = -n @ s[0]
        mask = np.abs(P @ n + d) < thresh
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask, best = mask, (n, d)
    return best[0], best[1], best_mask


def plane_basis(n: np.ndarray):
    """Two orthonormal in-plane axes for a plane with normal `n`."""
    a = np.array([1.0, 0.0, 0.0])
    a = a - n * (a @ n)
    if np.linalg.norm(a) < 1e-6:
        a = np.array([0.0, 1.0, 0.0])
        a = a - n * (a @ n)
    a /= np.linalg.norm(a)
    return a, np.cross(n, a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", required=True, help="COLMAP model dir, e.g. sparse/0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--thresh-frac", type=float, default=0.01)
    ap.add_argument("--trim-pct", type=float, default=0.5,
                    help="clip this percentile off each end of the extent so a "
                         "few stray points don't set the axis limits")
    args = ap.parse_args()

    rec = pycolmap.Reconstruction(args.sparse)
    P = np.array([p.xyz for p in rec.points3D.values()])
    ims = sorted(rec.images.values(), key=lambda i: i.name)
    C = np.array([im.projection_center() for im in ims])

    n, d, inl = fit_plane(P, thresh_frac=args.thresh_frac)
    a, b = plane_basis(n)
    uv = np.stack([P @ a, P @ b], 1)
    h = P @ n + d
    cuv = np.stack([C @ a, C @ b], 1)

    lo, hi = np.percentile(uv, [args.trim_pct, 100 - args.trim_pct], axis=0)
    m = np.all((uv > lo) & (uv < hi), axis=1)

    err = np.mean([p.error for p in rec.points3D.values()])
    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(uv[m, 0], uv[m, 1],
                    c=np.clip(h[m], *np.percentile(h[m], [2, 98])),
                    s=0.6, cmap="viridis", linewidths=0)
    ax.plot(cuv[:, 0], cuv[:, 1], "-", color="#B3400C", lw=1.6,
            label=f"camera path ({len(C)} frames)")
    ax.plot(*cuv[0], "go", ms=9, label="start")
    ax.plot(*cuv[-1], "rs", ms=9, label="end")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("in-plane u")
    ax.set_ylabel("in-plane v")
    ax.set_title(args.title or
                 f"{args.sparse} — sparse cloud, top-down on fitted floor plane\n"
                 f"{len(P):,} points · {len(C)}/{len(rec.images)} registered · "
                 f"{err:.2f} px reproj · floor inliers {100*inl.mean():.0f}% · "
                 "axes in SfM units")
    plt.colorbar(sc, ax=ax, label="height above floor", shrink=0.8)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}   ({len(P):,} pts, floor inliers {100*inl.mean():.1f}%)")


if __name__ == "__main__":
    main()

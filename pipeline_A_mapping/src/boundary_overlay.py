"""
  python boundary_overlay.py --sparse output/<NAME>/sparse --images <frames> \
      --metric output/<NAME>/<NAME>_metric.json --boundary output/<NAME>/<NAME>_boundary.csv \
      --wall-top 320 --out output/<NAME>/<NAME>_boundary_overlay.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pycolmap

COLOURS = {"outer": (40, 40, 230), "inner": (40, 40, 230), "midline": (0, 150, 255)}  # BGR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--wall-top", type=float, required=True, help="mm, from <NAME>_boundary.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=2, help="use every n-th registered frame")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=float, default=8.0)
    args = ap.parse_args()

    m = json.loads(Path(args.metric).read_text())
    R, t, s = np.array(m["R"]), np.array(m["t"]), m["scale_mm_per_unit"]
    loops: dict[str, list] = {}


    with open(args.boundary) as fh:

        for row in csv.DictReader(fh):
            loops.setdefault(row["loop"], []).append((float(row["x_mm"]), float(row["y_mm"])))


    lines = []                 

    for name, pts in loops.items():

        p = np.array(pts)

        for z in ((0.0, args.wall_top) if name != "midline" else (0.0,)):

            w = np.c_[p, np.full(len(p), z)]
            w = np.vstack([w, w[:1]])
            lines.append((name, z, ((w - t) @ R) / s))   # inverse of world = R (s X) + t

    rec = pycolmap.Reconstruction(args.sparse)
    ims = sorted(rec.images.values(), key=lambda i: i.name)[::args.every]
    tmp = tempfile.mkdtemp(prefix=".overlay_", dir=str(Path(args.out).parent))


    for n, im in enumerate(ims):

        img = cv2.imread(os.path.join(args.images, im.name))

        if img is None:
            continue
        cam = im.camera
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        P = T.matrix()

        for name, z, X in lines:

            Xc = X @ P[:, :3].T + P[:, 3]
            front = Xc[:, 2] > 0.05 * np.median(np.abs(Xc[:, 2]))
            uv = cam.img_from_cam(np.where(front[:, None], Xc, 1.0))
            thick = 5 if z == 0.0 else 3

            for a in range(len(uv) - 1):

                if front[a] and front[a + 1]:

                    p0, p1 = uv[a], uv[a + 1]

                    if np.all(np.abs(np.r_[p0, p1]) < 2e4):
                        cv2.line(img, tuple(np.int32(p0)), tuple(np.int32(p1)), COLOURS[name],
                                 thick if name != "midline" else 3, cv2.LINE_AA)

                        
        h = int(img.shape[0] * args.width / img.shape[1])
        img = cv2.resize(img, (args.width, h), interpolation=cv2.INTER_AREA)

        cv2.putText(img, f"{im.name}   red: wall centreline at floor (thick) and at {args.wall_top:.0f} mm "
                    f"(thin); orange: midline", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imwrite(os.path.join(tmp, f"{n:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (n + 1) % 50 == 0:
            print(f"  overlay {n + 1}/{len(ims)}", end="\r", flush=True)

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", os.path.join(tmp, "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "22", args.out], check=True)
    
    
    mid = sorted(os.listdir(tmp))[len(ims) // 3]
    os.replace(os.path.join(tmp, mid), str(Path(args.out).with_suffix(".jpg")))
    
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print(f"\nwrote {args.out} ({len(ims)} frames) and a still {Path(args.out).with_suffix('.jpg').name}")


if __name__ == "__main__":
    main()

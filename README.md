# External vision system for racetrack mapping and vehicle trajectory tracking

Two offline pipelines that turn GoPro footage of a temporary indoor racetrack into
(A) the vehicle's trajectory on the floor plane and (B) a 3D point cloud of the
track. Both rely on ArUco sheets laid flat on the floor.

| pipeline | camera | produces |
|---|---|---|
| `pipeline_A_tracking/` | fixed, overhead, ~3 m up | vehicle trajectory in floor-plane mm |
| `pipeline_B_mapping/` | handheld, walked around the track | COLMAP sparse model + OpenMVS dense cloud |

Each pipeline has the same two folders and the same convention:

```
capture/    put your videos here — any number of them
output/     one folder per video, named after it, with everything the run produced
```

You run a pipeline by **video name**, not by path. Frames and other intermediates
are created under `capture/` while a run is going and are **deleted when it
finishes**, so `capture/` only ever holds the videos. Both folders are tracked in
git; their contents are not.

---

## Requirements

### System

| tool | used by | notes |
|---|---|---|
| Python ≥ 3.10 | both | |
| ffmpeg / ffprobe | both | frame extraction; checking footage resolution |
| COLMAP ≥ 3.12 | B | `colmap` on PATH. CPU SIFT is used; no GPU needed |
| Docker | B | runs OpenMVS; the current user must be in the `docker` group (no sudo) |
| `openmvs/openmvs-ubuntu` image | B | pulled automatically on the first dense run |
| systemd-run | B (optional) | memory-caps the COLMAP stage; skipped if absent |
| `opensplat-rocm7` image | B, `run_splat.sh` only | AMD/ROCm-specific, optional |

### Python

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The runners find `.venv/bin/python` automatically. To use another interpreter:
`PYTHON=/path/to/python ./run_….sh …`.

### Footage

Both pipelines were developed on 3840×2160 @ 59.94 fps HEVC from GoPro HERO13s.
**Check every new clip with `ffprobe` before processing** — a cloud-synced
download can silently arrive at 1280×720, and the ArUco detector will then miss
every marker.

---

## The markers

Both pipelines rely on ArUco sheets (`DICT_4X4_50`) laid **flat on the floor**.
IDs are allocated by role and must not be reused across roles:

| IDs | role | size |
|---|---|---|
| 0–11 | ground control points (GCP), flat on the floor | 150 mm |
| 20, 21 | vehicle tags (20 = front) | 150 mm, white quiet zone, mounted flat |
| 30, 31 | scale-bar endpoints | 150 mm |

"Size" means the outer edge of the black square. Print at 100% / actual size,
never fit-to-page. Measure the printed square; if it differs from nominal, pass
the measured value to pipeline A with `--gcp-mm`.

A vehicle tag stuck straight onto dark bodywork with no white border detects in
4–16% of frames. The same tag with a generous quiet zone, mounted flat, detects
in 83–91%.

---

## Pipeline A — vehicle tracking

Fixed camera, roughly 3 m above the floor, looking down at the track.

```bash
cd pipeline_A_tracking
cp /wherever/Mono1_4k.MP4 capture/
./run_tracking.sh Mono1_4k                   # one video, by name (extension optional)
./run_tracking.sh Mono1_4k Mono2_4k          # several
./run_tracking.sh                            # every video in capture/
./run_tracking.sh Mono1_4k -- --cam-height-mm 3271 --tag-heights 20:57   # extra flags after --
./run_tracking.sh Mono1_4k --fallback        # re-enable the blob fallback (see below)
```

### Output

`output/<video-name>/`:

| file | what |
|---|---|
| `<name>_overlay.mp4` | the clip with the tracked path drawn — **look at this first** |
| `<name>_trajectory.csv` | per analysed frame: `frame, time_s, source, px_x, px_y, mm_x, mm_y, aruco_px_x, aruco_px_y` |
| `<name>_trajectory.png` | top-down plot in floor-plane mm, GCP squares drawn *(currently disabled)* |
| `<name>_distance.png` | displacement vs time *(currently disabled)* |
| `<name>.log` | everything the script printed |

The two PNG plots are disabled in `outputs.py` pending a rework.

Nothing is written into `capture/`; this pipeline reads the video directly.

### What it does

1. Detects every ArUco marker in every analysed frame (tuned detector — the
   OpenCV defaults miss 45 px markers in a 3840 px frame).
2. Identifies the static floor GCPs by detection rate and positional spread.
3. Builds the ground-plane homography from the best 150 mm sheet, then refits
   it on all GCP corners.
4. Fits radial lens distortion from the sheets themselves, by making every
   known square come out square.
5. Recovers the focal length and **camera height** from the homography (Zhang's
   orthonormality constraints). That recovered height against a tape-measured
   one is the pipeline's self-check.
6. Locates the vehicle tag, corrects its parallax (a tag *h* above the floor
   projects too far from the nadir by *H/(H−h)*), and smooths each contiguous
   run with a Savitzky-Golay filter.

The console output ends with a block like:

```
camera height       : 3002 mm recovered from the homography
vehicle ArUco fixes : 2829/3379 (83.7%)
located total       : 2829/3379 (83.7%)
scale check (rms)   : 7.38 mm on a 150 mm sheet (4.92%)
```

### Source layout

```
src/track_ground.py   command line + order of operations only
src/detector.py       tuned ArUco detector, ID parsing, pass 1 over the video
src/geometry.py       undistortion, square-error metric, camera-from-homography
src/floor.py          static-GCP selection, distortion fit, homography, camera pose
src/vehicle.py        ArUco vehicle fixes + parallax, blob fallback, combine + smooth
src/outputs.py        CSV, plots (currently disabled), overlay video, summary
src/aruco_config.py   the ArUco dictionary
```

### Defaults and why

- `--stride 2` — every 2nd frame (30 Hz at 59.94 fps).
- `--smooth 15` — Savitzky-Golay window in analysed frames.
- `--no-fallback` — **ArUco fixes only.** The median-background blob fallback
  is off. With a properly mounted tag it is not needed, and a bad blob fill
  *corrupts the ~7 ArUco frames either side of it* through the smoothing
  window (the CSV then shows `aruco20` rows with impossible steps while
  `aruco_px_*` stays smooth). Pass `--fallback` to the runner to re-enable it
  for a clip with long genuine dropouts, and compare `px_*` to `aruco_px_*`
  afterwards.
- `--cam-height-mm` is **optional** and only changes what the recovered height
  is printed against. Pass the tape measurement or nothing — not a guess.
- `--tag-heights 20:57,21:67` (mm above the floor) drives the parallax
  correction. Measure them for your mounting.

### Reading the CSV

`source` is `aruco20` / `aruco21` / `blob` / empty. `px_*` and `mm_*` are the
smoothed, merged track. `aruco_px_*` is the raw detection and is the ground
truth for whether a frame was really measured.

---

## Pipeline B — track mapping

Walk around the track with the camera, slowly, filming the floor and barriers.
One or more clips of the same scene go into **one** model.

```bash
cd pipeline_B_mapping
cp /wherever/GX011488.MP4 /wherever/GX011489.MP4 capture/

./run_mapping.sh GX011488                        # one clip            -> output/GX011488/
./run_mapping.sh GX011488 GX011489               # two walks, ONE model -> output/GX011488_GX011489/
NAME=TrackMap ./run_mapping.sh GX011488 GX011489 # pick the model name yourself

# recommended on a new scene: sparse first, read the component count, then dense
SPARSE_ONLY=1 ./run_mapping.sh GX011488 GX011489
grep "components:" output/GX011488_GX011489/GX011488_GX011489.log
./run_dense.sh GX011488_GX011489                 # sparse is kept in output/, only dense runs

# fragmented (components > 1)?  redo the sparse, matching every pair
EXHAUSTIVE=1 FORCE=1 ./run_mapping.sh GX011488 GX011489

# optional Gaussian splat (AMD ROCm OpenSplat image); -d 3 for ~600 4K frames
./run_splat.sh GX011488_GX011489 3               # <NAME> [downscale] [iterations]
```

Environment toggles for `run_mapping.sh`:

| variable | effect |
|---|---|
| `NAME=…` | model name (default: clip stems joined with `_`) |
| `SPARSE_ONLY=1` | stop after the sparse model and top-down plot |
| `EXHAUSTIVE=1` | match every image pair instead of sequential neighbours |
| `FORCE=1` | redo the sparse even if `output/<NAME>/sparse/` exists |
| `FPS=…` | frame extraction rate (default 4) |

### Output

`output/<NAME>/`:

| file | what |
|---|---|
| `<NAME>_topdown.png` | sparse cloud + camera path on the RANSAC floor plane, registration stats in the title |
| `<NAME>_sparse.ply` | sparse cloud, largest component |
| `<NAME>_dense.ply` | OpenMVS dense cloud. Open in CloudCompare, set Colors → RGB |
| `sparse/` | the COLMAP model itself (`cameras/images/points3D.bin`) — what `run_dense.sh`, `run_splat.sh` and any later metric scaling read |
| `splat_d<D>/` | OpenSplat output, if `run_splat.sh` was run |
| `sources.txt` | which clips built this model |
| `<NAME>.log` | everything every tool printed |

Frames and the COLMAP database live in `capture/<NAME>_work/` only while a run
is going and are **deleted when it completes** (kept only if it fails, so you can
look). `run_dense.sh` and `run_splat.sh` re-extract the frames from
`sources.txt` — about 15 s for a 4K clip — and delete them again. Consequence:
an `EXHAUSTIVE=1` re-match redoes feature extraction too.

### Stages

```
extract_frames.py         ffmpeg, 4 fps
colmap_pipeline.py        SIFT -> sequential or exhaustive matcher -> incremental mapper
(lib.sh)                  pick the largest connected component
plot_topdown.py           RANSAC floor plane + top-down render
dense_reconstruction.py   colmap image_undistorter -> OpenMVS InterfaceCOLMAP -> DensifyPointCloud (Docker)
```

The three runners share `lib.sh`.

### Settings and why

Each one is the result of a failure.

- `--max-image-size 3200 --peak-threshold 0.003` — what made SIFT work on a
  uniform tiled floor: features per image went from ~1.7 k to 4–10 k and
  registration to 100%.
- `--num-threads 8` — each SIFT worker holds a decoded 4K image and its pyramid
  (~1.2 GB). 24 threads reached 28 GB and the OOM killer took the session. Do
  not set this to "all cores" on 4K input.
- `--camera-model OPENCV --single-camera` — one GoPro, shared intrinsics.
- OpenMVS `--resolution-level 2` (quarter-res depth maps), `--number-views 5`,
  16 threads, 24 GB Docker cap. Fusion holds every depth map resident at once:
  half-res × ~450 views needs ~18 GB, quarter-res ~14 GB. The runner falls back
  to level 3 if level 2 dies.
- **Largest component, not `sparse/0`.** COLMAP writes one folder per connected
  component and does not sort by size; `sparse/0` has been a 2-image fragment.
  All three runners pick the biggest and say so when the model is fragmented —
  that also tells you the walk lost tracking.
- **Hard links, not symlinks**, for the combined frame set: symlinks do not
  resolve inside the OpenMVS container.
- `run_splat.sh` refuses a run that cannot fit in RAM. OpenSplat holds the input
  images as float32 at ~2.06× their naive size; at its default `-d 1`, 558 4K
  frames need 106 GB. Use `-d 3` or `-d 4`.

### Reading the result

The top-down title reports registered/total frames, points, reprojection error
and floor-inlier fraction. A healthy walk registers 100% in one component with
≥ 95% floor inliers, and the camera path closes on itself. The log's
`Depth-maps fused: N` line should equal the registered count — a low fusion
rate means a starved sparse model, not a dense-stage problem.

### Units

**Every `.ply` is in SfM units, not millimetres.** Metric scaling from the GCP
sheets is a separate step and is not part of this repo yet.

---

## Typical numbers

Measured on a 24-core CPU with 32 GB RAM.

| | value |
|---|---|
| Pipeline A, 71 s clip @ stride 2 | ~1.5 min |
| Pipeline A vehicle fix rate, 150 mm tag with quiet zone | 83–91% |
| Pipeline A recovered camera height, repeatability | ~55–70 mm spread |
| Pipeline B sparse, 577 frames, 8 threads | ~13 min |
| Pipeline B dense, 577 views, level 2 | ~14 min, 16 M points |
| Pipeline B splat, ~600 frames, `-d 3`, 30k iterations | ~20 min |

Memory, not compute, is the constraint on 4K input; the caps in the runners are
there for that reason.

---

## Not in this repo

Metric scaling of the clouds (Umeyama alignment on the GCP sheets), track
boundary extraction (RANSAC plane + DBSCAN), two-camera fusion, and the
learned-feature (GLUEMAP / SuperPoint) mapping variants.

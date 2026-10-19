# External vision system for racetrack mapping and vehicle trajectory tracking

Two offline pipelines that turn GoPro footage of a temporary indoor racetrack into
(A) a metric map of the track with its wall boundary and (B) the vehicle's
trajectory on the floor, both in millimetres in one shared world frame. Both use
ArUco markers laid flat on the floor.

| pipeline | camera | produces |
|---|---|---|
| `pipeline_A_mapping/` | handheld, walked around the track | COLMAP sparse model, OpenMVS dense cloud, metric scale, wall boundary + corridor width |
| `pipeline_B_tracking/` | two fixed cameras on stands at opposite ends of the track | fused vehicle trajectory in world mm |

Each pipeline has the same two folders and the same convention:

```
capture/    put your videos here
output/     one folder per run, named after it, with everything the run produced
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
| ffmpeg / ffprobe | both | frame extraction, timecode, checking footage resolution |
| COLMAP ≥ 3.12 | A | `colmap` on PATH. CPU SIFT is used; no GPU needed |
| Docker | A | runs OpenMVS; the current user must be in the `docker` group (no sudo) |
| `openmvs/openmvs-ubuntu` image | A | pulled automatically on the first dense run |
| systemd-run | A (optional) | memory-caps the COLMAP stage; skipped if absent |

### Python

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The runners find `.venv/bin/python` automatically. To use another interpreter:
`PYTHON=/path/to/python ./run_….sh …`.

`boundary.py` also needs `scikit-learn` (DBSCAN), which is not yet in
`requirements.txt`: `.venv/bin/pip install scikit-learn`.

### Footage

Both pipelines were developed on 3840×2160 @ 59.94 fps HEVC from GoPro HERO13s,
Linear lens, stabilisation off. **Check every new clip with `ffprobe` before
processing**: a cloud-synced download can silently arrive at 1280×720, and the
ArUco detector will then miss every marker.

---

## The markers

All markers are ArUco `DICT_4X4_50`. IDs are allocated by role and must not be
reused across roles:

| IDs | role | size |
|---|---|---|
| 0–11 | floor markers, flat on the floor | 150 mm |
| 20 | vehicle tag, mounted flat on the car's roof | 150 mm, white quiet zone |
| 30, 31 | scale-bar endpoints | 150 mm |

The lens calibration board is a ChArUco board (`DICT_5X5_100`).

"Size" means the outer edge of the black square. Print at 100% / actual size,
never fit-to-page. Measure the printed square; if it differs from nominal, pass
the measured value to pipeline B with `--gcp-mm`.

The print-ready PDFs are in `markers/print/`; regenerate them with
`python markers/generate_print_sheets.py` (and `markers/charuco_trimmed.py` for
the board-only ChArUco page). `markers/print/gcp_survey_template.csv` is a blank
sheet for tape-measuring the floor markers.

A vehicle tag stuck straight onto dark bodywork with no white border detects in
4–16% of frames. The same tag with a generous quiet zone, mounted flat, detects
in 83–91%.

---

## Pipeline A — track mapping

Walk around the track with the camera, slowly, filming the floor and walls, with
the floor markers down. One or more clips of the same scene go into **one** model.

```bash
cd pipeline_A_mapping
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

# metric scale + wall boundary (needs the sparse and dense models)
./run_boundary.sh TrackMap
BOUNDARY_ARGS="--min-count 10 --bridge-gaps 2000" ./run_boundary.sh TrackMap
```

Environment toggles for `run_mapping.sh`:

| variable | effect |
|---|---|
| `NAME=…` | model name (default: clip stems joined with `_`) |
| `SPARSE_ONLY=1` | stop after the sparse model and top-down plot |
| `EXHAUSTIVE=1` | match every image pair instead of sequential neighbours |
| `FORCE=1` | redo the sparse even if `output/<NAME>/sparse/` exists |
| `FPS=…` | frame extraction rate (default 4) |

Environment toggles for `run_boundary.sh`:

| variable | effect |
|---|---|
| `WORLD_ORIGIN=…` | floor marker whose centre is the world origin (default 1) |
| `WORLD_X=…` | floor marker the world +x axis points at (default 5) |
| `TRACKS=…` | glob of fused trajectory CSVs for the corridor check (default: all of pipeline B's) |
| `BOUNDARY_ARGS=…` | extra `boundary.py` options |

### Output

`output/<NAME>/`:

| file | what |
|---|---|
| `<NAME>_topdown.png` | sparse cloud + camera path on the RANSAC floor plane, registration stats in the title |
| `<NAME>_sparse.ply` | sparse cloud, largest component |
| `<NAME>_dense.ply` | OpenMVS dense cloud. Open in CloudCompare, set Colors → RGB |
| `sparse/` | the COLMAP model itself (`cameras/images/points3D.bin`), read by `run_dense.sh` and `run_boundary.sh` |
| `<NAME>_metric.json` | scale and transform from SfM units to world mm, the floor markers' corners in mm, leave-one-out hold-out errors |
| `<NAME>_boundary.csv` / `.dxf` / `.json` | outer wall and island centrelines in world mm, with a measured/interpolated flag per point |
| `<NAME>_boundary_topdown.png`, `_width.png` | the boundary drawn top-down; corridor width along the track |
| `<NAME>_walls3d.ply` | the wall points used, for checking in CloudCompare |
| `<NAME>_boundary_overlay.mp4` | the boundary drawn back onto the walk video — **look at this first** |
| `sources.txt` | which clips built this model |
| `<NAME>.log` | everything every tool printed |

Frames and the COLMAP database live in `capture/<NAME>_work/` only while a run
is going and are **deleted when it completes** (kept only if it fails, so you can
look). `run_dense.sh` and `run_boundary.sh` re-extract the frames from
`sources.txt` (about 15 s for a 4K clip) and delete them again. Consequence:
an `EXHAUSTIVE=1` re-match redoes feature extraction too.

### Stages

```
extract_frames.py         ffmpeg, 4 fps
colmap_pipeline.py        SIFT -> sequential or exhaustive matcher -> incremental mapper
(lib.sh)                  pick the largest connected component
plot_topdown.py           RANSAC floor plane + top-down render
dense_reconstruction.py   colmap image_undistorter -> OpenMVS InterfaceCOLMAP -> DensifyPointCloud (Docker)
metric_scale.py           detect the floor markers in the frames, triangulate their corners,
                          one scale per marker from its known side, similarity fit to world mm
boundary.py               floor plane, wall points above it, DBSCAN, outer/inner wall loops,
                          smoothed centrelines, corridor width
boundary_overlay.py       boundary projected back into the walk frames, as a video
```

The three runners share `lib.sh`.

### Settings and why

Each one is the result of a failure.

- `--max-image-size 3200 --peak-threshold 0.003`: what made SIFT work on a
  uniform tiled floor. Features per image went from ~1.7 k to 4–10 k and
  registration to 100%.
- `--num-threads 8`: each SIFT worker holds a decoded 4K image and its pyramid
  (~1.2 GB). 24 threads reached 28 GB and the OOM killer took the session. Do
  not set this to "all cores" on 4K input.
- `--camera-model OPENCV --single-camera`: one GoPro, shared intrinsics.
- OpenMVS `--resolution-level 2` (quarter-res depth maps), `--number-views 5`,
  16 threads, 24 GB Docker cap. Fusion holds every depth map resident at once:
  half-res × ~450 views needs ~18 GB, quarter-res ~14 GB. The runner falls back
  to level 3 if level 2 dies.
- **Largest component, not `sparse/0`.** COLMAP writes one folder per connected
  component and does not sort by size; `sparse/0` has been a 2-image fragment.
  All three runners pick the biggest and say so when the model is fragmented,
  which also tells you the walk lost tracking.
- **Hard links, not symlinks**, for the combined frame set: symlinks do not
  resolve inside the OpenMVS container.

### Reading the result

The top-down title reports registered/total frames, points, reprojection error
and floor-inlier fraction. A healthy walk registers 100% in one component with
≥ 95% floor inliers, and the camera path closes on itself. The log's
`Depth-maps fused: N` line should equal the registered count; a low fusion
rate means a starved sparse model, not a dense-stage problem.

### Units

The `.ply` clouds are in SfM units, not millimetres. `<NAME>_metric.json` holds
the transform to world mm, and everything `boundary.py` writes is already in
world mm.

### Checking against the car's SLAM map (test)

`src/SlamCompareTest.py` compares the extracted boundary with the occupancy map
the car built with SLAM (`.yaml` + image), after a similarity fit. Our world
frame is mirrored against the ROS map frame, and the fit allows for that.

```bash
cd pipeline_A_mapping/src
python SlamCompareTest.py --boundary ../output/<NAME>/<NAME>_boundary.csv \
    --map <path to the car's map>.yaml --out ../output/<NAME>_SlamTest [--dense-dir ../output/<NAME>]
```

---

## Pipeline B — vehicle tracking

Two fixed cameras on stands at opposite ends of the track, about 2.3–2.4 m up,
looking down at it at an angle. Each camera covers the far side of the track
the other sees poorly. Both clips are time-synced through the GoPro timecode:
scan the same QR timecode into both cameras before **every** session.

```bash
cd pipeline_B_tracking
cp /wherever/A-4.MP4 /wherever/B-4.MP4 capture/

./run_dual.sh A-4 B-4                          # by name, extension optional -> output/A-4_B-4/
NAME=Run4 ./run_dual.sh A-4 B-4                # choose the output folder name
./run_dual.sh A-4 B-4 -- --tag-heights 20:198  # extra flags after --
```

### Calibrate, then lift

The floor markers must be off the drivable floor while the car drives. Film one
calibration clip pair with the markers down, save its calibration, lift the
markers, and reuse the calibration for every later clip from the same camera
positions:

```bash
./run_dual.sh A-4 B-4 -- --min-rate 0.30 --map-anchor TrackMap_S4_L1 \
    --lens-a $PWD/calib/S4_camA_linear.json --save-cal S4_cal1mA     # markers down
./run_dual.sh A-5 B-5 -- --load-cal S4_cal1mA     # markers lifted
./run_dual.sh A-6 B-6 -- --load-cal S4_cal1mA --correct-motion
```

- `--map-anchor <NAME>` fits each camera to the floor-marker corners from
  pipeline A's `<NAME>_metric.json`, so the track and the map share one world
  frame and the scale comes from the map. Without it, the two cameras are
  joined through the floor markers they both see, with the origin at marker 1
  and +x towards marker 5 (`--world-origin`, `--world-x`).
- Every `--load-cal` run checks whether each camera moved since the calibration
  (log: `CAMERA-MOVED CHECK`; `<clip>/<clip>_movecheck.jpg`, red = calibration,
  cyan = this clip). `--correct-motion` folds the measured movement into the fit.
- Calibrations are saved as `calib/<NAME>.json`. The session-4 one in use is
  `calib/S4_cal1mA.json`, made by the first command above.
- Give lens files as absolute paths (`$PWD/calib/...`): the runner works from `src/`.

### Lens calibration

Without a lens file, the radial distortion is fitted from the floor markers by
making every known square come out square. With a ChArUco clip filmed by the
same camera in the same lens mode, `calibrate_charuco.py` writes a lens file
that is used instead (`--lens-a`, `--lens-b`, or `--lens` for one camera):

```bash
cd pipeline_B_tracking/src
python calibrate_charuco.py --camera A --out ../calib/S4_camA_linear.json "<ChArUco clip>.MP4" \
    --check ../capture/<a clip from that camera>.MP4
```

### Output

`output/<A>_<B>/`:

| file | what |
|---|---|
| `<A>_<B>_overlay.mp4` | both views side by side: each camera's own fixes (red), the fused track (yellow). **Look at this first** |
| `<A>_<B>_fused.csv` | `time_global_s, source (A/B/AB), x_mm, y_mm, xA, yA, xB, yB, disagree_mm, tagpx_A, tagpx_B, frame_A, frame_B` |
| `<A>_<B>_topdown.png` | fused track, both cameras' fixes and the floor-marker layout, world mm |
| `<A>/`, `<B>/` | the two single-camera runs |
| `<A>_<B>.log` | both floor fits, the world fit, the tag-height check, coverage and A–B disagreement |

Nothing is written into `capture/`; this pipeline reads the videos directly.

### What it does

For each camera:

1. Detects every ArUco marker in every analysed frame (tuned detector: the
   OpenCV defaults miss 45 px markers in a 3840 px frame), with a windowed,
   upscaled second look for the vehicle tag where it was expected.
2. Identifies the static floor markers by detection rate and positional spread.
3. Removes lens distortion (lens file, or fitted from the markers).
4. Builds the floor-to-image homography from the best-conditioned marker, then
   refits it on all floor-marker corners.
5. Recovers the focal length and **camera height** from the homography (Zhang's
   orthonormality constraints). The recovered height against a tape-measured
   one is a self-check.
6. Locates the vehicle tag and corrects its parallax: a tag *h* above the floor
   projects too far from the camera's nadir by *H/(H−h)*.

Then, for the pair:

7. Puts both cameras in one world frame (map anchor, or shared markers).
8. Aligns the two clips in time from their timecode.
9. Cross-guide: each camera searches again for the tag where the other camera
   saw the car (`--no-cross-guide` turns it off).
10. Fuses both cameras' fixes with a constant-velocity Kalman filter and a
    Rauch–Tung–Striebel smoother, bridging gaps of up to `--max-gap` samples
    (`--fusion mean` for the older tag-size-weighted mean).

### Source layout

```
src/track_dual.py      two-camera command line + order of operations
src/track_ground.py    one camera: command line, floor fit, vehicle fixes, outputs
src/detector.py        tuned ArUco detector, ID parsing, pass over the video
src/geometry.py        undistortion (division and OpenCV models), square-error metric, camera-from-homography
src/floor.py           static floor-marker selection, distortion fit, homography, camera pose
src/vehicle.py         vehicle-tag fixes + parallax, combine + smooth
src/world.py           2D rigid/similarity fits, the shared world frame, map anchoring
src/timecode.py        clip start time from the GoPro timecode
src/crossguide.py      search one camera's frames where the other saw the car
src/fuse.py            two-camera fusion: Kalman + RTS smoother, tag-height check
src/calcache.py        save/load calibrations, camera-moved check
src/calibrate_charuco.py  lens calibration from a ChArUco clip
src/outputs.py         single-camera CSV, overlay video, summary
src/dual_outputs.py    fused CSV, top-down plot, side-by-side overlay
src/gpu.py             OpenCL switch for OpenCV
src/aruco_config.py    the ArUco dictionary
```

`run_tracking.sh` runs `track_ground.py` on single clips (one camera only), into
`output/<video-name>/`.

### Defaults and why

- `--stride 2`: every 2nd frame (30 Hz at 59.94 fps).
- `--smooth 15`: Savitzky-Golay window in analysed frames (single-camera track).
- `--vehicle-ids 20`: only tag 20 is on the car.
- `--tag-heights 20:198`: the session-4 tag, 198 mm above the floor (tape,
  all corners). Measure it for your mounting.
- `--cam-height-mm` is **optional** and only changes what the recovered height
  is printed against. Pass the tape measurement or nothing, not a guess.
- **ArUco fixes only.** A median-background blob fallback was removed on
  18 Oct 2026: with a properly mounted tag it was not needed, and a bad blob fill
  *corrupted the ~7 ArUco frames either side of it* through the smoothing window.

### Checking against the car's onboard pose (test)

`src/BagOverlayTest.py` compares a fused track with the pose the car logged in a
rosbag (read with `rosbags`, no ROS install needed). It does NOT fit one track
onto the other: the onboard pose is taken into our world frame through the
boundary-to-SLAM-map fit from `SlamCompareTest.py`.

```bash
python src/BagOverlayTest.py --track <fused.csv> --bag <bag dir> \
    --slam-fit ../pipeline_A_mapping/output/TrackMap_S4_L1_SlamTest/TrackMap_S4_L1_boundary_vs_slam.json \
    --boundary ../pipeline_A_mapping/output/TrackMap_S4_L1/TrackMap_S4_L1_boundary.csv \
    --date 2026-10-17 --time-offset <per pair> --lever 18,-68 --out "output/ROSBAG compare/<name>"
```

The car's clock was ~44 s ahead of the cameras in session 4; pass the offset as
`--time-offset`. `--lever` is the tag position on the car (mm, base_link).
Writes `bag_overlay.png`, `matched.csv` and `report.json` (median, RMS, p95, max).

---

## Typical numbers

Measured on a 24-core CPU with 32 GB RAM.

| | value |
|---|---|
| Pipeline A sparse, 577 frames, 8 threads | ~13 min |
| Pipeline A dense, 577 views, level 2 | ~14 min, 16 M points |
| Pipeline A metric scale, leave-one-marker-out hold-out (session 2) | 0.36–0.65% |
| Pipeline A boundary vs the car's SLAM map (session 4) | 21 mm median |
| Pipeline B, 71 s clip @ stride 2, one camera | ~1.5 min |
| Pipeline B vehicle fix rate, 150 mm tag with quiet zone | 83–91% |
| Pipeline B fused track vs the car's onboard pose (session 4) | 76–128 mm median per run |

Memory, not compute, is the constraint on 4K input; the caps in the runners are
there for that reason.

---

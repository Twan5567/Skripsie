**External vision system for racetrack mapping and vehicle trajectory tracking**  
pipeline_A_tracking/    fixed overhead camera  ->  vehicle trajectory (mm, floor plane)  
 pipeline_B_mapping/     handheld walk-around   ->  COLMAP sparse + OpenMVS dense cloud  
   
Pipeline A's source is split by stage:  
src/track_ground.py   command line + order of operations only  
 src/detector.py       tuned ArUco detector, ID parsing, pass 1 over the video  
 src/geometry.py       undistortion, square-error metric, camera-from-homography  
 src/floor.py          static-GCP selection, distortion fit, homography, camera pose  
 src/vehicle.py        ArUco vehicle fixes + parallax, blob fallback, combine + smooth  
 src/outputs.py        CSV, plots (currently disabled), overlay video, summary  
   
Each pipeline has the same two folders:  
capture/    put your videos here — any number of them  
 output/     one folder per video, named after it, with everything the run produced  
   
You run a pipeline by **video name**, not by path. Frames and other  
   
 intermediates are created under capture/ while a run is going and are  
   
 **deleted when it finishes**, so capture/ only ever holds the videos.  
   
 Both folders are tracked in git; their contents are not.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAUBBAwSf8GGLWDWFDY3ixgjcRZhLMNjNHdQYAwF9cq1rV/vUEAIDX7gcRXAQ2s/16gwAAAABJRU5ErkJggg==)  
**Requirements**  
**System**  
| | | |  
|-|-|-|  
| **tool** | **used by** | **notes** |   
| Python ≥ 3.10 | both |   |   
| ffmpeg / ffprobe | both | frame extraction; checking footage resolution |   
| COLMAP ≥ 3.12 | B | colmap on PATH. CPU SIFT is used; no GPU needed |   
| Docker | B | runs OpenMVS; the current user must be in the docker group (no sudo) |   
| openmvs/openmvs-ubuntu image | B | pulled automatically on first dense run |   
| systemd-run | B (optional) | memory-caps the COLMAP stage; skipped if absent |   
| opensplat-rocm7 image | B run_splat.sh only | AMD/ROCm-specific, optional |   
   
**Python** — from the repo root:  
python3 -m venv .venv  
 .venv/bin/pip install -r requirements.txt  
   
The runners find .venv/bin/python automatically. To use another  
   
 interpreter: PYTHON=/path/to/python ./run_….sh ….  
**Footage.** Both pipelines were developed on 3840×2160 @ 59.94 fps HEVC  
   
 from GoPro HERO13s. Check every new clip with ffprobe before processing —  
   
 a cloud-synced download can silently arrive at 1280×720 and the ArUco  
   
 detector will then miss every marker.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSfYxKK/kJXEkyE8WcGbCFuCLTOzVXsAAPzFsVZ3dX4cAQDgvesB/vEF9H9odtUAAAAASUVORK5CYII=)  
**The markers**  
Both pipelines rely on ArUco sheets (DICT_4X4_50) laid **flat on the**  
 **  
 floor**. IDs are allocated by role and must not be reused across roles:  
| | | |  
|-|-|-|  
| **IDs** | **role** | **size** |   
| 0–11 | ground control points (GCP), flat on the floor | 150 mm |   
| 20, 21 | vehicle tags (20 = front) | 150 mm, white quiet zone, mounted flat |   
| 30, 31 | scale-bar endpoints | 150 mm |   
   
"Side" means the outer edge of the black square. Print at 100% / actual  
   
 size, never fit-to-page; measure the printed square and put the value in  
   
 pipeline_A_tracking/src/aruco_config.py if it differs from nominal.  
   
 A vehicle tag stuck straight onto dark bodywork with no white border  
   
 detects in 4–16% of frames; the same tag with a generous quiet zone,  
   
 mounted flat, detects in 83–91%.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhwgJuUPYDMpnRgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseaHEEM+cJoFcAAAAASUVORK5CYII=)  
**Pipeline A — vehicle tracking**  
Fixed camera, roughly 3 m above the floor, looking down at the track.  
cd pipeline_A_tracking  
 cp /wherever/Mono1_4k.MP4 capture/  
 ./run_tracking.sh Mono1_4k                   # one video, by name (extension optional)  
 ./run_tracking.sh                            # every video in capture/  
 ./run_tracking.sh Mono1_4k -- --cam-height-mm 3271 --tag-heights 20:57  
   
Output in output/<video-name>/:  
| | |  
|-|-|  
| **file** | **what** |   
| <stem>_overlay.mp4 | the clip with the tracked path drawn — **look at this first** |   
| <stem>_trajectory.png | top-down plot in floor-plane mm, GCP squares drawn |   
| <stem>_distance.png | displacement vs time |   
| <stem>_trajectory.csv | per analysed frame: frame, time_s, source, px_x, px_y, mm_x, mm_y, aruco_px_x, aruco_px_y |   
| <stem>.log | everything the script printed |   
   
(The two PNG plots are currently disabled in outputs.py pending a rework.)  
**What it does.** Detects every ArUco marker per frame; identifies the  
   
 static floor GCPs; builds the ground-plane homography from the 150 mm  
   
 sheets and refits it on all GCP corners; fits radial distortion from the  
   
 sheets themselves; recovers the focal length and **camera height** from the  
   
 homography (Zhang's constraints) — that recovered height against a  
   
 tape-measured one is the pipeline's self-check; corrects the parallax of  
   
 the elevated vehicle tag; smooths per contiguous run (Savitzky-Golay).  
**Console output** ends with a block like:  
camera height     : 3002 mm recovered from the homography  
 vehicle ArUco fixes : 2829/3379 (83.7%)  
 located total       : 2829/3379 (83.7%)  
 scale check (rms)   : 7.38 mm on a 150 mm sheet (4.92%)  
   
**Defaults and why**  
- --stride 2 — every 2nd frame (30 Hz at 59.94 fps).  
- --smooth 15 — Savitzky-Golay window in analysed frames.  
- --no-fallback — **ArUco fixes only.** The median-background blob  
   
 fallback is off. With a properly mounted tag it is not needed, and a bad  
   
 blob fill *corrupts the ~7 ArUco frames either side of it* through the  
   
 smoothing window (the CSV then shows aruco20 rows with impossible steps  
   
 while aruco_px stays smooth). Pass --fallback to re-enable it for a  
   
 clip with long genuine dropouts, and compare px_* to aruco_px_*  
   
 afterwards.  
- --cam-height-mm is **optional** and only changes what the recovered  
   
 height is printed against. Pass the tape measurement or nothing — not a  
   
 guess.  
- --tag-heights 20:57,21:67 (mm above the floor) drives the parallax  
   
 correction. Measure them for your mounting.  
**Reading the CSV.** source is aruco20 / aruco21 / blob / empty.  
   
 px_* and mm_* are the smoothed, merged track. aruco_px_* is the raw  
   
 detection and is the ground truth for whether a frame was really measured.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSeYxKS/kJkED6bwYAVvImwJtszMVu0BAPAXx1rd1fn1BACA164HHDwF+DpPyKwAAAAASUVORK5CYII=)  
**Pipeline B — track mapping**  
Walk around the track with the camera, slowly, filming the floor and  
   
 barriers. One or more clips of the same scene go into **one** model.  
cd pipeline_B_mapping  
 cp /wherever/GX011488.MP4 /wherever/GX011489.MP4 capture/  
   
 ./run_mapping.sh GX011488                    # one clip            -> output/GX011488/  
 ./run_mapping.sh GX011488 GX011489           # two walks, ONE model -> output/GX011488_GX011489/  
 NAME=TrackMap ./run_mapping.sh GX011488 GX011489      # pick the model name yourself  
   
 # recommended on a new scene: sparse first, read the component count, then dense  
 SPARSE_ONLY=1 ./run_mapping.sh GX011488 GX011489  
 grep "components:" output/GX011488_GX011489/GX011488_GX011489.log  
 ./run_dense.sh GX011488_GX011489             # sparse is kept in output/, only dense runs  
   
 # fragmented (components > 1)?  redo the sparse matching every pair  
 EXHAUSTIVE=1 FORCE=1 ./run_mapping.sh GX011488 GX011489  
   
 # optional Gaussian splat (AMD ROCm OpenSplat image); -d 3 for ~600 4K frames  
 ./run_splat.sh GX011488_GX011489 3  
   
Output in output/<NAME>/:  
| | |  
|-|-|  
| **file** | **what** |   
| <NAME>_topdown.png | sparse cloud + camera path on the RANSAC floor plane, with registration stats in the title |   
| <NAME>_sparse.ply | sparse cloud, largest component |   
| <NAME>_dense.ply | OpenMVS dense cloud. Open in CloudCompare, set Colors → RGB |   
| sparse/ | the COLMAP model itself (cameras/images/points3D.bin) — what run_dense.sh, run_splat.sh and any later metric scaling read |   
| sources.txt | which clips built this model |   
| <NAME>.log | everything every tool printed |   
   
Frames and the COLMAP database live in capture/<NAME>_work/ only while a  
   
 run is going, and are **deleted when it completes** (kept only if it fails,  
   
 so you can look). run_dense.sh and run_splat.sh re-extract the frames  
   
 from sources.txt — about 15 s for a 4K clip — and delete them again.  
   
 Consequence: an EXHAUSTIVE=1 re-match redoes feature extraction too.  
**Stages.** extract_frames.py (ffmpeg, 4 fps) → colmap_pipeline.py  
   
 (SIFT → sequential or exhaustive matcher → incremental mapper) → pick the  
   
 largest component → plot_topdown.py → dense_reconstruction.py  
   
 (colmap image_undistorter → OpenMVS InterfaceCOLMAP →  
   
 DensifyPointCloud in Docker).  
**Settings and why** (each one is the result of a failure)  
- --max-image-size 3200 --peak-threshold 0.003 — what made SIFT work on  
   
 a uniform tiled floor: features per image went from ~1.7 k to 4–10 k and  
   
 registration to 100%.  
- --num-threads 8 — each SIFT worker holds a decoded 4K image and its  
   
 pyramid (~1.2 GB). 24 threads reached 28 GB and the OOM killer took the  
   
 session. Do not set this to "all cores" on 4K input.  
- --camera-model OPENCV --single-camera — one GoPro, shared intrinsics.  
- OpenMVS --resolution-level 2 (quarter-res depth maps), --number-views 5,  
   
 16 threads, 24 GB Docker cap. Fusion holds every depth map resident at  
   
 once: half-res × ~450 views needs ~18 GB, quarter-res ~14 GB. The runner  
   
 falls back to level 3 if level 2 dies.  
- **Largest component, not ** **sparse/0** **.** COLMAP writes one folder per  
   
 connected component and does not sort by size; sparse/0 has been a  
   
 2-image fragment. All three runners pick the biggest and say so when the  
   
 model is fragmented — that also tells you the walk lost tracking.  
- **Hard links, not symlinks**, for the combined frame set: symlinks do not  
   
 resolve inside the OpenMVS container.  
**Reading the result.** The top-down title reports registered/total  
   
 frames, points, reprojection error and floor-inlier fraction. A healthy  
   
 walk registers 100% in one component with ≥ 95% floor inliers. The camera  
   
 path should close on itself. The log's Depth-maps fused: N depth-maps  
   
 line should equal the registered count — a low fusion rate means a starved  
   
 sparse model, not a dense-stage problem.  
**Units.** Every .ply is in SfM units, not millimetres. Metric scaling  
   
 from the GCP sheets is a separate step and is not part of this repo yet.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAALUlEQVR4nO3OQQ0AIAwEsAMlSJ0UrOFkGngRklZBR1WtJDsAAPzizNcDAADuNcKwAyU+nb+5AAAAAElFTkSuQmCC)  
**Typical numbers**  
| | |  
|-|-|  
|   | **value** |   
| Pipeline A, 71 s clip @ stride 2 | ~1.5 min |   
| Pipeline A vehicle fix rate, 150 mm tag with quiet zone | 83–91% |   
| Pipeline A recovered camera height, repeatability | ~55–70 mm spread |   
| Pipeline B sparse, 577 frames, 8 threads | ~13 min |   
| Pipeline B dense, 577 views, level 2 | ~14 min, 16 M points |   
   
Measured on a 24-core CPU with 32 GB RAM. Memory, not compute, is the  
   
 constraint on 4K input; the caps in the runners are there for that reason.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhwgJWEPcbJpnRgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseaIkEMIPgIvAAAAAASUVORK5CYII=)  
**Not in this repo**  
Metric scaling of the clouds (Umeyama alignment on the GCP sheets), track  
   
 boundary extraction (RANSAC plane + DBSCAN), two-camera fusion, and the  
   
 learned-feature (GLUEMAP / SuperPoint) mapping variants.  

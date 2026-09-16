#!/usr/bin/env bash
# Pipeline B — track mapping: video -> frames -> COLMAP sparse -> OpenMVS dense.
#
#   ./run_mapping.sh <name> <walk1.MP4> [<walk2.MP4> ...]
#
#   ./run_mapping.sh TrackMap_S2 ~/footage/GX011488.MP4 ~/footage/GX011489.MP4
#   SPARSE_ONLY=1 ./run_mapping.sh TrackMap_S2 walk1.MP4 walk2.MP4   # stop after sparse
#   EXHAUSTIVE=1  ./run_mapping.sh TrackMap_S2 walk1.MP4 walk2.MP4   # if it fragmented
#
# Several clips of the SAME scene are combined into ONE model: each is extracted
# to frames/_clips/<clip-stem>/ and hard-linked into frames/<name>/ with a
# w1_/w2_/... prefix so the source walk stays visible in every filename. Hard
# links, not symlinks — symlinks do not resolve inside the OpenMVS container.
#
# Output in output/<name>/:
#   <name>_sparse.ply     sparse cloud (largest COLMAP component)
#   <name>_topdown.png    sparse cloud + camera path on the RANSAC floor plane
#   <name>_dense.ply      OpenMVS dense cloud   (open in CloudCompare; Colors->RGB)
# Intermediates: frames/<name>/, colmap/<name>/{database.db, sparse/N/, dense/}
# Log: run_mapping.log (appended). Everything is in SfM units — NOT metric.
#
# Deliberate settings, each the result of something failing first:
#   fps 4                measured inter-frame motion 1-5% of image width — far
#                        inside the <10% rule for sequential matching
#   --num-threads 8      each COLMAP SIFT worker holds a decoded 4K image plus
#                        its pyramid (~1.2 GB); 24 threads hit 28 GB and the OOM
#                        killer took the whole session
#   --max-image-size 3200 / --peak-threshold 0.003
#                        the two settings that made SIFT work on a uniform tiled
#                        floor: features/image 1.7k -> 4.3k+, registration -> 100%
#   --camera-model OPENCV --single-camera
#                        one GoPro, one set of intrinsics shared by every frame
#   resolution-level 2   OpenMVS fusion holds every depth map resident at once;
#                        half-res x ~450 views needs ~18 GB, quarter-res ~14 GB.
#                        Falls back to level 3 if level 2 dies.
#   largest component    COLMAP writes sparse/0, sparse/1, ... one per connected
#                        component and does NOT sort by size. sparse/0 has been
#                        a 2-image fragment. Always pick the biggest. Say so when
#                        the model is fragmented — it means tracking was lost.
#
# Recommended: run with SPARSE_ONLY=1 first, read the "components:" line in the
# log, and only then run again (sparse is skipped, dense runs) — so a fragmented
# model is caught in ~15 min instead of after an hour of dense.
set -u

NAME="${1:?usage: run_mapping.sh <name> <clip.MP4> [<clip2.MP4> ...]}"; shift
[ $# -gt 0 ] || { echo "give at least one clip"; exit 1; }

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in "$ROOT/../.venv/bin/python" "$ROOT/../../.venv/bin/python"; do
    [ -x "$c" ] && PY="$c" && break
  done
  PY="${PY:-python3}"
fi
FPS="${FPS:-4}"
LOG="$ROOT/run_mapping.log"
EXHAUSTIVE=${EXHAUSTIVE:-0}
SPARSE_ONLY=${SPARSE_ONLY:-0}
MATCH_FLAG=""; [ "$EXHAUSTIVE" = "1" ] && MATCH_FLAG="--exhaustive"
# memory cap for the native COLMAP stage; no-op if systemd-run is unavailable
CAP=(); command -v systemd-run >/dev/null && CAP=(systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 --unit="colmap-$NAME-$RANDOM")

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

largest_model() {   # -> path of the sparse component with the most registered images
  local best="" bestn=-1 n
  for d in "$ROOT/colmap/$NAME/sparse"/*/; do
    [ -f "$d/points3D.bin" ] || continue
    n=$(colmap model_analyzer --path "$d" 2>&1 | grep -oP "Registered images: \K[0-9]+")
    n=${n:-0}
    if [ "$n" -gt "$bestn" ]; then bestn=$n; best=$d; fi
  done
  echo "$best"
}

dense_attempt() {   # model, resolution-level -> 0 ok / 1 fail
  local model="$1" level="$2" ws="$ROOT/colmap/$NAME/dense"
  # .dmap files are resolution-specific but reused by filename
  rm -f "$ws/undistorted"/*.dmap 2>/dev/null
  "$PY" "$SRC/dense_reconstruction.py" \
      --sparse "$model" --images "$ROOT/frames/$NAME" --workspace "$ws" \
      --memory-limit 24G --max-threads 16 \
      --resolution-level "$level" --number-views 5 >>"$LOG" 2>&1
}

say "================ $NAME ================"
out="$ROOT/output/$NAME"; mkdir -p "$out"

# ---- frames: extract each clip, then combine by hard link ----
if [ -d "$ROOT/frames/$NAME" ] && [ "$(ls -1 "$ROOT/frames/$NAME" 2>/dev/null | wc -l)" -gt 0 ]; then
  say "frames already present ($(ls -1 "$ROOT/frames/$NAME" | wc -l)) — skipping extraction"
else
  mkdir -p "$ROOT/frames/$NAME" "$ROOT/frames/_clips"
  i=0
  for clip in "$@"; do
    i=$((i+1))
    [ -f "$clip" ] || { say "no such clip: $clip"; exit 1; }
    cstem="$(basename "${clip%.*}")"
    if [ ! -d "$ROOT/frames/_clips/$cstem" ] || [ "$(ls -1 "$ROOT/frames/_clips/$cstem" | wc -l)" -eq 0 ]; then
      say "extracting $cstem @ ${FPS}fps"
      "$PY" "$SRC/extract_frames.py" --video "$clip" --out-dir "$ROOT/frames/_clips" --fps "$FPS" >>"$LOG" 2>&1 \
        || { say "FRAME EXTRACTION FAILED for $clip"; exit 1; }
    fi
    for f in "$ROOT/frames/_clips/$cstem"/*; do ln -f "$f" "$ROOT/frames/$NAME/w${i}_$(basename "$f")"; done
    say "  $cstem -> $(ls -1 "$ROOT/frames/_clips/$cstem" | wc -l) frames as w${i}_*"
  done
fi
say "frames: $(ls -1 "$ROOT/frames/$NAME" | wc -l)"

# ---- sparse ----
if ls "$ROOT/colmap/$NAME/sparse"/*/points3D.bin >/dev/null 2>&1; then
  say "sparse model already present, skipping"
else
  say "sparse: feature_extractor -> $([ -n "$MATCH_FLAG" ] && echo exhaustive || echo sequential)_matcher -> mapper"
  mkdir -p "$ROOT/colmap/$NAME/sparse"
  "${CAP[@]}" "$PY" "$SRC/colmap_pipeline.py" \
      --image "$ROOT/frames/$NAME" \
      --database "$ROOT/colmap/$NAME/database.db" \
      --output "$ROOT/colmap/$NAME/sparse" \
      --max-image-size 3200 --num-threads 8 --peak-threshold 0.003 \
      --camera-model OPENCV --single-camera $MATCH_FLAG >>"$LOG" 2>&1 \
    || { say "SPARSE FAILED"; exit 1; }
fi

ncomp=$(ls -d "$ROOT/colmap/$NAME/sparse"/*/ 2>/dev/null | wc -l)
model=$(largest_model)
[ -n "$model" ] || { say "no sparse model produced"; exit 1; }
say "components: $ncomp | using sparse/$(basename "${model%/}")"
[ "$ncomp" -gt 1 ] && say "  NOTE: model is FRAGMENTED — tracking was lost somewhere; consider EXHAUSTIVE=1"
colmap model_analyzer --path "$model" 2>&1 \
  | grep -oE "(Registered images|Points|Mean track length|Mean reprojection error): .*" | tee -a "$LOG"

colmap model_converter --input_path "$model" \
    --output_path "$out/${NAME}_sparse.ply" --output_type PLY >>"$LOG" 2>&1
"$PY" "$SRC/plot_topdown.py" --sparse "$model" --out "$out/${NAME}_topdown.png" 2>&1 | tee -a "$LOG"

if [ "$SPARSE_ONLY" = "1" ]; then say "SPARSE_ONLY set — stopping before dense"; exit 0; fi

# ---- dense ----
say "dense at resolution-level 2"
if ! dense_attempt "$model" 2; then
  say "level 2 failed (likely fusion memory) — retrying at level 3"
  dense_attempt "$model" 3 || { say "DENSE FAILED at both levels"; exit 1; }
fi
ply="$ROOT/colmap/$NAME/dense/undistorted/scene_dense.ply"
if [ -f "$ply" ]; then
  cp "$ply" "$out/${NAME}_dense.ply"; chmod 644 "$out/${NAME}_dense.ply"
  say "dense OK: $(head -c 400 "$ply" | strings | grep -oP 'element vertex \K[0-9]+') points"
else
  say "dense produced no ply"; exit 1
fi
say "$NAME done -> $out"

#!/usr/bin/env bash
# Pipeline B — track mapping: capture/<video> -> frames -> COLMAP sparse -> OpenMVS dense.
#
#   ./run_mapping.sh <video-name> [<video-name> ...]
#
#   ./run_mapping.sh GX011488                 # one clip      -> output/GX011488/
#   ./run_mapping.sh GX011488 GX011489        # two walks of the SAME track combined
#                                             #               -> output/GX011488_GX011489/
#   NAME=TrackMap ./run_mapping.sh GX011488 GX011489      # choose the model name
#   SPARSE_ONLY=1 ./run_mapping.sh GX011488   # stop after sparse; run again (or run_dense.sh) for dense
#   EXHAUSTIVE=1  ./run_mapping.sh GX011488   # match every pair — for a model that fragmented
#   FORCE=1       ./run_mapping.sh GX011488   # redo sparse even if output/<NAME>/sparse exists
#
# Videos go in capture/. Name them, do not path them. The extension is optional.
#
# Output in output/<NAME>/:
#   <NAME>_topdown.png   sparse cloud + camera path on the RANSAC floor plane
#   <NAME>_sparse.ply    sparse cloud, largest component
#   <NAME>_dense.ply     OpenMVS dense cloud  (CloudCompare, Colors -> RGB)
#   sparse/              the COLMAP model itself (cameras/images/points3D.bin) — needed
#                        by run_dense.sh, run_splat.sh and any later metric scaling
#   sources.txt          which clips built it;  <NAME>.log  everything the tools printed
# Everything is in SfM units, NOT metric.
#
# Frames and all other intermediates live in capture/<NAME>_work/ while the run is
# going and are DELETED when it finishes, so capture/ only ever holds the videos.
# On a failure the work folder is kept so you can look at it.
#
# Settings, each the result of something failing first:
#   fps 4                  inter-frame motion 1-5% of image width, well inside the
#                          <10% rule for sequential matching
#   --num-threads 8        each SIFT worker holds a decoded 4K image + pyramid (~1.2 GB);
#                          24 threads reached 28 GB and the OOM killer took the session
#   --max-image-size 3200 / --peak-threshold 0.003
#                          what made SIFT work on a uniform tiled floor (1.7k -> 4-10k
#                          features/image, registration -> 100%)
#   --camera-model OPENCV --single-camera   one GoPro, shared intrinsics
#   resolution-level 2     OpenMVS fusion holds every depth map resident; quarter-res
#                          is ~14 GB for ~450 views. Falls back to level 3.
#   largest component      COLMAP writes sparse/0, sparse/1, ... and does NOT sort by
#                          size; sparse/0 has been a 2-image fragment.
set -u
source "$(dirname "$0")/lib.sh"

[ $# -gt 0 ] || { sed -n '2,16p' "$0"; exit 1; }
CLIPS=(); for n in "$@"; do c="$(resolve_clip "$n")" || exit 1; CLIPS+=("$c"); done
set_name "${CLIPS[@]}"
EXHAUSTIVE=${EXHAUSTIVE:-0}; SPARSE_ONLY=${SPARSE_ONLY:-0}; FORCE=${FORCE:-0}
MATCH_FLAG=""; [ "$EXHAUSTIVE" = "1" ] && MATCH_FLAG="--exhaustive"
CAP=(); command -v systemd-run >/dev/null && CAP=(systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 --unit="colmap-$NAME-$RANDOM")

say "================ $NAME ================"
say "clips: ${CLIPS[*]}"
write_sources "${CLIPS[@]}"
ensure_frames "${CLIPS[@]}" || exit 1

# ---- sparse ----
if [ -f "$OUT/sparse/points3D.bin" ] && [ "$FORCE" != "1" ]; then
  say "sparse model already in $OUT/sparse — skipping (FORCE=1 to redo)"
else
  rm -rf "$OUT/sparse" "$WORK/colmap/sparse" "$WORK/colmap/database.db"
  mkdir -p "$WORK/colmap/sparse"
  say "sparse: feature_extractor -> $([ -n "$MATCH_FLAG" ] && echo exhaustive || echo sequential)_matcher -> mapper"
  "${CAP[@]}" "$PY" "$SRC/colmap_pipeline.py" \
      --image "$FRAMES" \
      --database "$WORK/colmap/database.db" \
      --output "$WORK/colmap/sparse" \
      --max-image-size 3200 --num-threads 8 --peak-threshold 0.003 \
      --camera-model OPENCV --single-camera $MATCH_FLAG >>"$LOG" 2>&1 \
    || { say "SPARSE FAILED — see $LOG; $WORK kept for inspection"; exit 1; }

  ncomp=$(ls -d "$WORK/colmap/sparse"/*/ 2>/dev/null | wc -l)
  model=$(largest_model "$WORK/colmap/sparse")
  [ -n "$model" ] || { say "no sparse model produced; $WORK kept"; exit 1; }
  say "components: $ncomp | using sparse/$(basename "${model%/}")"
  [ "$ncomp" -gt 1 ] && say "  NOTE: model is FRAGMENTED — tracking was lost somewhere; consider EXHAUSTIVE=1 FORCE=1"
  mkdir -p "$OUT/sparse"; cp "$model"/*.bin "$OUT/sparse/"
fi
model="$OUT/sparse"
model_stats "$model" | tee -a "$LOG"
colmap model_converter --input_path "$model" --output_path "$OUT/${NAME}_sparse.ply" --output_type PLY >>"$LOG" 2>&1
"$PY" "$SRC/plot_topdown.py" --sparse "$model" --out "$OUT/${NAME}_topdown.png" 2>&1 | tee -a "$LOG"

if [ "$SPARSE_ONLY" = "1" ]; then
  say "SPARSE_ONLY set — stopping before dense"
  cleanup_work; say "$NAME sparse -> $OUT"; exit 0
fi

# ---- dense ----
run_dense_stage "$model" || { say "$WORK kept for inspection"; exit 1; }
cleanup_work
say "$NAME done -> $OUT"

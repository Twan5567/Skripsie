#!/usr/bin/env bash
# Dense stage only, for a model run_mapping.sh already built.
#
#   ./run_dense.sh <NAME> [<NAME> ...]        NAME as in output/<NAME>/
#
# Rebuilds the frames from the clips listed in output/<NAME>/sources.txt (they
# were deleted when the previous run finished), runs OpenMVS against
# output/<NAME>/sparse at resolution-level 2 (fallback 3), copies the ply to
# output/<NAME>/<NAME>_dense.ply, then deletes the frames again.
set -u
source "$(dirname "$0")/lib.sh"
[ $# -gt 0 ] || { sed -n '2,4p' "$0"; exit 1; }

for NAME in "$@"; do
  OUT="$OUTPUT/$NAME"; WORK="$CAPTURE/${NAME}_work"; FRAMES="$WORK/frames"; LOG="$OUT/$NAME.log"
  [ -f "$OUT/sparse/points3D.bin" ] || { say "no model at $OUT/sparse — run run_mapping.sh first"; continue; }
  say "======== dense: $NAME ========"
  read_sources || continue
  ensure_frames "${CLIPS[@]}" || continue
  model_stats "$OUT/sparse" | tee -a "$LOG"
  run_dense_stage "$OUT/sparse" || { say "$WORK kept for inspection"; continue; }
  cleanup_work
  say "$NAME dense -> $OUT"
done

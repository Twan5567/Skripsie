#!/usr/bin/env bash
# Metric scale + track boundary for a model run_mapping.sh / run_dense.sh already built.
#
#   ./run_boundary.sh <NAME> [<NAME> ...]        NAME as in output/<NAME>/
#
# Needs output/<NAME>/sparse and output/<NAME>/<NAME>_dense.ply. Rebuilds the
# frames from sources.txt (only if capture/<NAME>_work/frames is empty), then:
#   metric_scale.py      -> <NAME>_metric.json   (mm, world frame of the tracker)
#   boundary.py          -> <NAME>_boundary.{csv,dxf,json}, _boundary_topdown.png,
#                           _width.png, _walls3d.ply
#   boundary_overlay.py  -> <NAME>_boundary_overlay.mp4 (+ a .jpg still)
# then deletes the frames again. Env: WORLD_ORIGIN (1), WORLD_X (5),
# TRACKS (glob of fused trajectory CSVs for the corridor check),
# BOUNDARY_ARGS (extra boundary.py options, e.g. "--min-count 10 --bridge-gaps 2000").
set -u
source "$(dirname "$0")/lib.sh"
[ $# -gt 0 ] || { sed -n '2,4p' "$0"; exit 1; }
TRACKS="${TRACKS:-$ROOT/../pipeline_B_tracking/output/*/*_fused.csv}"

for NAME in "$@"; do
  OUT="$OUTPUT/$NAME"; WORK="$CAPTURE/${NAME}_work"; FRAMES="$WORK/frames"; LOG="$OUT/$NAME.log"
  [ -f "$OUT/sparse/points3D.bin" ] || { say "no model at $OUT/sparse — run run_mapping.sh first"; continue; }
  [ -f "$OUT/${NAME}_dense.ply" ]   || { say "no $OUT/${NAME}_dense.ply — run run_dense.sh first"; continue; }
  say "======== boundary: $NAME ========"
  read_sources || continue
  ensure_frames "${CLIPS[@]}" || continue

  say "metric scale (origin GCP ${WORLD_ORIGIN:-1}, +x to GCP ${WORLD_X:-5})"
  "$PY" "$SRC/metric_scale.py" --sparse "$OUT/sparse" --images "$FRAMES" \
      --out "$OUT/${NAME}_metric.json" --origin "${WORLD_ORIGIN:-1}" --x-to "${WORLD_X:-5}" \
      2>&1 | grep -v '^  detect [0-9]*/' | tee "$OUT/${NAME}_metric.log" | tee -a "$LOG" \
      || { say "METRIC SCALE FAILED"; continue; }
  [ -f "$OUT/${NAME}_metric.json" ] || { say "METRIC SCALE FAILED"; continue; }

  say "boundary extraction"
  "$PY" "$SRC/boundary.py" --dense "$OUT/${NAME}_dense.ply" --sparse "$OUT/sparse" \
      --metric "$OUT/${NAME}_metric.json" --out-dir "$OUT" --name "$NAME" --tracks "$TRACKS" ${BOUNDARY_ARGS:-} \
      2>&1 | tee "$OUT/${NAME}_boundary.log" | tee -a "$LOG"
  [ -f "$OUT/${NAME}_boundary.csv" ] || { say "BOUNDARY FAILED"; continue; }

  WALL_TOP=$("$PY" -c "import json;print(json.load(open('$OUT/${NAME}_boundary.json'))['wall_top_mm'])")
  say "overlay video (wall top $WALL_TOP mm)"
  "$PY" "$SRC/boundary_overlay.py" --sparse "$OUT/sparse" --images "$FRAMES" \
      --metric "$OUT/${NAME}_metric.json" --boundary "$OUT/${NAME}_boundary.csv" \
      --wall-top "$WALL_TOP" --out "$OUT/${NAME}_boundary_overlay.mp4" 2>&1 | tail -1 | tee -a "$LOG"

  cleanup_work
  say "$NAME boundary -> $OUT"
done

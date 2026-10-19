#!/usr/bin/env bash
# Pipeline B, two cameras — vehicle trajectory from two fixed cameras at opposite
# ends of the track, time-synced by their embedded timecode.
#
#   ./run_dual.sh <clipA> <clipB>                 # by name, extension optional
#   ./run_dual.sh GX010165 GX011480
#   ./run_dual.sh A-1 B-1 --start-a 23           # ignore fixes in the first 23 s of A
#   ./run_dual.sh A-1 B-1 --solve-tag-height
#   NAME=Dual1 ./run_dual.sh A-1 B-1              # choose the output folder name
#   ./run_dual.sh A-1 B-1 -- --world-origin 0 --world-x 1 --tag-heights 20:150
#
# Calibrate, then lift (floor sheets down only for the calibration pair):
#   ./run_dual.sh A-1 B-1 -- --save-cal S3_cal1          # sheets down: writes calib/S3_cal1.json
#   ./run_dual.sh A-2 B-2 -- --load-cal S3_cal1          # sheets lifted: reuses it
#   ./run_dual.sh A-3 B-3 -- --load-cal S3_cal1 --correct-motion
# Map-anchored world frame (each camera fitted to the map's sheet corners, so the
# cameras need not share a sheet and the scale comes from the map):
#   ./run_dual.sh A-4 B-4 -- --map-anchor TrackMap_S4_L1 --save-cal S4_cal1m
# Every --load-cal run checks whether each camera moved since the calibration
# (log: CAMERA-MOVED CHECK; <clip>/<clip>_movecheck.jpg, red = calibration,
# cyan = this clip). --correct-motion folds the measured movement into the fit.
#
# Videos go in capture/. Output goes to output/<A>_<B>/:
#   <A>_<B>_overlay.mp4     both views side by side, each camera's own fix (red),
#                           the fused track drawn into both (yellow) — look at this first
#   <A>_<B>_fused.csv       time_global_s, source (A/B/AB), x_mm, y_mm, xA, yA, xB, yB,
#                           disagree_mm, tagpx_A, tagpx_B, frame_A, frame_B
#   <A>_<B>_topdown.png     fused track + both cameras' fixes + GCP layout, world mm
#   <A>/  <B>/              the two single-camera runs, as run_tracking.sh would make them
#   <A>_<B>.log             everything printed: both floor fits, the world fit, the
#                           tag-height solve, coverage and A-B disagreement
#
# World frame: origin at the centre of GCP 1, +x toward the centre of GCP 5
# (override with --world-origin / --world-x after "--").
#
# Defaults: --stride 2 --smooth 15 (as run_tracking.sh) and
# --tag-heights 20:198 — the session-4 tag, 198 mm above the floor (tape, all corners).
# Override after "--" for another mounting.
# --solve-tag-height applies the tag height that best reconciles the two cameras
# (it is always reported in the log; the default keeps --tag-heights as given).
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
CAPTURE="$ROOT/capture"; OUTPUT="$ROOT/output"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in "$ROOT/../.venv/bin/python" "$ROOT/../../.venv/bin/python"; do
    [ -x "$c" ] && PY="$c" && break
  done
  PY="${PY:-python3}"
fi

names=(); extra=(); solve=()
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; extra=("$@"); break ;;
    --solve-tag-height) solve=(--solve-tag-height) ;;
    --start-a|--start-b) extra+=("$1" "$2"); shift ;;
    *) names+=("$1") ;;
  esac
  shift
done
[ ${#names[@]} -eq 2 ] || { sed -n '2,9p' "$0"; exit 1; }

clips=()
for n in "${names[@]}"; do
  base="$(basename "${n%.*}")"; found=""
  for cand in "$CAPTURE/$base.MP4" "$CAPTURE/$base.mp4" "$CAPTURE/$base.MOV" "$CAPTURE/$base.mov"; do
    [ -f "$cand" ] && { found="$cand"; break; }
  done
  [ -n "$found" ] || { echo "no clip named '$base' in $CAPTURE"; exit 1; }
  clips+=("$found")
done
a="$(basename "${clips[0]%.*}")"; b="$(basename "${clips[1]%.*}")"
NAME="${NAME:-${a}_${b}}"

mkdir -p "$OUTPUT/$NAME"
LOG="$OUTPUT/$NAME/$NAME.log"
cd "$ROOT/src"
echo "############ $NAME  A=$a B=$b  $(date +%H:%M:%S) ############" | tee "$LOG"
"$PY" track_dual.py --video-a "${clips[0]}" --video-b "${clips[1]}" --outdir "$OUTPUT" --name "$NAME" \
    --stride 2 --smooth 15 --tag-heights 20:198 "${solve[@]}" "${extra[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "############ $NAME done rc=$rc -> $OUTPUT/$NAME ############" | tee -a "$LOG"
exit "$rc"

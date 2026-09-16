#!/usr/bin/env bash
# Pipeline A — vehicle trajectory tracking from a fixed overhead camera.
#
#   ./run_tracking.sh                          # every video in capture/
#   ./run_tracking.sh Mono1_4k                 # one, by name (extension optional)
#   ./run_tracking.sh Mono1_4k Mono2_4k -- --cam-height-mm 3005 --tag-heights 20:80
#
# Videos go in capture/. Output goes to output/<video-name>/:
#   <name>_trajectory.csv    frame, time_s, source, px_x/y (smoothed, merged),
#                            mm_x/y (floor plane), aruco_px_x/y (raw detection)
#   <name>_trajectory.png    top-down floor-plane plot   (plots currently disabled)
#   <name>_distance.png      displacement over time     (plots currently disabled)
#   <name>_overlay.mp4       the clip with the tracked path drawn — look at this first
#   <name>.log               everything the script printed
# Nothing is written into capture/; this pipeline reads the video directly.
#
# Defaults baked in here (override after "--"):
#   --stride 2        analyse every 2nd frame (30 Hz effective at 59.94 fps)
#   --smooth 15       Savitzky-Golay window in analysed frames
#   --no-fallback     ArUco fixes only. The blob fallback is OFF because with a
#                     properly mounted 150 mm tag detection runs at 83-91% and a
#                     bad blob fill CORRUPTS the ~7 ArUco frames either side of it
#                     through the smoothing window. Pass --fallback to re-enable
#                     it for a clip with long genuine dropouts.
#
# --cam-height-mm is optional: the script recovers the height from the floor GCPs
# itself; the flag only makes it print the comparison. Do not pass a guess.
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

names=(); extra=(); fallback=0
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; extra=("$@"); break ;;
    --fallback) fallback=1 ;;
    *) names+=("$1") ;;
  esac
  shift
done
clips=()
if [ ${#names[@]} -eq 0 ]; then
  while IFS= read -r -d '' f; do clips+=("$f"); done \
    < <(find "$CAPTURE" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' \) -print0 | sort -z)
  [ ${#clips[@]} -gt 0 ] || { echo "no videos in $CAPTURE"; exit 1; }
else
  for n in "${names[@]}"; do
    base="$(basename "${n%.*}")"; found=""
    for cand in "$CAPTURE/$base.MP4" "$CAPTURE/$base.mp4" "$CAPTURE/$base.MOV" "$CAPTURE/$base.mov"; do
      [ -f "$cand" ] && { found="$cand"; break; }
    done
    [ -n "$found" ] || { echo "no clip named '$base' in $CAPTURE"; exit 1; }
    clips+=("$found")
  done
fi

nofb=(--no-fallback); [ "$fallback" = 1 ] && nofb=()
cd "$ROOT/src"
for clip in "${clips[@]}"; do
  stem="$(basename "${clip%.*}")"
  mkdir -p "$OUTPUT/$stem"
  echo "############ $stem  $(date +%H:%M:%S) ############" | tee "$OUTPUT/$stem/$stem.log"
  "$PY" track_ground.py --video "$clip" --outdir "$OUTPUT" \
      --stride 2 --smooth 15 "${nofb[@]}" "${extra[@]}" 2>&1 | tee -a "$OUTPUT/$stem/$stem.log"
  rc=${PIPESTATUS[0]}
  echo "############ $stem done rc=$rc -> $OUTPUT/$stem ############" | tee -a "$OUTPUT/$stem/$stem.log"
done

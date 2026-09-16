#!/usr/bin/env bash
# Pipeline A — vehicle trajectory tracking from a fixed overhead camera.
#
#   ./run_tracking.sh <clip.MP4> [more clips or a directory of clips] [-- extra track_ground.py args]
#
#   ./run_tracking.sh ~/footage/Mono1_4k.MP4
#   ./run_tracking.sh ~/footage/                       # every *.MP4 / *.mp4 in the folder
#   ./run_tracking.sh clip.MP4 -- --cam-height-mm 3005 --tag-heights 20:80
#
# Output per clip, in output/<clip-stem>/:
#   <stem>_trajectory.csv    frame, time_s, source, px_x/y (smoothed, merged),
#                            mm_x/y (floor plane), aruco_px_x/y (raw detection)
#   <stem>_trajectory.png    top-down floor-plane plot with the GCP squares drawn
#   <stem>_distance.png      displacement over time
#   <stem>_overlay.mp4       the clip with the tracked path drawn — look at this first
#
# Defaults baked in here (override after "--"):
#   --stride 2        analyse every 2nd frame (30 Hz effective at 59.94 fps)
#   --smooth 15       Savitzky-Golay window in analysed frames
#   --no-fallback     ArUco fixes only. The median-background blob fallback is
#                     OFF because, with a properly mounted 150 mm vehicle tag,
#                     detection runs at 83-91% and the fallback is no longer
#                     needed — and a bad blob fill CORRUPTS the ~7 ArUco frames
#                     either side of it through the smoothing window. Only turn
#                     it back on (pass --fallback) for clips with long genuine
#                     dropouts, and check px_x vs aruco_px_x in the CSV afterwards.
#
# --cam-height-mm is optional. The script recovers the camera height from the
# floor GCPs on its own; passing the tape-measured value only makes it print the
# comparison. Do not pass a guess — the printed "error" is then meaningless.
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in "$ROOT/../.venv/bin/python" "$ROOT/../../.venv/bin/python"; do
    [ -x "$c" ] && PY="$c" && break
  done
  PY="${PY:-python3}"
fi

clips=(); extra=(); fallback=0
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; extra=("$@"); break ;;
    --fallback) fallback=1 ;;
    *) if [ -d "$1" ]; then
         while IFS= read -r -d '' f; do clips+=("$f"); done < <(find "$1" -maxdepth 1 -type f \( -iname '*.mp4' \) -print0 | sort -z)
       else
         clips+=("$1")
       fi ;;
  esac
  shift
done
[ ${#clips[@]} -gt 0 ] || { sed -n '2,12p' "$0"; exit 1; }

nofb=(--no-fallback); [ "$fallback" = 1 ] && nofb=()
mkdir -p "$ROOT/output"
cd "$ROOT/src"
for clip in "${clips[@]}"; do
  [ -f "$clip" ] || { echo "no such file: $clip"; continue; }
  stem="$(basename "${clip%.*}")"
  echo "############ $stem  $(date +%H:%M:%S) ############"
  "$PY" track_ground.py --video "$clip" --outdir "$ROOT/output" \
      --stride 2 --smooth 15 "${nofb[@]}" "${extra[@]}"
  echo "############ $stem done rc=$? -> $ROOT/output/$stem ############"
done

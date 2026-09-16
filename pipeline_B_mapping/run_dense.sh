#!/usr/bin/env bash
# Re-run ONLY the dense stage for a model that run_mapping.sh already built.
#
#   ./run_dense.sh <name> [<name2> ...]
#
# Use after SPARSE_ONLY=1, or to redo dense at a different level. Picks the
# largest COLMAP component (sparse/0 is NOT guaranteed to be it), re-exports the
# sparse .ply and top-down from that component, then runs OpenMVS at
# resolution-level 2 with a fallback to 3. Same settings as run_mapping.sh.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in "$ROOT/../.venv/bin/python" "$ROOT/../../.venv/bin/python"; do
    [ -x "$c" ] && PY="$c" && break
  done
  PY="${PY:-python3}"
fi
LOG="$ROOT/run_mapping.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

largest_model() {
  local stem="$1" best="" bestn=-1 n
  for d in "$ROOT/colmap/$stem/sparse"/*/; do
    [ -f "$d/points3D.bin" ] || continue
    n=$(colmap model_analyzer --path "$d" 2>&1 | grep -oP "Registered images: \K[0-9]+")
    n=${n:-0}
    if [ "$n" -gt "$bestn" ]; then bestn=$n; best=$d; fi
  done
  echo "$best"
}
dense_attempt() {
  local stem="$1" model="$2" level="$3" ws="$ROOT/colmap/$stem/dense"
  rm -f "$ws/undistorted"/*.dmap 2>/dev/null
  "$PY" "$SRC/dense_reconstruction.py" \
      --sparse "$model" --images "$ROOT/frames/$stem" --workspace "$ws" \
      --memory-limit 24G --max-threads 16 \
      --resolution-level "$level" --number-views 5 >>"$LOG" 2>&1
}

[ $# -gt 0 ] || { sed -n '2,5p' "$0"; exit 1; }
for stem in "$@"; do
  say "======== dense: $stem ========"
  out="$ROOT/output/$stem"; mkdir -p "$out"
  ncomp=$(ls -d "$ROOT/colmap/$stem/sparse"/*/ 2>/dev/null | wc -l)
  model=$(largest_model "$stem")
  [ -n "$model" ] || { say "no sparse model for $stem — run run_mapping.sh first"; continue; }
  stats=$(colmap model_analyzer --path "$model" 2>&1 \
          | grep -oE "(Registered images|Points|Mean track length|Mean reprojection error): .*" | tr '\n' ' ')
  say "components: $ncomp | using sparse/$(basename "${model%/}") | $stats"
  [ "$ncomp" -gt 1 ] && say "  NOTE: model is FRAGMENTED"
  rm -f "$out/${stem}_sparse.ply"
  colmap model_converter --input_path "$model" --output_path "$out/${stem}_sparse.ply" --output_type PLY >>"$LOG" 2>&1
  "$PY" "$SRC/plot_topdown.py" --sparse "$model" --out "$out/${stem}_topdown.png" 2>&1 | tee -a "$LOG"
  say "dense at resolution-level 2"
  if ! dense_attempt "$stem" "$model" 2; then
    say "level 2 failed (likely fusion memory) — retrying at level 3"
    dense_attempt "$stem" "$model" 3 || { say "DENSE FAILED for $stem"; continue; }
  fi
  ply="$ROOT/colmap/$stem/dense/undistorted/scene_dense.ply"
  if [ -f "$ply" ]; then
    cp "$ply" "$out/${stem}_dense.ply"; chmod 644 "$out/${stem}_dense.ply"
    say "dense OK: $(head -c 400 "$ply" | strings | grep -oP 'element vertex \K[0-9]+') points"
  else
    say "dense produced no ply for $stem"
  fi
done
say "DENSE PASS COMPLETE"

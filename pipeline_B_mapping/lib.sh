# Shared helpers for run_mapping.sh / run_dense.sh / run_splat.sh.  Source, don't run.
#
# Layout (all relative to this folder):
#   capture/<video>.MP4        put footage here; any number of clips
#   capture/<NAME>_work/       TRANSIENT: frames/ + COLMAP db + dense workspace.
#                              Removed when a run completes. Rebuilt on demand.
#   output/<NAME>/             the deliverables, plus sparse/ (the COLMAP model,
#                              largest component) and sources.txt (which clips
#                              built it) so later stages can rebuild frames.
#
# NAME = the video's stem, or the stems joined with "_" when several clips of
# the same scene are combined into one model (override with NAME=...).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/src"
CAPTURE="$ROOT/capture"
OUTPUT="$ROOT/output"
FPS="${FPS:-4}"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in "$ROOT/../.venv/bin/python" "$ROOT/../../.venv/bin/python"; do
    [ -x "$c" ] && PY="$c" && break
  done
  PY="${PY:-python3}"
fi

LOG=""                                  # set by set_name
say() { echo "[$(date +%H:%M:%S)] $*" | { if [ -n "$LOG" ]; then tee -a "$LOG"; else cat; fi; }; }

# resolve_clip <name-or-file> -> absolute path of capture/<name>.MP4 (any case of ext)
resolve_clip() {
  local n="$1" base cand
  base="$(basename "${n%.*}")"
  for cand in "$CAPTURE/$base.MP4" "$CAPTURE/$base.mp4" "$CAPTURE/$base.MOV" "$CAPTURE/$base.mov"; do
    [ -f "$cand" ] && { echo "$cand"; return 0; }
  done
  echo "no clip named '$base' in $CAPTURE  (have: $(ls "$CAPTURE" 2>/dev/null | grep -iE '\.(mp4|mov)$' | sed 's/\.[^.]*$//' | tr '\n' ' '))" >&2
  return 1
}

# set_name <clip> [<clip> ...]  -> sets NAME, OUT, WORK, LOG; creates OUT
set_name() {
  if [ -z "${NAME:-}" ]; then
    local stems=()
    for c in "$@"; do stems+=("$(basename "${c%.*}")"); done
    NAME="$(IFS=_; echo "${stems[*]}")"
  fi
  OUT="$OUTPUT/$NAME"
  WORK="$CAPTURE/${NAME}_work"
  FRAMES="$WORK/frames"
  mkdir -p "$OUT"
  LOG="$OUT/$NAME.log"
}

# record which clips built this model, so dense/splat can rebuild frames later
write_sources() { printf '%s\n' "$@" > "$OUT/sources.txt"; }
read_sources()  { [ -f "$OUT/sources.txt" ] || { say "no $OUT/sources.txt — run run_mapping.sh first"; return 1; }
                  mapfile -t CLIPS < "$OUT/sources.txt"; }

# ensure_frames <clip> [<clip> ...] -> $FRAMES populated (extracts only if missing)
ensure_frames() {
  if [ -d "$FRAMES" ] && [ "$(ls -1 "$FRAMES" 2>/dev/null | wc -l)" -gt 0 ]; then
    say "frames present ($(ls -1 "$FRAMES" | wc -l))"; return 0
  fi
  mkdir -p "$FRAMES" "$WORK/clips"
  local i=0 clip cstem
  for clip in "$@"; do
    i=$((i+1)); cstem="$(basename "${clip%.*}")"
    say "extracting $cstem @ ${FPS}fps"
    "$PY" "$SRC/extract_frames.py" --video "$clip" --out-dir "$WORK/clips" --fps "$FPS" >>"$LOG" 2>&1 \
      || { say "FRAME EXTRACTION FAILED for $clip"; return 1; }
    # hard links (not symlinks: those do not resolve inside the OpenMVS container),
    # prefixed so the source clip stays visible in every filename
    for f in "$WORK/clips/$cstem"/*; do ln -f "$f" "$FRAMES/w${i}_$(basename "$f")"; done
    say "  $cstem -> $(ls -1 "$WORK/clips/$cstem" | wc -l) frames as w${i}_*"
  done
  rm -rf "$WORK/clips"
  say "frames: $(ls -1 "$FRAMES" | wc -l)"
}

# largest_model <sparse-parent-dir> -> path of the component with most registered images
largest_model() {
  local best="" bestn=-1 n d
  for d in "$1"/*/; do
    [ -f "$d/points3D.bin" ] || continue
    n=$(colmap model_analyzer --path "$d" 2>&1 | grep -oP "Registered images: \K[0-9]+")
    n=${n:-0}
    if [ "$n" -gt "$bestn" ]; then bestn=$n; best=$d; fi
  done
  echo "$best"
}

model_stats() { colmap model_analyzer --path "$1" 2>&1 \
  | grep -oE "(Registered images|Points|Mean track length|Mean reprojection error): .*"; }

# dense_attempt <model> <resolution-level> -> 0 ok / 1 fail   (workspace under WORK)
dense_attempt() {
  local ws="$WORK/colmap/dense"
  rm -f "$ws/undistorted"/*.dmap 2>/dev/null       # .dmap is resolution-specific but reused by name
  "$PY" "$SRC/dense_reconstruction.py" \
      --sparse "$1" --images "$FRAMES" --workspace "$ws" \
      --memory-limit 24G --max-threads 16 \
      --resolution-level "$2" --number-views 5 >>"$LOG" 2>&1
}

# run dense with the level-3 fallback, copy the ply out. -> 0 ok / 1 fail
run_dense_stage() {
  local model="$1"
  say "dense at resolution-level 2"
  if ! dense_attempt "$model" 2; then
    say "level 2 failed (likely fusion memory) — retrying at level 3"
    dense_attempt "$model" 3 || { say "DENSE FAILED at both levels"; return 1; }
  fi
  local ply="$WORK/colmap/dense/undistorted/scene_dense.ply"
  [ -f "$ply" ] || { say "dense produced no ply"; return 1; }
  cp "$ply" "$OUT/${NAME}_dense.ply"; chmod 644 "$OUT/${NAME}_dense.ply"
  say "dense OK: $(head -c 400 "$ply" | strings | grep -oP 'element vertex \K[0-9]+') points"
}

# remove everything transient; capture/ keeps only the original videos
cleanup_work() {
  rm -rf "$WORK"
  say "removed $WORK (frames and intermediates); capture/ holds only the videos"
}

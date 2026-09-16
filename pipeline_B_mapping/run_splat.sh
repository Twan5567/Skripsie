#!/usr/bin/env bash
# Gaussian splat (OpenSplat, ROCm) for a capture already reconstructed by COLMAP.
#
#   ./run_splat.sh Capture4_4k 2        # stem, downscale factor (default 2)
#   ./run_splat.sh Capture1_4k 4
#
# Handles the three things that went wrong doing this by hand:
#
#  1. OpenSplat reads sparse/0. COLMAP writes one folder per connected
#     component and does NOT order them by size — Capture4_4k's sparse/0 is a
#     31-image fragment while sparse/1 has 558. This script finds the largest
#     component and builds a <stem>_best project pointing at it. The symlink
#     must be RELATIVE: an absolute host path does not resolve inside the
#     container, where only /workspace is mounted.
#
#  2. Every OOM in this pipeline has been input images held as float32, not
#     gaussians and not VRAM (the GPU never passed 3.8GB of 21.4GB). The cost
#     is images x width x height x 3 x 4 bytes — 93 MB PER IMAGE at 4K. The
#     default -d 1 means full resolution: 558 4K images is 51.7 GB. This script
#     computes that up front and refuses rather than dying 4 minutes in.
#
#  3. The docker invocation is long enough that pasting it into a terminal
#     splits it and silently drops arguments. Hence: a file.
set -euo pipefail

STEM="${1:?usage: run_splat.sh <capture-stem> [downscale-factor]}"
DOWN="${2:-2}"
ITERS="${3:-30000}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
FRAMES="$ROOT/frames/$STEM"
[ -d "$FRAMES" ] || { echo "no frames at $FRAMES"; exit 1; }

# --- pick the largest sparse component -----------------------------------
best=""; bestn=-1
for d in "$ROOT/colmap/$STEM/sparse"/*/; do
  [ -f "$d/points3D.bin" ] || continue
  n=$(colmap model_analyzer --path "$d" 2>&1 | grep -oP "Registered images: \K[0-9]+" || echo 0)
  if [ "${n:-0}" -gt "$bestn" ]; then bestn=${n:-0}; best="$d"; fi
done
[ -n "$best" ] || { echo "no sparse model for $STEM"; exit 1; }
ncomp=$(ls -d "$ROOT/colmap/$STEM/sparse"/*/ 2>/dev/null | wc -l)
comp=$(basename "${best%/}")
echo "sparse: $ncomp component(s); using sparse/$comp with $bestn registered images"
[ "$ncomp" -gt 1 ] && echo "  NOTE: fragmented model — OpenSplat would otherwise have used sparse/0"

proj="$ROOT/colmap/${STEM}_best"
mkdir -p "$proj/sparse"
rm -f "$proj/sparse/0"
ln -sfn "../../$STEM/sparse/$comp" "$proj/sparse/0"   # relative on purpose

# --- refuse to start a run that cannot fit --------------------------------
W=$(identify -format "%w" "$(ls -d "$FRAMES"/*.png | head -1)" 2>/dev/null || echo 3840)
H=$(identify -format "%h" "$(ls -d "$FRAMES"/*.png | head -1)" 2>/dev/null || echo 2160)
# The 2.06 factor is MEASURED, not theoretical: OpenSplat holds about two
# copies per image. A -d 2 run on 558 images died at 23.0GB having loaded ~460
# of them = 48.9 MB/image, against 23.7 MB for a single float32 tensor. The
# completed -d 4 run corroborates it (predicted 6.5GB, observed 7.7GB RSS).
# Without this factor the guard passes runs that cannot possibly fit.
imgs_gb=$(awk -v n="$bestn" -v w="$W" -v h="$H" -v d="$DOWN" \
  'BEGIN{printf "%.1f", 2.06*n*(w/d)*(h/d)*3*4/1073741824}')
avail_gb=$(awk '/^MemAvailable:/{printf "%.1f",$2/1048576}' /proc/meminfo)
cap_gb=$(awk -v i="$imgs_gb" -v a="$avail_gb" 'BEGIN{c=i*1.6+4; if(c<8)c=8; if(c>a-4)c=a-4; printf "%.0f", c}')

echo "images at -d $DOWN: ${W}x${H} -> $((W/DOWN))x$((H/DOWN)), ${imgs_gb} GB resident (measured 2.06x float32)"
echo "host available ${avail_gb} GB; container cap ${cap_gb} GB"
awk -v i="$imgs_gb" -v a="$avail_gb" 'BEGIN{exit !(i > a-6)}' && {
  echo "REFUSING: ${imgs_gb} GB of images leaves too little headroom on ${avail_gb} GB."
  echo "          Use a larger downscale factor, e.g. ./run_splat.sh $STEM $((DOWN+1))"
  exit 1
}

out="$ROOT/output/$STEM/splat_d${DOWN}"
mkdir -p "$out"
echo "output -> $out"
echo

exec docker run --rm --device /dev/kfd --device /dev/dri \
  --memory "${cap_gb}g" --memory-swap "${cap_gb}g" \
  --user "$(id -u):$(id -g)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  -v "$ROOT:/workspace" \
  opensplat-rocm7 /code/build/opensplat \
    "/workspace/colmap/${STEM}_best" \
    --colmap-image-path "/workspace/frames/$STEM" \
    -o "/workspace/output/$STEM/splat_d${DOWN}/${STEM}_splat_d${DOWN}.ply" \
    -n "$ITERS" -d "$DOWN" --save-every 5000

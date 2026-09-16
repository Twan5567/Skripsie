#!/usr/bin/env bash
# Gaussian splat (OpenSplat, AMD ROCm image) for a model run_mapping.sh already built.
#
#   ./run_splat.sh <NAME> [downscale-factor] [iterations]
#   ./run_splat.sh GX011488_GX011489 3          # -d 3 for ~600 4K frames, -d 4 for more
#
# Rebuilds the frames from output/<NAME>/sources.txt, points OpenSplat at
# output/<NAME>/sparse, writes output/<NAME>/splat_d<D>/, then deletes the frames.
#
# Refuses to start a run that cannot fit. Every OOM in this pipeline has been the
# input images held as float32, not gaussians and not VRAM: the cost is
# 2.06 x images x (w/d) x (h/d) x 3 x 4 bytes. The 2.06 is MEASURED — OpenSplat
# holds about two copies per image (a -d 2 run on 558 images died at 23 GB
# having loaded ~460 of them). OpenSplat's DEFAULT -d 1 is full resolution:
# 558 4K images = 106 GB.
#
# Needs: docker image "opensplat-rocm7", and the render + video groups passed
# in, or torch silently trains on CPU.
set -u
source "$(dirname "$0")/lib.sh"
NAME="${1:?usage: run_splat.sh <NAME> [downscale-factor] [iterations]}"
DOWN="${2:-3}"; ITERS="${3:-30000}"
OUT="$OUTPUT/$NAME"; WORK="$CAPTURE/${NAME}_work"; FRAMES="$WORK/frames"; LOG="$OUT/$NAME.log"
[ -f "$OUT/sparse/points3D.bin" ] || { say "no model at $OUT/sparse — run run_mapping.sh first"; exit 1; }

say "======== splat: $NAME  -d $DOWN  $ITERS iters ========"
read_sources || exit 1
ensure_frames "${CLIPS[@]}" || exit 1
nimg=$(model_stats "$OUT/sparse" | grep -oP "Registered images: \K[0-9]+")

# OpenSplat wants a COLMAP project dir with sparse/0; copy the (small) model in.
proj="$WORK/splat_proj"; mkdir -p "$proj/sparse/0"; cp "$OUT/sparse"/*.bin "$proj/sparse/0/"

# --- refuse to start a run that cannot fit --------------------------------
first="$(ls "$FRAMES"/*.png | head -1)"
read -r W H < <("$PY" -c "import cv2,sys;im=cv2.imread(sys.argv[1]);print(im.shape[1],im.shape[0])" "$first")
imgs_gb=$(awk -v n="$nimg" -v w="$W" -v h="$H" -v d="$DOWN" 'BEGIN{printf "%.1f", 2.06*n*(w/d)*(h/d)*3*4/1073741824}')
avail_gb=$(awk '/^MemAvailable:/{printf "%.1f",$2/1048576}' /proc/meminfo)
cap_gb=$(awk -v i="$imgs_gb" -v a="$avail_gb" 'BEGIN{c=i*1.6+4; if(c<8)c=8; if(c>a-4)c=a-4; printf "%.0f", c}')
say "images at -d $DOWN: ${W}x${H} -> $((W/DOWN))x$((H/DOWN)), ${nimg} views, ${imgs_gb} GB resident; host available ${avail_gb} GB; container cap ${cap_gb} GB"
awk -v i="$imgs_gb" -v a="$avail_gb" 'BEGIN{exit !(i > a-6)}' && {
  say "REFUSING: ${imgs_gb} GB of images leaves too little headroom. Try ./run_splat.sh $NAME $((DOWN+1))"
  cleanup_work; exit 1
}

sout="$OUT/splat_d${DOWN}"; mkdir -p "$sout"
docker run --rm --device /dev/kfd --device /dev/dri \
  --memory "${cap_gb}g" --memory-swap "${cap_gb}g" \
  --user "$(id -u):$(id -g)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  -v "$ROOT:/workspace" \
  opensplat-rocm7 /code/build/opensplat \
    "/workspace/capture/${NAME}_work/splat_proj" \
    --colmap-image-path "/workspace/capture/${NAME}_work/frames" \
    -o "/workspace/output/$NAME/splat_d${DOWN}/${NAME}_splat_d${DOWN}.ply" \
    -n "$ITERS" -d "$DOWN" --save-every 5000 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -eq 0 ]; then say "splat OK -> $sout"; cleanup_work; else say "SPLAT FAILED rc=$rc; $WORK kept"; fi
exit "$rc"

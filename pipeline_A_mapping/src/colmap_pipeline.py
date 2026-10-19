import argparse
import subprocess
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--database", type=Path, default=None, help="default: colmap/<video_name>/database.db")
    ap.add_argument("--output", type=Path, default=None, help="default: colmap/<video_name>/sparse")
    ap.add_argument(
        "--exhaustive", action="store_true",
        help="match every image pair instead of only nearby-in-sequence ones. "
             "Slower (O(n^2)), but recovers chains sequential_matcher misses on "
             "low-texture scenes (e.g. hallways) where frame-to-frame matching "
             "breaks down. Fine for small datasets (dozens of frames).",
    )
    ap.add_argument(
        "--gpu", action="store_true",
        help="use SiftGPU (OpenGL, works on this AMD card) for extraction/matching "
             "instead of CPU. Only works from a real display in a plain terminal — "
             "crashes headless (SSH) or from VS Code's snap-confined terminal. "
             "Does not affect the mapper stage, which is always CPU regardless.",
    )
    ap.add_argument(
        "--max-image-size", type=int, default=1600,
        help="cap on the larger image dimension for SIFT. 1600 was chosen when "
             "this machine had 15GB of RAM and six cores; on a larger host raise "
             "it (2400-3840) — more resolution is exactly what finds features on "
             "low-texture floors, which is what starved the V4 dense stage.",
    )
    ap.add_argument(
        "--num-threads", type=int, default=8,
        help="SIFT extraction threads. NOT 'all cores': each worker holds a full "
             "decoded 4K image plus its scale pyramid, ~1.2GB. 24 threads peaked "
             "at 28.1GB on a 32GB host and systemd-oomd killed the session "
             "(7 Sep 2026). Budget ~1.2GB per thread and leave room for the "
             "desktop.",
    )
    ap.add_argument(
        "--peak-threshold", type=float, default=None,
        help="SIFT detection threshold (COLMAP default 0.00667). Lower finds more "
             "keypoints on low-contrast surfaces — the same knob that, on the "
             "SuperPoint side, turned a 2.8x threshold drop into 2.2x the 3D "
             "points. Try 0.003 on blank floors.",
    )
    ap.add_argument(
        "--camera-model", default=None,
        help="e.g. OPENCV_FISHEYE for a GoPro; default lets COLMAP choose.",
    )
    ap.add_argument("--single-camera", action="store_true",
                    help="all frames come from one physical camera — share intrinsics")
    args = ap.parse_args()
    use_gpu = "1" if args.gpu else "0"

    video_name = args.image.name
    if args.database is None:
        args.database = Path("colmap") / video_name / "database.db"
    if args.output is None:
        args.output = Path("colmap") / video_name / "sparse"

    args.database.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    print("running feature_extractor...")
    
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(args.database),
        "--image_path", str(args.image),
        "--SiftExtraction.use_gpu", use_gpu,
        # 4K video frames blow up RAM: downscale for extraction (no quality
        # loss that matters for SIFT) and bound the number of concurrent
        # image buffers. Videos already smaller than 1600px are unaffected.
        "--SiftExtraction.max_image_size", str(args.max_image_size),
        "--SiftExtraction.num_threads", str(args.num_threads),
    ]
    if args.peak_threshold is not None:
        cmd += ["--SiftExtraction.peak_threshold", str(args.peak_threshold)]
    if args.camera_model:
        cmd += ["--ImageReader.camera_model", args.camera_model]
    if args.single_camera:
        cmd += ["--ImageReader.single_camera", "1"]
    subprocess.run(cmd, check=True)

    matcher = "exhaustive_matcher" if args.exhaustive else "sequential_matcher"
    print(f"running {matcher}...")

    cmd = [
        "colmap", matcher,
        "--database_path", str(args.database),
        "--SiftMatching.use_gpu", use_gpu,
    ]
    subprocess.run(cmd, check=True)

    print("running mapper...")

    cmd = [
        "colmap", "mapper",
        "--database_path", str(args.database),
        "--image_path", str(args.image),
        "--output_path", str(args.output),
        "--Mapper.num_threads", str(args.num_threads),
    ]
    subprocess.run(cmd, check=True)

    print(f"done — sparse model written under {args.output}/")


if __name__ == "__main__":
    main()
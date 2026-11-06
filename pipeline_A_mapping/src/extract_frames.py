import argparse
import subprocess
from pathlib import Path


# the clip as out_root/<stem>/frame_%06d.png at `fps` frames per second
def extract_video(video_path: Path, out_root: Path, fps: int, start: float = 0.0) -> Path:
    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    # -ss before -i: seek to the start time, then decode from there, so frame 1
    # is the first frame at or after `start` (the walk often begins mid-clip).
    seek = ["-ss", f"{start:g}"] if start > 0 else []
    subprocess.run(["ffmpeg", *seek, "-i", str(video_path), "-vf", f"fps={fps}",
                    str(out_dir / "frame_%06d.png")], check=True)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("frames"))
    ap.add_argument("--fps", type=int, required=True)
    ap.add_argument("--start", type=float, default=0.0, help="skip this many seconds of the clip")
    args = ap.parse_args()
    out_dir = extract_video(args.video, args.out_dir, args.fps, args.start)
    print(f"{len(list(out_dir.glob('*.png')))} frames written to {out_dir}/")


if __name__ == "__main__":
    main()
